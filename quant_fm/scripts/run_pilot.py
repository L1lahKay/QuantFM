"""
试点编排器：MinIO 清洗 -> 规范化 -> 分词 -> 清单。

将少量标的与日期的真实数据路径串联：

1. ``pylob`` 按 (date, symbol) 重建沪深订单簿并写出 ``events.parquet``
   （复用 ``build_clean_dataset``）。
2. 将事件与真实逐事件盘口状态转为 ``cn_l2_v2`` 规范分片。
3. 仅在训练日期上拟合冻结 FieldSpec，保存 ``vocab_v2.json``。
4. 确定性生成窄 token + Q16 scalar 分片。
5. 构建时间切分清单。

默认生产 V2；仅兼容性回放可显式传 ``--data-version v1``。

随后运行 ``python -m quant_fm.pretrain.train --config ...`` 进行预训练。

MinIO 凭据见 ``docs/data/minio_setup.md`` 与 ``quant_fm/scripts/minio_env.example.sh``：

- 读原始 L2：``zeus-cn-quote`` @ ``192.168.2.11:9000``（代码默认，无需配）
- 写产物：``model-cache`` @ ``192.168.2.11:9100``（上传脚本自动用）
- 只需配置凭据，见 ``quant_fm/scripts/minio_env.example.sh``
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pylob.pipeline.config import PipelineConfig
from pylob.pipeline.workflow import build_clean_dataset

from quant_fm.lob_rebuild.export_events import canonicalize_clean_dir
from quant_fm.manifest.build_manifest import build_manifest
from quant_fm.schema.cn_l2_v1 import SCHEMA_VERSION as V1_SCHEMA_VERSION
from quant_fm.schema.cn_l2_v2 import SCHEMA_VERSION as V2_SCHEMA_VERSION
from quant_fm.scripts.minio_config import describe_config, load_read_config, read_bucket
from quant_fm.tokenizer.field_spec import FULL_FIELD_SPECS_V2
from quant_fm.tokenizer.fit_bins import fit_bins
from quant_fm.tokenizer.fit_bins_v2 import fit_vocab_v2
from quant_fm.tokenizer.tokenize_events import assert_no_leakage, tokenize_path
from quant_fm.tokenizer.tokenize_events_v2 import (
    assert_no_leakage_v2,
    tokenize_path_v2,
)

logger = logging.getLogger(__name__)


def clean_one_day(
    date: str,
    symbols: tuple[str, ...],
    market: str,
    clean_dir: Path,
    *,
    capture_book_state: bool = True,
) -> None:
    """对单日运行 PyLOB 清洗流程，输出到 ``clean_dir``。"""
    cfg = PipelineConfig(
        bucket=read_bucket(),
        trade_prefix="",
        order_prefix="",
        output_dir=clean_dir,
        symbols=symbols,
        market=market,
        layout="zeus_default",
        date=date.replace("-", "."),  # zeus 布局使用 YYYY.MM.DD
        capture_book_state=capture_book_state,
    )
    build_clean_dataset(load_read_config(), cfg)


def run(
    *,
    dates: list[str],
    symbols: tuple[str, ...],
    market: str,
    workdir: Path,
    train_end: str,
    val_end: str,
    n_bins: int,
    skip_clean: bool,
    data_version: str = "v2",
    v2_max_samples_per_field: int = 5_000_000,
    v2_seed: int = 0,
) -> None:
    """执行试点数据流水线直至可训练清单就绪。"""
    workdir = Path(workdir)
    clean_dir = workdir / "clean"
    events_dir = workdir / "events"
    tokens_dir = workdir / "tokens"
    data_dir = workdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    schema_version = (
        V2_SCHEMA_VERSION if data_version == "v2" else V1_SCHEMA_VERSION
    )

    for date in dates:
        if not skip_clean:
            clean_one_day(
                date,
                symbols,
                market,
                clean_dir / date,
                capture_book_state=data_version == "v2",
            )
        canonicalize_clean_dir(
            clean_dir / date,
            events_dir,
            date=date,
            markets=(market,),
            symbols=symbols,
            schema_version=schema_version,
            strict=True,
        )

    train_paths = [p for p in events_dir.rglob("*.parquet") if p.stem <= train_end]
    fit_dates = [d for d in dates if d <= train_end]
    if data_version == "v2":
        vocab = fit_vocab_v2(
            train_paths,
            field_specs=FULL_FIELD_SPECS_V2,
            max_samples_per_field=v2_max_samples_per_field,
            fit_dates=fit_dates,
            seed=v2_seed,
        )
        vocab_path = data_dir / "vocab_v2.json"
    else:
        vocab = fit_bins(
            train_paths,
            n_bins=n_bins,
            fit_dates=fit_dates,
        )
        vocab_path = data_dir / "vocab.json"
    vocab.save(vocab_path)
    val_dates = [d for d in dates if train_end < d <= val_end]
    test_dates = [d for d in dates if d > val_end]
    if data_version == "v2":
        assert_no_leakage_v2(vocab, val_dates, test_dates)
    else:
        assert_no_leakage(vocab, val_dates, test_dates)

    for p in sorted(events_dir.rglob("*.parquet")):
        destination = tokens_dir / p.relative_to(events_dir)
        if data_version == "v2":
            tokenize_path_v2(p, destination, vocab)
        else:
            tokenize_path(p, destination, vocab)

    manifest = build_manifest(
        tokens_dir,
        train_end=train_end,
        val_end=val_end,
        markets=(market,),
        vocab_path=str(vocab_path),
    )
    manifest.save(data_dir / "manifest.json")
    if data_version == "v2":
        from quant_fm.scripts.audit_v2_artifacts import audit_v2_artifacts

        audit = audit_v2_artifacts(workdir, sample_shards=12, full_path_check=True)
        audit_path = workdir / "artifact_audit.json"
        audit_path.write_text(
            json.dumps(
                audit,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if not audit["contract_ready"]:
            msg = f"V2 artifact audit failed: {audit_path}"
            raise RuntimeError(msg)
    logger.info(
        "pilot ready: vocab=%s manifest=%s", vocab_path, data_dir / "manifest.json"
    )


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", required=True, help="逗号分隔 YYYY-MM-DD")
    parser.add_argument("--symbols", required=True, help="逗号分隔 6 位代码")
    parser.add_argument("--market", default="SZ")
    parser.add_argument("--workdir", type=Path, default=Path("quant_fm/runs/pilot"))
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--val-end", required=True)
    parser.add_argument("--n-bins", type=int, default=32)
    parser.add_argument(
        "--data-version",
        choices=("v1", "v2"),
        default="v2",
        help="默认 v2；仅兼容旧实验时显式选择 v1",
    )
    parser.add_argument("--v2-max-samples-per-field", type=int, default=5_000_000)
    parser.add_argument("--v2-seed", type=int, default=0)
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="复用已有 clean/ 输出（跳过 MinIO）",
    )
    args = parser.parse_args()
    logger.info("MinIO config:\n%s", describe_config())
    run(
        dates=[d.strip() for d in args.dates.split(",")],
        symbols=tuple(s.strip() for s in args.symbols.split(",")),
        market=args.market.upper(),
        workdir=args.workdir,
        train_end=args.train_end,
        val_end=args.val_end,
        n_bins=args.n_bins,
        skip_clean=args.skip_clean,
        data_version=args.data_version,
        v2_max_samples_per_field=args.v2_max_samples_per_field,
        v2_seed=args.v2_seed,
    )


if __name__ == "__main__":
    main()
