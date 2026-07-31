"""
试点编排器：MinIO 清洗 -> 规范化 -> 分词 -> 清单。

将少量标的与日期的真实数据路径串联：

1. ``pylob`` 按 (date, symbol) 重建沪深订单簿并写出 ``events.parquet``
   （复用 ``build_clean_dataset``）。
2. 将事件转为 cn_l2_v1 规范分片。
3. 仅在训练日期上拟合全局分箱，保存 ``vocab.json``。
4. 确定性分词全部 shard。
5. 构建时间切分清单。

随后运行 ``python -m quant_fm.pretrain.train --config ...`` 进行预训练。

MinIO 凭据见 ``docs/data/minio_setup.md`` 与 ``quant_fm/scripts/minio_env.example.sh``：

- 读原始 L2：``zeus-cn-quote`` @ ``192.168.2.11:9000``（代码默认，无需配）
- 写产物：``model-cache`` @ ``192.168.2.11:9100``（上传脚本自动用）
- 只需配置凭据，见 ``quant_fm/scripts/minio_env.example.sh``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pylob.pipeline.config import PipelineConfig
from pylob.pipeline.workflow import build_clean_dataset

from quant_fm.lob_rebuild.export_events import canonicalize_clean_dir
from quant_fm.manifest.build_manifest import build_manifest
from quant_fm.scripts.minio_config import describe_config, load_read_config, read_bucket
from quant_fm.tokenizer.fit_bins import fit_bins
from quant_fm.tokenizer.tokenize_events import assert_no_leakage, tokenize_path

logger = logging.getLogger(__name__)


def clean_one_day(
    date: str,
    symbols: tuple[str, ...],
    market: str,
    clean_dir: Path,
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
) -> None:
    """执行试点数据流水线直至可训练清单就绪。"""
    workdir = Path(workdir)
    clean_dir = workdir / "clean"
    events_dir = workdir / "events"
    tokens_dir = workdir / "tokens"
    data_dir = workdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for date in dates:
        if not skip_clean:
            clean_one_day(date, symbols, market, clean_dir / date)
        canonicalize_clean_dir(
            clean_dir / date,
            events_dir,
            date=date,
            markets=(market,),
            symbols=symbols,
        )

    train_paths = [p for p in events_dir.rglob("*.parquet") if p.stem <= train_end]
    vocab = fit_bins(
        train_paths,
        n_bins=n_bins,
        fit_dates=[d for d in dates if d <= train_end],
    )
    vocab_path = data_dir / "vocab.json"
    vocab.save(vocab_path)
    assert_no_leakage(
        vocab,
        [d for d in dates if train_end < d <= val_end],
        [d for d in dates if d > val_end],
    )

    for p in events_dir.rglob("*.parquet"):
        tokenize_path(p, tokens_dir / p.relative_to(events_dir), vocab)

    manifest = build_manifest(
        tokens_dir,
        train_end=train_end,
        val_end=val_end,
        markets=(market,),
        vocab_path=str(vocab_path),
    )
    manifest.save(data_dir / "manifest.json")
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
    )


if __name__ == "__main__":
    main()
