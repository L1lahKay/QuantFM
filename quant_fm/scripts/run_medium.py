"""
中等规模编排器：约全量 1/10（60 交易日 × 全市场标的）。

与 ``run_pilot.py`` 相同流水线，但面向沪深全市场、按日增量处理并可选
删除中间 ``clean/``、``events/`` 以控制磁盘峰值。

默认日期列表为 ``quant_fm/data/medium_60_dates.txt``，
包含 2025 年均匀抽样的 60 天（约总日历 1/10）。
默认标的列表：``quant_fm/data/medium_symbols_{sz,sh}.txt``。

磁盘提示：60 天 × ~5100 标的，events+tokens 峰值可达数百 GB；请先 ``df -h`` 确认空间，
可用 ``--max-symbols-per-market`` 做分阶段试跑。

MinIO 读 ``9000/zeus-cn-quote``，写 ``9100/model-cache``。
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
from pathlib import Path

from pylob.pipeline.config import PipelineConfig
from pylob.pipeline.workflow import build_clean_dataset

from quant_fm.lob_rebuild.export_events import canonicalize_clean_dir
from quant_fm.manifest.build_manifest import build_manifest
from quant_fm.scripts.minio_config import load_read_config, read_bucket
from quant_fm.scripts.suggest_model_size import estimate_events, suggest
from quant_fm.tokenizer.fit_bins import fit_bins
from quant_fm.tokenizer.tokenize_events import assert_no_leakage, tokenize_path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATES = ROOT / "quant_fm/data/medium_60_dates.txt"
DEFAULT_SZ = ROOT / "quant_fm/data/medium_symbols_sz.txt"
DEFAULT_SH = ROOT / "quant_fm/data/medium_symbols_sh.txt"


def _load_lines(path: Path) -> tuple[str, ...]:
    return tuple(ln.strip() for ln in path.read_text().splitlines() if ln.strip())


def _split_dates(
    dates: list[str],
    train_end: str | None,
    val_end: str | None,
) -> tuple[str, str]:
    """若未指定切分点，按 70% / 15% / 15% 自动划分。"""
    if train_end and val_end:
        return train_end, val_end
    n = len(dates)
    if n < 3:
        msg = f"need >=3 dates, got {n}"
        raise ValueError(msg)
    train_idx = max(0, math.floor(n * 0.70) - 1)
    val_idx = max(train_idx + 1, math.floor(n * 0.85) - 1)
    return dates[train_idx], dates[val_idx]


def clean_one_day(
    date: str,
    symbols: tuple[str, ...],
    market: str,
    clean_dir: Path,
) -> None:
    """单日单市场 PyLOB 清洗。"""
    cfg = PipelineConfig(
        bucket=read_bucket(),
        trade_prefix="",
        order_prefix="",
        output_dir=clean_dir,
        symbols=symbols,
        market=market,
        layout="zeus_default",
        date=date.replace("-", "."),
    )
    build_clean_dataset(load_read_config(), cfg)


def _date_done_marker(workdir: Path, date: str) -> Path:
    return workdir / "data" / ".done" / date


def _is_date_canonicalized(events_dir: Path, date: str) -> bool:
    return any(events_dir.rglob(f"{date}.parquet"))


def run(
    *,
    dates: list[str],
    symbols_sz: tuple[str, ...],
    symbols_sh: tuple[str, ...],
    workdir: Path,
    train_end: str | None,
    val_end: str | None,
    n_bins: int,
    skip_clean: bool,
    drop_clean: bool,
    drop_events: bool,
    fit_sample_days: int | None,
    resume: bool,
    estimate_only: bool,
    upload_minio: bool = False,
    upload_tag: str = "medium",
    upload_events: bool = False,
    delete_local_after_upload: bool = False,
) -> None:
    """Run the medium-scale data preparation and optional upload pipeline."""
    workdir = Path(workdir)
    clean_dir = workdir / "clean"
    events_dir = workdir / "events"
    tokens_dir = workdir / "tokens"
    data_dir = workdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_end, val_end = _split_dates(dates, train_end, val_end)
    train_dates = [d for d in dates if d <= train_end]
    val_dates = [d for d in dates if train_end < d <= val_end]
    test_dates = [d for d in dates if d > val_end]

    total_events_est = estimate_events(
        n_dates=len(dates),
        symbols_per_day=len(symbols_sz) + len(symbols_sh),
        events_per_symbol_day=62_000,
    )
    suggestion = suggest(total_events_est)
    logger.info(
        "scale: %d dates, SZ=%d SH=%d symbols, est events≈%.2fB → model %s (~%.0fM params)",
        len(dates),
        len(symbols_sz),
        len(symbols_sh),
        total_events_est / 1e9,
        suggestion.label,
        suggestion.approx_params_m,
    )
    logger.info("splits: train_end=%s val_end=%s", train_end, val_end)
    if estimate_only:
        return

    for date in dates:
        marker = _date_done_marker(workdir, date)
        if resume and marker.exists():
            logger.info("resume: skip %s (marker exists)", date)
            continue

        day_clean = clean_dir / date
        if not skip_clean:
            if symbols_sz:
                logger.info("clean %s SZ (%d symbols)", date, len(symbols_sz))
                clean_one_day(date, symbols_sz, "SZ", day_clean)
            if symbols_sh:
                logger.info("clean %s SH (%d symbols)", date, len(symbols_sh))
                clean_one_day(date, symbols_sh, "SH", day_clean)

        canonicalize_clean_dir(
            day_clean,
            events_dir,
            date=date,
            markets=("SZ", "SH"),
            symbols=symbols_sz + symbols_sh,
        )

        if drop_clean and day_clean.exists():
            shutil.rmtree(day_clean)
            logger.info("dropped clean/%s", date)

        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("canonicalized\n")

    fit_dates = train_dates
    train_paths = [p for p in events_dir.rglob("*.parquet") if p.stem <= train_end]
    if not train_paths:
        msg = "no train event shards found under events/"
        raise RuntimeError(msg)

    if fit_sample_days is not None and fit_sample_days < len(train_dates):
        sample_dates = set(train_dates[:fit_sample_days])
        train_paths = [p for p in train_paths if p.stem in sample_dates]
        fit_dates = [d for d in train_dates if d in sample_dates]
        logger.info(
            "fit_bins on first %d train days (%d shards)",
            fit_sample_days,
            len(train_paths),
        )

    vocab = fit_bins(train_paths, n_bins=n_bins, fit_dates=fit_dates)
    vocab_path = data_dir / "vocab.json"
    vocab.save(vocab_path)
    assert_no_leakage(vocab, val_dates, test_dates)

    for p in sorted(events_dir.rglob("*.parquet")):
        rel = p.relative_to(events_dir)
        dst = tokens_dir / rel
        if dst.exists() and resume:
            continue
        tokenize_path(p, dst, vocab)
        if drop_events:
            p.unlink()

    if drop_events:
        for empty in sorted(events_dir.rglob("*"), reverse=True):
            if empty.is_dir() and not any(empty.iterdir()):
                empty.rmdir()

    manifest = build_manifest(
        tokens_dir,
        train_end=train_end,
        val_end=val_end,
        markets=("SZ", "SH"),
        vocab_path=str(vocab_path),
    )
    manifest_path = data_dir / "manifest.json"
    manifest.save(manifest_path)
    logger.info("data ready: vocab=%s manifest=%s", vocab_path, manifest_path)

    if upload_minio:
        from quant_fm.scripts.upload_to_minio import remote_uri, upload_workdir

        uri = upload_workdir(
            workdir,
            tag=upload_tag,
            include_events=upload_events,
            delete_local=delete_local_after_upload,
        )
        logger.info("MinIO upload complete: %s", uri)
        logger.info("remote: %s", remote_uri(upload_tag))


def main() -> None:
    """Parse CLI arguments and run the medium-scale pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dates-file",
        type=Path,
        default=DEFAULT_DATES,
        help="每行一个 YYYY-MM-DD",
    )
    parser.add_argument(
        "--symbols-sz-file",
        type=Path,
        default=DEFAULT_SZ,
    )
    parser.add_argument(
        "--symbols-sh-file",
        type=Path,
        default=DEFAULT_SH,
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("quant_fm/runs/medium"),
    )
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--val-end", default=None)
    parser.add_argument("--n-bins", type=int, default=32)
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument(
        "--drop-clean",
        action="store_true",
        help="canonicalize 后删除当日 clean/（推荐）",
    )
    parser.add_argument(
        "--drop-events",
        action="store_true",
        help="tokenize 后删除 events parquet（推荐）",
    )
    parser.add_argument(
        "--fit-sample-days",
        type=int,
        default=None,
        help="仅用前 N 个训练日拟合分箱（省内存；默认用全部训练日）",
    )
    parser.add_argument(
        "--max-symbols-per-market",
        type=int,
        default=None,
        help="每市场最多处理 N 只标的（试跑用）",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="只打印规模与推荐模型，不跑流水线",
    )
    parser.add_argument(
        "--upload-minio",
        action="store_true",
        help="完成后上传到 model-cache（写 :9100）",
    )
    parser.add_argument(
        "--upload-tag",
        default="medium",
        help="MinIO 路径标签，如 medium / medium_try",
    )
    parser.add_argument(
        "--upload-events",
        action="store_true",
        help="同时上传 events/（默认只上传 tokens + vocab + manifest）",
    )
    parser.add_argument(
        "--delete-local-after-upload",
        action="store_true",
        help="上传成功后删除本地 tokens/（及 events/ 若上传）",
    )
    args = parser.parse_args()

    dates = list(_load_lines(args.dates_file))
    symbols_sz = _load_lines(args.symbols_sz_file)
    symbols_sh = _load_lines(args.symbols_sh_file)
    if args.max_symbols_per_market is not None:
        symbols_sz = symbols_sz[: args.max_symbols_per_market]
        symbols_sh = symbols_sh[: args.max_symbols_per_market]

    run(
        dates=dates,
        symbols_sz=symbols_sz,
        symbols_sh=symbols_sh,
        workdir=args.workdir,
        train_end=args.train_end,
        val_end=args.val_end,
        n_bins=args.n_bins,
        skip_clean=args.skip_clean,
        drop_clean=args.drop_clean,
        drop_events=args.drop_events,
        fit_sample_days=args.fit_sample_days,
        resume=args.resume,
        estimate_only=args.estimate_only,
        upload_minio=args.upload_minio,
        upload_tag=args.upload_tag,
        upload_events=args.upload_events,
        delete_local_after_upload=args.delete_local_after_upload,
    )


if __name__ == "__main__":
    main()
