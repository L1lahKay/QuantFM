"""
中等规模编排器：约全量 1/10（60 交易日 × 全市场标的）。

与 ``run_pilot.py`` 相同流水线，但面向沪深全市场、按日增量处理并可选
删除中间 ``clean/``、``events/`` 以控制磁盘峰值。

默认产出正式 ``cn_l2_v2``：真实逐事件 pre/post 盘口、冻结
``vocab_v2.json``、窄 token + Q16 scalar、版本化 manifest 和 V2 artifact
审计。仅旧实验兼容可显式使用 ``--data-version v1``。

默认日期列表为 ``quant_fm/data/medium_60_dates.txt``，
包含 2025 年均匀抽样的 60 天（约总日历 1/10）。
默认标的列表：``quant_fm/data/medium_symbols_{sz,sh}.txt``。

磁盘提示：60 天 × ~5100 标的，events+tokens 峰值可达数百 GB；请先 ``df -h`` 确认空间，
可用 ``--max-symbols-per-market`` 做分阶段试跑。

MinIO 读 ``9000/zeus-cn-quote``，写 ``9100/model-cache``。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

from pylob.event_ordering import (
    DEFAULT_EVENT_ORDERING_VERSION,
    SUPPORTED_EVENT_ORDERING_VERSIONS,
)
from pylob.pipeline.config import PipelineConfig
from pylob.pipeline.workflow import build_clean_dataset

from quant_fm.data_coverage import coverage_receipt_path, write_coverage_receipt
from quant_fm.lob_rebuild.export_events import (
    canonicalize_and_tokenize_clean_dir,
    canonicalize_clean_dir,
)
from quant_fm.manifest.build_manifest import build_manifest
from quant_fm.schema.cn_l2_v1 import SCHEMA_VERSION as V1_SCHEMA_VERSION
from quant_fm.schema.cn_l2_v2 import SCHEMA_VERSION as V2_SCHEMA_VERSION
from quant_fm.scripts.minio_config import load_read_config, read_bucket
from quant_fm.scripts.suggest_model_size import estimate_events, suggest
from quant_fm.tokenizer.artifact_contract import assert_token_contract_matches
from quant_fm.tokenizer.field_spec import FULL_FIELD_SPECS_V2
from quant_fm.tokenizer.fit_bins import fit_bins
from quant_fm.tokenizer.fit_bins_v2 import fit_vocab_v2
from quant_fm.tokenizer.tokenize_events import assert_no_leakage, tokenize_path
from quant_fm.tokenizer.tokenize_events_v2 import (
    assert_no_leakage_v2,
    tokenize_path_v2,
)
from quant_fm.tokenizer.transforms import (
    DEFAULT_FEATURE_TRANSFORM_VERSION,
    SUPPORTED_FEATURE_TRANSFORM_VERSIONS,
)
from quant_fm.tokenizer.vocab import Vocab
from quant_fm.tokenizer.vocab_v2 import VocabV2

logger = logging.getLogger(__name__)


@contextmanager
def _timed_stage(stage: str, *, date: str | None = None):
    """Log stable wall-clock timing for one pipeline stage."""
    started = time.perf_counter()
    status = "error"
    try:
        yield
        status = "ok"
    finally:
        logger.info(
            "stage timing stage=%s date=%s elapsed_s=%.3f status=%s",
            stage,
            date or "-",
            time.perf_counter() - started,
            status,
        )


def _load_vocab(path: Path) -> Vocab | VocabV2:
    """Load a V1/V2 vocab by its serialized version, never by filename."""
    artifact = Path(path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("vocab_version") == "2.0":
        return VocabV2.load(artifact)
    return Vocab.load(artifact)


def _default_tokenize_workers() -> int:
    """Tokenize 并行度：环境变量 ``TOKENIZE_WORKERS``，默认 min(16, cpu)。"""
    env = os.environ.get("TOKENIZE_WORKERS")
    if env:
        return max(1, int(env))
    cpu = os.cpu_count() or 8
    return max(1, min(16, cpu // 2))


def _tokenize_one_job(job: tuple[str, str, str]) -> str:
    """ProcessPool worker：``(src, dst, vocab_path)`` → 返回 src。"""
    src_s, dst_s, vocab_s = job
    vocab = _load_vocab(Path(vocab_s))
    if isinstance(vocab, VocabV2):
        tokenize_path_v2(Path(src_s), Path(dst_s), vocab)
    else:
        tokenize_path(Path(src_s), Path(dst_s), vocab)
    return src_s


def _tokenize_shards_parallel(
    jobs: list[tuple[Path, Path]],
    *,
    vocab_path: Path,
    drop_events: bool,
    n_workers: int,
) -> int:
    """并行 tokenize 一批 ``(src, dst)``；可选删除 src。返回处理数。"""
    if not jobs:
        return 0
    payload = [(str(s), str(d), str(vocab_path)) for s, d in jobs]
    if n_workers <= 1 or len(payload) == 1:
        for src, dst, _vp in ((Path(a), Path(b), c) for a, b, c in payload):
            vocab = _load_vocab(vocab_path)
            if isinstance(vocab, VocabV2):
                tokenize_path_v2(src, dst, vocab)
            else:
                tokenize_path(src, dst, vocab)
            if drop_events:
                src.unlink(missing_ok=True)
        return len(payload)

    done = 0
    # 必须用 spawn：默认 fork 会继承 polars/rayon 已锁定的线程池，子进程死锁
    # （表现为 tokenize 卡住、0 产出，主进程在 as_completed 上永久 futex_wait）。
    # 与 canonicalize_clean_dir 的做法保持一致。
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futs = [pool.submit(_tokenize_one_job, job) for job in payload]
        for fut in as_completed(futs):
            src = Path(fut.result())
            if drop_events:
                src.unlink(missing_ok=True)
            done += 1
            if done % 500 == 0 or done == len(payload):
                logger.info("tokenize progress %d/%d", done, len(payload))
    return done


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
    *,
    skip_existing: bool = False,
    n_workers: int | None = None,
    event_ordering_version: str = DEFAULT_EVENT_ORDERING_VERSION,
    capture_book_state: bool = True,
) -> None:
    """单日单市场 PyLOB 清洗。"""
    from pylob.pipeline.workflow import default_clean_workers

    cfg = PipelineConfig(
        bucket=read_bucket(),
        trade_prefix="",
        order_prefix="",
        output_dir=clean_dir,
        symbols=symbols,
        market=market,
        layout="zeus_default",
        date=date.replace("-", "."),
        skip_existing=skip_existing,
        # medium 流水线只用 events.parquet；跳过 debug 产物加快写盘
        write_debug_artifacts=False,
        n_workers=default_clean_workers() if n_workers is None else n_workers,
        event_ordering_version=event_ordering_version,
        capture_book_state=capture_book_state,
    )
    build_clean_dataset(load_read_config(), cfg)


def _date_done_marker(workdir: Path, date: str) -> Path:
    return workdir / "data" / ".done" / date


def _clean_done_marker(workdir: Path, date: str) -> Path:
    return workdir / "data" / ".clean_done" / date


def _is_date_canonicalized(events_dir: Path, date: str) -> bool:
    return any(events_dir.rglob(f"{date}.parquet"))


def _marker_matches(path: Path, schema_version: str) -> bool:
    """Only reuse completion markers written for the requested data schema."""
    if not path.is_file():
        return False
    return schema_version in path.read_text(encoding="utf-8")


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
    fast_clean: bool = False,
    events_only: bool = False,
    skip_manifest: bool = False,
    reuse_vocab: Path | None = None,
    upload_minio: bool = False,
    upload_tag: str = "medium",
    upload_events: bool = False,
    delete_local_after_upload: bool = False,
    event_ordering_version: str | None = None,
    feature_transform_version: str | None = None,
    data_version: str = "v2",
    v2_max_samples_per_field: int = 5_000_000,
    v2_seed: int = 0,
    v2_full_audit: bool = False,
) -> None:
    """Run the medium-scale data preparation and optional upload pipeline."""
    if data_version not in {"v1", "v2"}:
        msg = f"data_version must be v1 or v2, got {data_version!r}"
        raise ValueError(msg)
    if v2_max_samples_per_field < 1:
        msg = "v2_max_samples_per_field must be positive"
        raise ValueError(msg)
    if events_only and reuse_vocab is not None:
        msg = "--events-only cannot be combined with --reuse-vocab"
        raise ValueError(msg)
    if events_only and data_version != "v2":
        msg = "--events-only is reserved for formal V2 preparation"
        raise ValueError(msg)
    if events_only and not (fast_clean or skip_clean):
        msg = (
            "--events-only requires --fast-clean or an existing --skip-clean tree "
            "for exact symbol coverage receipts"
        )
        raise ValueError(msg)
    if events_only and (drop_events or upload_minio or delete_local_after_upload):
        msg = "--events-only cannot drop events or upload incomplete artifacts"
        raise ValueError(msg)
    schema_version = V2_SCHEMA_VERSION if data_version == "v2" else V1_SCHEMA_VERSION
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

    vocab_path = data_dir / ("vocab_v2.json" if data_version == "v2" else "vocab.json")
    vocab: Vocab | VocabV2 | None = None
    # 下游抽 embedding 场景：提前加载冻结 vocab，以便按日 tokenize 后立刻
    # drop events，避免 60+ 天 events 堆积爆盘（约 4GB/天）。
    if reuse_vocab is not None:
        vocab = _load_vocab(reuse_vocab)
        expected_type = VocabV2 if data_version == "v2" else Vocab
        if not isinstance(vocab, expected_type):
            msg = (
                f"--data-version {data_version} disagrees with reused vocab "
                f"schema={vocab.schema_version}"
            )
            raise ValueError(msg)
        if isinstance(vocab, VocabV2):
            assert_no_leakage_v2(vocab, val_dates, test_dates)
        else:
            assert_no_leakage(vocab, val_dates, test_dates)
        vocab.save(vocab_path)
        logger.info(
            "reuse frozen vocab: %s (fit_dates=%d, 按日 tokenize+drop)",
            reuse_vocab,
            len(vocab.fit_dates),
        )
        if (
            event_ordering_version is not None
            and event_ordering_version != vocab.event_ordering_version
        ):
            msg = (
                "requested event ordering disagrees with reused vocab: "
                f"requested={event_ordering_version}, "
                f"vocab={vocab.event_ordering_version}"
            )
            raise ValueError(msg)
        if (
            feature_transform_version is not None
            and feature_transform_version != vocab.feature_transform_version
        ):
            msg = (
                "requested feature transform disagrees with reused vocab: "
                f"requested={feature_transform_version}, "
                f"vocab={vocab.feature_transform_version}"
            )
            raise ValueError(msg)
        effective_event_ordering = vocab.event_ordering_version
        effective_feature_transform = vocab.feature_transform_version
    else:
        effective_event_ordering = (
            event_ordering_version or DEFAULT_EVENT_ORDERING_VERSION
        )
        effective_feature_transform = (
            feature_transform_version or DEFAULT_FEATURE_TRANSFORM_VERSION
        )

    def _tokenize_day(day: str) -> int:
        """
        Tokenize one trading day's event shards in a **fresh subprocess**.

        必须走子进程：本进程已跑过 clean/canonicalize，polars/rayon 残留上百线程、
        占用上百 GB 常驻内存，此时在本进程内开 ``ProcessPoolExecutor(spawn)`` 会在
        fork+exec 瞬间被父进程线程锁 / 巨大页表拖死（worker 全 futex_wait、0 产出）。
        交给 ``quant_fm.scripts.tokenize_dir`` 在干净解释器里跑即可规避。
        """
        assert vocab is not None
        pending = False
        for event_path in events_dir.rglob(f"{day}.parquet"):
            token_path = tokens_dir / event_path.relative_to(events_dir)
            if not token_path.exists():
                pending = True
                continue
            try:
                assert_token_contract_matches(token_path, vocab)
            except ValueError as exc:
                msg = (
                    f"refusing to reuse incompatible token shard: {token_path}; "
                    f"use a new workdir ({exc})"
                )
                raise RuntimeError(msg) from exc
        cmd = [
            sys.executable,
            "-m",
            "quant_fm.scripts.tokenize_dir",
            "--events-dir",
            str(events_dir),
            "--tokens-dir",
            str(tokens_dir),
            "--vocab",
            str(vocab_path),
            "--day",
            day,
            "--workers",
            str(_default_tokenize_workers()),
        ]
        if drop_events:
            cmd.append("--drop-events")
        if not resume:
            cmd.append("--no-resume")
        if pending or drop_events:
            subprocess.run(cmd, check=True, cwd=str(ROOT))
        return sum(1 for _ in tokens_dir.rglob(f"{day}.parquet"))

    def _prune_empty_dirs(root: Path) -> None:
        for empty in sorted(root.rglob("*"), reverse=True):
            if empty.is_dir() and not any(empty.iterdir()):
                empty.rmdir()

    for date in dates:
        marker = _date_done_marker(workdir, date)
        failure_path = data_dir / ".failed" / f"{date}.json"
        coverage_path = coverage_receipt_path(workdir, date)
        # 旧 marker 可能只表示 canonicalize 完成、events 仍在磁盘上；
        # reuse-vocab 路径下补做按日 tokenize + drop。
        marker_ready = (
            resume
            and _marker_matches(marker, schema_version)
            and not (events_only and failure_path.is_file())
            and (data_version != "v2" or coverage_path.is_file())
        )
        if (
            marker_ready
            and events_only
            and not _is_date_canonicalized(events_dir, date)
        ):
            logger.warning(
                "events-only marker exists but canonical events are missing; "
                "rebuilding %s",
                date,
            )
            marker_ready = False
        if marker_ready:
            if vocab is not None and any(events_dir.rglob(f"{date}.parquet")):
                with _timed_stage("resume_tokenize_day", date=date):
                    n = _tokenize_day(date)
                logger.info(
                    "resume: tokenized leftover events for %s (%d shards)", date, n
                )
                if drop_events:
                    _prune_empty_dirs(events_dir)
                marker.write_text(
                    f"tokenized:{schema_version}\n",
                    encoding="utf-8",
                )
            else:
                logger.info("resume: skip %s (marker exists)", date)
            continue

        day_clean = clean_dir / date
        failed_symbols_for_date: tuple[str, ...] = ()
        clean_marker = _clean_done_marker(workdir, date)
        if (
            resume
            and _marker_matches(clean_marker, schema_version)
            and not (events_only and failure_path.is_file())
        ):
            logger.info("resume: reuse completed clean/%s", date)
        elif not skip_clean and fast_clean:
            # P0/P1 高性能路径：读一次 MinIO、SZ+SH 同池清洗、带本地 raw 缓存。
            from quant_fm.lob_rebuild.clean_day_fast import (
                clean_day_fast,
                drop_day_cache,
            )

            raw_cache = data_dir / "raw_cache"
            logger.info(
                "clean(fast) %s SZ=%d SH=%d", date, len(symbols_sz), len(symbols_sh)
            )
            with _timed_stage("clean_fast", date=date):
                clean_stats = clean_day_fast(
                    date=date,
                    symbols_sz=symbols_sz,
                    symbols_sh=symbols_sh,
                    clean_dir=day_clean,
                    cache_dir=raw_cache,
                    minio_config=load_read_config(),
                    bucket=read_bucket(),
                    skip_existing=resume,
                    event_ordering_version=effective_event_ordering,
                    capture_book_state=data_version == "v2",
                )
            if int(clean_stats["errors"]):
                failed = clean_stats.get("failed_symbols", [])
                failed_symbols_for_date = tuple(str(value) for value in failed)
                if events_only:
                    msg = (
                        f"formal V2 preparation failed for {date}; symbol gaps={failed}"
                    )
                    raise RuntimeError(msg)
                logger.warning(
                    "clean accepted with %s explicit gap(s) on %s: %s",
                    clean_stats["errors"],
                    date,
                    failed,
                )
            clean_marker.parent.mkdir(parents=True, exist_ok=True)
            clean_marker.write_text(
                f"cleaned:{schema_version}\n",
                encoding="utf-8",
            )
            drop_day_cache(raw_cache, date)  # clean 完成后释放当日缓存
        elif not skip_clean:
            if symbols_sz:
                logger.info("clean %s SZ (%d symbols)", date, len(symbols_sz))
                with _timed_stage("clean_legacy_sz", date=date):
                    clean_one_day(
                        date,
                        symbols_sz,
                        "SZ",
                        day_clean,
                        skip_existing=resume,
                        event_ordering_version=effective_event_ordering,
                        capture_book_state=data_version == "v2",
                    )
            if symbols_sh:
                logger.info("clean %s SH (%d symbols)", date, len(symbols_sh))
                with _timed_stage("clean_legacy_sh", date=date):
                    clean_one_day(
                        date,
                        symbols_sh,
                        "SH",
                        day_clean,
                        skip_existing=resume,
                        event_ordering_version=effective_event_ordering,
                        capture_book_state=data_version == "v2",
                    )
            clean_marker.parent.mkdir(parents=True, exist_ok=True)
            clean_marker.write_text(
                f"cleaned:{schema_version}\n",
                encoding="utf-8",
            )

        if data_version == "v2":
            if skip_clean:
                if not coverage_path.is_file() and day_clean.is_dir():
                    write_coverage_receipt(
                        workdir=workdir,
                        clean_dir=day_clean,
                        date=date,
                        symbols_sz=symbols_sz,
                        symbols_sh=symbols_sh,
                    )
                if not coverage_path.is_file():
                    msg = (
                        f"formal V2 --skip-clean requires an exact coverage receipt: "
                        f"{coverage_path}"
                    )
                    raise RuntimeError(msg)
            else:
                write_coverage_receipt(
                    workdir=workdir,
                    clean_dir=day_clean,
                    date=date,
                    symbols_sz=symbols_sz,
                    symbols_sh=symbols_sh,
                    failed_symbols=failed_symbols_for_date,
                )

        if vocab is not None:
            with _timed_stage("canonicalize_tokenize", date=date):
                token_paths = canonicalize_and_tokenize_clean_dir(
                    day_clean,
                    tokens_dir,
                    vocab_path=vocab_path,
                    date=date,
                    markets=("SZ", "SH"),
                    symbols=symbols_sz + symbols_sh,
                    skip_existing=resume,
                )
            if not token_paths:
                msg = f"refusing to mark {date} done: no token shards were produced"
                raise RuntimeError(msg)
            from quant_fm.scripts.make_adhoc_manifest import write_day_index

            write_day_index(tokens_dir, date)
        else:
            with _timed_stage("canonicalize_events", date=date):
                canonicalize_clean_dir(
                    day_clean,
                    events_dir,
                    date=date,
                    markets=("SZ", "SH"),
                    symbols=symbols_sz + symbols_sh,
                    skip_existing=resume,
                    strict=True,
                    schema_version=schema_version,
                )

        if drop_clean and day_clean.exists():
            shutil.rmtree(day_clean)
            logger.info("dropped clean/%s", date)
        if drop_clean:
            clean_marker.unlink(missing_ok=True)

        marker.parent.mkdir(parents=True, exist_ok=True)
        if vocab is not None:
            marker.write_text(
                f"tokenized_with_gaps:{schema_version}\n"
                if failure_path.exists()
                else f"tokenized:{schema_version}\n",
                encoding="utf-8",
            )
            logger.info("day done (tokenized): %s (%d shards)", date, len(token_paths))
        else:
            marker.write_text(
                f"canonicalized:{schema_version}\n",
                encoding="utf-8",
            )

    if events_only:
        logger.info(
            "events-only ready: dates=%d events=%s; global vocab/tokenize deferred",
            len(dates),
            events_dir,
        )
        return

    if vocab is None:
        # 预训练数据路径：先 fit_bins，再一次性 tokenize。
        fit_dates = train_dates
        train_paths = [p for p in events_dir.rglob("*.parquet") if p.stem <= train_end]
        if not train_paths:
            msg = "no train event shards found under events/"
            raise RuntimeError(msg)

        if data_version == "v2" and fit_sample_days is not None:
            msg = (
                "formal V2 requires the complete training stream; "
                "--fit-sample-days is V1-only"
            )
            raise ValueError(msg)
        if fit_sample_days is not None and fit_sample_days < len(train_dates):
            sample_dates = set(train_dates[:fit_sample_days])
            train_paths = [p for p in train_paths if p.stem in sample_dates]
            fit_dates = [d for d in train_dates if d in sample_dates]
            logger.info(
                "fit_bins on first %d train days (%d shards)",
                fit_sample_days,
                len(train_paths),
            )

        with _timed_stage("fit_vocab"):
            if data_version == "v2":
                vocab = fit_vocab_v2(
                    train_paths,
                    field_specs=FULL_FIELD_SPECS_V2,
                    max_samples_per_field=v2_max_samples_per_field,
                    fit_dates=fit_dates,
                    seed=v2_seed,
                    event_ordering_version=effective_event_ordering,
                    feature_transform_version=effective_feature_transform,
                )
            else:
                vocab = fit_bins(
                    train_paths,
                    n_bins=n_bins,
                    fit_dates=fit_dates,
                    event_ordering_version=effective_event_ordering,
                    feature_transform_version=effective_feature_transform,
                )
        vocab.save(vocab_path)
        if isinstance(vocab, VocabV2):
            assert_no_leakage_v2(vocab, val_dates, test_dates)
        else:
            assert_no_leakage(vocab, val_dates, test_dates)

        jobs: list[tuple[Path, Path]] = []
        for p in sorted(events_dir.rglob("*.parquet")):
            rel = p.relative_to(events_dir)
            dst = tokens_dir / rel
            if dst.exists() and resume:
                try:
                    assert_token_contract_matches(dst, vocab)
                except ValueError as exc:
                    msg = (
                        f"refusing to reuse incompatible token shard: {dst}; "
                        f"use a new workdir ({exc})"
                    )
                    raise RuntimeError(msg) from exc
                continue
            jobs.append((p, dst))
        if jobs:
            logger.info(
                "tokenizing %d event shards (workers=%d)",
                len(jobs),
                _default_tokenize_workers(),
            )
            with _timed_stage("tokenize_all_events"):
                _tokenize_shards_parallel(
                    jobs,
                    vocab_path=vocab_path,
                    drop_events=drop_events,
                    n_workers=_default_tokenize_workers(),
                )

        if drop_events:
            _prune_empty_dirs(events_dir)
    else:
        # 扫尾：任何未 tokenize 的残留 events（如中断留下的半日）。
        # 只扫**本进程负责的日期**：跨日并行时多个 run_medium 共享 events_dir，
        # 若在此全局扫描会撞上别的组正在写/删的分片（FileNotFoundError / 竞态）。
        dates_set = set(dates)
        leftover_jobs: list[tuple[Path, Path]] = []
        for p in sorted(events_dir.rglob("*.parquet")):
            if p.stem not in dates_set:
                continue  # 别的并行组的日期，跳过
            rel = p.relative_to(events_dir)
            dst = tokens_dir / rel
            if dst.exists() and resume:
                try:
                    assert_token_contract_matches(dst, vocab)
                except ValueError as exc:
                    msg = (
                        f"refusing to reuse incompatible token shard: {dst}; "
                        f"use a new workdir ({exc})"
                    )
                    raise RuntimeError(msg) from exc
                if drop_events:
                    p.unlink(missing_ok=True)
                continue
            leftover_jobs.append((p, dst))
        if leftover_jobs:
            logger.info(
                "tokenizing %d leftover event shards (workers=%d)",
                len(leftover_jobs),
                _default_tokenize_workers(),
            )
            with _timed_stage("tokenize_leftover_events"):
                _tokenize_shards_parallel(
                    leftover_jobs,
                    vocab_path=vocab_path,
                    drop_events=drop_events,
                    n_workers=_default_tokenize_workers(),
                )
        if drop_events:
            _prune_empty_dirs(events_dir)

    if skip_manifest:
        # 跨日并行时：各组跳过收尾 manifest（扫全量 tokens + 逐文件 sha256 很重且会
        # race），由驱动脚本在所有组结束后统一构建一次。
        logger.info(
            "skip_manifest: 跳过收尾 manifest（vocab=%s，tokens 就绪）", vocab_path
        )
    else:
        with _timed_stage("build_manifest"):
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

        if data_version == "v2":
            from quant_fm.scripts.audit_v2_artifacts import audit_v2_artifacts

            with _timed_stage("audit_v2_artifacts"):
                audit = audit_v2_artifacts(
                    workdir,
                    sample_shards=12,
                    full_path_check=v2_full_audit or upload_minio,
                )
                audit_path = workdir / "artifact_audit.json"
                temporary = audit_path.with_name(f".{audit_path.name}.tmp")
                temporary.write_text(
                    json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
                temporary.replace(audit_path)
            if not audit["contract_ready"]:
                msg = f"V2 artifact audit failed: {audit_path}"
                raise RuntimeError(msg)
            logger.info("V2 artifact audit passed: %s", audit_path)

    if upload_minio:
        from quant_fm.scripts.upload_to_minio import remote_uri, upload_workdir

        with _timed_stage("upload_minio"):
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
        default=Path("quant_fm/runs/v2_shared"),
    )
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--val-end", default=None)
    parser.add_argument("--n-bins", type=int, default=32)
    parser.add_argument(
        "--data-version",
        choices=("v1", "v2"),
        default=None,
        help=("数据合约；默认新数据为 v2，--reuse-vocab 时从 artifact 自动识别"),
    )
    parser.add_argument(
        "--v2-max-samples-per-field",
        type=int,
        default=5_000_000,
        help="V2 每字段确定性分层 reservoir 上限",
    )
    parser.add_argument("--v2-seed", type=int, default=0)
    parser.add_argument(
        "--v2-full-audit",
        action="store_true",
        help="对 manifest 中全部 V2 token shards 做完整路径/内容合约审计",
    )
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
        "--event-ordering-version",
        choices=sorted(SUPPORTED_EVENT_ORDERING_VERSIONS),
        default=None,
        help=(
            "事件排序契约；新词表默认 exchange_time_sequence_v2，复用旧词表时"
            "自动采用该词表记录的版本"
        ),
    )
    parser.add_argument(
        "--feature-transform-version",
        choices=sorted(SUPPORTED_FEATURE_TRANSFORM_VERSIONS),
        default=None,
        help=(
            "派生特征/EW-VWAP 契约；新词表默认 ew_vwap_causal_nan_v2，"
            "复用词表时自动采用其中记录的版本"
        ),
    )
    parser.add_argument(
        "--fast-clean",
        action="store_true",
        help="P0/P1 高性能清洗：每天只读一次 MinIO、SZ+SH 同池、本地 raw 缓存续跑",
    )
    parser.add_argument(
        "--events-only",
        action="store_true",
        help=(
            "仅并行清洗并生成 canonical events；不拟合 vocab、不 tokenize、"
            "不构建 manifest。正式 V2 两阶段并行的准备阶段使用"
        ),
    )
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="跳过收尾 manifest 构建（跨日并行时用，由驱动脚本统一建一次）",
    )
    parser.add_argument(
        "--reuse-vocab",
        type=Path,
        default=None,
        help=(
            "复用已冻结 vocab.json/vocab_v2.json（抽 embedding 场景必须与"
            "预训练一致），并自动识别数据版本"
        ),
    )
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
        help=(
            "已禁用：自动递归删除无法形成原子安全边界；请停写、独立验收远端后离线清理"
        ),
    )
    args = parser.parse_args()

    dates = list(_load_lines(args.dates_file))
    symbols_sz = _load_lines(args.symbols_sz_file)
    symbols_sh = _load_lines(args.symbols_sh_file)
    if args.max_symbols_per_market is not None:
        symbols_sz = symbols_sz[: args.max_symbols_per_market]
        symbols_sh = symbols_sh[: args.max_symbols_per_market]

    data_version = args.data_version
    if data_version is None and args.reuse_vocab is not None:
        reused = _load_vocab(args.reuse_vocab)
        data_version = "v2" if isinstance(reused, VocabV2) else "v1"
    data_version = data_version or "v2"

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
        fast_clean=args.fast_clean,
        events_only=args.events_only,
        skip_manifest=args.skip_manifest,
        reuse_vocab=args.reuse_vocab,
        upload_minio=args.upload_minio,
        upload_tag=args.upload_tag,
        upload_events=args.upload_events,
        delete_local_after_upload=args.delete_local_after_upload,
        event_ordering_version=args.event_ordering_version,
        feature_transform_version=args.feature_transform_version,
        data_version=data_version,
        v2_max_samples_per_field=args.v2_max_samples_per_field,
        v2_seed=args.v2_seed,
        v2_full_audit=args.v2_full_audit,
    )


if __name__ == "__main__":
    main()
