"""
单日"读一次 + 双市场清洗"的高性能清洗（P0+P1 重构）。

相对旧路径（``run_medium.clean_one_day`` 对 SZ、SH 各调用一次
``build_clean_dataset`` → 每次都完整读 default/2 + default/3 两个全市场日文件）本模块：

* **只读一次 MinIO**：一次性拉取当日 trade+order，SZ/SH 共用同一份内存帧
  → 网络 IO 直接减半（这是 clean 阶段的头号浪费）。
* **本地 raw 缓存**：把过滤+列投影后的原始帧落到本地 parquet；进程中断/看门狗
  重启后**秒级续跑、绝不重下**（昨晚空转 12h 就是死于每次重启重下 2.8 亿行）。
* **列投影**：scan 只取 ``standardize`` 真正需要的列（``REQUIRED_COLUMNS``），
  再减少 20-40% 下载字节。
* **单进程池覆盖两市场**：SZ 用 ``OrderBookSZ``、SH 用 ``OrderBookSH``，
  在同一个 spawn ``ProcessPoolExecutor`` 里并行，减少池启停与再分区开销。

产物与旧路径完全一致：``{clean_dir}/{MKT}/{symbol}/events.parquet``。
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING

import polars as pl
from pylob.pipeline.config import PipelineConfig
from pylob.pipeline.s3_io import PolarsS3Reader

if TYPE_CHECKING:
    from pathlib import Path

    from pylob.pipeline.config import MinioConfig
from pylob.pipeline.standardize import (
    REQUIRED_COLUMNS,
    standardize_order_frame,
    standardize_trade_frame,
)
from pylob.pipeline.workflow import (
    _partition_by_symbol,
    _worker_clean_symbol,
    default_clean_workers,
)

logger = logging.getLogger(__name__)


def _read_one_projected(
    reader: PolarsS3Reader,
    bucket: str,
    keys: tuple[str, ...],
    symbols: tuple[str, ...],
    *,
    project: bool,
) -> pl.DataFrame:
    """读一组 object key，按 symbol 过滤 + 可选列投影，一次 ``collect``。"""
    sym_filter = tuple(s.zfill(6) for s in symbols)
    frames: list[pl.DataFrame] = []
    for key in keys:
        uri = f"s3://{bucket}/{key.lstrip('/')}"
        lf = pl.scan_parquet(uri, storage_options=reader.storage_options)
        symbol_dtype = None
        if project:
            try:
                schema = lf.collect_schema()
                avail = set(schema.names())
                symbol_dtype = schema.get("symbol")
                cols = [c for c in REQUIRED_COLUMNS if c in avail]
                if "symbol" in avail and cols:
                    lf = lf.select(cols)
            except Exception:
                logger.warning("列投影失败，退回全列读取: %s", key)
        if sym_filter:
            # Keep the native string column in the predicate so parquet/object_store
            # can push it into row-group filtering. cast+zfill forces a full scan.
            if symbol_dtype == pl.String:
                lf = lf.filter(pl.col("symbol").is_in(sym_filter))
            else:
                lf = lf.filter(
                    pl.col("symbol").cast(pl.String).str.zfill(6).is_in(sym_filter)
                )
        collected = lf.collect()
        logger.info("read %d rows from %s", collected.height, key)
        frames.append(collected)
    return pl.concat(frames, how="diagonal_relaxed")


def _load_raw_once(
    *,
    date: str,
    symbols: tuple[str, ...],
    minio_config: MinioConfig,
    bucket: str,
    cache_dir: Path,
    project_columns: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """读一次当日 trade+order（带本地缓存 + 列投影）。返回原始（未标准化）帧。"""
    pdate = date.replace("-", ".")
    key_cfg = PipelineConfig(
        bucket=bucket,
        trade_prefix="",
        order_prefix="",
        output_dir=cache_dir,
        symbols=symbols,
        market="SZ",  # zeus_default 的 key 与市场无关，仅占位
        layout="zeus_default",
        date=pdate,
    )
    trade_keys = key_cfg.resolved_trade_keys()
    order_keys = key_cfg.resolved_order_keys()

    day_cache = cache_dir / date
    trade_cache = day_cache / "trade.parquet"
    order_cache = day_cache / "order.parquet"

    if trade_cache.exists() and order_cache.exists():
        logger.info("raw 缓存命中，跳过 MinIO 下载: %s", day_cache)
        return pl.read_parquet(trade_cache), pl.read_parquet(order_cache)

    reader = PolarsS3Reader(minio_config)
    logger.info("单次读取 MinIO（union=%d 只，列投影=%s）", len(symbols), project_columns)
    raw_trade = _read_one_projected(
        reader, bucket, trade_keys, symbols, project=project_columns
    )
    raw_order = _read_one_projected(
        reader, bucket, order_keys, symbols, project=project_columns
    )

    # 原子写缓存：先写 .tmp 再 rename。
    day_cache.mkdir(parents=True, exist_ok=True)
    for frame, dst in ((raw_trade, trade_cache), (raw_order, order_cache)):
        tmp = dst.with_suffix(".parquet.tmp")
        frame.write_parquet(tmp)
        tmp.rename(dst)
    logger.info("raw 缓存已落盘: %s", day_cache)
    return raw_trade, raw_order


def clean_day_fast(
    *,
    date: str,
    symbols_sz: tuple[str, ...],
    symbols_sh: tuple[str, ...],
    clean_dir: Path,
    cache_dir: Path,
    minio_config: MinioConfig,
    bucket: str,
    n_workers: int | None = None,
    skip_existing: bool = True,
    project_columns: bool = True,
    cut_time: int = 151000000,
    cut_serial: int | None = None,
) -> dict[str, int | list[str]]:
    """
    单日清洗：读一次原始数据，SZ+SH 共用内存帧，在同一进程池里并行清洗。

    返回 ``{"written":..,"empty":..,"errors":..,"skipped":..}`` 统计。
    产物写到 ``{clean_dir}/{MKT}/{symbol}/events.parquet``（与旧路径一致）。
    """
    clean_dir.mkdir(parents=True, exist_ok=True)
    union = tuple(symbols_sz) + tuple(symbols_sh)
    market_of: dict[str, str] = {}
    for s in symbols_sz:
        market_of[s] = "SZ"
    for s in symbols_sh:
        market_of[s] = "SH"

    raw_trade, raw_order = _load_raw_once(
        date=date,
        symbols=union,
        minio_config=minio_config,
        bucket=bucket,
        cache_dir=cache_dir,
        project_columns=project_columns,
    )
    logger.info(
        "standardize: trade_rows=%d order_rows=%d", raw_trade.height, raw_order.height
    )
    trade_df = standardize_trade_frame(raw_trade)
    order_df = standardize_order_frame(raw_order)
    del raw_trade, raw_order

    # 已完成的 symbol 跳过（resume）。
    pending: list[str] = []
    skipped = 0
    for sym in union:
        out = clean_dir / market_of[sym] / sym / "events.parquet"
        if skip_existing and out.exists():
            skipped += 1
            continue
        pending.append(sym)
    if skipped:
        logger.info("skip_existing: 复用 %d/%d 只", skipped, len(union))
    if not pending:
        logger.info("nothing to clean for %s", date)
        return {
            "written": 0,
            "empty": 0,
            "errors": 0,
            "skipped": skipped,
            "failed_symbols": [],
        }

    # 一次分区覆盖两市场。
    trade_parts, order_parts = _partition_by_symbol(trade_df, order_df, pending)
    empty_trade = trade_df.clear()
    empty_order = order_df.clear()
    del trade_df, order_df

    payloads = [
        {
            "symbol": sym,
            "trade_df": trade_parts.get(sym, empty_trade),
            "order_df": order_parts.get(sym, empty_order),
            "market": market_of[sym],
            "output_dir": str(clean_dir),
            "cut_time": cut_time,
            "cut_serial": cut_serial,
            "write_debug_artifacts": False,
        }
        for sym in pending
    ]
    del trade_parts, order_parts

    n_workers = default_clean_workers() if n_workers is None else max(1, n_workers)
    logger.info("并行清洗 %d 只（SZ+SH 同池），n_workers=%d", len(pending), n_workers)

    written = empty = errors = 0
    failed_symbols: list[str] = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futs = [pool.submit(_worker_clean_symbol, p) for p in payloads]
        total = len(futs)
        for i, fut in enumerate(as_completed(futs), start=1):
            sym, status, err = fut.result()
            if status == "written":
                written += 1
            elif status == "empty":
                empty += 1
            else:
                errors += 1
                failed_symbols.append(sym)
                logger.error("failed symbol=%s: %s", sym, err)
            # 尾盘更密打点，避免「停在 900/1000」假象。
            if i % 50 == 0 or i == total or (total - i) <= 20:
                logger.info(
                    "clean progress %d/%d written=%d empty=%d errors=%d",
                    i, total, written, empty, errors,
                )

    # Retry only the failed tail once while the already-downloaded frames are alive.
    # A persistent pathological stock is recorded as an explicit gap instead of
    # silently disappearing or blocking the other ~999 names indefinitely.
    if failed_symbols:
        retry_timeout = float(os.environ.get("CLEAN_RETRY_TIMEOUT", "300"))
        by_symbol = {str(payload["symbol"]): payload for payload in payloads}
        retry_payloads = []
        for symbol in failed_symbols:
            payload = dict(by_symbol[symbol])
            payload["timeout_s"] = retry_timeout
            retry_payloads.append(payload)
        logger.warning(
            "retry failed symbols once: %s timeout=%.0fs",
            failed_symbols,
            retry_timeout,
        )
        still_failed: list[str] = []
        with ProcessPoolExecutor(
            max_workers=min(2, len(retry_payloads)), mp_context=ctx
        ) as retry_pool:
            futures = [
                retry_pool.submit(_worker_clean_symbol, payload)
                for payload in retry_payloads
            ]
            for future in as_completed(futures):
                symbol, status, err = future.result()
                if status == "written":
                    written += 1
                    errors -= 1
                elif status == "empty":
                    empty += 1
                    errors -= 1
                else:
                    still_failed.append(symbol)
                    logger.error("retry failed symbol=%s: %s", symbol, err)
        failed_symbols = sorted(still_failed)
    logger.info(
        "clean done(fast) date=%s written=%d empty=%d errors=%d skipped=%d",
        date, written, empty, errors, skipped,
    )
    failure_dir = clean_dir.parents[1] / "data" / ".failed"
    failure_path = failure_dir / f"{date}.json"
    if failed_symbols:
        failure_dir.mkdir(parents=True, exist_ok=True)
        tmp = failure_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(sorted(failed_symbols)), encoding="utf-8")
        tmp.replace(failure_path)
    else:
        failure_path.unlink(missing_ok=True)
    return {
        "written": written,
        "empty": empty,
        "errors": errors,
        "skipped": skipped,
        "failed_symbols": failed_symbols,
    }


def drop_day_cache(cache_dir: Path, date: str) -> None:
    """清洗完成后删除当日 raw 缓存（释放磁盘）。"""
    import shutil

    day_cache = cache_dir / date
    if day_cache.exists():
        shutil.rmtree(day_cache, ignore_errors=True)
        logger.info("已删除 raw 缓存: %s", day_cache)
