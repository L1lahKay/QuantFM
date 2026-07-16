"""End-to-end cleaning workflow from MinIO objects to local parquet outputs."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from pylob.orderbook_builder_sh import OrderBookSH
from pylob.orderbook_builder_sz import OrderBookSZ
from pylob.pipeline.events import build_event_stream
from pylob.pipeline.minio_io import MinioDataSource
from pylob.pipeline.s3_io import PolarsS3Reader, split_combined_frame
from pylob.pipeline.standardize import standardize_order_frame, standardize_trade_frame
from pylob.pipeline.tokenizer import build_field_tokens

if TYPE_CHECKING:
    from pylob.pipeline.config import MinioConfig, PipelineConfig

logger = logging.getLogger(__name__)


def _load_raw_frames(
    config: PipelineConfig,
    minio_config: MinioConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    reader = PolarsS3Reader(minio_config)

    if config.layout in {"hds", "archive", "combined", "zeus_default"}:
        trade_keys = config.resolved_trade_keys()
        order_keys = config.resolved_order_keys()
        logger.info(
            "Loading layout=%s trade_keys=%d order_keys=%d",
            config.layout,
            len(trade_keys),
            len(order_keys),
        )
        filter_symbols = config.symbols
        if config.layout != "zeus_default":
            combined_keys = set(trade_keys) | set(order_keys)
            if not any(
                "/default/2/" in key or "/default/3/" in key for key in combined_keys
            ):
                filter_symbols = ()
        raw_trade = reader.read_object_keys(
            config.bucket, trade_keys, symbols=filter_symbols
        )
        if config.layout in {"archive", "combined"} and trade_keys == order_keys:
            logger.info("Splitting combined archive frame into trade/order subsets")
            raw_trade, raw_order = split_combined_frame(raw_trade)
        else:
            raw_order = reader.read_object_keys(
                config.bucket, order_keys, symbols=filter_symbols
            )
        return raw_trade, raw_order

    minio_source = MinioDataSource(minio_config)
    trade_prefix = config.resolved_trade_prefix()
    order_prefix = config.resolved_order_prefix()
    logger.info("Listing trade objects under s3://%s/%s", config.bucket, trade_prefix)
    trade_objects = minio_source.list_objects(
        config.bucket, trade_prefix, config.file_suffixes
    )
    logger.info("Listing order objects under s3://%s/%s", config.bucket, order_prefix)
    order_objects = minio_source.list_objects(
        config.bucket, order_prefix, config.file_suffixes
    )
    logger.info(
        "Downloading %d trade files and %d order files",
        len(trade_objects),
        len(order_objects),
    )
    raw_trade = minio_source.read_objects(config.bucket, trade_objects)
    raw_order = minio_source.read_objects(config.bucket, order_objects)
    return raw_trade, raw_order


def _clean_one_symbol(
    *,
    symbol: str,
    trade_df: pl.DataFrame,
    order_df: pl.DataFrame,
    market: str,
    output_dir: Path,
    cut_time: int,
    cut_serial: int | None,
    write_debug_artifacts: bool,
) -> str:
    """Clean a single symbol; returns status ``written`` or ``empty``."""
    orderbook = OrderBookSH() if market == "SH" else OrderBookSZ()
    order_data = orderbook.prepare_market_data(
        trade_df,
        order_df,
        symbol=symbol,
        cut_time=cut_time,
        cut_serial=cut_serial,
    )
    if order_data is None or len(order_data) == 0:
        return "empty"

    orderbook.process_workflow(order_data)
    events = build_event_stream(order_data, market=market)
    if events is None or len(events) == 0:
        return "empty"

    symbol_dir = output_dir / market / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    events_out = symbol_dir / "events.parquet"
    pl.from_pandas(events).write_parquet(events_out)
    if write_debug_artifacts:
        tokens = build_field_tokens(events)
        pl.from_pandas(order_data).write_parquet(symbol_dir / "market_rows.parquet")
        pl.from_pandas(tokens).write_parquet(symbol_dir / "tokens.parquet")
    return "written"


def _worker_clean_symbol(payload: dict[str, Any]) -> tuple[str, str, str | None]:
    """ProcessPool worker entrypoint. Returns ``(symbol, status, error)``."""
    try:
        status = _clean_one_symbol(
            symbol=payload["symbol"],
            trade_df=payload["trade_df"],
            order_df=payload["order_df"],
            market=payload["market"],
            output_dir=Path(payload["output_dir"]),
            cut_time=payload["cut_time"],
            cut_serial=payload["cut_serial"],
            write_debug_artifacts=payload["write_debug_artifacts"],
        )
        return payload["symbol"], status, None
    except Exception as exc:  # noqa: BLE001 - keep day job alive on bad symbols
        return payload["symbol"], "error", f"{type(exc).__name__}: {exc}"


def _partition_by_symbol(
    trade_df: pl.DataFrame,
    order_df: pl.DataFrame,
    symbols: list[str],
) -> tuple[dict[str, pl.DataFrame], dict[str, pl.DataFrame]]:
    """Split market frames into per-symbol subsets for parallel workers."""
    symbol_set = set(symbols)
    trade_sub = trade_df.filter(pl.col("symbol").is_in(symbols))
    order_sub = order_df.filter(pl.col("symbol").is_in(symbols))
    trade_parts = trade_sub.partition_by("symbol", as_dict=True, include_key=True)
    order_parts = order_sub.partition_by("symbol", as_dict=True, include_key=True)
    # polars may key by the value itself or a single-element tuple depending on version
    def _normalize(parts: dict) -> dict[str, pl.DataFrame]:
        out: dict[str, pl.DataFrame] = {}
        for key, frame in parts.items():
            sym = key[0] if isinstance(key, tuple) else key
            sym = str(sym)
            if sym in symbol_set:
                out[sym] = frame
        return out

    return _normalize(trade_parts), _normalize(order_parts)


def _empty_like(df: pl.DataFrame) -> pl.DataFrame:
    return df.clear()


def build_clean_dataset(minio_config: MinioConfig, config: PipelineConfig) -> None:
    """Read raw MinIO data and write cleaned per-symbol parquet artifacts."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    raw_trade, raw_order = _load_raw_frames(config, minio_config)
    logger.info(
        "Standardizing raw frames: trade_rows=%d order_rows=%d",
        raw_trade.height,
        raw_order.height,
    )

    trade_df = standardize_trade_frame(raw_trade, field_mapping=config.field_mapping)
    order_df = standardize_order_frame(raw_order, field_mapping=config.field_mapping)
    del raw_trade, raw_order

    pending: list[str] = []
    skipped = 0
    for symbol in config.symbols:
        events_out = config.output_dir / config.market / symbol / "events.parquet"
        if config.skip_existing and events_out.exists():
            skipped += 1
            continue
        pending.append(symbol)

    if skipped:
        logger.info(
            "skip_existing: reused %d/%d symbols under %s/%s",
            skipped,
            len(config.symbols),
            config.output_dir,
            config.market,
        )

    if not pending:
        logger.info("nothing to clean for %s/%s", config.output_dir, config.market)
        return

    n_workers = max(1, int(config.n_workers))
    logger.info(
        "cleaning %d symbols for %s with n_workers=%d",
        len(pending),
        config.market,
        n_workers,
    )

    if n_workers == 1:
        written = empty = errors = 0
        for symbol in pending:
            logger.info("Processing symbol=%s market=%s", symbol, config.market)
            try:
                status = _clean_one_symbol(
                    symbol=symbol,
                    trade_df=trade_df,
                    order_df=order_df,
                    market=config.market,
                    output_dir=config.output_dir,
                    cut_time=config.cut_time,
                    cut_serial=config.cut_serial,
                    write_debug_artifacts=config.write_debug_artifacts,
                )
                if status == "written":
                    written += 1
                    logger.info(
                        "Wrote cleaned artifacts to %s",
                        config.output_dir / config.market / symbol,
                    )
                else:
                    empty += 1
            except Exception as exc:  # noqa: BLE001
                errors += 1
                logger.exception("failed symbol=%s: %s", symbol, exc)
        logger.info(
            "clean done market=%s written=%d empty=%d errors=%d skipped=%d",
            config.market,
            written,
            empty,
            errors,
            skipped,
        )
        return

    trade_parts, order_parts = _partition_by_symbol(trade_df, order_df, pending)
    empty_trade = _empty_like(trade_df)
    empty_order = _empty_like(order_df)
    # release parent copies of full frames after partition to free RAM for workers
    del trade_df, order_df

    payloads = [
        {
            "symbol": symbol,
            "trade_df": trade_parts.get(symbol, empty_trade),
            "order_df": order_parts.get(symbol, empty_order),
            "market": config.market,
            "output_dir": str(config.output_dir),
            "cut_time": config.cut_time,
            "cut_serial": config.cut_serial,
            "write_debug_artifacts": config.write_debug_artifacts,
        }
        for symbol in pending
    ]
    del trade_parts, order_parts

    written = empty = errors = 0
    # spawn avoids Polars thread-pool + fork deadlocks after large reads
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = [pool.submit(_worker_clean_symbol, payload) for payload in payloads]
        total = len(futures)
        for i, fut in enumerate(as_completed(futures), start=1):
            symbol, status, err = fut.result()
            if status == "written":
                written += 1
            elif status == "empty":
                empty += 1
            else:
                errors += 1
                logger.error("failed symbol=%s: %s", symbol, err)
            if i % 50 == 0 or i == total:
                logger.info(
                    "clean progress %s %d/%d written=%d empty=%d errors=%d",
                    config.market,
                    i,
                    total,
                    written,
                    empty,
                    errors,
                )

    logger.info(
        "clean done market=%s written=%d empty=%d errors=%d skipped=%d",
        config.market,
        written,
        empty,
        errors,
        skipped,
    )


def default_clean_workers() -> int:
    """Default worker count for parallel symbol cleaning."""
    env = os.environ.get("CLEAN_WORKERS")
    if env:
        return max(1, int(env))
    cpu = os.cpu_count() or 8
    # leave headroom for OS / MinIO / polars; each worker holds one symbol book
    return max(1, min(32, cpu // 2))
