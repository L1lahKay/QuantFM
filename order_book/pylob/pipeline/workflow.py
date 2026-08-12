"""End-to-end cleaning workflow from MinIO objects to local parquet outputs."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from pylob.event_ordering import DEFAULT_EVENT_ORDERING_VERSION
from pylob.orderbook_builder_sh import OrderBookSH
from pylob.orderbook_builder_sz import OrderBookSZ
from pylob.pipeline.events import (
    build_event_stream,
    event_stream_contract_matches,
    write_event_stream_contract,
)
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
    capture_book_state: bool = False,
    capture_regime_atomic: bool = False,
    date: str | None = None,
    timeout_s: float | None = None,
    event_ordering_version: str = DEFAULT_EVENT_ORDERING_VERSION,
) -> str:
    """
    Clean a single symbol; returns status ``written`` or ``empty``.

    ``timeout_s``：单票回放墙钟上限（秒）。超时抛 ``TimeoutError``，由 worker
    捕获为 ``error``，避免个别活跃股拖死整天进程池。默认读环境变量
    ``CLEAN_SYMBOL_TIMEOUT``（缺省 900；设为 0 关闭）。
    """
    import os
    import time

    if timeout_s is None:
        timeout_s = float(os.environ.get("CLEAN_SYMBOL_TIMEOUT", "300"))
    if capture_regime_atomic and not capture_book_state:
        msg = "capture_regime_atomic requires capture_book_state"
        raise ValueError(msg)
    if capture_regime_atomic and not date:
        msg = "capture_regime_atomic requires an explicit ISO date"
        raise ValueError(msg)

    orderbook = OrderBookSH() if market == "SH" else OrderBookSZ()
    order_data = orderbook.prepare_market_data(
        trade_df,
        order_df,
        symbol=symbol,
        cut_time=cut_time,
        cut_serial=cut_serial,
        event_ordering_version=event_ordering_version,
    )
    if order_data is None or len(order_data) == 0:
        return "empty"

    if timeout_s and timeout_s > 0:
        deadline = time.monotonic() + timeout_s
        orig = orderbook.process_single_market_record

        def _guarded(row_data):
            if time.monotonic() > deadline:
                msg = f"clean timeout after {timeout_s:.0f}s symbol={symbol}"
                raise TimeoutError(msg)
            return orig(row_data)

        orderbook.process_single_market_record = _guarded  # type: ignore[method-assign]

    transitions = orderbook.process_workflow(
        order_data,
        capture_book_state=capture_book_state,
    )
    events = build_event_stream(order_data, market=market)
    if events is None or len(events) == 0:
        return "empty"

    symbol_dir = output_dir / market / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    events_out = symbol_dir / "events.parquet"
    temporary_events = events_out.with_suffix(".parquet.tmp")
    pl.from_pandas(events).write_parquet(temporary_events)
    temporary_events.replace(events_out)
    write_event_stream_contract(
        events_out,
        event_ordering_version=event_ordering_version,
    )
    if capture_book_state:
        from quant_fm.tokenizer.lob_transforms import transitions_to_feature_frame

        if transitions is None or len(transitions) != len(events):
            msg = "captured book transitions are not aligned with clean events"
            raise RuntimeError(msg)
        book_features = transitions_to_feature_frame(
            transitions,
            event_prices=events["price"].tolist(),
        )
        book_features_out = symbol_dir / "book_features.parquet"
        temporary_features = book_features_out.with_suffix(".parquet.tmp")
        book_features.write_parquet(
            temporary_features,
            compression="zstd",
            compression_level=3,
            statistics=True,
        )
        temporary_features.replace(book_features_out)
        if capture_regime_atomic:
            from quant_fm.regime.atomic import build_stock_day_atomic

            atomic = build_stock_day_atomic(
                transitions,
                events["int_time"].tolist(),
                date=str(date),
                symbol=symbol,
                market=market,
                event_ordering_version=event_ordering_version,
            )
            atomic_out = symbol_dir / "regime_atomic.parquet"
            temporary_atomic = atomic_out.with_suffix(".parquet.tmp")
            atomic.write_parquet(
                temporary_atomic,
                compression="zstd",
                compression_level=3,
                statistics=True,
            )
            temporary_atomic.replace(atomic_out)
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
            capture_book_state=bool(payload.get("capture_book_state", False)),
            capture_regime_atomic=bool(
                payload.get("capture_regime_atomic", False)
            ),
            date=payload.get("date"),
            timeout_s=payload.get("timeout_s"),
            event_ordering_version=payload.get(
                "event_ordering_version", DEFAULT_EVENT_ORDERING_VERSION
            ),
        )
        return payload["symbol"], status, None
    except Exception as exc:
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
            try:
                event_compatible = event_stream_contract_matches(
                    events_out,
                    version=config.event_ordering_version,
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                event_compatible = False
                logger.warning("invalid event contract %s: %s", events_out, exc)
            book_features_ready = (
                not config.capture_book_state
                or (events_out.parent / "book_features.parquet").is_file()
            )
            regime_atomic_ready = (
                not config.capture_regime_atomic
                or (events_out.parent / "regime_atomic.parquet").is_file()
            )
            if event_compatible and book_features_ready and regime_atomic_ready:
                skipped += 1
                continue
            if event_compatible and (not book_features_ready or not regime_atomic_ready):
                logger.info(
                    "resume: regenerating missing V2 book/regime features for %s",
                    events_out,
                )
                pending.append(symbol)
                continue
            msg = (
                f"refusing to overwrite incompatible clean event artifact during "
                f"resume: {events_out}; requested ordering="
                f"{config.event_ordering_version}. Use a new output root or disable "
                "skip_existing explicitly."
            )
            raise RuntimeError(msg)
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
                    capture_book_state=config.capture_book_state,
                    capture_regime_atomic=config.capture_regime_atomic,
                    date=(
                        str(config.date).replace(".", "-") if config.date else None
                    ),
                    event_ordering_version=config.event_ordering_version,
                )
                if status == "written":
                    written += 1
                    logger.info(
                        "Wrote cleaned artifacts to %s",
                        config.output_dir / config.market / symbol,
                    )
                else:
                    empty += 1
            except Exception:
                errors += 1
                logger.exception("failed symbol=%s", symbol)
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
            "capture_book_state": config.capture_book_state,
            "capture_regime_atomic": config.capture_regime_atomic,
            "date": str(config.date).replace(".", "-") if config.date else None,
            "event_ordering_version": config.event_ordering_version,
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
            # 尾盘更密打点，避免「停在 900/1000」假象（实际卡在最后几只）。
            if i % 50 == 0 or i == total or (total - i) <= 20:
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
