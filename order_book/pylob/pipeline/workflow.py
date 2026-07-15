"""End-to-end cleaning workflow from MinIO objects to local parquet outputs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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

    for symbol in config.symbols:
        logger.info("Processing symbol=%s market=%s", symbol, config.market)
        orderbook = OrderBookSH() if config.market == "SH" else OrderBookSZ()
        order_data = orderbook.prepare_market_data(
            trade_df,
            order_df,
            symbol=symbol,
            cut_time=config.cut_time,
            cut_serial=config.cut_serial,
        )
        orderbook.process_workflow(order_data)

        events = build_event_stream(order_data, market=config.market)
        tokens = build_field_tokens(events)

        symbol_dir = config.output_dir / config.market / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        pl.from_pandas(order_data).write_parquet(symbol_dir / "market_rows.parquet")
        pl.from_pandas(events).write_parquet(symbol_dir / "events.parquet")
        pl.from_pandas(tokens).write_parquet(symbol_dir / "tokens.parquet")
        logger.info("Wrote cleaned artifacts to %s", symbol_dir)
