"""Polars-backed S3 readers for zeus-cn-quote style object layouts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

import polars as pl

from pylob.pipeline.paths import normalize_endpoint

if TYPE_CHECKING:
    from pylob.pipeline.config import MinioConfig

logger = logging.getLogger(__name__)


class ObjectReader(Protocol):
    """Minimal interface consumed by the cleaning workflow."""

    def read_object_keys(
        self, bucket: str, object_keys: tuple[str, ...]
    ) -> pl.DataFrame:
        """Read explicit object keys from a bucket."""
        ...

    def read_prefix(
        self, bucket: str, prefix: str, suffixes: tuple[str, ...]
    ) -> pl.DataFrame:
        """Read matching objects beneath a bucket prefix."""
        ...


def build_storage_options(config: MinioConfig) -> dict[str, str]:
    """Return Polars/object_store credentials for an S3-compatible endpoint."""
    endpoint = normalize_endpoint(config.endpoint)
    scheme = "https" if config.secure else "http"
    opts: dict[str, str] = {
        "aws_access_key_id": config.access_key,
        "aws_secret_access_key": config.secret_key,
        "aws_endpoint_url": f"{scheme}://{endpoint}",
        "aws_region": "us-east-1",
    }
    if not config.secure:
        # MinIO plain HTTP：Polars/object_store 需显式允许
        opts["aws_allow_http"] = "true"
    return opts


class PolarsS3Reader:
    """Read parquet/csv objects through Polars' S3 integration."""

    def __init__(self, config: MinioConfig):
        self.config = config
        self.storage_options = build_storage_options(config)

    def _s3_uri(self, bucket: str, object_key: str) -> str:
        return f"s3://{bucket}/{object_key.lstrip('/')}"

    def read_object_keys(
        self,
        bucket: str,
        object_keys: tuple[str, ...],
        *,
        symbols: tuple[str, ...] = (),
    ) -> pl.DataFrame:
        """Read explicit parquet objects, optionally filtering symbols."""
        if not object_keys:
            msg = f"no object keys provided for bucket={bucket}"
            raise FileNotFoundError(msg)

        symbol_filter = tuple(symbol.zfill(6) for symbol in symbols)
        frames: list[pl.DataFrame] = []
        for index, object_key in enumerate(object_keys, start=1):
            uri = self._s3_uri(bucket, object_key)
            logger.info(
                "Reading object %s/%s (%d/%d)",
                bucket,
                object_key,
                index,
                len(object_keys),
            )
            frame = pl.scan_parquet(uri, storage_options=self.storage_options)
            if symbol_filter:
                logger.info("Filtering symbols=%s", ",".join(symbol_filter))
                schema = frame.collect_schema()
                if schema.get("symbol") == pl.String:
                    frame = frame.filter(pl.col("symbol").is_in(symbol_filter))
                else:
                    frame = frame.filter(
                        pl.col("symbol")
                        .cast(pl.String)
                        .str.zfill(6)
                        .is_in(symbol_filter)
                    )
            collected = frame.collect()
            logger.info("Loaded %d rows from %s", collected.height, object_key)
            frames.append(collected)
        return pl.concat(frames, how="diagonal_relaxed")

    def read_prefix(
        self, bucket: str, prefix: str, suffixes: tuple[str, ...]
    ) -> pl.DataFrame:
        """List and read matching objects beneath a bucket prefix."""
        from pylob.pipeline.minio_io import MinioDataSource

        source = MinioDataSource(self.config)
        object_names = source.list_objects(bucket, prefix, suffixes)
        logger.info(
            "Found %d objects under s3://%s/%s", len(object_names), bucket, prefix
        )
        return self.read_object_keys(bucket, tuple(object_names))


def split_combined_frame(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a combined archive frame into trade-like and order-like subsets."""
    if "type" in df.columns:
        trade_df = df.filter(pl.col("type").cast(pl.String).str.to_uppercase() == "T")
        order_df = df.filter(pl.col("type").cast(pl.String).str.to_uppercase() == "O")
        if not trade_df.is_empty() or not order_df.is_empty():
            return trade_df, order_df

    trade_mask = pl.col("trade_volume").fill_null(0).gt(0) | pl.col("trade_type").cast(
        pl.String
    ).fill_null("").ne("")
    order_mask = pl.col("order_volume").fill_null(0).gt(0) | pl.col("order_type").cast(
        pl.String
    ).fill_null("").ne("")
    return df.filter(trade_mask), df.filter(order_mask)
