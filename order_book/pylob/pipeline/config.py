"""Configuration objects for the data cleaning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pylob.pipeline.paths import (
    archive_object_key,
    endpoint_is_secure,
    hds_order_prefix,
    hds_trade_prefix,
    normalize_endpoint,
    symbol_object_keys,
    zeus_default_order_key,
    zeus_default_trade_key,
)

SUPPORTED_LAYOUTS = {"prefix", "hds", "archive", "combined", "zeus_default"}


@dataclass(slots=True, frozen=True)
class MinioConfig:
    """Connection settings for an S3-compatible MinIO endpoint."""

    endpoint: str
    access_key: str
    secret_key: str
    secure: bool = False
    region: str | None = None

    def __post_init__(self) -> None:
        raw = self.endpoint
        secure = self.secure or endpoint_is_secure(raw)
        object.__setattr__(self, "endpoint", normalize_endpoint(raw))
        object.__setattr__(self, "secure", secure)


@dataclass(slots=True, frozen=True)
class PipelineConfig:
    """Runtime settings for one market data cleaning job."""

    bucket: str
    trade_prefix: str
    order_prefix: str
    output_dir: Path
    symbols: tuple[str, ...]
    market: str
    layout: str = "prefix"
    date: str | None = None
    mic: str = "XSHE"
    dataset: str = "china_stock"
    archive_partition: str = "1"
    archive_filename: str = "all.parquet"
    combined_object_key: str | None = None
    trade_object_keys: tuple[str, ...] = ()
    order_object_keys: tuple[str, ...] = ()
    cut_time: int = 151000000
    cut_serial: int | None = None
    file_suffixes: tuple[str, ...] = (".parquet", ".csv")
    field_mapping: dict[str, str] = field(default_factory=dict)
    skip_existing: bool = False
    write_debug_artifacts: bool = True
    n_workers: int = 1

    def __post_init__(self) -> None:
        market = self.market.upper()
        if market not in {"SH", "SZ"}:
            msg = "market must be 'SH' or 'SZ'"
            raise ValueError(msg)
        layout = self.layout.lower()
        if layout not in SUPPORTED_LAYOUTS:
            msg = f"layout must be one of {sorted(SUPPORTED_LAYOUTS)}"
            raise ValueError(msg)
        object.__setattr__(self, "market", market)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    def resolved_trade_keys(self) -> tuple[str, ...]:
        """Return explicit or layout-derived trade object keys."""
        if self.trade_object_keys:
            return self.trade_object_keys
        if self.layout == "zeus_default":
            if not self.date:
                msg = "PYLOB_DATE is required when PYLOB_LAYOUT=zeus_default"
                raise ValueError(msg)
            return (zeus_default_trade_key(self.date, dataset=self.dataset),)
        if self.layout == "hds":
            if not self.date:
                msg = "PYLOB_DATE is required when PYLOB_LAYOUT=hds"
                raise ValueError(msg)
            prefix = hds_trade_prefix(self.date, mic=self.mic, dataset=self.dataset)
            return symbol_object_keys(prefix, self.market, self.symbols)
        if self.layout in {"archive", "combined"}:
            if not self.date and not self.combined_object_key:
                msg = "PYLOB_DATE or MINIO_COMBINED_KEY is required for archive/combined layout"
                raise ValueError(msg)
            key = self.combined_object_key or archive_object_key(
                self.date,
                partition=self.archive_partition,
                name=self.archive_filename,
            )
            return (key,)
        return ()

    def resolved_order_keys(self) -> tuple[str, ...]:
        """Return explicit or layout-derived order object keys."""
        if self.order_object_keys:
            return self.order_object_keys
        if self.layout == "zeus_default":
            if not self.date:
                msg = "PYLOB_DATE is required when PYLOB_LAYOUT=zeus_default"
                raise ValueError(msg)
            return (zeus_default_order_key(self.date, dataset=self.dataset),)
        if self.layout == "hds":
            if not self.date:
                msg = "PYLOB_DATE is required when PYLOB_LAYOUT=hds"
                raise ValueError(msg)
            prefix = hds_order_prefix(self.date, mic=self.mic, dataset=self.dataset)
            return symbol_object_keys(prefix, self.market, self.symbols)
        if self.layout in {"archive", "combined"}:
            if not self.date and not self.combined_object_key:
                msg = "PYLOB_DATE or MINIO_COMBINED_KEY is required for archive/combined layout"
                raise ValueError(msg)
            key = self.combined_object_key or archive_object_key(
                self.date,
                partition=self.archive_partition,
                name=self.archive_filename,
            )
            return (key,)
        return ()

    def resolved_trade_prefix(self) -> str:
        """Return the effective trade prefix for the configured layout."""
        if self.layout == "hds" and self.date:
            return hds_trade_prefix(self.date, mic=self.mic, dataset=self.dataset)
        return self.trade_prefix

    def resolved_order_prefix(self) -> str:
        """Return the effective order prefix for the configured layout."""
        if self.layout == "hds" and self.date:
            return hds_order_prefix(self.date, mic=self.mic, dataset=self.dataset)
        return self.order_prefix
