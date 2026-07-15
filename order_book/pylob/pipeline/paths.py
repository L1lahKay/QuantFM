"""Object-key helpers for common zeus-cn-quote layouts."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date


def normalize_endpoint(endpoint: str) -> str:
    """Strip scheme and trailing slashes from a MinIO endpoint."""
    endpoint = endpoint.strip()
    for prefix in ("https://", "http://"):
        if endpoint.startswith(prefix):
            return endpoint[len(prefix) :].rstrip("/")
    return endpoint.rstrip("/")


def endpoint_is_secure(endpoint: str) -> bool:
    """Return whether the endpoint URL uses HTTPS."""
    return endpoint.strip().lower().startswith("https://")


def parse_date(value: str) -> date:
    """Parse ``YYYY-MM-DD`` or ``YYYY.MM.DD`` into a date."""
    for fmt in ("%Y-%m-%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    msg = f"unsupported date format: {value!r} (expected YYYY-MM-DD or YYYY.MM.DD)"
    raise ValueError(msg)


def hds_trade_prefix(
    value: str,
    *,
    mic: str = "XSHE",
    dataset: str = "china_stock",
) -> str:
    """Build the HDS transaction prefix for one trading day."""
    day = parse_date(value)
    date_str = day.strftime("%Y-%m-%d")
    return (
        f"HDS/SOURCE=zeus/DOMAIN=quote/DATASET={dataset}/"
        f"DATE={date_str}/MIC={mic}/TYPE=transaction/"
    )


def hds_order_prefix(
    value: str,
    *,
    mic: str = "XSHE",
    dataset: str = "china_stock",
) -> str:
    """Build the HDS order prefix for one trading day."""
    day = parse_date(value)
    date_str = day.strftime("%Y-%m-%d")
    return (
        f"HDS/SOURCE=zeus/DOMAIN=quote/DATASET={dataset}/"
        f"DATE={date_str}/MIC={mic}/TYPE=order/"
    )


def archive_object_key(
    value: str, *, partition: str = "1", name: str = "all.parquet"
) -> str:
    """Build an archive key: ``YYYY.MM/YYYY.MM.DD/default/<part>/<name>``."""
    day = parse_date(value)
    month_prefix = day.strftime("%Y.%m")
    day_prefix = day.strftime("%Y.%m.%d")
    return f"{month_prefix}/{day_prefix}/default/{partition}/{name}"


def zeus_default_object_key(
    value: str,
    *,
    partition: str,
    dataset: str = "china_stock",
    name: str = "all.parquet",
) -> str:
    """Build zeus-cn-quote HDS keys such as ``.../default/2/all.parquet``."""
    day = parse_date(value)
    month_prefix = day.strftime("%Y.%m")
    day_prefix = day.strftime("%Y.%m.%d")
    return (
        f"HDS/SOURCE=zeus/DOMAIN=quote/DATASET={dataset}/"
        f"{month_prefix}/{day_prefix}/default/{partition}/{name}"
    )


def zeus_default_trade_key(value: str, *, dataset: str = "china_stock") -> str:
    """Return the combined transaction file for one trading day (partition 2)."""
    return zeus_default_object_key(value, partition="2", dataset=dataset)


def zeus_default_order_key(value: str, *, dataset: str = "china_stock") -> str:
    """Return the combined order file for one trading day (partition 3)."""
    return zeus_default_object_key(value, partition="3", dataset=dataset)


def symbol_filename(market: str, symbol: str) -> str:
    """Return ``SH600000.parquet`` / ``SZ000001.parquet`` style filenames."""
    market_code = market.upper()
    if market_code not in {"SH", "SZ"}:
        msg = "market must be 'SH' or 'SZ'"
        raise ValueError(msg)
    return f"{market_code}{symbol.zfill(6)}.parquet"


def symbol_object_keys(
    prefix: str, market: str, symbols: tuple[str, ...]
) -> tuple[str, ...]:
    """Join a prefix with per-symbol parquet filenames."""
    if prefix and not prefix.endswith("/"):
        prefix = f"{prefix}/"
    return tuple(f"{prefix}{symbol_filename(market, symbol)}" for symbol in symbols)
