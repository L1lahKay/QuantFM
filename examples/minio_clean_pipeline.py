"""Download MinIO parquet and export cleaned PyLOB artifacts.

Minimal usage (same credentials as your Polars script)::

    export MINIO_ENDPOINT=192.168.2.11:9000
    export MINIO_ACCESS_KEY=...
    export MINIO_SECRET_KEY=...
    export MINIO_OBJECT_KEY=2026.02/2026.02.02/default/1/all.parquet
    export PYLOB_SYMBOLS=000001,000002
    export PYLOB_MARKET=SZ

    python examples/minio_clean_pipeline.py          # probe + clean
    python examples/minio_clean_pipeline.py --ls-only  # probe only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import polars as pl

from pylob.pipeline import MinioConfig, PipelineConfig, build_clean_dataset
from pylob.pipeline.paths import (
    archive_object_key,
    hds_order_prefix,
    hds_trade_prefix,
    symbol_object_keys,
    zeus_default_order_key,
    zeus_default_trade_key,
)
from pylob.pipeline.s3_io import build_storage_options

logger = logging.getLogger(__name__)


def _split_csv(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _minio_config_from_env() -> MinioConfig:
    return MinioConfig(
        endpoint=os.environ["MINIO_ENDPOINT"],
        access_key=os.environ["MINIO_ACCESS_KEY"],
        secret_key=os.environ["MINIO_SECRET_KEY"],
        secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
    )


def _candidate_keys() -> list[str]:
    bucket_keys: list[str] = []
    if key := os.getenv("MINIO_OBJECT_KEY"):
        bucket_keys.append(key.strip())

    for item in _split_csv(os.getenv("MINIO_TRADE_KEYS")):
        bucket_keys.append(item)
    for item in _split_csv(os.getenv("MINIO_ORDER_KEYS")):
        bucket_keys.append(item)

    date = os.getenv("PYLOB_DATE")
    if date:
        bucket_keys.append(zeus_default_trade_key(date))
        bucket_keys.append(zeus_default_order_key(date))

    if date:
        bucket_keys.append(archive_object_key(date))

    if date and (symbols := _split_csv(os.getenv("PYLOB_SYMBOLS", "000001"))):
        market = os.getenv("PYLOB_MARKET", "SZ")
        mic = os.getenv("PYLOB_MIC", "XSHE")
        trade_prefix = hds_trade_prefix(date, mic=mic)
        order_prefix = hds_order_prefix(date, mic=mic)
        bucket_keys.extend(symbol_object_keys(trade_prefix, market, symbols))
        bucket_keys.extend(symbol_object_keys(order_prefix, market, symbols))

    # de-dup while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for key in bucket_keys:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _probe_key(minio_config: MinioConfig, bucket: str, object_key: str) -> str:
    uri = f"s3://{bucket}/{object_key.lstrip('/')}"
    storage_options = build_storage_options(minio_config)
    try:
        schema = pl.scan_parquet(uri, storage_options=storage_options).collect_schema()
        rows = (
            pl.scan_parquet(uri, storage_options=storage_options)
            .select(pl.len())
            .collect()
            .item()
        )
        return f"OK rows={rows} cols={len(schema.names())}"
    except Exception as exc:
        message = str(exc)
        if (
            "403" in message
            or "AccessDenied" in message
            or "privileges" in message.lower()
        ):
            return "DENIED"
        if "404" in message or "not found" in message.lower():
            return "MISSING"
        return f"ERROR {type(exc).__name__}"


def probe_remote(minio_config: MinioConfig) -> list[str]:
    """Print probe results and return object keys that are readable."""
    bucket = os.getenv("MINIO_BUCKET", "zeus-cn-quote")
    print(f"endpoint={minio_config.endpoint} bucket={bucket}")
    print("Probing known object keys (ListBucket not required):")

    accessible: list[str] = []
    for object_key in _candidate_keys():
        status = _probe_key(minio_config, bucket, object_key)
        mark = "✓" if status.startswith("OK") else "✗"
        print(f"  {mark} s3://{bucket}/{object_key}  [{status}]")
        if status.startswith("OK"):
            accessible.append(object_key)

    if not accessible:
        print(
            "\nNo readable objects. Set MINIO_OBJECT_KEY to the exact path "
            "from your working Polars script."
        )
    return accessible


def _pipeline_config_for_keys(accessible: list[str]) -> PipelineConfig:
    bucket = os.getenv("MINIO_BUCKET", "zeus-cn-quote")
    symbols = tuple(os.getenv("PYLOB_SYMBOLS", "000001").split(","))
    explicit_trade = _split_csv(os.getenv("MINIO_TRADE_KEYS"))
    explicit_order = _split_csv(os.getenv("MINIO_ORDER_KEYS"))
    date = os.getenv("PYLOB_DATE")
    layout = os.getenv("PYLOB_LAYOUT")

    zeus_trade = [k for k in accessible if "/default/2/all.parquet" in k]
    zeus_order = [k for k in accessible if "/default/3/all.parquet" in k]

    if layout == "zeus_default" or (zeus_trade and zeus_order):
        layout = "zeus_default"
        explicit_trade = explicit_trade or tuple(zeus_trade[:1])
        explicit_order = explicit_order or tuple(zeus_order[:1])
    elif explicit_trade and explicit_order:
        layout = layout or "hds"
    elif len(accessible) == 1 or os.getenv("MINIO_OBJECT_KEY"):
        layout = layout or "archive"
    elif len(accessible) >= 2:
        trade_keys = [
            k for k in accessible if "/TYPE=transaction/" in k or "/trade/" in k.lower()
        ]
        order_keys = [
            k for k in accessible if "/TYPE=order/" in k or "/order/" in k.lower()
        ]
        if trade_keys and order_keys:
            layout = layout or "hds"
            explicit_trade = explicit_trade or tuple(trade_keys)
            explicit_order = explicit_order or tuple(order_keys)
        else:
            layout = layout or "archive"
    elif date:
        layout = layout or "zeus_default"
    else:
        layout = layout or "archive"

    combined = os.getenv("MINIO_OBJECT_KEY") or (
        accessible[0] if len(accessible) == 1 else None
    )

    return PipelineConfig(
        bucket=bucket,
        trade_prefix=os.getenv("MINIO_TRADE_PREFIX", "raw/trade/"),
        order_prefix=os.getenv("MINIO_ORDER_PREFIX", "raw/order/"),
        output_dir=Path(os.getenv("PYLOB_OUTPUT_DIR", "data/clean")),
        symbols=symbols,
        market=os.getenv("PYLOB_MARKET", "SZ"),
        layout=layout,
        date=date,
        mic=os.getenv("PYLOB_MIC", "XSHE"),
        combined_object_key=combined,
        trade_object_keys=explicit_trade,
        order_object_keys=explicit_order,
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Probe MinIO, download parquet, export cleaned data."
    )
    parser.add_argument(
        "--ls-only",
        action="store_true",
        help="Only probe remote objects, do not clean.",
    )
    args = parser.parse_args()

    for var in ("MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY"):
        if not os.getenv(var):
            print(f"Missing required env var: {var}", file=sys.stderr)
            return 1

    minio_config = _minio_config_from_env()
    accessible = probe_remote(minio_config)
    if args.ls_only or not accessible:
        return 0 if accessible else 1

    config = _pipeline_config_for_keys(accessible)
    logger.info("Cleaning symbols=%s -> %s", config.symbols, config.output_dir)
    build_clean_dataset(minio_config, config)

    output_dir = Path(os.getenv("PYLOB_OUTPUT_DIR", "data/clean"))
    files = sorted(output_dir.rglob("*.parquet"))
    print(f"\nDone. Wrote {len(files)} parquet file(s):")
    for path in files:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
