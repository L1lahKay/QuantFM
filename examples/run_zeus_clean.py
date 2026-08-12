#!/usr/bin/env python3
"""Run PyLOB cleaning for zeus-cn-quote default/2 + default/3 layout."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pylob.pipeline import PipelineConfig, build_clean_dataset  # noqa: E402

from quant_fm.scripts.minio_config import load_read_config, read_bucket  # noqa: E402


def main() -> int:
    """Run one environment-configured zeus-default cleaning job."""
    minio_config = load_read_config()
    date = os.getenv("PYLOB_DATE", "2026-02-02")
    symbols = tuple(os.getenv("PYLOB_SYMBOLS", "000001,000002").split(","))

    config = PipelineConfig(
        bucket=os.getenv("MINIO_BUCKET", read_bucket()),
        trade_prefix="",
        order_prefix="",
        output_dir=Path(os.getenv("PYLOB_OUTPUT_DIR", "data/clean")),
        symbols=symbols,
        market=os.getenv("PYLOB_MARKET", "SZ"),
        layout="zeus_default",
        date=date,
    )

    print(f"date={date} symbols={symbols} layout=zeus_default")
    print(f"trade={config.resolved_trade_keys()[0]}")
    print(f"order={config.resolved_order_keys()[0]}")
    build_clean_dataset(minio_config, config)

    outputs = sorted(Path(config.output_dir).rglob("*.parquet"))
    print(f"\nDone. Wrote {len(outputs)} files:")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
