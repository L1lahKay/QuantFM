"""Fail closed unless a pretraining noninferiority artifact is an explicit PASS."""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_fm.monitoring.acceptance import validate_pretrain_acceptance


def main() -> None:
    """Validate the persisted acceptance schema for strict downstream entrypoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = validate_pretrain_acceptance(args.path)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"PASS: {args.path} ({payload['primary_metric']})")


if __name__ == "__main__":
    main()
