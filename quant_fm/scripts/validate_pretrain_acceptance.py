"""Fail closed unless a pretraining noninferiority artifact is an explicit PASS."""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_fm.monitoring.acceptance import (
    DEFAULT_NONINFERIORITY_TOLERANCE,
    validate_pretrain_acceptance,
)


def main() -> None:
    """Validate the persisted acceptance schema for strict downstream entrypoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument(
        "--expected-tolerance",
        type=float,
        default=DEFAULT_NONINFERIORITY_TOLERANCE,
        help="independent noninferiority policy threshold (default: 0.01)",
    )
    args = parser.parse_args()
    try:
        payload = validate_pretrain_acceptance(
            args.path,
            expected_noninferiority_tolerance=args.expected_tolerance,
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"PASS: {args.path} ({payload['primary_metric']})")


if __name__ == "__main__":
    main()
