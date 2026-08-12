"""比较同一固定窗口上的候选/基线预训练评估并执行非劣门槛。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_fm._io import atomic_write_text
from quant_fm.monitoring.acceptance import (
    DEFAULT_NONINFERIORITY_TOLERANCE,
    compare_pretrain_evaluations,
    render_acceptance_report,
)


def main() -> None:
    """写出 JSON/Markdown；非劣失败时以退出码 2 阻止 test 阶段。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_NONINFERIORITY_TOLERANCE,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    result = compare_pretrain_evaluations(
        args.candidate,
        args.baseline,
        noninferiority_tolerance=args.tolerance,
    )
    atomic_write_text(
        args.out,
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    markdown = args.markdown or args.out.with_suffix(".md")
    atomic_write_text(markdown, render_acceptance_report(result))
    print(args.out)
    if not result["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
