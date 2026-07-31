"""比较同一固定窗口上的候选/基线预训练评估并执行非劣门槛。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_fm.monitoring.acceptance import (
    compare_pretrain_evaluations,
    render_acceptance_report,
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    """写出 JSON/Markdown；非劣失败时以退出码 2 阻止 test 阶段。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    result = compare_pretrain_evaluations(
        args.candidate,
        args.baseline,
        noninferiority_tolerance=args.tolerance,
    )
    _atomic_text(
        args.out,
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    markdown = args.markdown or args.out.with_suffix(".md")
    _atomic_text(markdown, render_acceptance_report(result))
    print(args.out)
    if not result["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
