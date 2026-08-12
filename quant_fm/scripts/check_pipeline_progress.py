"""Report quant_fm data-pipeline progress (clean → events → tokens → manifest)."""

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKDIR = ROOT / "quant_fm" / "runs" / "medium_300m"
DEFAULT_DATES = ROOT / "quant_fm" / "data" / "medium_300m_22_dates.txt"

_CLEAN_PROGRESS = re.compile(
    r"clean progress (?P<market>SZ|SH) (?P<done>\d+)/(?P<total>\d+)"
)
_CLEAN_START = re.compile(r"clean (?P<date>\d{4}-\d{2}-\d{2}) (?P<market>SZ|SH)")
_CANON = re.compile(
    r"canonicaliz(?:e|ing) (?P<n>\d+) symbols for (?P<date>\d{4}-\d{2}-\d{2})"
    r"|canonicalize progress (?P<date2>\d{4}-\d{2}-\d{2}) (?P<done>\d+)/(?P<total>\d+)"
)


def _load_dates(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _dir_size_gb(path: Path) -> float | None:
    if not path.exists():
        return None
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024**3)


def _pipeline_running(workdir: Path) -> bool:
    needle = f"run_medium.*{workdir.name}"
    try:
        out = subprocess.run(
            ["pgrep", "-af", needle],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    for line in out.stdout.splitlines():
        if "check_pipeline_progress" in line:
            continue
        if "pgrep" in line:
            continue
        return True
    return False


def _parse_log_tail(log_path: Path, *, tail_lines: int = 4000) -> dict:
    info: dict = {
        "clean_market": None,
        "clean_done": None,
        "clean_total": None,
        "clean_date": None,
        "canon_date": None,
        "canon_done": None,
        "canon_total": None,
        "last_line": "",
    }
    if not log_path.is_file():
        return info

    lines = log_path.read_text(errors="replace").splitlines()
    info["last_line"] = lines[-1] if lines else ""
    for line in reversed(lines[-tail_lines:]):
        m = _CLEAN_PROGRESS.search(line)
        if m and info["clean_done"] is None:
            info["clean_market"] = m.group("market")
            info["clean_done"] = int(m.group("done"))
            info["clean_total"] = int(m.group("total"))
        m = _CLEAN_START.search(line)
        if m and info["clean_date"] is None:
            info["clean_date"] = m.group("date")
        m = _CANON.search(line)
        if m and info["canon_date"] is None:
            if m.group("date"):
                info["canon_date"] = m.group("date")
                info["canon_total"] = int(m.group("n"))
            else:
                info["canon_date"] = m.group("date2")
                info["canon_done"] = int(m.group("done"))
                info["canon_total"] = int(m.group("total"))
    return info


def _fmt_pct(done: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{100.0 * done / total:.1f}%"


def report(
    *,
    workdir: Path,
    dates: list[str],
    log_path: Path | None = None,
) -> int:
    """Print progress summary; return exit code 0 ok / 1 if data incomplete."""
    workdir = workdir.resolve()
    log_path = log_path or (workdir / "pipeline.log")
    done_dir = workdir / "data" / ".done"
    events_dir = workdir / "events"
    tokens_dir = workdir / "tokens"
    manifest = workdir / "data" / "manifest.json"
    vocab = workdir / "data" / "vocab.json"

    done_dates = (
        sorted(p.name for p in done_dir.glob("*") if p.is_file())
        if done_dir.is_dir()
        else []
    )
    done_set = set(done_dates)
    pending = [d for d in dates if d not in done_set]
    current = pending[0] if pending else None

    running = _pipeline_running(workdir)
    log_info = _parse_log_tail(log_path)

    # per-day event shard counts for current / recent
    event_counts: dict[str, int] = {}
    if events_dir.is_dir():
        for p in events_dir.rglob("*.parquet"):
            event_counts[p.stem] = event_counts.get(p.stem, 0) + 1

    print("== QuantFM 数据流水线进度 ==")
    print(f"workdir:  {workdir}")
    print(f"log:      {log_path}")
    print(f"进程:     {'运行中' if running else '未检测到'}")
    print()
    print(
        f"日期:     {len(done_dates)}/{len(dates)} 完成 ({_fmt_pct(len(done_dates), len(dates))})"
    )
    if done_dates:
        print(f"已完成:   {', '.join(done_dates)}")
    if current:
        print(f"进行中:   {current}")
    if len(pending) > 1:
        print(f"待处理:   {', '.join(pending[1:])}")
    print()

    # stage detail for current day
    if current:
        n_events = event_counts.get(current, 0)
        clean_exists = (workdir / "clean" / current).is_dir()
        print(f"— 当日细项 ({current}) —")
        if log_info["clean_date"] == current and log_info["clean_done"] is not None:
            print(
                f"  洗股:     {log_info['clean_market']} "
                f"{log_info['clean_done']}/{log_info['clean_total']} "
                f"({_fmt_pct(log_info['clean_done'], log_info['clean_total'] or 1)})"
            )
        elif clean_exists:
            print("  洗股:     clean/ 目录存在（详见 log）")
        if log_info["canon_date"] == current:
            if log_info["canon_done"] is not None and log_info["canon_total"]:
                print(
                    f"  规范化:   {log_info['canon_done']}/{log_info['canon_total']} "
                    f"({_fmt_pct(log_info['canon_done'], log_info['canon_total'])})"
                )
            elif log_info["canon_total"]:
                print(f"  规范化:   待处理 {log_info['canon_total']} 股")
        print(f"  events:   {n_events} 个 symbol-day parquet")
        print()

    # ETA from .done mtimes
    if len(done_dates) >= 2 and pending:
        stamps = []
        for d in done_dates[-4:]:
            p = done_dir / d
            if p.exists():
                stamps.append((d, datetime.fromtimestamp(p.stat().st_mtime, tz=UTC)))
        if len(stamps) >= 2:
            deltas = [
                (stamps[i][1] - stamps[i - 1][1]).total_seconds() / 60.0
                for i in range(1, len(stamps))
            ]
            avg_min = sum(deltas) / len(deltas)
            eta_h = avg_min * len(pending) / 60.0
            print(f"近期均速: ~{avg_min:.0f} min/天（基于最近 {len(deltas)} 天）")
            print(f"预估剩余: ~{eta_h:.1f} h（剩 {len(pending)} 天，不含 tokenize）")
            print()

    # downstream artifacts
    total_event_shards = sum(event_counts.values())
    print("— 产物 —")
    for label, path in (
        ("events", events_dir),
        ("clean", workdir / "clean"),
        ("tokens", tokens_dir),
    ):
        gb = _dir_size_gb(path)
        if gb is not None:
            print(f"  {label:8s} {gb:6.1f} GB")
    print(f"  event shards (all dates): {total_event_shards:,}")
    print(f"  manifest: {'就绪' if manifest.is_file() else '未生成'}")
    print(f"  vocab:    {'就绪' if vocab.is_file() else '未生成'}")
    print(
        f"  tokens:   {'就绪' if tokens_dir.is_dir() and any(tokens_dir.rglob('*.parquet')) else '未生成'}"
    )
    print()

    if log_info["last_line"]:
        print("— 日志末行 —")
        print(f"  {log_info['last_line'][:200]}")
        print()

    if manifest.is_file() and vocab.is_file():
        print("状态: 数据阶段已完成，可开始/继续训练")
        return 0
    if running:
        print("状态: 数据阶段进行中")
        return 0
    if pending:
        print("状态: 数据未完成且进程未运行 — 可用 --resume 重启流水线")
        return 1
    return 0


def main() -> None:
    """Parse CLI arguments and print the current data-pipeline status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=DEFAULT_WORKDIR,
        help=f"流水线工作目录（默认 {DEFAULT_WORKDIR.relative_to(ROOT)}）",
    )
    parser.add_argument(
        "--dates-file",
        type=Path,
        default=None,
        help="交易日列表；默认从 workdir 推断或 medium_300m_22_dates.txt",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="pipeline.log 路径（默认 workdir/pipeline.log）",
    )
    args = parser.parse_args()

    dates_file = args.dates_file
    if dates_file is None:
        candidate = args.workdir / "dates.txt"
        dates_file = candidate if candidate.is_file() else DEFAULT_DATES

    dates = _load_dates(dates_file)
    raise SystemExit(report(workdir=args.workdir, dates=dates, log_path=args.log))


if __name__ == "__main__":
    main()
