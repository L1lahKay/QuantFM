"""
低干扰训练看门狗：输出结构化状态、checkpoint 登记和 Markdown 报告。

该入口默认只执行一次，适合 cron/systemd/tmux 周期调用。它不会杀进程、重启训练、
加载大 checkpoint 或启动 GPU 评估。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from quant_fm._io import atomic_write_text
from quant_fm.monitoring.training import (
    build_run_metadata,
    collect_training_status,
    render_training_report,
)


def _paths(config_path: Path) -> tuple[Path, Path, Path, Path]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(cfg["runtime"]["out_dir"])
    return (
        run_dir,
        run_dir / "training_status.json",
        run_dir / "run_metadata.json",
        run_dir / "training_report.md",
    )


def _write_observation(
    *,
    config: Path,
    log: Path | None,
    tmux_session: str | None,
    world_size: int,
    stall_seconds: int,
    disk_warning_percent: float,
    status_path: Path,
    metadata_path: Path,
    report_path: Path,
    refresh_metadata: bool,
) -> dict[str, object]:
    status = collect_training_status(
        config,
        log_path=log,
        tmux_session=tmux_session,
        stall_seconds=stall_seconds,
        disk_warning_percent=disk_warning_percent,
    )
    if refresh_metadata or not metadata_path.is_file():
        metadata = build_run_metadata(config, world_size=world_size)
        atomic_write_text(
            metadata_path,
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    atomic_write_text(
        status_path,
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(report_path, render_training_report(status, metadata))
    latest = status["progress"].get("latest_update")  # type: ignore[index]
    summary = {
        "state": status["state"],
        "update": latest.get("update") if latest else None,
        "loss": latest.get("loss") if latest else None,
        "alerts": status["alerts"],
        "status_path": str(status_path.resolve()),
        "report_path": str(report_path.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return status


def main() -> None:
    """执行一次或周期性采集；告警通过退出码暴露给外部调度器。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--tmux-session")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--stall-seconds", type=int, default=900)
    parser.add_argument("--disk-warning-percent", type=float, default=20.0)
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="0 表示只检查一次；正数表示循环间隔秒数",
    )
    parser.add_argument("--status-out", type=Path)
    parser.add_argument("--metadata-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--refresh-metadata", action="store_true")
    args = parser.parse_args()
    if args.world_size < 1:
        parser.error("--world-size must be positive")
    if args.interval < 0:
        parser.error("--interval must be non-negative")

    _, default_status, default_metadata, default_report = _paths(args.config)
    while True:
        status = _write_observation(
            config=args.config,
            log=args.log,
            tmux_session=args.tmux_session,
            world_size=args.world_size,
            stall_seconds=args.stall_seconds,
            disk_warning_percent=args.disk_warning_percent,
            status_path=args.status_out or default_status,
            metadata_path=args.metadata_out or default_metadata,
            report_path=args.report_out or default_report,
            refresh_metadata=args.refresh_metadata,
        )
        if args.interval == 0:
            if status["state"] == "critical":
                raise SystemExit(2)
            if status["state"] == "warning":
                raise SystemExit(1)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
