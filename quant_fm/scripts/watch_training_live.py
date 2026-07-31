"""在终端低干扰地实时展示 QuantFM 预训练成效。"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import yaml

from quant_fm.monitoring.training import ERROR_PATTERNS, UPDATE_RE, VALIDATION_RE

if TYPE_CHECKING:
    from collections.abc import Sequence


DEFAULT_CONFIG = Path("quant_fm/runs/backbone_moe_v1/config.yaml")
STEP_CHECKPOINT_RE = re.compile(r"^step(?P<step>\d+)\.pt$")
SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _tail_text(path: Path, *, max_bytes: int) -> str:
    """只读取日志尾部，避免随训练时长增加而持续加重 I/O。"""
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - max_bytes))
        payload = stream.read()
    return payload.decode("utf-8", errors="replace")


def parse_log_series(
    text: str,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    """解析用于看板的 update 与 validation 序列。"""
    updates: list[dict[str, float | int]] = [
        {
            "update": int(match.group("update")),
            "micro": int(match.group("micro")),
            "tokens": int(match.group("tokens")),
            "lr": float(match.group("lr")),
            "loss": float(match.group("loss")),
            "aux": float(match.group("aux")),
        }
        for match in UPDATE_RE.finditer(text)
    ]
    validations: list[dict[str, float | int]] = [
        {
            "update": int(match.group("update")),
            "val_loss": float(match.group("val_loss")),
        }
        for match in VALIDATION_RE.finditer(text)
    ]
    return updates, validations


def active_error_codes(text: str) -> list[str]:
    """只报告最后一条 update 之后的错误，忽略已重启恢复的旧故障。"""
    matches = list(UPDATE_RE.finditer(text))
    active_text = text[matches[-1].start() :] if matches else text
    return [
        code
        for code, pattern in ERROR_PATTERNS
        if pattern.search(active_text) is not None
    ]


def sparkline(values: Sequence[float]) -> str:
    """把数值序列压缩成单行 Unicode 趋势图。"""
    if not values:
        return "尚无数据"
    if len(values) == 1 or math.isclose(min(values), max(values)):
        return SPARK_CHARS[len(SPARK_CHARS) // 2] * len(values)
    low = min(values)
    span = max(values) - low
    return "".join(
        SPARK_CHARS[round((value - low) / span * (len(SPARK_CHARS) - 1))]
        for value in values
    )


def progress_bar(current: int, total: int, *, width: int = 38) -> str:
    """渲染固定宽度进度条。"""
    if total <= 0:
        return f"[{'-' * width}] n/a"
    fraction = min(1.0, max(0.0, current / total))
    filled = round(width * fraction)
    return f"[{'█' * filled}{'·' * (width - filled)}] {fraction:6.2%}"


def checkpoint_speed(run_dir: Path, *, sample_count: int = 5) -> float | None:
    """用最近几个周期 checkpoint 的落盘时间估计 update/min。"""
    samples: list[tuple[float, int]] = []
    if not run_dir.is_dir():
        return None
    for path in run_dir.glob("step*.pt"):
        match = STEP_CHECKPOINT_RE.match(path.name)
        if match is None:
            continue
        try:
            samples.append((path.stat().st_mtime, int(match.group("step"))))
        except FileNotFoundError:
            continue
    samples.sort(key=lambda item: item[1])
    selected = samples[-sample_count:]
    if len(selected) < 2:
        return None
    elapsed_minutes = (selected[-1][0] - selected[0][0]) / 60
    advanced = selected[-1][1] - selected[0][1]
    if elapsed_minutes <= 0 or advanced <= 0:
        return None
    return advanced / elapsed_minutes


def runtime_speed(samples: Sequence[tuple[float, int]]) -> float | None:
    """用看板启动后的观测计算短期 update/min。"""
    if len(samples) < 2:
        return None
    elapsed_minutes = (samples[-1][0] - samples[0][0]) / 60
    advanced = samples[-1][1] - samples[0][1]
    if elapsed_minutes < 0.5 or advanced <= 0:
        return None
    return advanced / elapsed_minutes


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours}小时{minutes:02d}分" if hours else f"{minutes}分钟"


def _format_bytes(value: int) -> str:
    gib = value / (1024**3)
    return f"{gib:.1f} GiB"


def _change(current: float, previous: float) -> str:
    delta = current - previous
    percent = delta / previous * 100 if previous else 0.0
    direction = "↓ 改善" if delta < 0 else ("↑ 变差" if delta > 0 else "→ 持平")
    return f"{delta:+.4f} ({percent:+.2f}%, {direction})"


def _gpu_rows() -> tuple[list[str], str | None]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return [], "nvidia-smi 不可用"
    result = subprocess.run(
        [
            executable,
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return [], result.stderr.strip() or "nvidia-smi 查询失败"
    rows: list[str] = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 5:
            continue
        index, utilization, used, total, temperature = values
        rows.append(
            f"GPU {index}: util {utilization:>3}%  "
            f"显存 {used:>6}/{total:<6} MiB  温度 {temperature:>2}°C"
        )
    return rows, None


def _checkpoints(run_dir: Path) -> tuple[str, int | None]:
    numbered: list[tuple[int, Path]] = []
    for path in run_dir.glob("step*.pt") if run_dir.is_dir() else ():
        match = STEP_CHECKPOINT_RE.match(path.name)
        if match is not None:
            numbered.append((int(match.group("step")), path))
    latest_step = max(numbered, default=(None, None), key=lambda item: item[0])[0]
    names = []
    if latest_step is not None:
        names.append(f"step{latest_step}.pt")
    if (run_dir / "best.pt").is_file():
        names.append("best.pt")
    if (run_dir / "final.pt").is_file():
        names.append("final.pt")
    return ", ".join(names) if names else "尚无 checkpoint", latest_step


def render_dashboard(
    *,
    config_path: Path,
    log_path: Path,
    run_dir: Path,
    max_updates: int,
    updates: Sequence[dict[str, float | int]],
    validations: Sequence[dict[str, float | int]],
    speed: float | None,
    speed_source: str,
    timezone: ZoneInfo,
    train_window: int,
    gpu_rows: Sequence[str],
    gpu_error: str | None,
    errors: Sequence[str],
) -> str:
    """渲染一次完整看板。"""
    now = datetime.now(tz=timezone)
    latest = updates[-1] if updates else None
    current = int(latest["update"]) if latest else 0
    remaining = max(0, max_updates - current)
    eta_text = "等待速度样本"
    if speed and speed > 0:
        seconds_left = remaining / speed * 60
        eta = datetime.fromtimestamp(time.time() + seconds_left, tz=timezone)
        eta_text = f"{eta:%m-%d %H:%M}（约 {_format_duration(seconds_left)}）"

    lines = [
        "================ QuantFM 训练实时成效看板 ================",
        f"北京时间: {now:%Y-%m-%d %H:%M:%S}    刷新后按 Ctrl-C 仅退出看板",
        f"配置: {config_path}",
        f"日志: {log_path}",
        "",
        "1) 总进度",
        f"   {progress_bar(current, max_updates)}",
        f"   update {current:,}/{max_updates:,}，剩余 {remaining:,}",
        f"   速度: {f'{speed:.2f} update/min' if speed else '采样中'} ({speed_source})",
        f"   预计完成: {eta_text}",
        "",
        "2) 训练 loss（单步波动大，应看滑动均值）",
    ]
    if latest:
        losses = [float(item["loss"]) for item in updates]
        recent = losses[-train_window:]
        previous = losses[-2 * train_window : -train_window]
        lines.extend(
            [
                f"   最新: {float(latest['loss']):.4f}    "
                f"最近 {len(recent)} 点均值: {fmean(recent):.4f}",
                f"   lr: {float(latest['lr']):.3e}    aux: {float(latest['aux']):.4f}    "
                f"tokens: {int(latest['tokens']):,}",
                f"   最近趋势: {sparkline(recent)}  （低=好）",
            ]
        )
        if previous:
            lines.append(
                f"   均值对比前 {len(previous)} 点: "
                f"{_change(fmean(recent), fmean(previous))}"
            )
    else:
        lines.append("   尚未解析到 INFO update 日志")

    lines.extend(["", "3) 验证成效（判断模型是否真正改善的主要指标）"])
    if validations:
        values = [float(item["val_loss"]) for item in validations]
        latest_val = validations[-1]
        best_val = min(validations, key=lambda item: float(item["val_loss"]))
        lines.extend(
            [
                f"   最新: {float(latest_val['val_loss']):.4f} @ {int(latest_val['update']):,}",
                f"   最佳: {float(best_val['val_loss']):.4f} @ {int(best_val['update']):,}",
                f"   val 趋势: {sparkline(values[-24:])}  （左→右，低=好）",
            ]
        )
        if len(validations) >= 2:
            lines.append("   较上次验证: " + _change(values[-1], values[-2]))
        recent_vals = validations[-6:]
        lines.append(
            "   最近记录: "
            + "  ".join(
                f"{int(item['update'])}:{float(item['val_loss']):.4f}"
                for item in recent_vals
            )
        )
    else:
        lines.append("   尚未产生 validation 记录")

    disk = shutil.disk_usage(run_dir if run_dir.exists() else run_dir.parent)
    checkpoint_text, checkpoint_step = _checkpoints(run_dir)
    log_age = time.time() - log_path.stat().st_mtime if log_path.is_file() else None
    final_exists = (run_dir / "final.pt").is_file()
    if final_exists:
        health = "已完成"
    elif log_age is None:
        health = "异常：日志不存在"
    elif log_age < 300:
        health = "运行中"
    else:
        health = f"注意：日志 {_format_duration(log_age)} 未更新"

    lines.extend(
        [
            "",
            "4) 运行健康",
            f"   状态: {health}    日志延迟: "
            f"{f'{log_age:.0f}s' if log_age is not None else 'n/a'}",
            f"   checkpoint: {checkpoint_text}",
            f"   磁盘可用: {_format_bytes(disk.free)} ({disk.free / disk.total:.1%})",
        ]
    )
    if checkpoint_step is not None and current - checkpoint_step > 4000:
        lines.append(
            f"   注意：最新可恢复 checkpoint 落后 {current - checkpoint_step:,} updates"
        )
    if gpu_rows:
        lines.extend(f"   {row}" for row in gpu_rows)
    elif gpu_error:
        lines.append(f"   GPU: {gpu_error}")
    if errors:
        lines.append("   严重日志信号: " + ", ".join(errors))
    lines.append("===========================================================")
    return "\n".join(lines)


def _load_config(config_path: Path) -> tuple[Path, int]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(cfg["runtime"]["out_dir"])
    optim = cfg["optim"]
    max_updates = int(optim.get("max_update_steps", optim.get("max_steps", 0)))
    if max_updates <= 0:
        msg = "配置中缺少有效的 optim.max_update_steps/max_steps"
        raise ValueError(msg)
    return run_dir, max_updates


def main() -> None:
    """持续刷新训练看板，Ctrl-C 安全退出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--log", type=Path, help="默认使用 out_dir 上一级的 train.log")
    parser.add_argument("--interval", type=float, default=10.0, help="刷新秒数")
    parser.add_argument("--train-window", type=int, default=20, help="loss 均值窗口")
    parser.add_argument(
        "--max-log-mib", type=int, default=16, help="最多读取日志尾部 MiB"
    )
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--once", action="store_true", help="只打印一次，便于检查")
    parser.add_argument("--no-clear", action="store_true", help="刷新时不清屏")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval 必须大于 0")
    if args.train_window <= 0:
        parser.error("--train-window 必须大于 0")
    if args.max_log_mib <= 0:
        parser.error("--max-log-mib 必须大于 0")

    config_path = args.config
    run_dir, max_updates = _load_config(config_path)
    log_path = args.log or run_dir.parent / "train.log"
    timezone = ZoneInfo(args.timezone)
    historical_speed = checkpoint_speed(run_dir)
    observations: deque[tuple[float, int]] = deque(maxlen=12)
    last_update: int | None = None

    try:
        while True:
            text = _tail_text(log_path, max_bytes=args.max_log_mib << 20)
            updates, validations = parse_log_series(text)
            current = int(updates[-1]["update"]) if updates else 0
            if current != last_update:
                observations.append((time.time(), current))
                last_update = current
            live_speed = runtime_speed(observations)
            speed = live_speed or historical_speed
            speed_source = "看板实时观测" if live_speed else "最近 checkpoint 历史"
            if speed is None:
                speed_source = "等待至少两个进度样本"

            gpu_rows, gpu_error = _gpu_rows()
            errors = active_error_codes(text)
            output = render_dashboard(
                config_path=config_path,
                log_path=log_path,
                run_dir=run_dir,
                max_updates=max_updates,
                updates=updates,
                validations=validations,
                speed=speed,
                speed_source=speed_source,
                timezone=timezone,
                train_window=args.train_window,
                gpu_rows=gpu_rows,
                gpu_error=gpu_error,
                errors=errors,
            )
            if not args.no_clear and not args.once:
                print("\033[H\033[2J", end="")
            print(output, flush=True)
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n看板已退出；训练进程未受影响。")


if __name__ == "__main__":
    main()
