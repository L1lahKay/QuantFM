"""低干扰的训练状态采集、checkpoint 登记和验收报告。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from quant_fm._io import atomic_write_json

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


UPDATE_RE = re.compile(
    r"\bINFO update (?P<update>\d+) micro (?P<micro>\d+) "
    r"tokens (?P<tokens>\d+) lr (?P<lr>[-+0-9.eE]+) "
    r"loss (?P<loss>[-+0-9.eE]+) aux (?P<aux>[-+0-9.eE]+)"
)
VALIDATION_RE = re.compile(
    r"\bINFO update (?P<update>\d+) val_loss (?P<val_loss>[-+0-9.eE]+)"
)
ERROR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("traceback", re.compile(r"Traceback \(most recent call last\):")),
    ("cuda_oom", re.compile(r"CUDA out of memory|OutOfMemoryError", re.I)),
    ("nccl_error", re.compile(r"NCCL.*(?:error|timeout)|DistBackendError", re.I)),
    ("non_finite_loss", re.compile(r"\bloss\s+(?:nan|[-+]?inf)\b", re.I)),
)
STEP_CHECKPOINT_RE = re.compile(r"^step(?P<step>\d+)\.pt$")


@dataclass(frozen=True, slots=True)
class Alert:
    """一个结构化训练告警。"""

    severity: str
    code: str
    message: str


def _utc_iso(timestamp: float | None = None) -> str:
    value = (
        datetime.now(tz=UTC)
        if timestamp is None
        else datetime.fromtimestamp(timestamp, tz=UTC)
    )
    return value.isoformat()


def _tail_text(path: Path, *, max_bytes: int = 2 << 20) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - max_bytes))
        payload = stream.read()
    return payload.decode("utf-8", errors="replace")


def parse_training_log(text: str) -> dict[str, Any]:
    """解析训练日志中的 update、验证和致命错误。"""
    updates = [
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
    validations = [
        {
            "update": int(match.group("update")),
            "val_loss": float(match.group("val_loss")),
        }
        for match in VALIDATION_RE.finditer(text)
    ]
    errors = [
        {"code": code, "match": match.group(0)}
        for code, pattern in ERROR_PATTERNS
        if (match := pattern.search(text)) is not None
    ]
    best_validation = (
        min(validations, key=lambda item: (item["val_loss"], item["update"]))
        if validations
        else None
    )
    return {
        "first_update": updates[0] if updates else None,
        "latest_update": updates[-1] if updates else None,
        "latest_validation": validations[-1] if validations else None,
        "best_validation": best_validation,
        "updates_observed": len(updates),
        "validations_observed": len(validations),
        "errors": errors,
    }


def _checkpoint_kind(path: Path) -> tuple[str, int | None, bool]:
    match = STEP_CHECKPOINT_RE.match(path.name)
    if match is not None:
        return "periodic_resume", int(match.group("step")), True
    if path.name == "final_resume.pt":
        return "final_resume", None, True
    if path.name == "best.pt":
        return "best_inference", None, False
    if path.name == "final.pt":
        return "final_inference", None, False
    return "other", None, False


class CheckpointRegistry:
    """只读取文件元数据的轻量 checkpoint 登记表。"""

    VERSION = "1.0"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        """加载既有登记表；不存在时返回空表。"""
        if not self.path.is_file():
            return {"registry_version": self.VERSION, "checkpoints": []}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("registry_version") != self.VERSION:
            msg = "unsupported checkpoint registry version"
            raise ValueError(msg)
        return payload

    def update(
        self,
        run_dir: Path,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """扫描 ``*.pt`` 并以连续两次大小不变作为写盘稳定信号。"""
        observed_at = now or datetime.now(tz=UTC)
        previous = {item["name"]: item for item in self.load().get("checkpoints", [])}
        checkpoints: list[dict[str, Any]] = []
        for path in sorted(Path(run_dir).glob("*.pt")):
            stat = path.stat()
            old = previous.get(path.name, {})
            unchanged = (
                old.get("size_bytes") == stat.st_size
                and old.get("mtime_ns") == stat.st_mtime_ns
            )
            observations = (
                int(old.get("stable_observations", 1)) + 1 if unchanged else 1
            )
            kind, step, resumable = _checkpoint_kind(path)
            checkpoints.append(
                {
                    "name": path.name,
                    "path": str(path.resolve()),
                    "kind": kind,
                    "step": step,
                    "resumable": resumable,
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "mtime_utc": _utc_iso(stat.st_mtime),
                    "first_seen_utc": old.get(
                        "first_seen_utc", observed_at.isoformat()
                    ),
                    "last_seen_utc": observed_at.isoformat(),
                    "stable_observations": observations,
                    "stable": observations >= 2,
                }
            )
        payload = {
            "registry_version": self.VERSION,
            "updated_utc": observed_at.isoformat(),
            "stability_rule": "same size and mtime in two consecutive observations",
            "checkpoints": checkpoints,
        }
        atomic_write_json(self.path, payload)
        return payload


def probe_tmux_session(session: str | None) -> bool | None:
    """检查 tmux 会话；未指定或 tmux 不可用时返回 ``None``。"""
    if not session or shutil.which("tmux") is None:
        return None
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def training_process_alive(config_path: Path) -> bool:
    """通过 ``/proc`` 查找使用指定配置的预训练 rank。"""
    wanted = str(Path(config_path))
    wanted_resolved = str(Path(config_path).resolve())
    own_pid = os.getpid()
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return False
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            tokens = [
                value.decode("utf-8", errors="replace")
                for value in (entry / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "quant_fm.pretrain.train" not in tokens:
            continue
        if any(
            token in {wanted, wanted_resolved}
            or (token.endswith(wanted) and token.endswith(".yaml"))
            for token in tokens
        ):
            return True
    return False


def probe_gpus() -> tuple[list[dict[str, int]], str | None]:
    """读取 GPU 利用率；命令不可用或权限不足时返回错误文本。"""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return [], "nvidia-smi is unavailable"
    result = subprocess.run(
        [
            executable,
            "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return [], result.stderr.strip() or "nvidia-smi failed"
    rows: list[dict[str, int]] = []
    for line in result.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != 4 or any(not value.lstrip("-").isdigit() for value in values):
            continue
        rows.append(
            {
                "index": int(values[0]),
                "memory_used_mib": int(values[1]),
                "utilization_percent": int(values[2]),
                "temperature_c": int(values[3]),
            }
        )
    return rows, None


def collect_training_status(
    config_path: Path,
    *,
    log_path: Path | None = None,
    tmux_session: str | None = None,
    stall_seconds: int = 900,
    disk_warning_percent: float = 20.0,
    now: datetime | None = None,
    process_alive: bool | None = None,
    tmux_alive: bool | None = None,
    gpu_rows: Sequence[dict[str, int]] | None = None,
    gpu_error: str | None = None,
    update_registry: bool = True,
) -> dict[str, Any]:
    """采集一次只读训练状态，并可更新轻量 checkpoint 登记表。"""
    observed_at = now or datetime.now(tz=UTC)
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(cfg["runtime"]["out_dir"])
    selected_log = (
        Path(log_path) if log_path is not None else run_dir.parent / "train.log"
    )
    parsed = parse_training_log(_tail_text(selected_log))
    process_state = (
        training_process_alive(config_path) if process_alive is None else process_alive
    )
    tmux_state = probe_tmux_session(tmux_session) if tmux_alive is None else tmux_alive
    if gpu_rows is None and gpu_error is None:
        detected_gpus, detected_error = probe_gpus()
    else:
        detected_gpus = list(gpu_rows or [])
        detected_error = gpu_error

    registry_path = run_dir / "checkpoint_registry.json"
    registry = CheckpointRegistry(registry_path)
    checkpoint_payload = (
        registry.update(run_dir, now=observed_at)
        if update_registry
        else registry.load()
    )
    checkpoints = checkpoint_payload.get("checkpoints", [])
    names = {item["name"] for item in checkpoints}
    complete = {"final.pt", "final_resume.pt"}.issubset(names)

    alerts: list[Alert] = []
    if not complete and process_state is False:
        alerts.append(Alert("critical", "process_missing", "未发现训练 rank"))
    if not complete and tmux_state is False:
        alerts.append(Alert("critical", "tmux_missing", "训练 tmux 会话不存在"))
    for error in parsed["errors"]:
        alerts.append(
            Alert("critical", str(error["code"]), f"日志命中 {error['match']}")
        )

    log_age_seconds: float | None = None
    if selected_log.is_file():
        log_age_seconds = max(
            0.0, observed_at.timestamp() - selected_log.stat().st_mtime
        )
        if process_state and log_age_seconds > stall_seconds:
            alerts.append(
                Alert(
                    "warning",
                    "log_stalled",
                    f"日志已 {int(log_age_seconds)} 秒无更新",
                )
            )
    elif not complete:
        alerts.append(Alert("critical", "log_missing", "训练日志不存在"))

    disk = shutil.disk_usage(run_dir if run_dir.exists() else run_dir.parent)
    disk_free_percent = 100.0 * disk.free / disk.total
    if disk_free_percent < disk_warning_percent:
        alerts.append(
            Alert(
                "warning",
                "disk_low",
                f"磁盘剩余 {disk_free_percent:.1f}%",
            )
        )

    latest = parsed["latest_update"]
    checkpoint_every = int(cfg["runtime"].get("ckpt_every", 0))
    log_every = int(cfg["runtime"].get("log_every", 1))
    resumable = [item for item in checkpoints if item.get("resumable")]
    if (
        latest is not None
        and checkpoint_every > 0
        and latest["update"] >= checkpoint_every + max(log_every, 1)
        and not resumable
    ):
        alerts.append(
            Alert(
                "warning",
                "checkpoint_overdue",
                "已越过首个 checkpoint 周期但没有可恢复文件",
            )
        )

    severities = {alert.severity for alert in alerts}
    if complete and not process_state:
        state = "complete"
    elif "critical" in severities:
        state = "critical"
    elif "warning" in severities:
        state = "warning"
    else:
        state = "running"
    max_updates = int(
        cfg["optim"].get("max_update_steps", cfg["optim"].get("max_steps", 0))
    )
    progress_fraction = (
        min(1.0, latest["update"] / max_updates)
        if latest is not None and max_updates > 0
        else None
    )
    return {
        "status_version": "1.0",
        "observed_utc": observed_at.isoformat(),
        "state": state,
        "config": str(config_path.resolve()),
        "log": str(selected_log.resolve()),
        "run_dir": str(run_dir.resolve()),
        "process_alive": process_state,
        "tmux_session": tmux_session,
        "tmux_alive": tmux_state,
        "log_age_seconds": log_age_seconds,
        "max_update_steps": max_updates,
        "progress_fraction": progress_fraction,
        "progress": parsed,
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "free_percent": disk_free_percent,
        },
        "gpus": detected_gpus,
        "gpu_probe_error": detected_error,
        "checkpoint_registry": str(registry_path.resolve()),
        "checkpoints": checkpoints,
        "alerts": [asdict(alert) for alert in alerts],
    }


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
    }


def build_run_metadata(
    config_path: Path,
    *,
    world_size: int,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """冻结配置、数据 artifact、预算和 Git 身份元数据。"""
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = Path(cfg["data"]["manifest"])
    vocab_path = Path(cfg["data"]["vocab"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_stats: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        shards = [item for item in manifest["shards"] if item["split"] == split]
        split_stats[split] = {
            "shards": len(shards),
            "events": sum(int(item["rows"]) for item in shards),
            "dates": len({str(item["date"]) for item in shards}),
        }
    opt = cfg["optim"]
    micro_batch = int(opt.get("micro_batch_size", opt.get("batch_size", 1)))
    accumulation = int(opt.get("grad_accum", 1))
    context = int(cfg["data"]["context"])
    max_updates = int(opt.get("max_update_steps", opt.get("max_steps", 0)))
    run_dir = Path(cfg["runtime"]["out_dir"])
    validation_plan = Path(
        cfg["data"].get("validation_plan", run_dir / "validation_windows.json")
    )
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    files: dict[str, dict[str, Any]] = {}
    for name, path in (
        ("config", config_path),
        ("manifest", manifest_path),
        ("vocab", vocab_path),
        ("validation_plan", validation_plan),
    ):
        files[name] = {
            "path": str(path.resolve()),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
    return {
        "metadata_version": "1.0",
        "created_utc": _utc_iso(),
        "run_dir": str(run_dir.resolve()),
        "git": _git_metadata(root),
        "schema_version": manifest.get("schema_version"),
        "seed": int(cfg["seed"]),
        "model": cfg["model"],
        "book": cfg.get("book", {}),
        "pooling": cfg.get("pooling", {}),
        "data_splits": split_stats,
        "budget": {
            "world_size": world_size,
            "micro_batch_size": micro_batch,
            "grad_accum": accumulation,
            "effective_sequences_per_update": micro_batch * accumulation * world_size,
            "context": context,
            "scheduled_tokens_per_update": (
                micro_batch * accumulation * world_size * context
            ),
            "max_update_steps": max_updates,
            "scheduled_tokens": (
                micro_batch * accumulation * world_size * context * max_updates
            ),
        },
        "files": files,
    }


def render_training_report(
    status: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    """把结构化状态和冻结元数据渲染成简洁 Markdown 验收报告。"""
    progress = status["progress"]
    latest = progress.get("latest_update")
    best = progress.get("best_validation")
    stable_resumable = [
        item
        for item in status.get("checkpoints", [])
        if item.get("resumable") and item.get("stable")
    ]
    critical = [
        item for item in status.get("alerts", []) if item["severity"] == "critical"
    ]
    gate_health = "PASS" if not critical else "FAIL"
    gate_checkpoint = "PASS" if stable_resumable else "PENDING"
    gate_quality = "READY" if best is not None else "PENDING"
    model = metadata.get("model", {})
    budget = metadata.get("budget", {})
    lines = [
        "# 训练运行验收报告",
        "",
        f"生成时间：{status['observed_utc']}  ",
        f"状态：`{status['state']}`  ",
        f"配置：`{status['config']}`",
        "",
        "## 运行摘要",
        "",
        f"- 当前 update：{latest['update'] if latest else '尚无日志'}",
        f"- 当前 loss：{latest['loss'] if latest else '尚无日志'}",
        f"- 最佳 validation：{best['val_loss'] if best else '尚未产生'}",
        f"- 进程存活：{status['process_alive']}",
        f"- 磁盘剩余：{status['disk']['free_percent']:.1f}%",
        f"- 稳定可恢复 checkpoint：{len(stable_resumable)}",
        "",
        "## 冻结配置",
        "",
        f"- 模型：d_model={model.get('d_model')}，layers={model.get('n_layers')}，"
        f"MoE={model.get('backbone_moe', {}).get('enabled', False)}",
        f"- world size：{budget.get('world_size')}",
        f"- 有效序列/update：{budget.get('effective_sequences_per_update')}",
        f"- 计划 token：{budget.get('scheduled_tokens')}",
        f"- schema：{metadata.get('schema_version')}",
        f"- Git commit：{metadata.get('git', {}).get('commit')}",
        "",
        "## 决策门",
        "",
        f"- Gate 1 运行健康：**{gate_health}**",
        f"- Gate 1 Checkpoint 可恢复：**{gate_checkpoint}**",
        f"- Gate 2 预训练质量材料：**{gate_quality}**",
        "- Gate 3 下游价值：**PENDING**",
        "- Gate 4 V2/MoE 晋级：**PENDING**",
        "",
        "## 告警",
        "",
    ]
    alerts = status.get("alerts", [])
    if alerts:
        lines.extend(
            f"- [{item['severity'].upper()}] {item['code']}: {item['message']}"
            for item in alerts
        )
    else:
        lines.append("- 无")
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "本报告只使用日志和文件元数据，不加载大 checkpoint，不执行 GPU 评估。",
            "validation/test、embedding、RankIC 和回测应在训练结束后的串行评估阶段补充。",
            "",
        ]
    )
    return "\n".join(lines)
