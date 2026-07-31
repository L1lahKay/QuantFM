"""
训练完成后的串行预训练评估队列。

默认只生成计划。只有显式传入 ``--execute``、训练进程已退出并且
``final.pt``/``final_resume.pt`` 均存在时，才会依次运行 validation 和 test 评估。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from quant_fm.monitoring.training import training_process_alive

if TYPE_CHECKING:
    from typing import Any


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_evaluation_plan(
    config_path: Path,
    *,
    device: str,
    max_batches: int,
    unigram_max_batches: int,
    gradient_norm_batches: int,
    baseline_checkpoint: Path | None = None,
    noninferiority_tolerance: float = 0.01,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """根据冻结训练配置生成 val→test 串行命令，不执行任何模型计算。"""
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_dir = Path(cfg["runtime"]["out_dir"])
    checkpoint = run_dir / "best.pt"
    if not checkpoint.is_file():
        checkpoint = run_dir / "final.pt"
    completion_markers = [run_dir / "final.pt", run_dir / "final_resume.pt"]
    missing = [str(path) for path in completion_markers if not path.is_file()]
    if not checkpoint.is_file():
        missing.append(str(checkpoint))
    if baseline_checkpoint is not None and not baseline_checkpoint.is_file():
        missing.append(str(baseline_checkpoint))
    executable = python_executable or sys.executable
    validation_plan = Path(
        cfg["data"].get("validation_plan", run_dir / "validation_windows.json")
    )
    jobs: list[dict[str, Any]] = []

    def evaluation_job(
        *,
        name: str,
        split: str,
        selected_checkpoint: Path,
        plan_path: Path,
        out_path: Path,
    ) -> dict[str, Any]:
        command = [
            executable,
            "-m",
            "quant_fm.pretrain.eval",
            "--checkpoint",
            str(selected_checkpoint),
            "--config",
            str(config_path),
            "--split",
            split,
            "--max-batches",
            str(max_batches),
            "--unigram-max-batches",
            str(unigram_max_batches),
            "--gradient-norm-batches",
            str(gradient_norm_batches),
            "--validation-plan",
            str(plan_path),
            "--device",
            device,
            "--out",
            str(out_path),
        ]
        return {
            "name": name,
            "split": split,
            "command": command,
            "validation_plan": str(plan_path.resolve()),
            "output": str(out_path.resolve()),
            "state": "pending",
        }

    candidate_val = run_dir / "eval_val.json"
    jobs.append(
        evaluation_job(
            name="candidate_pretrain_val",
            split="val",
            selected_checkpoint=checkpoint,
            plan_path=validation_plan,
            out_path=candidate_val,
        )
    )
    if baseline_checkpoint is not None:
        baseline_val = run_dir / "eval_val_baseline.json"
        jobs.append(
            evaluation_job(
                name="baseline_pretrain_val",
                split="val",
                selected_checkpoint=baseline_checkpoint,
                plan_path=validation_plan,
                out_path=baseline_val,
            )
        )
        acceptance_out = run_dir / "pretrain_acceptance.json"
        jobs.append(
            {
                "name": "pretrain_noninferiority_gate",
                "split": "val",
                "command": [
                    executable,
                    "-m",
                    "quant_fm.scripts.compare_pretrain_evaluations",
                    "--candidate",
                    str(candidate_val),
                    "--baseline",
                    str(baseline_val),
                    "--tolerance",
                    str(noninferiority_tolerance),
                    "--out",
                    str(acceptance_out),
                ],
                "output": str(acceptance_out.resolve()),
                "state": "pending",
            }
        )
    jobs.append(
        evaluation_job(
            name="candidate_pretrain_test",
            split="test",
            selected_checkpoint=checkpoint,
            plan_path=run_dir / "test_windows.json",
            out_path=run_dir / "eval_test.json",
        )
    )
    return {
        "plan_version": "1.0",
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "config": str(config_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "baseline_checkpoint": (
            str(baseline_checkpoint.resolve())
            if baseline_checkpoint is not None
            else None
        ),
        "noninferiority_tolerance": noninferiority_tolerance,
        "completion_markers": [str(path.resolve()) for path in completion_markers],
        "missing_required_artifacts": missing,
        "runnable": not missing,
        "device": device,
        "jobs": jobs,
    }


def execute_evaluation_plan(
    plan: dict[str, Any],
    *,
    config_path: Path,
    plan_path: Path,
) -> dict[str, Any]:
    """在训练完成且进程退出后串行执行计划，并持续落盘状态。"""
    if not plan.get("runnable"):
        missing = plan.get("missing_required_artifacts", [])
        msg = f"post-train evaluation is blocked; missing: {missing}"
        raise RuntimeError(msg)
    if training_process_alive(config_path):
        msg = "training process is still alive; refusing to compete for resources"
        raise RuntimeError(msg)
    plan["state"] = "running"
    plan["started_utc"] = datetime.now(tz=UTC).isoformat()
    _atomic_json(plan_path, plan)
    for job in plan["jobs"]:
        job["state"] = "running"
        job["started_utc"] = datetime.now(tz=UTC).isoformat()
        _atomic_json(plan_path, plan)
        result = subprocess.run(job["command"], check=False)
        job["returncode"] = result.returncode
        job["finished_utc"] = datetime.now(tz=UTC).isoformat()
        job["state"] = "complete" if result.returncode == 0 else "failed"
        _atomic_json(plan_path, plan)
        if result.returncode != 0:
            plan["state"] = "failed"
            _atomic_json(plan_path, plan)
            msg = f"evaluation job failed: {job['name']}"
            raise RuntimeError(msg)
    plan["state"] = "complete"
    plan["finished_utc"] = datetime.now(tz=UTC).isoformat()
    _atomic_json(plan_path, plan)
    return plan


def main() -> None:
    """生成计划，或在训练安全结束后显式执行计划。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--unigram-max-batches", type=int, default=200)
    parser.add_argument("--gradient-norm-batches", type=int, default=1)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--noninferiority-tolerance", type=float, default=0.01)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.max_batches < 1 or args.unigram_max_batches < 1:
        parser.error("batch limits must be positive")
    if args.gradient_norm_batches < 0:
        parser.error("--gradient-norm-batches must be non-negative")
    if args.noninferiority_tolerance < 0:
        parser.error("--noninferiority-tolerance must be non-negative")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_dir = Path(cfg["runtime"]["out_dir"])
    plan_path = args.out or run_dir / "posttrain_evaluation_plan.json"
    plan = build_evaluation_plan(
        args.config,
        device=args.device,
        max_batches=args.max_batches,
        unigram_max_batches=args.unigram_max_batches,
        gradient_norm_batches=args.gradient_norm_batches,
        baseline_checkpoint=args.baseline_checkpoint,
        noninferiority_tolerance=args.noninferiority_tolerance,
    )
    _atomic_json(plan_path, plan)
    if args.execute:
        execute_evaluation_plan(plan, config_path=args.config, plan_path=plan_path)
    print(plan_path)


if __name__ == "__main__":
    main()
