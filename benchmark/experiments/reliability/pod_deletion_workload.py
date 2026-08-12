#!/usr/bin/env python3
"""Sustained one-GPU workload for the Job reconciliation after Pod deletion test."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

EVENT_PREFIX = "POD_DELETE_EVENT_JSON="
RESULT_PREFIX = "POD_DELETE_RESULT_JSON="


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def emit(prefix: str, payload: Mapping[str, Any]) -> None:
    print(
        prefix + json.dumps(payload, sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def visible_gpu() -> dict[str, str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=10,
    )
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one visible NVIDIA GPU, found {len(rows)}")
    fields = [value.strip() for value in rows[0].split(",", maxsplit=2)]
    if len(fields) != 3 or not all(fields):
        raise RuntimeError("nvidia-smi returned an invalid GPU identity")
    return {"gpu_uuid": fields[0], "gpu_name": fields[1], "driver_version": fields[2]}


def train(base: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    total_steps = int(os.environ.get("TOTAL_STEPS", "300"))
    delete_after_step = int(os.environ.get("DELETE_AFTER_STEP", "75"))
    step_delay = float(os.environ.get("STEP_DELAY_SECONDS", "0.03"))
    if total_steps != 300 or delete_after_step != 75:
        raise RuntimeError("workload requires total_steps=300 and delete_after_step=75")
    if step_delay < 0.02 or step_delay > 0.05:
        raise RuntimeError("STEP_DELAY_SECONDS must be between 0.02 and 0.05")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Pod deletion workload requires exactly one CUDA device")

    torch.manual_seed(20260806)
    torch.cuda.manual_seed_all(20260806)
    device = torch.device("cuda:0")
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 512),
        torch.nn.GELU(),
        torch.nn.Linear(512, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    features = torch.randn(4096, 100, device=device)
    target = torch.randn(4096, 1, device=device)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.mse_loss(model(features), target)
        loss.backward()
        optimizer.step()

    for _ in range(10):
        step()
    torch.cuda.synchronize()
    emit(EVENT_PREFIX, {**base, "event": "training_started", "timestamp": utc_now()})
    started = time.perf_counter()
    cuda_step_seconds = 0.0
    for current_step in range(1, total_steps + 1):
        cuda_started = time.perf_counter()
        step()
        torch.cuda.synchronize()
        cuda_step_seconds += time.perf_counter() - cuda_started
        if current_step == delete_after_step:
            emit(
                EVENT_PREFIX,
                {
                    **base,
                    "event": "deletion_ready",
                    "step": current_step,
                    "total_steps": total_steps,
                    "timestamp": utc_now(),
                },
            )
        elif current_step % 75 == 0:
            emit(
                EVENT_PREFIX,
                {
                    **base,
                    "event": "training_progress",
                    "step": current_step,
                    "total_steps": total_steps,
                    "timestamp": utc_now(),
                },
            )
        time.sleep(step_delay)
    elapsed = time.perf_counter() - started
    return {
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "training_steps": total_steps,
        "warmup_steps": 10,
        "deletion_marker_step": delete_after_step,
        "training_time_seconds": elapsed,
        "cuda_step_time_seconds": cuda_step_seconds,
        "configured_step_delay_seconds": step_delay,
        "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
    }


def main() -> int:
    base = {
        "schema_version": 1,
        "experiment": "pod-deletion-job-reconciliation",
        "run_token": os.environ.get("RUN_TOKEN", "unknown"),
        "pod_name": os.environ.get("POD_NAME", "unknown"),
        "pod_uid": os.environ.get("POD_UID", "unknown"),
        **visible_gpu(),
    }
    emit(EVENT_PREFIX, {**base, "event": "attempt_started", "timestamp": utc_now()})
    training = train(base)
    emit(
        RESULT_PREFIX,
        {**base, **training, "status": "success", "timestamp": utc_now()},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
