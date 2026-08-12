#!/usr/bin/env python3
"""One-GPU workload used to verify Kubernetes Job retry after process loss."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

EVENT_PREFIX = "POD_RETRY_EVENT_JSON="
RESULT_PREFIX = "POD_RETRY_RESULT_JSON="


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
        raise RuntimeError(
            f"expected exactly one visible NVIDIA GPU, found {len(rows)}"
        )
    fields = [value.strip() for value in rows[0].split(",", maxsplit=2)]
    if len(fields) != 3 or not all(fields):
        raise RuntimeError("nvidia-smi returned an invalid GPU identity")
    return {"gpu_uuid": fields[0], "gpu_name": fields[1], "driver_version": fields[2]}


def train() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("retry workload requires exactly one CUDA device")
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
    started = time.perf_counter()
    for _ in range(80):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "training_steps": 80,
        "warmup_steps": 10,
        "training_time_seconds": elapsed,
        "throughput_samples_per_second": 4096 * 80 / elapsed,
        "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
    }


def main() -> int:
    pod_name = os.environ.get("POD_NAME", "unknown")
    pod_uid = os.environ.get("POD_UID", "unknown")
    run_token = os.environ.get("RUN_TOKEN", "unknown")
    injection_window = int(os.environ.get("INJECTION_WINDOW_SECONDS", "45"))
    if injection_window < 20 or injection_window > 90:
        raise RuntimeError("INJECTION_WINDOW_SECONDS must be between 20 and 90")
    gpu = visible_gpu()
    base = {
        "schema_version": 1,
        "experiment": "pod-process-failure-job-retry",
        "run_token": run_token,
        "pod_name": pod_name,
        "pod_uid": pod_uid,
        **gpu,
    }
    emit(
        EVENT_PREFIX,
        {
            **base,
            "event": "attempt_started",
            "timestamp": utc_now(),
            "injection_window_seconds": injection_window,
        },
    )
    time.sleep(injection_window)
    emit(
        EVENT_PREFIX,
        {**base, "event": "training_started", "timestamp": utc_now()},
    )
    training = train()
    emit(
        RESULT_PREFIX,
        {
            **base,
            **training,
            "status": "success",
            "timestamp": utc_now(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
