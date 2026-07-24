"""按真实 non-pad token 计量模型吞吐和延迟。"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class ModelBenchmark:
    """训练或推理主循环的硬件无关基准结果。"""

    non_pad_tokens_per_second: float
    windows_per_second: float
    optimizer_updates_per_second: float
    mean_step_time_seconds: float
    p95_step_time_seconds: float
    peak_allocated_gpu_bytes: int
    peak_reserved_gpu_bytes: int

    def to_dict(self) -> dict[str, float | int]:
        """转换为实验 registry 可写的映射。"""
        return asdict(self)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_model(
    step: Callable[[], dict[str, torch.Tensor] | torch.Tensor],
    *,
    attention_mask: torch.Tensor,
    device: torch.device,
    measured_steps: int = 20,
    warmup_steps: int = 5,
    update_every: int = 1,
) -> ModelBenchmark:
    """计量任意前向/训练 step；token 数来自 attention_mask.sum()。"""
    if measured_steps < 1 or warmup_steps < 0 or update_every < 1:
        msg = "invalid benchmark step counts"
        raise ValueError(msg)
    for _ in range(warmup_steps):
        step()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations: list[float] = []
    for _ in range(measured_steps):
        started = time.perf_counter()
        step()
        _sync(device)
        durations.append(time.perf_counter() - started)
    elapsed = sum(durations)
    tokens = int(attention_mask.sum().item()) * measured_steps
    windows = int(attention_mask.size(0)) * measured_steps
    p95_index = max(0, int(0.95 * len(durations) + 0.999999) - 1)
    ordered = sorted(durations)
    allocated = reserved = 0
    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
    return ModelBenchmark(
        non_pad_tokens_per_second=tokens / elapsed,
        windows_per_second=windows / elapsed,
        optimizer_updates_per_second=(measured_steps / update_every) / elapsed,
        mean_step_time_seconds=statistics.fmean(durations),
        p95_step_time_seconds=ordered[p95_index],
        peak_allocated_gpu_bytes=allocated,
        peak_reserved_gpu_bytes=reserved,
    )
