"""MoE 路由健康度指标。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class MoETelemetry:
    """可写入训练日志的路由健康度快照。"""

    expert_fraction: tuple[float, ...]
    normalized_entropy: float
    mean_top1_probability: float
    overflow_rate: float


def summarize_moe(
    probabilities: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    overflow_rate: torch.Tensor | float = 0.0,
) -> MoETelemetry:
    """将 batch 路由结果转换成可记录、可告警的标量。"""
    n_experts = probabilities.size(-1)
    selected = torch.nn.functional.one_hot(
        topk_indices.reshape(-1), num_classes=n_experts
    ).float()
    fraction = selected.mean(dim=0)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    entropy = entropy.mean() / torch.log(probabilities.new_tensor(max(n_experts, 2)))
    overflow = float(torch.as_tensor(overflow_rate).detach().cpu().item())
    return MoETelemetry(
        expert_fraction=tuple(float(value) for value in fraction.cpu()),
        normalized_entropy=float(entropy.detach().cpu()),
        mean_top1_probability=float(
            probabilities.max(dim=-1).values.mean().detach().cpu()
        ),
        overflow_rate=overflow,
    )
