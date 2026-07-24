"""稳定、可诊断的 Top-K 稀疏路由器。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class RouterOutput:
    """路由概率、稀疏选择和辅助正则。"""

    logits: torch.Tensor
    probabilities: torch.Tensor
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor
    load_balance_loss: torch.Tensor
    router_z_loss: torch.Tensor
    entropy: torch.Tensor

    def auxiliary_loss(
        self,
        *,
        load_balance_weight: float,
        router_z_loss_weight: float,
    ) -> torch.Tensor:
        """按配置聚合 router 正则。"""
        return (
            load_balance_weight * self.load_balance_loss
            + router_z_loss_weight * self.router_z_loss
        )


class TopKRouter(nn.Module):
    """以 FP32 概率计算执行 Top-K 路由，输出权重归一化到 1。"""

    def __init__(
        self,
        input_dim: int,
        n_experts: int,
        *,
        top_k: int,
        hidden_dim: int | None = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim < 1 or n_experts < 1:
            msg = "input_dim and n_experts must be positive"
            raise ValueError(msg)
        if not 1 <= top_k <= n_experts:
            msg = "top_k must be in [1, n_experts]"
            raise ValueError(msg)
        if temperature <= 0.0:
            msg = "temperature must be positive"
            raise ValueError(msg)
        self.n_experts = n_experts
        self.top_k = top_k
        self.temperature = float(temperature)
        self.net = (
            nn.Linear(input_dim, n_experts, bias=False)
            if hidden_dim is None
            else nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, n_experts, bias=False),
            )
        )

    def forward(self, inputs: torch.Tensor) -> RouterOutput:
        """返回与 inputs 前置维度一致的专家路由。"""
        if inputs.size(-1) < 1:
            msg = "router inputs must have a non-empty feature dimension"
            raise ValueError(msg)
        logits = self.net(inputs).float() / self.temperature
        probabilities = logits.softmax(dim=-1)
        topk_probabilities, topk_indices = probabilities.topk(self.top_k, dim=-1)
        topk_weights = topk_probabilities / topk_probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(topk_probabilities.dtype).eps)

        flat_probabilities = probabilities.reshape(-1, self.n_experts)
        flat_indices = topk_indices.reshape(-1, self.top_k)
        importance = flat_probabilities.mean(dim=0)
        hard_load = torch.nn.functional.one_hot(
            flat_indices,
            num_classes=self.n_experts,
        ).to(flat_probabilities.dtype)
        hard_load = hard_load.sum(dim=1).mean(dim=0) / float(self.top_k)
        load_balance = self.n_experts * (importance * hard_load.detach()).sum()
        router_z = torch.logsumexp(logits, dim=-1).square().mean()
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(
            dim=-1
        ).mean() / math.log(max(self.n_experts, 2))
        return RouterOutput(
            logits=logits,
            probabilities=probabilities,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
            load_balance_loss=load_balance,
            router_z_loss=router_z,
            entropy=entropy,
        )
