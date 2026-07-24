"""可选顶部层使用的本地稀疏 SwiGLU MoE。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from quant_fm.moe.router import TopKRouter

if TYPE_CHECKING:
    from quant_fm.moe.config import BackboneMoEConfig
    from quant_fm.moe.router import RouterOutput


class _SwiGLUExpert(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.value = nn.Linear(d_model, hidden, bias=False)
        self.output = nn.Linear(hidden, d_model, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(
            torch.nn.functional.silu(self.gate(inputs)) * self.value(inputs)
        )


@dataclass(slots=True)
class SparseMoEOutput:
    """稀疏 FFN 输出与可加入训练目标的 router 损失。"""

    hidden: torch.Tensor
    auxiliary_loss: torch.Tensor
    router: RouterOutput
    overflow_rate: torch.Tensor


class SparseMoEFeedForward(nn.Module):
    """Shared expert + capacity-limited Top-K routed experts。"""

    def __init__(self, d_model: int, config: BackboneMoEConfig) -> None:
        super().__init__()
        if not config.enabled:
            msg = "SparseMoEFeedForward requires an enabled config"
            raise ValueError(msg)
        self.config = config
        self.router = TopKRouter(
            d_model,
            config.n_routed_experts,
            top_k=config.top_k,
        )
        self.shared = (
            _SwiGLUExpert(d_model, config.shared_expert_hidden)
            if config.shared_expert_hidden
            else None
        )
        self.experts = nn.ModuleList(
            _SwiGLUExpert(d_model, config.routed_expert_hidden)
            for _ in range(config.n_routed_experts)
        )

    def forward(self, inputs: torch.Tensor) -> SparseMoEOutput:
        """稀疏分派扁平 token，并恢复原始前置形状。"""
        original_shape = inputs.shape
        flat = inputs.reshape(-1, original_shape[-1])
        route = self.router(flat)
        routed = torch.zeros_like(flat)
        total_assignments = flat.size(0) * self.config.top_k
        capacity = max(
            1,
            math.ceil(
                self.config.capacity_factor
                * total_assignments
                / self.config.n_routed_experts
            ),
        )
        accepted = 0
        for expert_index, expert in enumerate(self.experts):
            matches = route.topk_indices.eq(expert_index)
            token_index, slot_index = torch.nonzero(matches, as_tuple=True)
            if token_index.numel() == 0:
                continue
            weights = route.topk_weights[token_index, slot_index]
            if token_index.numel() > capacity:
                keep = weights.topk(capacity, sorted=False).indices
                token_index = token_index[keep]
                weights = weights[keep]
            accepted += token_index.numel()
            expert_output = expert(flat[token_index])
            routed.index_add_(
                0,
                token_index,
                expert_output * weights.to(expert_output.dtype).unsqueeze(-1),
            )
        hidden = routed if self.shared is None else routed + self.shared(flat)
        auxiliary = route.auxiliary_loss(
            load_balance_weight=self.config.load_balance_weight,
            router_z_loss_weight=self.config.router_z_loss_weight,
        )
        overflow = flat.new_tensor(1.0 - accepted / max(total_assignments, 1))
        return SparseMoEOutput(
            hidden=hidden.reshape(original_shape),
            auxiliary_loss=auxiliary,
            router=route,
            overflow_rate=overflow,
        )
