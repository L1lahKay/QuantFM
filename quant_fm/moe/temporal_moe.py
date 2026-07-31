"""股日时间聚合后的轻量 Regime-MoE。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from quant_fm.moe.router import TopKRouter

if TYPE_CHECKING:
    from quant_fm.embedding.intraday_aggregator import IntradayAggregator
    from quant_fm.moe.config import RegimeMoEConfig
    from quant_fm.moe.router import RouterOutput


class _RegimeExpert(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


@dataclass(slots=True)
class MoEOutput:
    """MoE 表示、路由诊断与辅助损失。"""

    hidden: torch.Tensor
    router: RouterOutput
    auxiliary_loss: torch.Tensor
    overflow_rate: torch.Tensor


class TemporalRegimeMoE(nn.Module):
    """共享专家保底、按行情特征 Top-K 路由的轻量适配层。"""

    def __init__(
        self,
        hidden_dim: int,
        regime_feature_dim: int,
        config: RegimeMoEConfig,
    ) -> None:
        super().__init__()
        if not config.enabled:
            msg = "TemporalRegimeMoE requires an enabled config"
            raise ValueError(msg)
        self.config = config
        self.router = TopKRouter(
            regime_feature_dim,
            config.n_experts,
            top_k=config.top_k,
            hidden_dim=config.router_hidden,
            temperature=config.temperature,
        )
        self.shared_expert = _RegimeExpert(
            hidden_dim, config.expert_hidden, config.dropout
        )
        self.experts = nn.ModuleList(
            _RegimeExpert(hidden_dim, config.expert_hidden, config.dropout)
            for _ in range(config.n_experts)
        )

    def forward(self, hidden: torch.Tensor, regime_features: torch.Tensor) -> MoEOutput:
        """用同一时点可用的 regime 特征选择股日专家。"""
        if hidden.shape[:-1] != regime_features.shape[:-1]:
            msg = "hidden and regime_features leading shapes must match"
            raise ValueError(msg)
        original_shape = hidden.shape
        flat_hidden = hidden.reshape(-1, hidden.size(-1))
        route = self.router(regime_features.reshape(-1, regime_features.size(-1)))
        routed = torch.zeros_like(flat_hidden)
        assignments = flat_hidden.size(0) * self.config.top_k
        capacity = max(
            1,
            math.ceil(
                self.config.capacity_factor * assignments / self.config.n_experts
            ),
        )
        accepted = 0
        for expert_index, expert in enumerate(self.experts):
            token_index, slot_index = torch.nonzero(
                route.topk_indices.eq(expert_index), as_tuple=True
            )
            if token_index.numel() == 0:
                continue
            weights = route.topk_weights[token_index, slot_index]
            if self.training and token_index.numel() > capacity:
                keep = weights.topk(capacity, sorted=False).indices
                token_index, weights = token_index[keep], weights[keep]
            accepted += token_index.numel()
            expert_hidden = expert(flat_hidden[token_index])
            # Autocast may produce BF16/FP16 expert outputs for an FP32 input
            # buffer. Accumulate in ``routed`` dtype because index_add_ does not
            # permit mixed source/destination dtypes.
            contribution = expert_hidden.to(routed.dtype) * weights.to(
                routed.dtype
            ).unsqueeze(-1)
            routed.index_add_(
                0,
                token_index,
                contribution,
            )
        output = flat_hidden + self.shared_expert(flat_hidden) + routed
        auxiliary = route.auxiliary_loss(
            load_balance_weight=self.config.load_balance_weight,
            router_z_loss_weight=self.config.router_z_loss_weight,
        )
        overflow = flat_hidden.new_tensor(1.0 - accepted / max(assignments, 1))
        return MoEOutput(
            hidden=output.reshape(original_shape),
            router=route,
            auxiliary_loss=auxiliary,
            overflow_rate=overflow,
        )


class RegimeIntradayModel(nn.Module):
    """组合严格因果的日内聚合器与股日 Regime-MoE。"""

    def __init__(
        self,
        aggregator: IntradayAggregator,
        regime_feature_dim: int,
        config: RegimeMoEConfig,
    ) -> None:
        super().__init__()
        self.aggregator = aggregator
        self.moe = TemporalRegimeMoE(
            aggregator.d_model,
            regime_feature_dim,
            config,
        )

    def forward(
        self,
        chunk_summaries: torch.Tensor,
        chunk_time: torch.Tensor,
        chunk_session: torch.Tensor,
        chunk_mask: torch.Tensor,
        regime_features: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], MoEOutput]:
        """返回原始多尺度摘要、regime 摘要与路由诊断。"""
        summaries = self.aggregator(
            chunk_summaries, chunk_time, chunk_session, chunk_mask
        )
        output = self.moe(summaries["full_day_summary"], regime_features)
        summaries = dict(summaries)
        summaries["regime_summary"] = output.hidden
        return summaries, output
