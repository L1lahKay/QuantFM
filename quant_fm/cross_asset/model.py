"""基于同步 interval embedding 的线性复杂度跨股票模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from quant_fm.cross_asset.context_pool import build_synchronous_context

if TYPE_CHECKING:
    from quant_fm.cross_asset.context_pool import CrossAssetContext


@dataclass(frozen=True, slots=True)
class CrossAssetModelConfig:
    """轻量跨股票上下文组合器配置。"""

    input_dim: int
    hidden_dim: int = 128
    output_dim: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.hidden_dim < 1 or self.output_dim < 1:
            msg = "input_dim, hidden_dim and output_dim must be positive"
            raise ValueError(msg)
        if not 0.0 <= self.dropout < 1.0:
            msg = "dropout must be in [0, 1)"
            raise ValueError(msg)


@dataclass(slots=True)
class CrossAssetModelOutput:
    """interval 级因果表示和每股片段末状态。"""

    interval_embeddings: torch.Tensor
    stock_summary: torch.Tensor
    context: CrossAssetContext


class LinearCrossAssetModel(nn.Module):
    """
    O(T*N) 的 market/industry context 与逐股因果时间聚合器。

    输入必须是单股模型预先汇总后的 ``[T, N, D]`` interval embedding。
    模块不包含 ``MultiheadAttention``，也没有接收全市场原始事件序列的 API。
    横截面交互仅调用 sum/count pool，参数量与股票数 N 无关。
    """

    cross_section_complexity = "O(T*N*D)"
    accepts_raw_events = False

    def __init__(self, config: CrossAssetModelConfig) -> None:
        super().__init__()
        self.config = config
        context_dim = 5 * config.input_dim
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.context_gate = nn.Linear(context_dim, config.hidden_dim)
        self.own_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.temporal_cell = nn.GRUCell(config.hidden_dim, config.hidden_dim)
        self.output_norm = nn.LayerNorm(config.hidden_dim)
        self.output_projection = nn.Linear(config.hidden_dim, config.output_dim)

    def forward(
        self,
        own: torch.Tensor,
        industry_id: torch.Tensor,
        *,
        active_mask: torch.Tensor | None = None,
    ) -> CrossAssetModelOutput:
        """
        组合同步横截面上下文并沿 T 轴做严格因果聚合。

        参数
        ----------
        own
            ``[T, N, D]`` 的单股 interval embedding；不允许传逐笔事件张量。
        industry_id
            ``[T, N]`` 或 ``[N]`` 的 PIT 行业编号，未知行业为负数。
        active_mask
            ``[T, N]``；缺失/停牌位置为 False，不进入 pool，也不更新 GRU 状态。
        """
        if own.ndim != 3:
            msg = "own must have shape [time, stock, feature]"
            raise ValueError(msg)
        n_time, n_stock, feature_dim = own.shape
        if n_time == 0 or n_stock == 0:
            msg = "time and stock axes must not be empty"
            raise ValueError(msg)
        if feature_dim != self.config.input_dim:
            msg = (
                f"expected feature dimension {self.config.input_dim}, got {feature_dim}"
            )
            raise ValueError(msg)
        if not own.is_floating_point():
            msg = "own interval embeddings must use a floating dtype"
            raise TypeError(msg)

        active = (
            torch.ones((n_time, n_stock), dtype=torch.bool, device=own.device)
            if active_mask is None
            else active_mask.to(device=own.device, dtype=torch.bool)
        )
        if active.shape != (n_time, n_stock):
            msg = "active_mask must have shape [time, stock]"
            raise ValueError(msg)
        industries = industry_id.to(device=own.device, dtype=torch.long)
        if industries.ndim == 1:
            if industries.shape != (n_stock,):
                msg = "one-dimensional industry_id must have shape [stock]"
                raise ValueError(msg)
        elif industries.shape != (n_time, n_stock):
            msg = "industry_id must have shape [stock] or [time, stock]"
            raise ValueError(msg)

        masked_own = torch.where(active.unsqueeze(-1), own, 0.0)
        context = build_synchronous_context(
            masked_own,
            industries,
            active_mask=active,
        )
        combined = context.concatenate(masked_own)
        proposal = self.context_encoder(combined)
        gate = torch.sigmoid(self.context_gate(combined))
        interval_input = self.own_projection(masked_own) + gate * proposal
        interval_input = torch.where(active.unsqueeze(-1), interval_input, 0.0)

        state = torch.zeros(
            (n_stock, self.config.hidden_dim),
            dtype=own.dtype,
            device=own.device,
        )
        causal_rows: list[torch.Tensor] = []
        for time_index in range(n_time):
            candidate = self.temporal_cell(interval_input[time_index], state)
            time_active = active[time_index].unsqueeze(-1)
            state = torch.where(time_active, candidate, state)
            causal_rows.append(torch.where(time_active, state, 0.0))

        interval_hidden = self.output_norm(torch.stack(causal_rows))
        interval_embeddings = self.output_projection(interval_hidden)
        interval_embeddings = torch.where(
            active.unsqueeze(-1), interval_embeddings, 0.0
        )
        ever_active = active.any(dim=0).unsqueeze(-1)
        stock_summary = self.output_projection(self.output_norm(state))
        stock_summary = torch.where(ever_active, stock_summary, 0.0)
        return CrossAssetModelOutput(
            interval_embeddings=interval_embeddings,
            stock_summary=stock_summary,
            context=context,
        )
