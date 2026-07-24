"""同一交易 interval 内的 O(N) 市场/行业上下文池化。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class CrossAssetContext:
    """每只股票每个同步 interval 的跨截面上下文。"""

    market: torch.Tensor
    industry_leave_one_out: torch.Tensor
    own_minus_market: torch.Tensor
    own_minus_industry: torch.Tensor
    industry_has_peer: torch.Tensor

    def concatenate(self, own: torch.Tensor) -> torch.Tensor:
        """按设计稿顺序拼接 own、上下文及相对表示。"""
        return torch.cat(
            (
                own,
                self.market,
                self.industry_leave_one_out,
                self.own_minus_market,
                self.own_minus_industry,
            ),
            dim=-1,
        )


def _industry_leave_one_out(
    own: torch.Tensor,
    industry_id: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对一个时点按行业做 sum/count，再严格扣除自身。"""
    valid = active & industry_id.ge(0)
    indices = torch.nonzero(valid, as_tuple=False).flatten()
    result = torch.zeros_like(own)
    has_peer = torch.zeros(own.size(0), dtype=torch.bool, device=own.device)
    if indices.numel() == 0:
        return result, has_peer

    valid_industry = industry_id[indices]
    _, inverse = torch.unique(valid_industry, sorted=True, return_inverse=True)
    n_groups = int(inverse.max().item()) + 1
    sums = torch.zeros((n_groups, own.size(-1)), device=own.device, dtype=own.dtype)
    sums.index_add_(0, inverse, own[indices])
    counts = torch.zeros(n_groups, device=own.device, dtype=own.dtype)
    counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=own.dtype))

    peer_counts = counts[inverse] - 1.0
    valid_peer = peer_counts.gt(0)
    peer_mean = (sums[inverse] - own[indices]) / peer_counts.clamp_min(1).unsqueeze(-1)
    result[indices] = torch.where(valid_peer.unsqueeze(-1), peer_mean, 0.0)
    has_peer[indices] = valid_peer
    return result, has_peer


def build_synchronous_context(
    own: torch.Tensor,
    industry_id: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
) -> CrossAssetContext:
    """
    在每个独立 interval 内生成市场均值和行业 leave-one-out 表示。

    参数
    ----------
    own
        ``[T, N, D]``，已经按交易所时钟同步的单股 interval embedding。
    industry_id
        ``[T, N]`` 或 ``[N]`` 的 PIT 行业 id，未知行业使用负数。
    active_mask
        可选 ``[T, N]``；停牌/缺失股票为 False，不进入任何 pool。
    """
    if own.ndim != 3:
        msg = "own must have shape [time, stock, feature]"
        raise ValueError(msg)
    n_time, n_stock, _ = own.shape
    if industry_id.ndim == 1:
        industry_id = industry_id.unsqueeze(0).expand(n_time, -1)
    if industry_id.shape != (n_time, n_stock):
        msg = "industry_id must have shape [stock] or [time, stock]"
        raise ValueError(msg)
    active = (
        torch.ones((n_time, n_stock), dtype=torch.bool, device=own.device)
        if active_mask is None
        else active_mask.to(device=own.device, dtype=torch.bool)
    )
    if active.shape != (n_time, n_stock):
        msg = "active_mask must have shape [time, stock]"
        raise ValueError(msg)

    market_rows: list[torch.Tensor] = []
    industry_rows: list[torch.Tensor] = []
    peer_rows: list[torch.Tensor] = []
    for time_index in range(n_time):
        time_active = active[time_index]
        count = time_active.sum().clamp_min(1).to(own.dtype)
        market_vector = own[time_index, time_active].sum(dim=0) / count
        market_rows.append(
            market_vector.unsqueeze(0).expand(n_stock, -1) * time_active.unsqueeze(-1)
        )
        industry_row, peer_row = _industry_leave_one_out(
            own[time_index], industry_id[time_index], time_active
        )
        industry_rows.append(industry_row)
        peer_rows.append(peer_row)

    market = torch.stack(market_rows)
    industry = torch.stack(industry_rows)
    has_peer = torch.stack(peer_rows)
    active_feature = active.unsqueeze(-1)
    own_minus_market = torch.where(active_feature, own - market, 0.0)
    own_minus_industry = torch.where(has_peer.unsqueeze(-1), own - industry, 0.0)
    return CrossAssetContext(
        market=market,
        industry_leave_one_out=industry,
        own_minus_market=own_minus_market,
        own_minus_industry=own_minus_industry,
        industry_has_peer=has_peer,
    )
