"""事件内多字段表示的可配置融合。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class FieldFusionConfig:
    """字段融合配置；``legacy_sum`` 与 v1 checkpoint 完全兼容。"""

    method: str = "legacy_sum"
    field_dim: int = 32
    field_dropout: float = 0.0
    input_norm: bool = True

    def validate(self) -> None:
        """校验融合方式和数值范围。"""
        choices = {"legacy_sum", "scaled_sum", "gated_sum", "concat_mlp"}
        if self.method not in choices:
            msg = f"unknown field fusion {self.method!r}; choose from {sorted(choices)}"
            raise ValueError(msg)
        if self.field_dim < 1:
            msg = "field_dim must be positive"
            raise ValueError(msg)
        if not 0.0 <= self.field_dropout < 1.0:
            msg = "field_dropout must be in [0, 1)"
            raise ValueError(msg)


class _RMSNorm(nn.Module):
    """避免与模型模块形成循环依赖的轻量 RMSNorm。"""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """按最后一维归一化。"""
        normalized = value * torch.rsqrt(
            value.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized * self.weight


class EventFieldFusion(nn.Module):
    """融合同一事件的字段向量，不改变时间维度。"""

    def __init__(
        self,
        *,
        n_fields: int,
        input_dim: int,
        d_model: int,
        config: FieldFusionConfig,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.n_fields = n_fields
        self.gate_logits = (
            nn.Parameter(torch.zeros(n_fields))
            if config.method == "gated_sum"
            else None
        )
        self.projection = (
            nn.Sequential(
                nn.Linear(n_fields * input_dim, 2 * d_model, bias=False),
                nn.GELU(),
                nn.Linear(2 * d_model, d_model, bias=False),
            )
            if config.method == "concat_mlp"
            else None
        )
        self.norm = (
            _RMSNorm(d_model)
            if config.input_norm and config.method != "legacy_sum"
            else nn.Identity()
        )

    def _field_mask(self, device: torch.device) -> torch.Tensor:
        """生成一次 forward 共用的字段 dropout mask，且至少保留一个字段。"""
        if not self.training or self.config.field_dropout <= 0:
            return torch.ones(self.n_fields, device=device)
        keep = torch.rand(self.n_fields, device=device) >= self.config.field_dropout
        if not bool(keep.any()):
            keep[torch.randint(self.n_fields, (1,), device=device)] = True
        return keep.to(dtype=torch.float32)

    def forward(self, parts: list[torch.Tensor]) -> torch.Tensor:
        """融合字段列表；所有元素形状均为 ``[B, L, D]``。"""
        if len(parts) != self.n_fields:
            msg = f"expected {self.n_fields} fields, got {len(parts)}"
            raise ValueError(msg)
        if self.config.method == "legacy_sum":
            return torch.stack(parts, dim=0).sum(dim=0)

        mask = self._field_mask(parts[0].device).to(parts[0].dtype)
        if self.config.method == "concat_mlp":
            masked = [part * mask[index] for index, part in enumerate(parts)]
            if self.projection is None:  # pragma: no cover - constructor invariant
                msg = "concat projection is not initialised"
                raise RuntimeError(msg)
            return self.norm(self.projection(torch.cat(masked, dim=-1)))

        stacked = torch.stack(parts, dim=-2)
        if self.config.method == "scaled_sum":
            weights = mask
        else:
            if self.gate_logits is None:  # pragma: no cover - constructor invariant
                msg = "gated fusion parameters are not initialised"
                raise RuntimeError(msg)
            weights = torch.sigmoid(self.gate_logits).to(stacked.dtype) * mask
        denominator = weights.square().sum().sqrt().clamp_min(1e-6)
        fused = (stacked * weights.view(1, 1, -1, 1)).sum(dim=-2) / denominator
        return self.norm(fused)


def embedding_dim_for_fusion(config: FieldFusionConfig, d_model: int) -> int:
    """返回各字段 lookup embedding 的维度。"""
    return config.field_dim if config.method == "concat_mlp" else d_model
