"""多任务输出头，以及兼容 v1 的下一事件损失。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from quant_fm.tokenizer.vocab import PAD_ID


class MultiHead(nn.Module):
    """每个目标字段一个线性分类头。"""

    def __init__(self, d_model: int, field_sizes: dict[str, int]) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {name: nn.Linear(d_model, size) for name, size in field_sizes.items()}
        )

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """将隐状态投影为各字段 logits。"""
        return {name: head(hidden) for name, head in self.heads.items()}


@dataclass(slots=True)
class LossOutput:
    """聚合损失及各字段分解的容器。"""

    total: torch.Tensor
    per_field: dict[str, torch.Tensor]
    normalized_per_field: dict[str, torch.Tensor] = field(default_factory=dict)
    ordinal_per_field: dict[str, torch.Tensor] = field(default_factory=dict)
    weighted_per_field: dict[str, torch.Tensor] = field(default_factory=dict)
    valid_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """一个预训练目标的损失、权重和有效范围声明。"""

    name: str
    loss_type: str = "ce"
    weight: float = 1.0
    entropy: float = 1.0
    ordinal_weight: float = 0.0
    ordinal_start_id: int = 0
    applicable_event_ids: tuple[int, ...] = ()
    ignore_ids: tuple[int, ...] = (PAD_ID,)
    mask_field: str | None = None

    def validate(self) -> None:
        """尽早拒绝会静默改变训练目标的错误配置。"""
        if self.loss_type not in {"ce", "ordinal_ce"}:
            msg = f"unsupported loss type {self.loss_type!r} for {self.name}"
            raise ValueError(msg)
        if self.weight < 0:
            msg = f"target weight must be non-negative: {self.name}"
            raise ValueError(msg)
        if self.entropy <= 0:
            msg = f"target entropy must be positive: {self.name}"
            raise ValueError(msg)
        if self.ordinal_weight < 0:
            msg = f"ordinal weight must be non-negative: {self.name}"
            raise ValueError(msg)
        if self.ordinal_start_id < 0:
            msg = f"ordinal_start_id must be non-negative: {self.name}"
            raise ValueError(msg)


def target_specs_from_config(
    target_fields: tuple[str, ...],
    loss_config: dict[str, Any] | None,
    *,
    default_ignore_ids: tuple[int, ...] = (PAD_ID,),
    default_ordinal_start_id: int = 0,
) -> tuple[TargetSpec, ...] | None:
    """
    将 YAML ``loss.targets`` 转成冻结声明。

    没有 ``loss.targets`` 时返回 ``None``，调用方继续使用 v1 等权 CE。显式启用
    v2 后，未声明的 head 权重为零，因此 ``tok_session`` 不会再成为默认主目标。
    """
    if not loss_config or not loss_config.get("targets"):
        return None
    configured = loss_config["targets"]
    entropy_by_field = loss_config.get("train_entropy", {})
    specs: list[TargetSpec] = []
    for name in target_fields:
        item = configured.get(name)
        if item is None:
            specs.append(
                TargetSpec(
                    name=name,
                    weight=0.0,
                    ignore_ids=default_ignore_ids,
                    ordinal_start_id=default_ordinal_start_id,
                )
            )
            continue
        if not isinstance(item, dict):
            msg = f"loss.targets.{name} must be a mapping"
            raise TypeError(msg)
        applicable = item.get("applicable_event_ids", ())
        if any(not isinstance(value, int) for value in applicable):
            msg = f"loss.targets.{name}.applicable_event_ids must contain token ids"
            raise TypeError(msg)
        ignore_ids = item.get("ignore_ids", default_ignore_ids)
        spec = TargetSpec(
            name=name,
            loss_type=str(item.get("type", "ce")),
            weight=float(item.get("weight", 1.0)),
            entropy=float(item.get("entropy", entropy_by_field.get(name, 1.0))),
            ordinal_weight=float(item.get("ordinal_weight", 0.0)),
            ordinal_start_id=int(
                item.get("ordinal_start_id", default_ordinal_start_id)
            ),
            applicable_event_ids=tuple(int(value) for value in applicable),
            ignore_ids=tuple(int(value) for value in ignore_ids),
            mask_field=item.get("mask_field"),
        )
        spec.validate()
        specs.append(spec)
    return tuple(specs)


def _target_valid_mask(
    batch: dict[str, torch.Tensor],
    spec: TargetSpec,
    *,
    event_type_field: str,
) -> torch.Tensor:
    """组合序列边界、NA、事件适用范围和显式目标 mask。"""
    attention = batch["attention_mask"]
    valid = attention[:, :-1] & attention[:, 1:]
    target = batch[spec.name][:, 1:]
    for token_id in spec.ignore_ids:
        valid = valid & target.ne(token_id)
    if spec.applicable_event_ids:
        event_type = batch[event_type_field][:, 1:]
        applicable = torch.zeros_like(valid)
        for token_id in spec.applicable_event_ids:
            applicable = applicable | event_type.eq(token_id)
        valid = valid & applicable

    mask_name = spec.mask_field or f"mask_{spec.name}"
    explicit = batch.get(mask_name)
    if explicit is not None:
        if explicit.shape != attention.shape:
            msg = f"{mask_name} shape {explicit.shape} != {attention.shape}"
            raise ValueError(msg)
        valid = valid & explicit[:, 1:].bool()
    return valid


def _ordinal_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """惩罚预测期望 bin 与真实 bin 的距离，并按词表宽度缩放。"""
    n_classes = prediction.size(-1)
    indices = torch.arange(n_classes, device=prediction.device, dtype=prediction.dtype)
    expected = (prediction.softmax(dim=-1) * indices).sum(dim=-1)
    scale = float(max(n_classes - 1, 1))
    return F.smooth_l1_loss(expected / scale, target.to(expected.dtype) / scale)


def next_event_loss_v2(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    target_specs: tuple[TargetSpec, ...],
    *,
    event_type_field: str = "tok_evt_type",
) -> LossOutput:
    """计算熵归一化、可按事件 mask 的 CE/ordinal 多任务损失。"""
    device = batch["attention_mask"].device
    total = torch.zeros((), device=device)
    raw_losses: dict[str, torch.Tensor] = {}
    normalized_losses: dict[str, torch.Tensor] = {}
    ordinal_losses: dict[str, torch.Tensor] = {}
    weighted_losses: dict[str, torch.Tensor] = {}
    valid_counts: dict[str, int] = {}

    for spec in target_specs:
        spec.validate()
        prediction = logits[spec.name][:, :-1, :]
        target = batch[spec.name][:, 1:]
        valid = _target_valid_mask(batch, spec, event_type_field=event_type_field)
        count = int(valid.sum().item())
        valid_counts[spec.name] = count
        if count:
            selected_prediction = prediction[valid]
            selected_target = target[valid]
            raw = F.cross_entropy(selected_prediction, selected_target)
            if spec.loss_type == "ordinal_ce":
                ordinal_target = selected_target - spec.ordinal_start_id
                if (ordinal_target < 0).any():
                    msg = f"{spec.name} contains a special id in ordinal targets"
                    raise ValueError(msg)
                ordinal = _ordinal_huber(
                    selected_prediction[:, spec.ordinal_start_id :],
                    ordinal_target,
                )
            else:
                ordinal = prediction.sum() * 0.0
        else:
            # 保留计算图，确保全 NA/不适用 task 可以安全 backward。
            raw = prediction.sum() * 0.0
            ordinal = prediction.sum() * 0.0

        normalized = raw / spec.entropy
        weighted = spec.weight * (normalized + spec.ordinal_weight * ordinal)
        raw_losses[spec.name] = raw
        normalized_losses[spec.name] = normalized
        ordinal_losses[spec.name] = ordinal
        weighted_losses[spec.name] = weighted
        total = total + weighted

    return LossOutput(
        total=total,
        per_field=raw_losses,
        normalized_per_field=normalized_losses,
        ordinal_per_field=ordinal_losses,
        weighted_per_field=weighted_losses,
        valid_counts=valid_counts,
    )


def next_event_loss(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    target_fields: tuple[str, ...],
) -> LossOutput:
    """
    各字段下一事件交叉熵之和，并屏蔽填充。

    ``logits[f]`` 形状为 ``[B, L, V_f]``；目标为 ``batch[f]`` 右移一位。
    当前位或下一位为填充的位置均忽略。

    参数
    ----------
    logits
        全序列各字段 logits。
    batch
        含字段 token 张量与 ``attention_mask`` 的批数据。
    target_fields
        配备预测头的字段。

    返回
    -------
    LossOutput
    """
    mask = batch["attention_mask"]  # [B, L] bool
    # [导读] 有效预测位：当前和下一格都必须是真实事件（不能跨过 padding 边界）
    valid = mask[:, :-1] & mask[:, 1:]  # [B, L-1]
    per_field: dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=mask.device)

    for f in target_fields:
        # [导读] 用位置 0..L-2 的 hidden 预测位置 1..L-1 的 token（右移一位）
        pred = logits[f][:, :-1, :]  # 形状 [B, L-1, 该字段词表大小]
        target = batch[f][:, 1:]  # 下一事件的正确 token id
        target = target.masked_fill(~valid, PAD_ID)  # 无效位置标成 PAD，loss 会忽略
        loss = F.cross_entropy(
            pred.reshape(-1, pred.size(-1)),
            target.reshape(-1),
            ignore_index=PAD_ID,
        )
        per_field[f] = loss
        total = total + loss  # 6 个字段的 CE 相加

    return LossOutput(total=total, per_field=per_field)
