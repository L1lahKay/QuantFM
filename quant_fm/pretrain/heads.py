"""
多任务输出投影与求和的下一事件交叉熵损失。

不设单一巨大组合 softmax，每个预测字段拥有独立线性头与独立交叉熵项。
总损失为各字段 CE 之和（TradeFM 风格），在下一事件目标上计算，并屏蔽填充位。
"""

from __future__ import annotations

from dataclasses import dataclass

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
