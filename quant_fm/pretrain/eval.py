"""
预训练评估：各字段困惑度与风格化事实检验。

困惑度为过程指标（真正验收在下游 RankIC 提升）。
风格化事实（厚尾收益、波动率聚集）用于检验模型/数据是否再现已知微观结构规律。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from quant_fm.pretrain.heads import next_event_loss

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from quant_fm.pretrain.model import OrderFlowFM

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PerplexityReport:
    """各字段交叉熵与困惑度。"""

    ce: dict[str, float]

    @property
    def perplexity(self) -> dict[str, float]:
        """各字段 CE 的指数。"""
        return {f: math.exp(v) for f, v in self.ce.items()}

    @property
    def total_ce(self) -> float:
        """各字段 CE 之和（训练目标）。"""
        return float(sum(self.ce.values()))


@torch.no_grad()
def field_perplexity(
    model: OrderFlowFM,
    loader: DataLoader,
    device: torch.device,
    target_fields: tuple[str, ...],
    *,
    max_batches: int = 200,
) -> PerplexityReport:
    """在 DataLoader 上平均各字段交叉熵。"""
    model.eval()
    sums = dict.fromkeys(target_fields, 0.0)
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        out = next_event_loss(logits, batch, target_fields)
        for f, fl in out.per_field.items():
            sums[f] += float(fl.item())
        n += 1
    ce = {f: sums[f] / max(n, 1) for f in target_fields}
    return PerplexityReport(ce=ce)


def stylized_facts(returns: np.ndarray, *, max_lag: int = 50) -> dict[str, float]:
    """
    计算超额峰度与波动率聚集自相关。

    参数
    ----------

    Returns
    -------
        一维（对数）收益序列，如中间价变化。
    max_lag
        |收益| 自相关平均的最大滞后。

    返回
    -------
    dict
        ``excess_kurtosis``（厚尾 > 0）与 ``vol_clustering_acf``（|收益| 自相关均值；
        > 0 表示聚集）。
    """
    r = returns[np.isfinite(returns)]
    if r.size < max_lag + 2:
        return {"excess_kurtosis": float("nan"), "vol_clustering_acf": float("nan")}
    r = r - r.mean()
    var = r.var()
    kurt = (np.mean(r**4) / (var**2)) - 3.0 if var > 0 else float("nan")

    absr = np.abs(r)
    absr = absr - absr.mean()
    denom = np.sum(absr**2)
    acfs = []
    for lag in range(1, max_lag + 1):
        num = np.sum(absr[:-lag] * absr[lag:])
        acfs.append(num / denom if denom > 0 else 0.0)
    return {
        "excess_kurtosis": float(kurt),
        "vol_clustering_acf": float(np.mean(acfs)),
    }
