"""
仅在训练窗口上拟合全局分位数分箱边界（防泄漏）。

读取规范 cn_l2_v1 事件 parquet（训练切分），流式处理派生连续字段，
子采样以控制内存，并用 1%/99% 缩尾计算分位数边界。结果边界冻结到 :class:`Vocab`。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from quant_fm.tokenizer.transforms import add_derived_fields
from quant_fm.tokenizer.vocab import CONTINUOUS_FIELDS, default_vocab

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from quant_fm.tokenizer.vocab import Vocab

logger = logging.getLogger(__name__)

# 双尾缩尾 vs 仅上尾缩尾的字段。
_TWO_SIDED = {"price_rel"}


def _quantile_edges(
    values: np.ndarray,
    n_bins: int,
    *,
    two_sided: bool,
) -> list[float]:
    """通过缩尾分位数计算 ``n_bins - 1`` 个内部分箱边界。"""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return list(np.linspace(-1.0, 1.0, n_bins - 1))
    lo = np.quantile(values, 0.01) if two_sided else np.quantile(values, 0.0)
    hi = np.quantile(values, 0.99)
    clipped = np.clip(values, lo, hi)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(clipped, qs)
    edges = np.unique(edges)
    # 若过多重复分位数坍缩，用 linspace 补齐。
    if edges.size < n_bins - 1:
        edges = np.linspace(lo, hi if hi > lo else lo + 1.0, n_bins - 1)
    return [float(e) for e in edges]


def fit_bins(
    paths: Iterable[Path],
    *,
    n_bins: int = 32,
    max_samples_per_field: int = 5_000_000,
    fit_dates: Sequence[str] = (),
    seed: int = 0,
) -> Vocab:
    """
    从规范事件 parquet 拟合 :class:`Vocab`。

    参数
    ----------
    paths
        属于**训练**切分的规范事件 parquet 文件。
    n_bins
        每个连续字段的分箱数。
    max_samples_per_field
        每字段蓄水池上限，以控制内存。
    fit_dates
        用于拟合的交易日（记入词表供审计）。
    seed
        子采样 RNG 种子（可复现）。

    返回
    -------
    Vocab
        含冻结连续边界的词表。
    """
    rng = np.random.default_rng(seed)
    pools: dict[str, list[np.ndarray]] = {f: [] for f in CONTINUOUS_FIELDS}
    counts: dict[str, int] = dict.fromkeys(CONTINUOUS_FIELDS, 0)

    for path in paths:
        df = pl.read_parquet(path)
        if df.height == 0:
            continue
        df = add_derived_fields(df)
        for f in CONTINUOUS_FIELDS:
            vals = df[f].to_numpy().astype(np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            # 超预算时按蓄水池方式子采样。
            budget = max_samples_per_field - counts[f]
            if budget <= 0:
                continue
            if vals.size > budget:
                idx = rng.choice(vals.size, size=budget, replace=False)
                vals = vals[idx]
            pools[f].append(vals)
            counts[f] += vals.size

    vocab = default_vocab(n_bins=n_bins)
    for f in CONTINUOUS_FIELDS:
        pooled = (
            np.concatenate(pools[f]) if pools[f] else np.array([], dtype=np.float64)
        )
        vocab.edges[f] = _quantile_edges(pooled, n_bins, two_sided=f in _TWO_SIDED)
        logger.info(
            "fitted %s: %d samples -> %d edges", f, pooled.size, len(vocab.edges[f])
        )

    vocab.fit_dates = tuple(fit_dates)
    return vocab
