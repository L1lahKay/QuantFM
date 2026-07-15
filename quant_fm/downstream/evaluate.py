"""
门控 3：横截面指标、CPCV 与 Deflated Sharpe Ratio。

提供报告中的验收机制：

* ``rank_ic`` / ``rank_icir`` -- 日度横截面 Spearman IC 及其 ICIR
* ``group_monotonicity``       -- 得分分位组收益的单调性
* ``cpcv_splits``              -- 组合式 purged CV（purge + embargo）
* ``deflated_sharpe_ratio``    -- 对 N 次试验中最佳结果的运气折扣
* ``correlation_gate``         -- 拒绝与已知因子过度相关的信号
"""

from __future__ import annotations

import logging
from itertools import combinations
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

_NORM = NormalDist()
_EULER_GAMMA = 0.5772156649015329


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """两数组的 Spearman 秩相关。"""
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def rank_ic(
    predictions: pl.DataFrame,
    panel: pl.DataFrame,
    *,
    score_col: str = "score",
    ret_col: str = "fwd_ret",
) -> pl.DataFrame:
    """``score`` 与前瞻收益的日度横截面 Spearman IC。"""
    df = predictions.join(panel, on=["date", "symbol"], how="inner")
    rows = []
    for (date,), sub in df.group_by(["date"], maintain_order=True):
        ic = _spearman(sub[score_col].to_numpy(), sub[ret_col].to_numpy())
        rows.append({"date": str(date), "ic": ic})
    return pl.DataFrame(rows)


def rank_icir(ic_frame: pl.DataFrame) -> float:
    """日度 IC 序列的信息比率（均值 / 标准差）。"""
    ic = ic_frame["ic"].to_numpy()
    ic = ic[np.isfinite(ic)]
    if ic.size < 2 or ic.std(ddof=1) == 0:
        return float("nan")
    return float(ic.mean() / ic.std(ddof=1))


def group_monotonicity(
    predictions: pl.DataFrame,
    panel: pl.DataFrame,
    *,
    n_groups: int = 5,
    score_col: str = "score",
    ret_col: str = "fwd_ret",
) -> list[float]:
    """各得分分位组的平均前瞻收益（期望单调上升）。"""
    df = predictions.join(panel, on=["date", "symbol"], how="inner")
    df = df.with_columns(
        (
            (pl.col(score_col).rank("ordinal").over("date") - 1)
            * n_groups
            // pl.len().over("date")
        ).alias("grp")
    )
    grouped = df.group_by("grp").agg(pl.col(ret_col).mean().alias("ret")).sort("grp")
    return grouped["ret"].to_list()


def cpcv_splits(
    dates: Sequence[str],
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    purge: int = 5,
    embargo: int = 2,
) -> list[tuple[list[str], list[str]]]:
    """
    时间块上的组合式 purged 交叉验证切分。

    参数
    ----------
    dates
        全部可用交易日（任意顺序；内部排序）。
    n_groups
        连续时间块数量。
    n_test_groups
        每次组合中作为测试集的块数。
    purge
        每个测试块周围从训练集中剔除的天数（标签重叠 purge）。
    embargo
        每个测试块之后额外屏蔽的天数。

    返回
    -------
    list of ``(train_dates, test_dates)``。
    """
    uniq = sorted(set(dates))
    n = len(uniq)
    if n < n_groups:
        msg = f"need >= {n_groups} dates, got {n}"
        raise ValueError(msg)
    bounds = np.linspace(0, n, n_groups + 1).astype(int)
    blocks = [list(range(bounds[i], bounds[i + 1])) for i in range(n_groups)]

    splits: list[tuple[list[str], list[str]]] = []
    for combo in combinations(range(n_groups), n_test_groups):
        test_idx = sorted(i for g in combo for i in blocks[g])
        test_set = set(test_idx)
        blocked = set(test_idx)
        for i in test_idx:
            for d in range(1, purge + 1):
                blocked.add(i - d)
                blocked.add(i + d)
            for d in range(1, embargo + 1):
                blocked.add(i + purge + d)
        train_idx = [i for i in range(n) if i not in blocked and i not in test_set]
        splits.append(
            ([uniq[i] for i in train_idx], [uniq[i] for i in sorted(test_idx)])
        )
    logger.info("generated %d CPCV splits", len(splits))
    return splits


def deflated_sharpe_ratio(
    observed_sr: float,
    *,
    n_trials: int,
    n_obs: int,
    sr_variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Deflated Sharpe Ratio：真实 SR 超过选择门槛的概率。

    参数
    ----------
    observed_sr
        最佳候选 Sharpe（每次观测，非年化）。
    n_trials
        独立尝试的策略配置数（选择偏差）。
    n_obs
        收益观测次数。
    sr_variance
        各次试验 Sharpe 的方差。
    skew, kurtosis
        收益分布高阶矩（kurtosis 为原始值，正态为 3）。

    返回
    -------
    float
        DSR，范围 ``[0, 1]``；通常要求约 > 0.95。
    """
    if n_trials < 1 or n_obs < 2:
        return float("nan")
    std = float(np.sqrt(max(sr_variance, 0.0)))
    z1 = _NORM.inv_cdf(1 - 1.0 / n_trials)
    z2 = _NORM.inv_cdf(1 - 1.0 / (n_trials * np.e))
    sr_star = std * ((1 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)

    num = (observed_sr - sr_star) * np.sqrt(n_obs - 1)
    denom = np.sqrt(1 - skew * observed_sr + (kurtosis - 1) / 4.0 * observed_sr**2)
    if denom <= 0:
        return float("nan")
    return float(_NORM.cdf(num / denom))


def correlation_gate(
    signal: np.ndarray,
    existing_factors: dict[str, np.ndarray],
    *,
    max_abs_corr: float = 0.7,
) -> dict[str, float | bool]:
    """拒绝与任一已有因子过度相关的信号。"""
    corrs = {}
    for name, fac in existing_factors.items():
        m = np.isfinite(signal) & np.isfinite(fac)
        if m.sum() < 3:
            corrs[name] = float("nan")
            continue
        corrs[name] = float(np.corrcoef(signal[m], fac[m])[0, 1])
    max_corr = max((abs(v) for v in corrs.values() if np.isfinite(v)), default=0.0)
    result: dict[str, float | bool] = dict(corrs)
    result["max_abs_corr"] = max_corr
    result["passed"] = max_corr <= max_abs_corr
    return result
