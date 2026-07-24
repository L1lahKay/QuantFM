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
from dataclasses import dataclass
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


@dataclass(slots=True)
class ICStats:
    """日度 IC 的集中趋势、稳定性和自相关稳健显著性。"""

    n_periods: int
    mean_ic: float
    std_ic: float
    icir: float
    positive_rate: float
    naive_t: float
    newey_west_t: float
    hac_lags: int


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """两数组的 Spearman 秩相关。"""
    if a.size < 3:
        return float("nan")

    def _average_rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        ranks = np.empty(values.size, dtype=np.float64)
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return ranks

    ra = _average_rank(np.asarray(a, dtype=np.float64))
    rb = _average_rank(np.asarray(b, dtype=np.float64))
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
    df = df.filter(
        pl.col(score_col).is_not_null()
        & pl.col(ret_col).is_not_null()
        & pl.col(score_col).is_finite()
        & pl.col(ret_col).is_finite()
    )
    if "eligible_at_signal" in df.columns:
        df = df.filter(pl.col("eligible_at_signal").fill_null(False))
    rows = []
    for (date,), sub in df.group_by(["date"], maintain_order=True):
        ic = _spearman(sub[score_col].to_numpy(), sub[ret_col].to_numpy())
        rows.append({"date": str(date), "ic": ic})
    if not rows:
        return pl.DataFrame(schema={"date": pl.Utf8, "ic": pl.Float64})
    return pl.DataFrame(rows)


def rank_icir(ic_frame: pl.DataFrame) -> float:
    """日度 IC 序列的信息比率（均值 / 标准差）。"""
    ic = ic_frame["ic"].to_numpy()
    ic = ic[np.isfinite(ic)]
    if ic.size < 2 or ic.std(ddof=1) == 0:
        return float("nan")
    return float(ic.mean() / ic.std(ddof=1))


def ic_statistics(
    ic_frame: pl.DataFrame,
    *,
    hac_lags: int | None = None,
) -> ICStats:
    """计算 ICIR、朴素 t 值和 Newey-West/HAC t 值。"""
    values = ic_frame["ic"].to_numpy().astype(np.float64)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return ICStats(0, float("nan"), float("nan"), float("nan"), 0.0, 0.0, 0.0, 0)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    icir = mean / std if std > 0 else float("nan")
    naive_t = mean / (std / np.sqrt(n)) if std > 0 else 0.0
    lag = (
        max(int(np.floor(4 * (n / 100) ** (2 / 9))), 0)
        if hac_lags is None
        else max(min(int(hac_lags), n - 1), 0)
    )
    centered = values - mean
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_variance = gamma0
    for offset in range(1, lag + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / n)
        weight = 1.0 - offset / (lag + 1)
        long_run_variance += 2.0 * weight * covariance
    hac_se = np.sqrt(max(long_run_variance, 0.0) / n)
    newey_west_t = mean / hac_se if hac_se > 0 else 0.0
    return ICStats(
        n_periods=n,
        mean_ic=mean,
        std_ic=std,
        icir=float(icir),
        positive_rate=float(np.mean(values > 0)),
        naive_t=float(naive_t),
        newey_west_t=float(newey_west_t),
        hac_lags=lag,
    )


def block_bootstrap_mean_ci(
    values: np.ndarray,
    *,
    block_size: int = 5,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """移动块 bootstrap 的均值 95% 置信区间。"""
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan")
    size = max(min(int(block_size), clean.size), 1)
    starts = np.arange(clean.size - size + 1)
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    blocks_needed = int(np.ceil(clean.size / size))
    for index in range(n_bootstrap):
        selected = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([clean[start : start + size] for start in selected])
        means[index] = sample[: clean.size].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def quantile_return_panel(
    predictions: pl.DataFrame,
    panel: pl.DataFrame,
    *,
    n_groups: int = 10,
    score_col: str = "score",
    ret_col: str = "fwd_ret",
) -> pl.DataFrame:
    """返回逐日逐分位收益；group=0 最低分、group=n-1 最高分。"""
    if n_groups < 2:
        msg = "n_groups must be >= 2"
        raise ValueError(msg)
    frame = predictions.join(panel, on=["date", "symbol"], how="inner")
    frame = frame.filter(pl.col(ret_col).is_not_null())
    if "eligible_at_signal" in frame.columns:
        frame = frame.filter(pl.col("eligible_at_signal").fill_null(False))
    frame = frame.with_columns(
        (
            (pl.col(score_col).rank("ordinal").over("date") - 1)
            * n_groups
            // pl.len().over("date")
        )
        .clip(0, n_groups - 1)
        .cast(pl.Int32)
        .alias("group")
    )
    return (
        frame.group_by(["date", "group"])
        .agg(
            pl.len().alias("n_names"),
            pl.col(ret_col).mean().alias("mean_return"),
        )
        .sort(["date", "group"])
    )


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
