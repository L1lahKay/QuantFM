"""
考虑成本的 Top-K 回测，T+1 执行与涨跌停过滤。

给定日度得分与时点正确面板（前瞻收益 + 可交易性），构建多头 Top-K
（或多空）组合，按换手收取交易成本，并报告标准验收指标：年化 Sharpe、
换手、最大回撤与累计收益。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

TRADING_DAYS = 244  # A 股惯例
MIN_DAYS_FOR_ANNUALIZATION = 60  # 少于该天数不年化，避免小样本指数外推爆表


@dataclass(slots=True)
class BacktestResult:
    """
    回测汇总统计与日收益序列。

    指标分两类：
    * **原始/区间指标**（任何样本长度都成立）：``mean_daily_return``、
      ``daily_vol``、``sharpe_daily``、``cum_return``（区间累计收益）、
      ``hit_rate``（正收益日占比）、``max_drawdown``、``turnover``。
    * **年化指标**（仅在 ``n_days >= min_days`` 时给出，否则为 ``None``）：
      ``sharpe``（= sharpe_daily × √TRADING_DAYS）、``ann_return``（CAGR）。

    ``reliable`` 标记年化指标是否可信。
    """

    daily_returns: np.ndarray
    dates: list[str]
    # 原始/区间指标
    mean_daily_return: float
    daily_vol: float
    sharpe_daily: float
    cum_return: float
    hit_rate: float
    max_drawdown: float
    turnover: float
    # 年化指标（小样本时为 None）
    sharpe: float | None
    ann_return: float | None
    reliable: bool
    min_days: int

    def as_dict(self) -> dict[str, float | None]:
        """返回可 JSON 序列化的摘要。"""
        return {
            "n_days": float(len(self.dates)),
            "reliable": self.reliable,
            "min_days_for_annualization": float(self.min_days),
            "mean_daily_return": self.mean_daily_return,
            "daily_vol": self.daily_vol,
            "sharpe_daily": self.sharpe_daily,
            "cum_return": self.cum_return,
            "hit_rate": self.hit_rate,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "sharpe": self.sharpe,
            "ann_return": self.ann_return,
        }


def _max_drawdown(equity: np.ndarray) -> float:
    """返回权益曲线最大峰谷回撤。"""
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min()) if dd.size else 0.0


def backtest_topk(
    scores: pl.DataFrame,
    panel: pl.DataFrame,
    *,
    top_k: int = 50,
    long_short: bool = False,
    cost_bps: float = 15.0,
    min_days: int = MIN_DAYS_FOR_ANNUALIZATION,
    trading_days: int = TRADING_DAYS,
) -> BacktestResult:
    """
    运行 Top-K 回测。

    参数
    ----------
    scores
        ``date, symbol, score``（越高越看好）。
    panel
        时点正确日频面板，含 ``date, symbol, fwd_ret`` 及可选
        ``limit_locked``（涨跌停锁死无法在 T+1 建仓）。
    top_k
        多头 leg 标的数（``long_short`` 时空头亦为 bottom-K）。
    long_short
        为真时同时做空 bottom-K（美元中性）。
    cost_bps
        按换手收取的往返交易成本（基点）。
    min_days
        样本天数低于该值时不给出年化 Sharpe/收益（置 ``None``、``reliable=False``），
        避免用极短样本做指数外推得到爆表数字。
    trading_days
        年化用的年交易日数。

    返回
    -------
    BacktestResult
    """
    df = scores.join(panel, on=["date", "symbol"], how="inner")
    if "limit_locked" in df.columns:
        df = df.filter(~pl.col("limit_locked").cast(pl.Boolean).fill_null(False))

    prev_long: set[str] = set()
    prev_short: set[str] = set()
    daily_returns: list[float] = []
    dates: list[str] = []
    turnovers: list[float] = []

    for (date,), sub in df.group_by(["date"], maintain_order=True):
        sub = sub.sort("score", descending=True)
        longs = sub.head(top_k)
        long_ret = float(longs["fwd_ret"].mean() or 0.0)
        long_set = set(longs["symbol"].to_list())

        if long_short:
            shorts = sub.tail(top_k)
            short_ret = float(shorts["fwd_ret"].mean() or 0.0)
            short_set = set(shorts["symbol"].to_list())
            gross = 0.5 * (long_ret - short_ret)
        else:
            short_set = set()
            gross = long_ret

        # 换手 = 自昨日以来组合中变更的仓位比例。
        long_turn = len(long_set ^ prev_long) / max(2 * top_k, 1)
        short_turn = len(short_set ^ prev_short) / max(2 * top_k, 1)
        turn = long_turn + short_turn
        cost = turn * cost_bps / 1e4

        daily_returns.append(gross - cost)
        turnovers.append(turn)
        dates.append(str(date))
        prev_long, prev_short = long_set, short_set

    returns = np.asarray(daily_returns, dtype=np.float64)
    n = returns.size
    equity = np.cumprod(1.0 + returns) if n else np.asarray([], dtype=np.float64)

    # 原始/区间指标：任何样本长度都成立，不做年化外推。
    mean = float(returns.mean()) if n else 0.0
    std = float(returns.std(ddof=1)) if n > 1 else 0.0
    sharpe_daily = float(mean / std) if std > 0 else 0.0
    cum_return = float(equity[-1] - 1.0) if n else 0.0
    hit_rate = float(np.mean(returns > 0.0)) if n else 0.0

    # 年化指标：仅在样本足够长时给出，否则置 None 并标记不可信。
    reliable = n >= min_days
    if reliable and std > 0:
        sharpe = sharpe_daily * float(np.sqrt(trading_days))
        ann_return = float(equity[-1] ** (trading_days / n) - 1.0)
    else:
        sharpe = None
        ann_return = None

    result = BacktestResult(
        daily_returns=returns,
        dates=dates,
        mean_daily_return=mean,
        daily_vol=std,
        sharpe_daily=sharpe_daily,
        cum_return=cum_return,
        hit_rate=hit_rate,
        max_drawdown=_max_drawdown(equity) if n else 0.0,
        turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        sharpe=sharpe,
        ann_return=ann_return,
        reliable=reliable,
        min_days=min_days,
    )
    if reliable:
        logger.info(
            "backtest: days=%d cum=%.2f%% sharpe_d=%.3f sharpe_ann=%.2f "
            "ann=%.2f%% hit=%.2f mdd=%.2f%% turnover=%.2f",
            n,
            cum_return * 100,
            sharpe_daily,
            sharpe,
            (ann_return or 0.0) * 100,
            hit_rate,
            result.max_drawdown * 100,
            result.turnover,
        )
    else:
        logger.info(
            "backtest: days=%d cum=%.2f%% sharpe_d=%.3f hit=%.2f mdd=%.2f%% "
            "turnover=%.2f [年化略过：样本<%d日，不可信]",
            n,
            cum_return * 100,
            sharpe_daily,
            hit_rate,
            result.max_drawdown * 100,
            result.turnover,
            min_days,
        )
    return result
