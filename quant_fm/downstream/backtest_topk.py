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


@dataclass(slots=True)
class BacktestResult:
    """回测汇总统计与日收益序列。"""

    daily_returns: np.ndarray
    dates: list[str]
    sharpe: float
    ann_return: float
    max_drawdown: float
    turnover: float

    def as_dict(self) -> dict[str, float]:
        """返回可 JSON 序列化的摘要。"""
        return {
            "sharpe": self.sharpe,
            "ann_return": self.ann_return,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "n_days": float(len(self.dates)),
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
    equity = np.cumprod(1.0 + returns)
    mean, std = returns.mean(), returns.std(ddof=1) if returns.size > 1 else 0.0
    sharpe = float(mean / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0
    ann_return = (
        float(equity[-1] ** (TRADING_DAYS / len(returns)) - 1) if returns.size else 0.0
    )

    result = BacktestResult(
        daily_returns=returns,
        dates=dates,
        sharpe=sharpe,
        ann_return=ann_return,
        max_drawdown=_max_drawdown(equity) if returns.size else 0.0,
        turnover=float(np.mean(turnovers)) if turnovers else 0.0,
    )
    logger.info(
        "backtest: days=%d sharpe=%.2f ann=%.2f%% mdd=%.2f%% turnover=%.2f",
        len(dates),
        result.sharpe,
        result.ann_return * 100,
        result.max_drawdown * 100,
        result.turnover,
    )
    return result
