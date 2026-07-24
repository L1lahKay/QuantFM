"""带持仓缓冲、成交拒绝和显式成本的日频研究组合模拟器。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class PortfolioConfig:
    """稳健 Top-K 组合规则。"""

    candidate_top_k: int = 150
    target_holdings: int = 100
    entry_rank: int = 80
    exit_rank: int = 180
    rebalance_interval: int = 1
    score_smoothing_days: int = 3
    max_turnover: float = 1.0

    def validate(self) -> None:
        """校验排名和换仓参数。"""
        if not 1 <= self.entry_rank <= self.candidate_top_k:
            msg = "entry_rank must be within candidate_top_k"
            raise ValueError(msg)
        if self.exit_rank < self.entry_rank:
            msg = "exit_rank must be >= entry_rank"
            raise ValueError(msg)
        if not 1 <= self.target_holdings <= self.candidate_top_k:
            msg = "target_holdings must be within candidate_top_k"
            raise ValueError(msg)
        if self.rebalance_interval < 1 or self.score_smoothing_days < 1:
            msg = "rebalance interval and smoothing days must be positive"
            raise ValueError(msg)
        if not 0 < self.max_turnover <= 1:
            msg = "max_turnover must be in (0, 1]"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CostConfig:
    """按成交名义金额收取的简化、可审计成本。"""

    buy_bps: float = 15.0
    sell_bps: float = 15.0
    stamp_duty_bps_sell: float = 0.0

    def validate(self) -> None:
        """成本不得为负。"""
        if min(self.buy_bps, self.sell_bps, self.stamp_duty_bps_sell) < 0:
            msg = "cost rates must be non-negative"
            raise ValueError(msg)


@dataclass(slots=True)
class PortfolioSimulationResult:
    """组合汇总及可追溯逐日明细。"""

    daily: pl.DataFrame
    holdings: pl.DataFrame
    trades: pl.DataFrame
    summary: dict[str, float | int] = field(default_factory=dict)


def _smooth_scores(scores: pl.DataFrame, window: int) -> pl.DataFrame:
    """对每只股票做仅使用当日及历史值的滚动平滑。"""
    frame = scores.sort(["symbol", "date"])
    if window == 1:
        return frame.with_columns(pl.col("score").alias("smooth_score"))
    return frame.with_columns(
        pl.col("score")
        .rolling_mean(window_size=window, min_samples=1)
        .over("symbol")
        .alias("smooth_score")
    )


def simulate_buffered_topk(
    scores: pl.DataFrame,
    panel: pl.DataFrame,
    *,
    portfolio: PortfolioConfig | None = None,
    costs: CostConfig | None = None,
) -> PortfolioSimulationResult:
    """
    生成目标、模拟拒单并按未投资现金计算组合收益。

    选股只读取信号日已知的 ``eligible_at_signal``。买卖调仓均发生在当前
    interval 的 ``entry_date``，只由 ``entry_fillable`` 决定能否成交；买入失败
    不会事后用更低排名股票补位。``exit_fillable`` 不参与当前 interval 的调仓。
    """
    cfg = portfolio or PortfolioConfig()
    cost_cfg = costs or CostConfig()
    cfg.validate()
    cost_cfg.validate()
    required_scores = {"date", "symbol", "score"}
    required_panel = {
        "date",
        "symbol",
        "fwd_ret",
        "eligible_at_signal",
        "entry_fillable",
        "exit_fillable",
    }
    if missing := required_scores - set(scores.columns):
        msg = f"scores missing columns: {sorted(missing)}"
        raise ValueError(msg)
    if missing := required_panel - set(panel.columns):
        msg = f"execution panel missing columns: {sorted(missing)}"
        raise ValueError(msg)

    joined = _smooth_scores(scores, cfg.score_smoothing_days).join(
        panel, on=["date", "symbol"], how="inner"
    )
    dates = sorted(str(value) for value in joined["date"].unique())
    positions: set[str] = set()
    daily_rows: list[dict[str, float | int | str]] = []
    holding_rows: list[dict[str, float | int | str]] = []
    trade_rows: list[dict[str, float | int | str | bool]] = []
    nav = 1.0

    for day_index, date in enumerate(dates):
        day = joined.filter(pl.col("date") == date).sort(
            "smooth_score", descending=True
        )
        day = day.with_row_index("rank", offset=1)
        by_symbol = {row["symbol"]: row for row in day.iter_rows(named=True)}
        rebalance = day_index % cfg.rebalance_interval == 0
        desired = set(positions)
        if rebalance:
            desired = {
                symbol
                for symbol in positions
                if symbol in by_symbol
                and int(by_symbol[symbol]["rank"]) <= cfg.exit_rank
            }
            candidates = day.filter(
                pl.col("eligible_at_signal").fill_null(False)
                & (pl.col("rank") <= cfg.entry_rank)
            )["symbol"].to_list()
            for symbol in candidates:
                if len(desired) >= cfg.target_holdings:
                    break
                desired.add(str(symbol))

        sell_orders = sorted(positions - desired)
        buy_orders = sorted(desired - positions)
        max_side_names = max(int(cfg.max_turnover * cfg.target_holdings), 1)
        sell_orders = sell_orders[:max_side_names]
        buy_orders = buy_orders[:max_side_names]

        filled_sells: set[str] = set()
        for symbol in sell_orders:
            row = by_symbol.get(symbol)
            fillable = bool(row and row.get("entry_fillable"))
            trade_rows.append(
                {
                    "signal_date": date,
                    "symbol": symbol,
                    "side": "SELL",
                    "filled": fillable,
                    "reject_reason": (
                        "" if fillable else "entry_not_fillable_for_sell"
                    ),
                }
            )
            if fillable:
                filled_sells.add(symbol)
        positions -= filled_sells

        filled_buys: set[str] = set()
        for symbol in buy_orders:
            row = by_symbol.get(symbol)
            fillable = bool(row and row.get("entry_fillable"))
            trade_rows.append(
                {
                    "signal_date": date,
                    "symbol": symbol,
                    "side": "BUY",
                    "filled": fillable,
                    "reject_reason": "" if fillable else "entry_not_fillable",
                }
            )
            if fillable:
                filled_buys.add(symbol)
        positions |= filled_buys

        denominator = max(cfg.target_holdings, len(positions), 1)
        weight = 1.0 / denominator
        gross = 0.0
        for symbol in sorted(positions):
            row = by_symbol.get(symbol)
            value = row.get("fwd_ret") if row else None
            ret = float(value) if value is not None and np.isfinite(value) else 0.0
            gross += weight * ret
            holding_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "weight": weight,
                    "rank": int(row["rank"]) if row else -1,
                    "score": float(row["smooth_score"]) if row else float("nan"),
                    "fwd_ret": ret,
                }
            )
        buy_notional = len(filled_buys) / denominator
        sell_notional = len(filled_sells) / denominator
        cost = (
            buy_notional * cost_cfg.buy_bps
            + sell_notional * (cost_cfg.sell_bps + cost_cfg.stamp_duty_bps_sell)
        ) / 1e4
        net = gross - cost
        nav *= 1.0 + net
        daily_rows.append(
            {
                "date": date,
                "gross_return": gross,
                "cost": cost,
                "net_return": net,
                "nav": nav,
                "n_holdings": len(positions),
                "cash_weight": max(1.0 - len(positions) * weight, 0.0),
                "buy_notional": buy_notional,
                "sell_notional": sell_notional,
                "turnover": 0.5 * (buy_notional + sell_notional),
                "failed_buys": len(buy_orders) - len(filled_buys),
                "failed_sells": len(sell_orders) - len(filled_sells),
            }
        )

    daily_frame = pl.DataFrame(daily_rows)
    returns = (
        daily_frame["net_return"].to_numpy()
        if daily_frame.height
        else np.asarray([], dtype=float)
    )
    std = float(returns.std(ddof=1)) if returns.size > 1 else 0.0
    mean = float(returns.mean()) if returns.size else 0.0
    nav_values = (
        daily_frame["nav"].to_numpy()
        if daily_frame.height
        else np.asarray([], dtype=float)
    )
    drawdown = (
        nav_values / np.maximum.accumulate(nav_values) - 1.0
        if nav_values.size
        else np.asarray([], dtype=float)
    )
    summary: dict[str, float | int] = {
        "n_days": int(returns.size),
        "cum_return": float(nav_values[-1] - 1.0) if nav_values.size else 0.0,
        "mean_daily_return": mean,
        "daily_vol": std,
        "sharpe": mean / std * np.sqrt(244) if std > 0 else 0.0,
        "max_drawdown": float(drawdown.min()) if drawdown.size else 0.0,
        "mean_turnover": (
            float(daily_frame["turnover"].mean()) if daily_frame.height else 0.0
        ),
        "total_cost": float(daily_frame["cost"].sum()) if daily_frame.height else 0.0,
    }
    return PortfolioSimulationResult(
        daily=daily_frame,
        holdings=pl.DataFrame(holding_rows),
        trades=pl.DataFrame(trade_rows),
        summary=summary,
    )
