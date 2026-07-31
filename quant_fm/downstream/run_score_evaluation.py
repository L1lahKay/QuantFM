"""对冻结 OOS score 做独立、可复现的研究评估。"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from quant_fm.downstream.evaluate import (
    block_bootstrap_mean_ci,
    ic_statistics,
    quantile_return_panel,
    rank_ic,
)
from quant_fm.downstream.portfolio_simulator import (
    CostConfig,
    PortfolioConfig,
    simulate_buffered_topk,
)
from quant_fm.downstream.return_spec import (
    read_trading_calendar,
    validate_execution_panel_contract,
)
from quant_fm.downstream.risk_attribution import (
    portfolio_exposures,
    residualize_returns,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def _json_safe(value: object) -> object:
    """递归把 numpy 标量转成内置类型，并以 null 代替非有限浮点数。"""
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _return_summary(values: np.ndarray) -> dict[str, float | int]:
    """汇总不做事后年化收益外推的日收益序列。"""
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {
            "n_days": 0,
            "cum_return": 0.0,
            "mean_daily_return": 0.0,
            "daily_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }
    nav = np.cumprod(1.0 + clean)
    drawdown = nav / np.maximum.accumulate(nav) - 1.0
    std = float(clean.std(ddof=1)) if clean.size > 1 else 0.0
    mean = float(clean.mean())
    return {
        "n_days": int(clean.size),
        "cum_return": float(nav[-1] - 1.0),
        "mean_daily_return": mean,
        "daily_vol": std,
        "sharpe": mean / std * np.sqrt(244) if std > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
    }


def _eligible_benchmark(
    panel: pl.DataFrame,
    *,
    score_dates: list[str],
    score_keys: pl.DataFrame | None,
    return_name: str,
) -> pl.DataFrame:
    """构造全股票池或有 score 股票池的时点一致等权基准。"""
    frame = panel.filter(pl.col("date").is_in(score_dates))
    if score_keys is not None:
        frame = score_keys.join(frame, on=["date", "symbol"], how="inner")
    return (
        frame.filter(
            pl.col("eligible_at_signal").fill_null(False)
            & pl.col("fwd_ret").is_not_null()
            & pl.col("fwd_ret").is_finite()
        )
        .group_by("date")
        .agg(pl.col("fwd_ret").mean().alias(return_name))
        .sort("date")
    )


def _topk_grid(
    scores: pl.DataFrame,
    panel: pl.DataFrame,
    *,
    top_ks: list[int],
    rebalance_intervals: list[int],
    smoothing_windows: list[int],
    cost_bps: list[float],
) -> pl.DataFrame:
    rows: list[dict[str, float | int]] = []
    for top_k in top_ks:
        for interval in rebalance_intervals:
            for smoothing in smoothing_windows:
                for cost in cost_bps:
                    result = simulate_buffered_topk(
                        scores,
                        panel,
                        portfolio=PortfolioConfig(
                            candidate_top_k=top_k,
                            target_holdings=top_k,
                            entry_rank=top_k,
                            exit_rank=top_k,
                            rebalance_interval=interval,
                            score_smoothing_days=smoothing,
                            max_turnover=1.0,
                        ),
                        costs=CostConfig(buy_bps=cost, sell_bps=cost),
                    )
                    rows.append(
                        {
                            "top_k": top_k,
                            "rebalance_interval": interval,
                            "smoothing_days": smoothing,
                            "cost_bps": cost,
                            **result.summary,
                        }
                    )
    return pl.DataFrame(rows)


def _head_ranking_metrics(
    joined: pl.DataFrame,
    *,
    cutoffs: tuple[int, ...] = (50, 300, 350),
) -> tuple[dict[str, float], pl.DataFrame]:
    """按训练同口径计算 exact NDCG 与 Top-K 超额收益。"""
    daily_rows: list[dict[str, float | str]] = []
    for (date,), day in joined.group_by("date", maintain_order=True):
        score = day["score"].to_numpy().astype(np.float64)
        realized = day["fwd_ret"].to_numpy().astype(np.float64)
        n_names = realized.size
        percentile = (
            day.select(
                ((pl.col("fwd_ret").rank("average") - 1.0) / max(n_names - 1, 1)).alias(
                    "percentile"
                )
            )["percentile"]
            .to_numpy()
            .astype(np.float64)
        )
        gain = np.square(np.clip((percentile - 0.5) / 0.5, 0.0, None))
        row: dict[str, float | str] = {"date": str(date)}
        for cutoff in cutoffs:
            effective_k = min(cutoff, n_names)
            discounts = 1.0 / np.log2(np.arange(2, effective_k + 2))
            predicted = np.argsort(-score, kind="stable")[:effective_k]
            ideal = np.argsort(-gain, kind="stable")[:effective_k]
            ideal_dcg = float(np.dot(gain[ideal], discounts))
            dcg = float(np.dot(gain[predicted], discounts))
            row[f"ndcg_{cutoff}"] = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            row[f"top_excess_bps_{cutoff}"] = float(
                (realized[predicted].mean() - realized.mean()) * 1e4
            )
        daily_rows.append(row)
    daily = pl.DataFrame(daily_rows)
    summary = {
        column: float(daily[column].mean())
        for column in daily.columns
        if column != "date"
    }
    return summary, daily


def evaluate_scores(
    *,
    scores_path: Path,
    panel_path: Path,
    calendar_path: Path | None = None,
    out_dir: Path,
    quantile_groups: int = 10,
    top_ks: list[int] | None = None,
    rebalance_intervals: list[int] | None = None,
    smoothing_windows: list[int] | None = None,
    cost_bps: list[float] | None = None,
    factors_path: Path | None = None,
    market_benchmark_panel_path: Path | None = None,
    require_complete_horizon: bool = True,
    allow_legacy_return_panel: bool = False,
) -> Path:
    """评估 score 并落盘指标、逐日序列、网格和账本。"""
    scores = (
        pl.read_parquet(scores_path)
        .select(["date", "symbol", "score"])
        .with_columns(pl.col("score").cast(pl.Float64, strict=False))
    )
    panel = pl.read_parquet(panel_path)
    trading_calendar = (
        read_trading_calendar(calendar_path) if calendar_path is not None else None
    )
    execution_contract = validate_execution_panel_contract(
        panel,
        trading_calendar=trading_calendar,
        require_calendar_verification=calendar_path is not None,
        allow_legacy=allow_legacy_return_panel,
    )
    if scores.is_empty():
        msg = "scores must contain at least one row"
        raise ValueError(msg)
    invalid_scores = scores.filter(
        pl.col("score").is_null() | ~pl.col("score").is_finite()
    ).height
    if invalid_scores:
        msg = f"scores contain {invalid_scores} null or non-finite score values"
        raise ValueError(msg)
    required_panel = {"date", "symbol", "fwd_ret", "eligible_at_signal"}
    if missing := required_panel - set(panel.columns):
        msg = f"panel missing required evaluation columns: {sorted(missing)}"
        raise ValueError(msg)
    key_columns = ["date", "symbol"]
    if scores.select(pl.struct(key_columns).is_duplicated().any()).item():
        msg = "scores contain duplicate (date, symbol) keys"
        raise ValueError(msg)
    if panel.select(pl.struct(key_columns).is_duplicated().any()).item():
        msg = "panel contains duplicate (date, symbol) keys"
        raise ValueError(msg)

    score_dates = sorted(str(value) for value in scores["date"].unique())
    panel_keys = panel.select(key_columns)
    matched_key_rows = (
        scores.select(key_columns).join(panel_keys, on=key_columns, how="inner").height
    )
    key_coverage = matched_key_rows / scores.height
    if matched_key_rows != scores.height:
        msg = (
            "incomplete evaluation horizon: execution panel key coverage is "
            f"{matched_key_rows}/{scores.height}"
        )
        raise ValueError(msg)

    joined = scores.join(panel, on=key_columns, how="inner")
    eligible = joined.filter(pl.col("eligible_at_signal").fill_null(False))
    finite_labels = eligible.filter(
        pl.col("fwd_ret").is_not_null() & pl.col("fwd_ret").is_finite()
    )
    eligible_by_date = {
        str(row["date"]): int(row["eligible_rows"])
        for row in eligible.group_by("date")
        .agg(pl.len().alias("eligible_rows"))
        .to_dicts()
    }
    valid_by_date = {
        str(row["date"]): int(row["valid_rows"])
        for row in finite_labels.group_by("date")
        .agg(pl.len().alias("valid_rows"))
        .to_dicts()
    }
    incomplete_label_dates = [
        date
        for date in score_dates
        if valid_by_date.get(date, 0) < eligible_by_date.get(date, 0)
    ]
    insufficient_cross_section_dates = [
        date for date in score_dates if valid_by_date.get(date, 0) < 3
    ]
    eligible_score_rows = eligible.height
    valid_label_rows = finite_labels.height
    label_coverage = (
        valid_label_rows / eligible_score_rows if eligible_score_rows else 0.0
    )
    if require_complete_horizon and valid_label_rows != eligible_score_rows:
        msg = (
            "incomplete evaluation horizon: finite fwd_ret coverage for eligible "
            f"score rows is {valid_label_rows}/{eligible_score_rows}; dates="
            f"{incomplete_label_dates[:5]}"
        )
        raise ValueError(msg)
    if require_complete_horizon and insufficient_cross_section_dates:
        msg = (
            "incomplete evaluation horizon: fewer than 3 eligible finite labels on "
            f"score dates {insufficient_cross_section_dates[:5]}"
        )
        raise ValueError(msg)

    ic = rank_ic(scores, panel)
    stats = ic_statistics(ic)
    if stats.n_periods == 0:
        msg = "evaluation produced zero valid IC periods"
        raise ValueError(msg)
    ci_low, ci_high = block_bootstrap_mean_ci(ic["ic"].to_numpy())
    quantiles = quantile_return_panel(scores, panel, n_groups=quantile_groups)
    head_metrics, daily_head_metrics = _head_ranking_metrics(finite_labels)
    grid = _topk_grid(
        scores,
        panel,
        top_ks=top_ks or [50, 100, 200, 300, 350],
        rebalance_intervals=rebalance_intervals or [1, 5],
        smoothing_windows=smoothing_windows or [1, 3],
        cost_bps=cost_bps or [0.0, 15.0, 30.0],
    )
    primary_portfolio = PortfolioConfig(
        candidate_top_k=350,
        target_holdings=300,
        entry_rank=300,
        exit_rank=350,
        rebalance_interval=3,
        score_smoothing_days=3,
        max_turnover=0.15,
    )
    primary_costs = CostConfig()
    primary = simulate_buffered_topk(
        scores, panel, portfolio=primary_portfolio, costs=primary_costs
    )
    score_keys = scores.select(["date", "symbol"])
    benchmark = _eligible_benchmark(
        panel,
        score_dates=score_dates,
        score_keys=score_keys,
        return_name="scored_universe_return",
    )
    active = primary.daily.join(benchmark, on="date", how="left").with_columns(
        (pl.col("net_return") - pl.col("scored_universe_return")).alias("active_return")
    )
    benchmark_daily = benchmark
    market_benchmark_summary: dict[str, float | int] | None = None
    if market_benchmark_panel_path is not None:
        market_panel = pl.read_parquet(market_benchmark_panel_path)
        market_contract = validate_execution_panel_contract(
            market_panel,
            trading_calendar=trading_calendar,
            require_calendar_verification=calendar_path is not None,
            allow_legacy=allow_legacy_return_panel,
        )
        comparable_fields = (
            "return_spec",
            "entry_day_lag",
            "exit_day_lag",
            "entry_price_field",
            "exit_price_field",
        )
        if any(
            market_contract.get(field) != execution_contract.get(field)
            for field in comparable_fields
        ):
            msg = "market benchmark panel uses a different execution contract"
            raise ValueError(msg)
        market_benchmark = _eligible_benchmark(
            market_panel,
            score_dates=score_dates,
            score_keys=None,
            return_name="all_market_return",
        )
        benchmark_daily = benchmark_daily.join(market_benchmark, on="date", how="left")
        market_benchmark_summary = _return_summary(
            market_benchmark["all_market_return"].to_numpy()
        )
    active_values = active["active_return"].drop_nulls().to_numpy()
    active_std = float(active_values.std(ddof=1)) if active_values.size > 1 else 0.0
    active_mean = float(active_values.mean()) if active_values.size else 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    ic.write_parquet(out_dir / "daily_ic.parquet")
    quantiles.write_parquet(out_dir / "quantile_returns.parquet")
    daily_head_metrics.write_parquet(out_dir / "daily_head_metrics.parquet")
    grid.write_parquet(out_dir / "topk_grid.parquet")
    benchmark_daily.write_parquet(out_dir / "benchmark_daily.parquet")
    active.write_parquet(out_dir / "portfolio_daily.parquet")
    primary.holdings.write_parquet(out_dir / "holdings.parquet")
    primary.trades.write_parquet(out_dir / "trades.parquet")

    neutralized_metrics: dict[str, object] | None = None
    if factors_path is not None:
        factors = pl.read_parquet(factors_path)
        factor_cols = [name for name in factors.columns if name.startswith("factor_")]
        if factor_cols:
            exposures = portfolio_exposures(
                primary.holdings, factors, exposure_cols=factor_cols
            )
            exposures.write_parquet(out_dir / "exposure_daily.parquet")
            neutral = residualize_returns(panel, factors, exposure_cols=factor_cols)
            neutral_ic = rank_ic(
                scores,
                neutral,
                ret_col="neutralized_ret",
            )
            neutral_ic.write_parquet(out_dir / "daily_neutralized_ic.parquet")
            neutralized_metrics = asdict(ic_statistics(neutral_ic))

    created_utc = datetime.now(tz=UTC).isoformat()
    raw_metrics = {
        "created_utc": created_utc,
        "signal_days": len(score_dates),
        "evaluated_days": stats.n_periods,
        "missing_horizon_dates": [
            date for date in score_dates if valid_by_date.get(date, 0) == 0
        ],
        "incomplete_label_dates": incomplete_label_dates,
        "insufficient_cross_section_dates": insufficient_cross_section_dates,
        "score_rows": scores.height,
        "panel_rows": panel.height,
        "matched_key_rows": matched_key_rows,
        "join_coverage": key_coverage,
        "eligible_score_rows": eligible_score_rows,
        "valid_label_rows": valid_label_rows,
        "label_coverage": label_coverage,
        "execution_contract": execution_contract,
        "ic": {
            **asdict(stats),
            "block_bootstrap_ci_95": [ci_low, ci_high],
        },
        "head_ranking": head_metrics,
        "primary_portfolio": primary.summary,
        "primary_portfolio_config": asdict(primary_portfolio),
        "primary_cost_config": asdict(primary_costs),
        "scored_universe_benchmark": _return_summary(
            benchmark["scored_universe_return"].to_numpy()
        ),
        "all_market_benchmark": market_benchmark_summary,
        "active": {
            "mean_daily_return": active_mean,
            "daily_tracking_error": active_std,
            "information_ratio": (
                active_mean / active_std * np.sqrt(244) if active_std > 0 else 0.0
            ),
        },
        "neutralized_ic": neutralized_metrics,
    }
    metrics = _json_safe(raw_metrics)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    manifest = {
        "created_utc": created_utc,
        "scores": {"path": str(scores_path.resolve()), "sha256": _sha256(scores_path)},
        "panel": {"path": str(panel_path.resolve()), "sha256": _sha256(panel_path)},
        "calendar": (
            {"path": str(calendar_path.resolve()), "sha256": _sha256(calendar_path)}
            if calendar_path is not None
            else None
        ),
        "market_benchmark_panel": (
            {
                "path": str(market_benchmark_panel_path.resolve()),
                "sha256": _sha256(market_benchmark_panel_path),
            }
            if market_benchmark_panel_path is not None
            else None
        ),
        "factors": (
            {"path": str(factors_path.resolve()), "sha256": _sha256(factors_path)}
            if factors_path is not None
            else None
        ),
        "require_complete_horizon": require_complete_horizon,
        "allow_legacy_return_panel": allow_legacy_return_panel,
        "execution_contract": execution_contract,
        "outputs": [
            "metrics.json",
            "daily_ic.parquet",
            "quantile_returns.parquet",
            "daily_head_metrics.parquet",
            "topk_grid.parquet",
            "benchmark_daily.parquet",
            "portfolio_daily.parquet",
            "holdings.parquet",
            "trades.parquet",
        ],
    }
    (out_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return metrics_path


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument(
        "--calendar",
        type=Path,
        help="构建 execution panel 使用的完整交易日历；传入后逐行重验 T+1/T+2",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--factors", type=Path)
    parser.add_argument("--market-benchmark-panel", type=Path)
    parser.add_argument("--quantile-groups", type=int, default=10)
    parser.add_argument("--top-k-grid", default="50,100,200,300,350")
    parser.add_argument("--rebalance-grid", default="1,5")
    parser.add_argument("--smoothing-grid", default="1,3")
    parser.add_argument("--cost-bps-grid", default="0,15,30")
    parser.add_argument("--allow-incomplete-horizon", action="store_true")
    parser.add_argument(
        "--allow-legacy-return-panel",
        action="store_true",
        help="仅诊断：允许缺少 entry/exit 契约的旧 fwd_ret panel",
    )
    args = parser.parse_args()
    path = evaluate_scores(
        scores_path=args.scores,
        panel_path=args.panel,
        calendar_path=args.calendar,
        out_dir=args.out_dir,
        quantile_groups=args.quantile_groups,
        top_ks=_parse_ints(args.top_k_grid),
        rebalance_intervals=_parse_ints(args.rebalance_grid),
        smoothing_windows=_parse_ints(args.smoothing_grid),
        cost_bps=_parse_floats(args.cost_bps_grid),
        factors_path=args.factors,
        market_benchmark_panel_path=args.market_benchmark_panel,
        require_complete_horizon=not args.allow_incomplete_horizon,
        allow_legacy_return_panel=args.allow_legacy_return_panel,
    )
    print(path)


if __name__ == "__main__":
    main()
