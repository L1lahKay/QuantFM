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


def evaluate_scores(
    *,
    scores_path: Path,
    panel_path: Path,
    out_dir: Path,
    quantile_groups: int = 10,
    top_ks: list[int] | None = None,
    rebalance_intervals: list[int] | None = None,
    smoothing_windows: list[int] | None = None,
    cost_bps: list[float] | None = None,
    factors_path: Path | None = None,
    require_complete_horizon: bool = True,
) -> Path:
    """评估 score 并落盘指标、逐日序列、网格和账本。"""
    scores = (
        pl.read_parquet(scores_path)
        .select(["date", "symbol", "score"])
        .with_columns(pl.col("score").cast(pl.Float64, strict=False))
    )
    panel = pl.read_parquet(panel_path)
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
    grid = _topk_grid(
        scores,
        panel,
        top_ks=top_ks or [20, 50, 100, 150, 200],
        rebalance_intervals=rebalance_intervals or [1, 5],
        smoothing_windows=smoothing_windows or [1, 3],
        cost_bps=cost_bps or [0.0, 15.0, 30.0],
    )
    primary_portfolio = PortfolioConfig()
    primary_costs = CostConfig()
    primary = simulate_buffered_topk(
        scores, panel, portfolio=primary_portfolio, costs=primary_costs
    )
    benchmark = (
        scores.select(["date", "symbol"])
        .join(panel, on=["date", "symbol"], how="inner")
        .filter(pl.col("eligible_at_signal").fill_null(False))
        .group_by("date")
        .agg(pl.col("fwd_ret").mean().alias("benchmark_return"))
        .sort("date")
    )
    active = primary.daily.join(benchmark, on="date", how="left").with_columns(
        (pl.col("net_return") - pl.col("benchmark_return")).alias("active_return")
    )
    active_values = active["active_return"].drop_nulls().to_numpy()
    active_std = float(active_values.std(ddof=1)) if active_values.size > 1 else 0.0
    active_mean = float(active_values.mean()) if active_values.size else 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    ic.write_parquet(out_dir / "daily_ic.parquet")
    quantiles.write_parquet(out_dir / "quantile_returns.parquet")
    grid.write_parquet(out_dir / "topk_grid.parquet")
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
        "ic": {
            **asdict(stats),
            "block_bootstrap_ci_95": [ci_low, ci_high],
        },
        "primary_portfolio": primary.summary,
        "primary_portfolio_config": asdict(primary_portfolio),
        "primary_cost_config": asdict(primary_costs),
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
        "factors": (
            {"path": str(factors_path.resolve()), "sha256": _sha256(factors_path)}
            if factors_path is not None
            else None
        ),
        "require_complete_horizon": require_complete_horizon,
        "return_spec": (
            panel.select(["return_spec", "entry_price_field", "exit_price_field"])
            .unique()
            .to_dicts()
            if {
                "return_spec",
                "entry_price_field",
                "exit_price_field",
            }
            <= set(panel.columns)
            else []
        ),
        "outputs": [
            "metrics.json",
            "daily_ic.parquet",
            "quantile_returns.parquet",
            "topk_grid.parquet",
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
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--factors", type=Path)
    parser.add_argument("--quantile-groups", type=int, default=10)
    parser.add_argument("--top-k-grid", default="20,50,100,150,200")
    parser.add_argument("--rebalance-grid", default="1,5")
    parser.add_argument("--smoothing-grid", default="1,3")
    parser.add_argument("--cost-bps-grid", default="0,15,30")
    parser.add_argument("--allow-incomplete-horizon", action="store_true")
    args = parser.parse_args()
    path = evaluate_scores(
        scores_path=args.scores,
        panel_path=args.panel,
        out_dir=args.out_dir,
        quantile_groups=args.quantile_groups,
        top_ks=_parse_ints(args.top_k_grid),
        rebalance_intervals=_parse_ints(args.rebalance_grid),
        smoothing_windows=_parse_ints(args.smoothing_grid),
        cost_bps=_parse_floats(args.cost_bps_grid),
        factors_path=args.factors,
        require_complete_horizon=not args.allow_incomplete_horizon,
    )
    print(path)


if __name__ == "__main__":
    main()
