from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from quant_fm.downstream.evaluate import (
    ic_statistics,
    quantile_return_panel,
    rank_ic,
)
from quant_fm.downstream.portfolio_simulator import (
    CostConfig,
    PortfolioConfig,
    simulate_buffered_topk,
)
from quant_fm.downstream.risk_attribution import residualize_returns
from quant_fm.downstream.run_score_evaluation import evaluate_scores


def test_quantiles_and_ic_statistics() -> None:
    rows = [
        {"date": date, "symbol": f"{index:06d}", "score": float(index)}
        for date in ("2026-01-05", "2026-01-06")
        for index in range(20)
    ]
    scores = pl.DataFrame(rows)
    panel = scores.select("date", "symbol").with_columns(
        pl.col("symbol").cast(pl.Int64).cast(pl.Float64).alias("fwd_ret"),
        pl.lit(True).alias("eligible_at_signal"),
    )
    ic = rank_ic(scores, panel)
    stats = ic_statistics(ic)
    quantiles = quantile_return_panel(scores, panel, n_groups=5)
    means = quantiles.group_by("group").agg(pl.col("mean_return").mean()).sort("group")
    assert stats.n_periods == 2
    assert stats.mean_ic == 1.0
    assert means["mean_return"].to_list() == sorted(means["mean_return"].to_list())


def test_unfillable_top_name_is_not_replaced_with_lower_rank() -> None:
    scores = pl.DataFrame(
        {
            "date": ["2026-01-05"] * 3,
            "symbol": ["000001", "000002", "000003"],
            "score": [3.0, 2.0, 1.0],
        }
    )
    panel = pl.DataFrame(
        {
            "date": ["2026-01-05"] * 3,
            "symbol": ["000001", "000002", "000003"],
            "fwd_ret": [0.1, 0.02, 0.5],
            "eligible_at_signal": [True, True, True],
            "entry_fillable": [False, True, True],
            "exit_fillable": [True, True, True],
        }
    )
    result = simulate_buffered_topk(
        scores,
        panel,
        portfolio=PortfolioConfig(
            candidate_top_k=2,
            target_holdings=2,
            entry_rank=2,
            exit_rank=2,
            score_smoothing_days=1,
        ),
        costs=CostConfig(buy_bps=0, sell_bps=0),
    )
    assert result.holdings["symbol"].to_list() == ["000002"]
    assert result.daily["cash_weight"][0] == 0.5
    assert result.daily["gross_return"][0] == 0.01
    assert result.daily["failed_buys"][0] == 1


def test_rebalance_sell_uses_entry_fillability_not_future_exit() -> None:
    scores = pl.DataFrame(
        {
            "date": ["2026-01-05", "2026-01-05", "2026-01-06", "2026-01-06"],
            "symbol": ["000001", "000002", "000001", "000002"],
            "score": [2.0, 1.0, 1.0, 2.0],
        }
    )
    base_panel = scores.select(["date", "symbol"]).with_columns(
        pl.Series("fwd_ret", [0.01, 0.02, 0.10, 0.20]),
        pl.lit(True).alias("eligible_at_signal"),
        pl.lit(True).alias("entry_fillable"),
        pl.lit(True).alias("exit_fillable"),
    )
    changed_exit = base_panel.with_columns(
        pl.when((pl.col("date") == "2026-01-06") & (pl.col("symbol") == "000001"))
        .then(False)
        .otherwise(pl.col("exit_fillable"))
        .alias("exit_fillable")
    )
    config = PortfolioConfig(
        candidate_top_k=2,
        target_holdings=1,
        entry_rank=1,
        exit_rank=1,
        score_smoothing_days=1,
    )
    costs = CostConfig(buy_bps=0, sell_bps=0)

    expected = simulate_buffered_topk(scores, base_panel, portfolio=config, costs=costs)
    actual = simulate_buffered_topk(scores, changed_exit, portfolio=config, costs=costs)

    assert actual.daily.to_dicts() == expected.daily.to_dicts()
    assert actual.holdings.to_dicts() == expected.holdings.to_dicts()
    assert actual.trades.to_dicts() == expected.trades.to_dicts()
    sell = actual.trades.filter(
        (pl.col("signal_date") == "2026-01-06") & (pl.col("side") == "SELL")
    ).row(0, named=True)
    assert sell["symbol"] == "000001"
    assert sell["filled"]
    assert actual.daily.filter(pl.col("date") == "2026-01-06")["gross_return"][0] == 0.2


def test_newey_west_statistic_is_finite() -> None:
    values = np.sin(np.arange(30) / 5) * 0.02 + 0.03
    frame = pl.DataFrame({"date": [str(i) for i in range(30)], "ic": values})
    stats = ic_statistics(frame, hac_lags=5)
    assert np.isfinite(stats.newey_west_t)
    assert stats.positive_rate > 0.5


def test_ranker_early_stopping_restores_best_epoch() -> None:
    pytest.importorskip("torch")
    from quant_fm.downstream.train_ranker import fit_ranker

    def features(dates: list[str]) -> pl.DataFrame:
        rows = []
        for date in dates:
            for index in range(8):
                rows.append(
                    {
                        "date": date,
                        "symbol": f"{index:06d}",
                        "label": index / 8,
                        "emb_0": float(index),
                        "emb_1": float(index % 2),
                    }
                )
        return pl.DataFrame(rows)

    result = fit_ranker(
        features(["2025-01-01", "2025-01-02"]),
        val_features=features(["2025-01-03"]),
        epochs=5,
        patience=1,
        lr=0.0,
        hidden=8,
        depth=1,
        dropout=0.0,
        use_attention=False,
        device="cpu",
        seed=7,
    )
    assert result.stopped_early
    assert result.best_epoch == 0
    assert len(result.history) == 2


def test_cross_sectional_return_residualization_removes_factor() -> None:
    factors = pl.DataFrame(
        {
            "date": ["2026-01-05"] * 10,
            "symbol": [f"{index:06d}" for index in range(10)],
            "factor_size": [float(index) for index in range(10)],
        }
    )
    panel = factors.select(["date", "symbol"]).with_columns(
        (pl.Series([float(index) for index in range(10)]) * 0.02 + 0.01).alias(
            "fwd_ret"
        )
    )
    residual = residualize_returns(panel, factors)
    assert residual.height == 10
    assert abs(float(residual["neutralized_ret"].mean())) < 1e-12
    assert float(residual["neutralized_ret"].abs().max()) < 1e-12


def test_score_evaluation_rejects_missing_horizon_and_writes_report(tmp_path) -> None:
    scores = pl.DataFrame(
        [
            {"date": date, "symbol": f"{index:06d}", "score": float(index)}
            for date in ("2026-01-05", "2026-01-06")
            for index in range(20)
        ]
    )
    panel = scores.select(["date", "symbol"]).with_columns(
        (pl.col("symbol").cast(pl.Int64) / 1000).alias("fwd_ret"),
        pl.lit(True).alias("eligible_at_signal"),
        pl.lit(True).alias("entry_fillable"),
        pl.lit(True).alias("exit_fillable"),
    )
    scores_path = tmp_path / "scores.parquet"
    panel_path = tmp_path / "panel.parquet"
    scores.write_parquet(scores_path)
    panel.filter(pl.col("date") == "2026-01-05").write_parquet(panel_path)
    with pytest.raises(ValueError, match="incomplete evaluation horizon"):
        evaluate_scores(
            scores_path=scores_path,
            panel_path=panel_path,
            out_dir=tmp_path / "bad",
        )

    panel.write_parquet(panel_path)
    metrics = evaluate_scores(
        scores_path=scores_path,
        panel_path=panel_path,
        out_dir=tmp_path / "good",
        top_ks=[5],
        rebalance_intervals=[1],
        smoothing_windows=[1],
        cost_bps=[0.0],
    )
    assert metrics.exists()
    assert (tmp_path / "good" / "topk_grid.parquet").exists()


def test_strict_evaluation_checks_label_coverage_and_writes_no_nan(tmp_path) -> None:
    scores = pl.DataFrame(
        {
            "date": ["2026-01-05"] * 5,
            "symbol": [f"{index:06d}" for index in range(5)],
            "score": [float(index) for index in range(5)],
        }
    )
    panel = scores.select(["date", "symbol"]).with_columns(
        pl.Series("fwd_ret", [0.0, 0.01, 0.02, 0.03, None]),
        pl.lit(True).alias("eligible_at_signal"),
        pl.lit(True).alias("entry_fillable"),
        pl.lit(True).alias("exit_fillable"),
    )
    scores_path = tmp_path / "scores.parquet"
    panel_path = tmp_path / "panel.parquet"
    scores.write_parquet(scores_path)
    panel.write_parquet(panel_path)

    with pytest.raises(ValueError, match="finite fwd_ret coverage"):
        evaluate_scores(
            scores_path=scores_path,
            panel_path=panel_path,
            out_dir=tmp_path / "strict",
            top_ks=[3],
            rebalance_intervals=[1],
            smoothing_windows=[1],
            cost_bps=[0.0],
        )

    metrics_path = evaluate_scores(
        scores_path=scores_path,
        panel_path=panel_path,
        out_dir=tmp_path / "incomplete",
        top_ks=[3],
        rebalance_intervals=[1],
        smoothing_windows=[1],
        cost_bps=[0.0],
        require_complete_horizon=False,
    )
    raw = metrics_path.read_text(encoding="utf-8")
    metrics = json.loads(raw)
    assert "NaN" not in raw
    assert metrics["join_coverage"] == 1.0
    assert metrics["label_coverage"] == 0.8
    assert metrics["valid_label_rows"] == 4
    assert metrics["ic"]["n_periods"] == 1
    assert metrics["ic"]["icir"] is None


def test_strict_evaluation_requires_three_names_and_nonzero_ic_periods(
    tmp_path,
) -> None:
    scores = pl.DataFrame(
        {
            "date": ["2026-01-05", "2026-01-05"],
            "symbol": ["000001", "000002"],
            "score": [1.0, 2.0],
        }
    )
    panel = scores.select(["date", "symbol"]).with_columns(
        pl.Series("fwd_ret", [0.01, 0.02]),
        pl.lit(True).alias("eligible_at_signal"),
        pl.lit(True).alias("entry_fillable"),
        pl.lit(True).alias("exit_fillable"),
    )
    scores_path = tmp_path / "small_scores.parquet"
    panel_path = tmp_path / "small_panel.parquet"
    scores.write_parquet(scores_path)
    panel.write_parquet(panel_path)

    with pytest.raises(ValueError, match="fewer than 3"):
        evaluate_scores(
            scores_path=scores_path,
            panel_path=panel_path,
            out_dir=tmp_path / "small_strict",
        )
    with pytest.raises(ValueError, match="zero valid IC periods"):
        evaluate_scores(
            scores_path=scores_path,
            panel_path=panel_path,
            out_dir=tmp_path / "small_incomplete",
            require_complete_horizon=False,
        )


@pytest.mark.parametrize("bad_score", [None, float("nan"), float("inf"), -float("inf")])
def test_score_evaluation_rejects_invalid_scores(tmp_path, bad_score) -> None:
    scores = pl.DataFrame(
        {
            "date": ["2026-01-05"] * 3,
            "symbol": ["000001", "000002", "000003"],
            "score": [1.0, 2.0, bad_score],
        }
    )
    panel = scores.select(["date", "symbol"]).with_columns(
        pl.Series("fwd_ret", [0.01, 0.02, 0.03]),
        pl.lit(True).alias("eligible_at_signal"),
        pl.lit(True).alias("entry_fillable"),
        pl.lit(True).alias("exit_fillable"),
    )
    scores_path = tmp_path / "invalid_scores.parquet"
    panel_path = tmp_path / "panel.parquet"
    scores.write_parquet(scores_path)
    panel.write_parquet(panel_path)

    with pytest.raises(ValueError, match="null or non-finite score"):
        evaluate_scores(
            scores_path=scores_path,
            panel_path=panel_path,
            out_dir=tmp_path / "invalid",
        )
