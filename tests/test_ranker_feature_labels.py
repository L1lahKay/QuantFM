from __future__ import annotations

import math

import polars as pl
import pytest

from quant_fm.downstream.make_features import (
    build_features,
    build_scoring_features,
    build_training_features,
)


def _embeddings(symbols: list[str], *, date: str = "2025-01-02") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date] * len(symbols),
            "symbol": symbols,
            "emb_0": [float(index) for index in range(len(symbols))],
        }
    )


def test_strict_panel_filters_only_on_signal_time_eligibility() -> None:
    symbols = ["1", "2", "3", "4", "5"]
    panel = pl.DataFrame(
        {
            "date": ["2025-01-02"] * 5,
            "symbol": symbols,
            "fwd_ret": [0.01, 0.02, 0.03, 0.04, 0.05],
            "eligible_at_signal": [True, True, False, None, True],
            # Future execution outcomes must never change the training universe.
            "entry_fillable": [False, True, True, True, True],
            "exit_fillable": [True, False, True, True, True],
            # eligible_at_signal is authoritative when both schemas are present.
            "is_st": [False, False, False, False, True],
        }
    )

    result = build_training_features(_embeddings(symbols), panel, min_names_per_day=1)

    assert result["symbol"].to_list() == ["000001", "000002", "000005"]


def test_legacy_panel_flags_remain_supported() -> None:
    symbols = ["1", "2", "3", "4"]
    panel = pl.DataFrame(
        {
            "date": ["2025-01-02"] * 4,
            "symbol": symbols,
            "fwd_ret": [0.01, 0.02, 0.03, 0.04],
            "is_st": [False, True, False, None],
            "is_halt": [False, False, False, False],
            "is_new": [False, False, False, False],
            "limit_locked": [False, False, True, False],
        }
    )

    result = build_training_features(_embeddings(symbols), panel, min_names_per_day=1)

    assert result["symbol"].to_list() == ["000001", "000004"]


def test_nonfinite_returns_are_removed_before_daily_targets() -> None:
    symbols = ["1", "2", "3", "4", "5"]
    panel = pl.DataFrame(
        {
            "date": ["2025-01-02"] * 5,
            "symbol": symbols,
            "fwd_ret": [0.1, None, float("nan"), float("inf"), -float("inf")],
        }
    )

    result = build_training_features(_embeddings(symbols), panel, min_names_per_day=1)

    assert result["symbol"].to_list() == ["000001"]
    assert result["target_return"].item() == pytest.approx(0.0)
    assert result["label"].item() == pytest.approx(0.0)
    assert result["aux_target"].item() == pytest.approx(0.0)
    assert result["head_gain"].item() == pytest.approx(0.0)
    for column in ("target_return", "label", "aux_target", "head_gain"):
        assert math.isfinite(result[column].item())


def test_all_invalid_returns_are_rejected() -> None:
    symbols = ["1", "2"]
    panel = pl.DataFrame(
        {
            "date": ["2025-01-02"] * 2,
            "symbol": symbols,
            "fwd_ret": [None, None],
        },
        schema={
            "date": pl.Utf8,
            "symbol": pl.Utf8,
            "fwd_ret": pl.Float64,
        },
    )

    with pytest.raises(ValueError, match="no rows with finite fwd_ret"):
        build_training_features(_embeddings(symbols), panel, min_names_per_day=1)


def test_daily_percentile_aux_target_and_head_gain() -> None:
    symbols = ["1", "2", "3", "4"]
    panel = pl.DataFrame(
        {
            "date": ["2025-01-02"] * 4,
            "symbol": symbols,
            "fwd_ret": [0.0, 1.0, 2.0, 100.0],
        }
    )

    result = build_training_features(_embeddings(symbols), panel, min_names_per_day=1)

    assert result["label"].to_list() == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    assert result["target_return"].to_list() == pytest.approx(
        [-25.75, -24.75, -23.75, 74.25]
    )
    assert result["aux_target"].to_list() == pytest.approx(
        [-1.5 / 1.4826, -0.5 / 1.4826, 0.5 / 1.4826, 3.0]
    )
    assert result["head_gain"].to_list() == pytest.approx([0.0, 0.0, 1 / 9, 1.0])


def test_targets_are_computed_independently_per_day() -> None:
    embeddings = pl.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"],
            "symbol": ["1", "2", "1", "2"],
            "emb_0": [0.0, 1.0, 2.0, 3.0],
        }
    )
    panel = embeddings.select("date", "symbol").with_columns(
        pl.Series("fwd_ret", [0.0, 2.0, 100.0, 200.0])
    )

    result = build_training_features(embeddings, panel, min_names_per_day=1)

    assert result["label"].to_list() == pytest.approx([0.0, 1.0, 0.0, 1.0])
    assert result["target_return"].to_list() == pytest.approx([-1.0, 1.0, -50.0, 50.0])
    assert result["aux_target"].to_list() == pytest.approx(
        [-1 / 1.4826, 1 / 1.4826, -1 / 1.4826, 1 / 1.4826]
    )


def test_aux_target_uses_daily_std_when_mad_is_zero() -> None:
    symbols = ["1", "2", "3", "4"]
    panel = pl.DataFrame(
        {
            "date": ["2025-01-02"] * 4,
            "symbol": symbols,
            "fwd_ret": [0.0, 0.0, 0.0, 10.0],
        }
    )

    result = build_training_features(_embeddings(symbols), panel, min_names_per_day=1)

    # Polars' daily std uses ddof=1: std([0, 0, 0, 10]) == 5.
    assert result["aux_target"].to_list() == pytest.approx([0.0, 0.0, 0.0, 2.0])


def test_training_universe_is_applied_per_signal_day() -> None:
    dates = ["2025-01-02"] * 3 + ["2025-01-03"] * 3
    symbols = ["1", "2", "3"] * 2
    embeddings = pl.DataFrame(
        {
            "date": dates,
            "symbol": symbols,
            "emb_0": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    panel = embeddings.select("date", "symbol").with_columns(
        pl.Series("fwd_ret", [0.0, 1.0, 2.0, 20.0, 10.0, 0.0])
    )
    universe = pl.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"],
            "symbol": ["1", "2", "2", "3"],
            "pit_liquidity_rank": [1, 2, 1, 2],
        }
    )

    result = build_training_features(
        embeddings, panel, universe=universe, min_names_per_day=1
    )
    legacy_result = build_features(
        embeddings, panel, universe=universe, min_names_per_day=1
    )

    expected_keys = [
        {"date": "2025-01-02", "symbol": "000001"},
        {"date": "2025-01-02", "symbol": "000002"},
        {"date": "2025-01-03", "symbol": "000002"},
        {"date": "2025-01-03", "symbol": "000003"},
    ]
    assert result.select("date", "symbol").to_dicts() == expected_keys
    assert legacy_result.equals(result)
    assert result["label"].to_list() == pytest.approx([0.0, 1.0, 1.0, 0.0])


def test_training_universe_must_cover_every_joined_signal_date() -> None:
    embeddings = pl.concat(
        [
            _embeddings(["1", "2"], date="2025-01-02"),
            _embeddings(["1", "2"], date="2025-01-03"),
        ]
    )
    panel = embeddings.select("date", "symbol").with_columns(
        pl.Series("fwd_ret", [0.0, 1.0, 0.0, 1.0])
    )
    incomplete_universe = pl.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02"],
            "symbol": ["1", "2"],
        }
    )

    with pytest.raises(ValueError, match="missing training signal dates"):
        build_training_features(
            embeddings,
            panel,
            universe=incomplete_universe,
            min_names_per_day=1,
        )


def test_training_inputs_reject_duplicate_normalised_keys() -> None:
    embeddings = _embeddings(["1", "2"])
    panel = pl.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02"],
            "symbol": ["1", "2"],
            "fwd_ret": [0.0, 1.0],
        }
    )
    factors = panel.select("date", "symbol").with_columns(
        pl.Series("factor_value", [0.0, 1.0])
    )
    universe = panel.select("date", "symbol")

    duplicate_inputs = {
        "embeddings": {
            "embeddings": pl.concat([embeddings, embeddings.head(1)]),
            "panel": panel,
        },
        "panel": {
            "embeddings": embeddings,
            "panel": pl.concat([panel, panel.head(1)]),
        },
        "factors": {
            "embeddings": embeddings,
            "panel": panel,
            "factors": pl.concat([factors, factors.head(1)]),
        },
        # These two spellings collide after symbol normalisation.
        "universe": {
            "embeddings": embeddings,
            "panel": panel,
            "universe": pl.concat(
                [
                    universe,
                    pl.DataFrame({"date": ["2025-01-02"], "symbol": ["000001"]}),
                ]
            ),
        },
    }

    for input_name, arguments in duplicate_inputs.items():
        with pytest.raises(
            ValueError, match=rf"{input_name} contains duplicate \(date, symbol\)"
        ):
            build_training_features(**arguments, min_names_per_day=1)


def test_scoring_inputs_reject_duplicate_keys() -> None:
    embeddings = _embeddings(["1", "2"])
    duplicate_universe = pl.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02"],
            "symbol": ["1", "000001"],
        }
    )

    with pytest.raises(ValueError, match="universe contains duplicate"):
        build_scoring_features(embeddings, universe=duplicate_universe)


def test_scoring_universe_must_cover_every_embedding_date() -> None:
    embeddings = pl.concat(
        [
            _embeddings(["1", "2"], date="2025-01-02"),
            _embeddings(["1", "2"], date="2025-01-03"),
        ]
    )
    universe = pl.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02"],
            "symbol": ["1", "2"],
        }
    )

    with pytest.raises(ValueError, match="missing scoring signal dates"):
        build_scoring_features(embeddings, universe=universe)


@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf")])
def test_training_features_must_be_finite(bad_value: float | None) -> None:
    symbols = ["1", "2"]
    embeddings = _embeddings(symbols).with_columns(
        pl.Series("emb_0", [0.0, bad_value], dtype=pl.Float64)
    )
    panel = pl.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02"],
            "symbol": symbols,
            "fwd_ret": [0.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match=r"training features.*emb_0"):
        build_training_features(embeddings, panel, min_names_per_day=1)


def test_training_rejects_missing_or_nonfinite_factor_values() -> None:
    symbols = ["1", "2"]
    embeddings = _embeddings(symbols)
    panel = pl.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-02"],
            "symbol": symbols,
            "fwd_ret": [0.0, 1.0],
        }
    )
    incomplete_factors = pl.DataFrame(
        {
            "date": ["2025-01-02"],
            "symbol": ["1"],
            "factor_value": [0.0],
        }
    )

    with pytest.raises(ValueError, match=r"training features.*factor_value"):
        build_training_features(
            embeddings,
            panel,
            factors=incomplete_factors,
            min_names_per_day=1,
        )


@pytest.mark.parametrize("bad_value", [None, float("nan"), -float("inf")])
def test_scoring_features_must_be_finite(bad_value: float | None) -> None:
    embeddings = _embeddings(["1", "2"]).with_columns(
        pl.Series("emb_0", [0.0, bad_value], dtype=pl.Float64)
    )

    with pytest.raises(ValueError, match=r"scoring features.*emb_0"):
        build_scoring_features(embeddings)


@pytest.mark.parametrize(
    "column",
    [
        "label",
        "fwd_ret",
        "xs_ret",
        "target_return",
        "aux_target",
        "head_gain",
        "target_future_alpha",
    ],
)
def test_scoring_rejects_all_target_columns(column: str) -> None:
    embeddings = _embeddings(["1", "2"]).with_columns(pl.lit(0.0).alias(column))

    with pytest.raises(ValueError, match=rf"forbidden future columns.*{column}"):
        build_scoring_features(embeddings)


def test_scoring_rejects_target_columns_in_factors() -> None:
    embeddings = _embeddings(["1", "2"])
    factors = embeddings.select("date", "symbol").with_columns(
        pl.lit(0.0).alias("factor_value"),
        pl.lit(1.0).alias("target_leak"),
    )

    with pytest.raises(ValueError, match=r"scoring factors.*target_leak"):
        build_scoring_features(embeddings, factors=factors)
