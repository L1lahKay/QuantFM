from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import torch

from quant_fm.downstream.train_ranker import (
    TemporalRegimeRanker,
    fit_ranker,
    predict,
)
from quant_fm.moe.config import RegimeMoEConfig, TemporalRegimeTrainingConfig
from quant_fm.moe.regime_features import (
    RegimeFeatureSpec,
    attach_regime_features,
    validate_regime_feature_frame,
)


def _ranker_features(
    dates: tuple[str, ...], *, regime_shift: float = 0.0
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "date": date,
                "symbol": f"{index + 1:06d}",
                "label": index / 5,
                "head_gain": max((index / 5 - 0.5) / 0.5, 0.0) ** 2,
                "aux_target": (index - 2.5) / 2.5,
                "target_return": (index - 2.5) / 100.0,
                "emb_0": float(index),
                "emb_1": float(index % 2),
                "market_vol": regime_shift + float(day),
                "stock_ofi": regime_shift + float(index - 2),
            }
            for day, date in enumerate(dates)
            for index in range(6)
        ]
    )


def _moe_config() -> RegimeMoEConfig:
    return RegimeMoEConfig(
        enabled=True,
        n_experts=2,
        top_k=1,
        expert_hidden=8,
        router_hidden=4,
        dropout=0.0,
        capacity_factor=2.0,
    )


def test_temporal_regime_config_loads_standard_yaml() -> None:
    config = TemporalRegimeTrainingConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "quant_fm/moe/config_regime_v1.yaml"
    )
    assert config.placement == "temporal_aggregator"
    assert config.moe.enabled is True
    assert len(config.feature_specs) == 9


def test_temporal_regime_ranker_fits_normalizer_on_train_only() -> None:
    specs = (RegimeFeatureSpec("market_vol"), RegimeFeatureSpec("stock_ofi"))
    train = _ranker_features(("2025-01-02", "2025-01-03"))
    validation = _ranker_features(("2025-01-06",), regime_shift=10_000.0)

    result = fit_ranker(
        train,
        val_features=validation,
        epochs=1,
        hidden=8,
        depth=1,
        dropout=0.0,
        use_attention=False,
        device="cpu",
        seed=7,
        regime_moe=_moe_config(),
        regime_feature_specs=specs,
    )

    assert isinstance(result.model, TemporalRegimeRanker)
    expected = torch.tensor(
        train.select([spec.name for spec in specs]).mean().row(0),
        dtype=torch.float32,
    )
    assert torch.allclose(result.model.normalizer.mean, expected)
    assert result.model.normalizer.fit_end == "2025-01-03"
    assert result.history[0]["train_moe_aux_loss"] > 0
    assert 0 <= result.history[0]["train_moe_entropy"] <= 1
    assert "train_moe_expert_0_fraction" in result.history[0]
    assert any(
        parameter.grad is not None
        for parameter in result.model.temporal_moe.router.parameters()
    )

    scores = predict(result.model, validation, device="cpu")
    assert scores.shape == (6, 3)
    assert scores.columns == ["date", "symbol", "score"]


def test_regime_feature_frame_enforces_asof_and_complete_join() -> None:
    specs = (
        RegimeFeatureSpec("market_vol"),
        RegimeFeatureSpec("stock_ofi", availability_lag=1),
    )
    regime = pl.DataFrame(
        {
            "date": ["2025-01-03"],
            "symbol": ["000001"],
            "asof_date": ["2025-01-02"],
            "market_vol": [1.0],
            "stock_ofi": [2.0],
        }
    )
    validated = validate_regime_feature_frame(regime, specs)
    assert validated.columns == ["date", "symbol", "market_vol", "stock_ofi"]

    features = pl.DataFrame(
        {"date": ["2025-01-03"], "symbol": ["000001"], "emb_0": [0.0]}
    )
    attached = attach_regime_features(features, regime, specs)
    assert attached["stock_ofi"].item() == 2.0

    leaking = regime.with_columns(pl.lit("2025-01-03").alias("asof_date"))
    with pytest.raises(ValueError, match="availability_lag=1"):
        validate_regime_feature_frame(leaking, specs)

    with pytest.raises(ValueError, match="missing ranker keys"):
        attach_regime_features(
            features.with_columns(pl.lit("000002").alias("symbol")),
            regime,
            specs,
        )
