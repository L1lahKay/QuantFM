from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

torch = pytest.importorskip("torch")

from quant_fm.cross_asset.dataset import (  # noqa: E402
    align_interval_embeddings,
    build_cross_asset_panel,
    join_pit_industry,
    validate_pit_industry_assignments,
)
from quant_fm.cross_asset.model import (  # noqa: E402
    CrossAssetModelConfig,
    LinearCrossAssetModel,
)


def _time(hour: int, minute: int) -> datetime:
    return datetime(2026, 1, 5, hour, minute)


def _interval_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["A", "B", "A"],
            "prediction_time": [_time(9, 35), _time(9, 35), _time(9, 40)],
            "embedding": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        }
    )


def _industry_history() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["A", "A", "B"],
            "effective_time": [_time(9, 0), _time(9, 40), _time(9, 0)],
            "industry_id": [10, 99, 10],
        }
    )


def test_strict_pit_join_skips_exact_and_future_industry_rows() -> None:
    joined = join_pit_industry(_interval_frame(), _industry_history())

    assert joined["industry_id"].to_list() == [10, 10, 10]
    assert joined["industry_effective_time"].to_list() == [
        _time(9, 0),
        _time(9, 0),
        _time(9, 0),
    ]
    assert joined["industry_effective_time"].max() < joined["prediction_time"].max()


def test_pit_validator_rejects_equal_or_future_effective_time() -> None:
    leaked = pl.DataFrame(
        {
            "prediction_time": [_time(9, 35), _time(9, 40)],
            "industry_effective_time": [_time(9, 0), _time(9, 40)],
        }
    )

    with pytest.raises(ValueError, match="PIT industry leakage"):
        validate_pit_industry_assignments(leaked)


def test_effective_date_is_compared_to_intraday_prediction_timestamp() -> None:
    intervals = pl.DataFrame(
        {
            "symbol": ["A"],
            "prediction_time": [_time(9, 35)],
            "embedding": [[1.0]],
        }
    )
    history = pl.DataFrame(
        {
            "symbol": ["A"],
            "effective_time": [date(2026, 1, 5)],
            "industry_id": [10],
        }
    )

    joined = join_pit_industry(intervals, history)

    assert joined["industry_id"].to_list() == [10]
    assert joined["industry_effective_time"].to_list() == [date(2026, 1, 5)]


def test_alignment_builds_t_n_d_and_marks_missing_stock_interval() -> None:
    panel = build_cross_asset_panel(_interval_frame(), _industry_history())

    assert panel.embeddings.shape == (2, 2, 2)
    assert panel.symbols == ("A", "B")
    assert panel.active_mask.tolist() == [[True, True], [True, False]]
    assert torch.equal(panel.embeddings[1, 1], torch.zeros(2))
    assert panel.industry_id.tolist() == [[10, 10], [10, -1]]
    assert panel.max_industry_effective_time == _time(9, 0)


def test_alignment_requires_pit_audit_column() -> None:
    unsafe = _interval_frame().with_columns(pl.lit(10).alias("industry_id"))

    with pytest.raises(ValueError, match="industry_effective_time"):
        align_interval_embeddings(unsafe)


def test_model_uses_no_dense_stock_attention_and_is_n_independent() -> None:
    config = CrossAssetModelConfig(
        input_dim=2,
        hidden_dim=4,
        output_dim=3,
        dropout=0.0,
    )
    model = LinearCrossAssetModel(config)

    assert not any(
        isinstance(module, torch.nn.MultiheadAttention) for module in model.modules()
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    for n_stock in (2, 7):
        output = model(
            torch.randn(3, n_stock, 2),
            torch.zeros(n_stock, dtype=torch.long),
        )
        assert output.interval_embeddings.shape == (3, n_stock, 3)
        assert output.stock_summary.shape == (n_stock, 3)
        assert (
            sum(parameter.numel() for parameter in model.parameters())
            == parameter_count
        )


def test_model_is_causal_across_intervals_and_masks_missing_inputs() -> None:
    torch.manual_seed(7)
    model = LinearCrossAssetModel(
        CrossAssetModelConfig(
            input_dim=1,
            hidden_dim=4,
            output_dim=2,
            dropout=0.0,
        )
    ).eval()
    own = torch.tensor(
        [
            [[1.0], [3.0], [999.0]],
            [[2.0], [4.0], [999.0]],
        ]
    )
    active = torch.tensor([[True, True, False], [True, True, False]])
    changed_future = own.clone()
    changed_future[1, 1] = 100_000.0

    baseline = model(own, torch.tensor([0, 0, 0]), active_mask=active)
    modified = model(
        changed_future,
        torch.tensor([0, 0, 0]),
        active_mask=active,
    )

    assert torch.equal(
        baseline.interval_embeddings[0],
        modified.interval_embeddings[0],
    )
    assert torch.equal(
        baseline.interval_embeddings[:, 2],
        torch.zeros(2, 2),
    )
    assert torch.equal(baseline.stock_summary[2], torch.zeros(2))
    assert torch.allclose(baseline.context.market[0, 0], torch.tensor([2.0]))
