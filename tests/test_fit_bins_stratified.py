from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from quant_fm.tokenizer.field_spec import FieldSpec
from quant_fm.tokenizer.fit_bins_v2 import fit_vocab_v2


def _write_shard(
    path,
    *,
    date: str,
    value: float,
    n: int = 20,
    add_missing: bool = False,
) -> None:
    values = np.full(n, value, dtype=np.float64)
    if add_missing:
        values[-1] = np.nan
    pl.DataFrame(
        {
            "date": [date] * n,
            "exchange": ["XSHG"] * n,
            "board": ["MAIN"] * n,
            "evt_type": ["ADD"] * n,
            "symbol": ["600000.SH"] * n,
            "event_idx": np.arange(n, dtype=np.int64),
            "x": values,
        }
    ).write_parquet(path)


def _specs(n_bins: int = 32) -> tuple[FieldSpec, ...]:
    return (FieldSpec("x", "x", "ordinal", n_bins=n_bins),)


def test_repeated_quantiles_reduce_actual_bins_without_fake_linspace(tmp_path) -> None:
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    _write_shard(first, date="2025-01-02", value=0.0)
    _write_shard(second, date="2025-01-03", value=1.0)

    vocab = fit_vocab_v2(
        [first, second], field_specs=_specs(32), max_samples_per_field=40
    )
    field = vocab.binned["x"]
    assert field.requested_n_bins == 32
    assert field.actual_n_bins == 2
    assert field.edges == pytest.approx((0.5,))
    assert field.occupancy == (20, 20)
    assert vocab.size("x") == 6 + field.actual_n_bins


def test_sampling_is_path_order_invariant_and_reads_late_shards(tmp_path) -> None:
    paths = []
    for index, value in enumerate((0.0, 10.0, 1000.0)):
        path = tmp_path / f"shard-{index}.parquet"
        _write_shard(path, date=f"2025-01-0{index + 2}", value=value)
        paths.append(path)

    first = fit_vocab_v2(
        paths,
        field_specs=_specs(4),
        max_samples_per_field=6,
        seed=7,
    )
    reversed_order = fit_vocab_v2(
        list(reversed(paths)),
        field_specs=_specs(4),
        max_samples_per_field=6,
        seed=7,
    )

    assert first.to_json() == reversed_order.to_json()
    assert first.binned["x"].max_value == 1000.0
    assert first.binned["x"].n_observed == 60
    assert first.sampling["sample_counts"]["x"] == 6
    assert first.sampling["strata_counts"]["x"] == 3


def test_stratified_budget_must_cover_every_nonempty_stratum(tmp_path) -> None:
    paths = []
    for index in range(3):
        path = tmp_path / f"shard-{index}.parquet"
        _write_shard(path, date=f"2025-01-0{index + 2}", value=float(index))
        paths.append(path)

    with pytest.raises(ValueError, match="smaller than 3 non-empty strata"):
        fit_vocab_v2(
            paths,
            field_specs=_specs(4),
            max_samples_per_field=2,
        )


def test_occupancy_and_missing_rate_use_full_stream_not_reservoir(tmp_path) -> None:
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    _write_shard(first, date="2025-01-02", value=0.0, add_missing=True)
    _write_shard(second, date="2025-01-03", value=2.0, add_missing=True)

    vocab = fit_vocab_v2(
        [first, second], field_specs=_specs(4), max_samples_per_field=4
    )
    field = vocab.binned["x"]
    assert sum(field.occupancy) == 38
    assert field.n_observed == 38
    assert field.n_missing == 2
    assert field.missing_rate == pytest.approx(0.05)
    assert field.normalizer.count == 38
