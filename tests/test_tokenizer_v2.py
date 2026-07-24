from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from quant_fm.tokenizer.field_spec import FieldSpec
from quant_fm.tokenizer.fit_bins_v2 import fit_vocab_v2
from quant_fm.tokenizer.tokenize_events_v2 import (
    coverage_report_v2,
    tokenize_frame_v2,
)
from quant_fm.tokenizer.vocab import N_SPECIAL as V1_N_SPECIAL
from quant_fm.tokenizer.vocab import PAD_ID as V1_PAD_ID
from quant_fm.tokenizer.vocab_v2 import (
    BOS_ID,
    EOS_ID,
    N_SPECIAL,
    NA_ID,
    PAD_ID,
    SESSION_BREAK_ID,
    UNK_ID,
    BinnedFieldVocab,
    ContinuousNormalizer,
    VocabV2,
)


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2025-01-02"] * 5,
            "exchange": ["XSHG"] * 5,
            "board": ["MAIN"] * 5,
            "symbol": ["600000.SH"] * 5,
            "event_idx": np.arange(5, dtype=np.int64),
            "evt_type": ["ADD", "EXEC", "EXEC", "CANCEL", "EXEC"],
            "side": ["B", "S", None, "B", "unexpected"],
            "feature": [0.0, 1.0, np.nan, -1.0, 2.0],
            "raw_feature": [0.0, 1.0, np.nan, -1.0, 2.0],
        }
    )


def _specs() -> tuple[FieldSpec, ...]:
    return (
        FieldSpec("evt_type", "evt_type", "categorical", is_target=True),
        FieldSpec("side", "side", "categorical"),
        FieldSpec(
            "feature",
            "feature",
            "ordinal",
            n_bins=8,
            applicable_events=("EXEC",),
            is_target=True,
        ),
        FieldSpec("raw", "raw_feature", "continuous"),
    )


def test_v2_special_ids_do_not_change_v1_contract() -> None:
    assert (V1_PAD_ID, V1_N_SPECIAL) == (0, 1)
    assert (
        PAD_ID,
        UNK_ID,
        NA_ID,
        BOS_ID,
        EOS_ID,
        SESSION_BREAK_ID,
        N_SPECIAL,
    ) == (0, 1, 2, 3, 4, 5, 6)


def test_field_spec_rejects_ambiguous_declarations() -> None:
    with pytest.raises(ValueError, match="requires n_bins"):
        FieldSpec("x", "x", "ordinal")
    with pytest.raises(ValueError, match="cannot define n_bins"):
        FieldSpec("x", "x", "categorical", n_bins=4)
    with pytest.raises(ValueError, match="neither input nor target"):
        FieldSpec("x", "x", "continuous", is_input=False, is_target=False)


def test_binned_vocab_keeps_na_distinct_from_real_zero() -> None:
    vocab = BinnedFieldVocab(
        requested_n_bins=4,
        edges=(-0.5, 0.5),
        occupancy=(1, 1, 1),
        normalizer=ContinuousNormalizer(mean=0.0, std=1.0, count=3),
        min_value=-1.0,
        max_value=1.0,
        n_observed=3,
        n_missing=1,
    )
    encoded = vocab.encode(np.array([np.nan, 0.0, -1.0, 1.0]))
    assert encoded.tolist() == [NA_ID, N_SPECIAL + 1, N_SPECIAL, N_SPECIAL + 2]
    assert vocab.actual_n_bins == 3
    assert vocab.size == N_SPECIAL + 3
    assert vocab.missing_rate == pytest.approx(0.25)


def test_vocab_roundtrip_preserves_specs_occupancy_and_normalizer(tmp_path) -> None:
    source = tmp_path / "events.parquet"
    _frame().write_parquet(source)
    vocab = fit_vocab_v2(
        [source],
        field_specs=_specs(),
        max_samples_per_field=20,
        categorical_values={
            "evt_type": ("ADD", "CANCEL", "EXEC"),
            "side": ("B", "S", "N"),
        },
    )
    artifact = tmp_path / "vocab_v2.json"
    vocab.save(artifact)
    loaded = VocabV2.load(artifact)

    assert loaded.to_json() == vocab.to_json()
    assert loaded.field_specs == vocab.field_specs
    assert loaded.binned["feature"].occupancy == vocab.binned["feature"].occupancy
    assert loaded.binned["raw"].normalizer == vocab.binned["raw"].normalizer
    assert loaded.categorical_occupancy == vocab.categorical_occupancy
    assert loaded.categorical_unknown_counts == vocab.categorical_unknown_counts
    assert loaded.categorical_missing_counts == vocab.categorical_missing_counts
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["vocab_version"] == "2.0"
    assert payload["binned"]["feature"]["missing_rate"] == pytest.approx(1 / 3)


def test_vocab_exposes_token_parquet_training_contract(tmp_path) -> None:
    source = tmp_path / "events.parquet"
    _frame().write_parquet(source)
    vocab = fit_vocab_v2(
        [source],
        field_specs=_specs(),
        max_samples_per_field=20,
        categorical_values={
            "evt_type": ("ADD", "CANCEL", "EXEC"),
            "side": ("B", "S", "N"),
        },
    )

    assert tuple(vocab.field_sizes()) == (
        "tok_evt_type",
        "tok_side",
        "tok_feature_bin",
    )
    assert vocab.field_sizes() == vocab.token_field_sizes()
    assert tuple(vocab.logical_field_sizes()) == ("evt_type", "side", "feature")
    assert vocab.input_token_fields == (
        "tok_evt_type",
        "tok_side",
        "tok_feature_bin",
    )
    assert vocab.target_token_fields == ("tok_evt_type", "tok_feature_bin")
    assert vocab.input_value_fields == ("val_feature", "val_raw")
    assert vocab.target_value_fields == ("val_feature",)
    assert vocab.token_to_value_fields == {"tok_feature_bin": "val_feature"}
    assert vocab.train_entropy("feature") == pytest.approx(np.log(2.0))
    assert vocab.train_entropy("tok_feature_bin") == vocab.train_entropy("feature")
    assert vocab.train_entropy("side") == pytest.approx(
        -(0.5 * np.log(0.5) + 2 * 0.25 * np.log(0.25))
    )
    assert vocab.train_entropy("side", include_missing=True) > vocab.train_entropy(
        "side"
    )


def test_tokenize_v2_outputs_bin_and_scalar_channels(tmp_path) -> None:
    source = tmp_path / "events.parquet"
    frame = _frame()
    frame.write_parquet(source)
    vocab = fit_vocab_v2(
        [source],
        field_specs=_specs(),
        max_samples_per_field=20,
        categorical_values={
            "evt_type": ("ADD", "CANCEL", "EXEC"),
            "side": ("B", "S", "N"),
        },
    )
    tokens = tokenize_frame_v2(frame, vocab)

    assert {"tok_feature_bin", "val_feature", "val_raw"} <= set(tokens.columns)
    assert "tok_raw_bin" not in tokens.columns
    # ADD/CANCEL 对 feature 不适用，真实 0 也会变成 NA；EXEC 的数值 1 是普通 bin。
    assert tokens["tok_feature_bin"].to_list()[0] == NA_ID
    assert tokens["tok_feature_bin"].to_list()[1] >= N_SPECIAL
    # EXEC 上的 NaN 仍是显式 NA，标量通道只作为数值 0 占位。
    assert tokens["tok_feature_bin"].to_list()[2] == NA_ID
    assert tokens["val_feature"].to_list()[2] == 0.0
    # 类别缺失与词表外类别分别使用 NA/UNK。
    assert tokens["tok_side"].to_list()[2] == NA_ID
    assert tokens["tok_side"].to_list()[4] == UNK_ID

    report = coverage_report_v2(tokens, vocab)
    assert report.actual_n_bins["feature"] == vocab.binned["feature"].actual_n_bins
    assert report.missing_rate["feature"] == pytest.approx(3 / 5)
    assert report.unknown_rate["side"] == pytest.approx(1 / 5)


def test_vocab_v2_loader_rejects_v1_artifact(tmp_path) -> None:
    artifact = tmp_path / "v1.json"
    artifact.write_text('{"schema_version": "cn_l2_v1"}', encoding="utf-8")
    with pytest.raises(ValueError, match="vocab_version"):
        VocabV2.load(artifact)
