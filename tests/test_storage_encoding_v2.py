from __future__ import annotations

import copy

import numpy as np
import polars as pl
import pytest

from quant_fm.tokenizer.artifact_contract import (
    read_token_contract,
    write_token_contract,
)
from quant_fm.tokenizer.field_spec import FieldSpec
from quant_fm.tokenizer.storage_encoding_v2 import (
    Q16_MAX,
    StorageEncodingMetadataV2,
    assert_storage_metadata_matches_vocab_v2,
    build_storage_metadata_v2,
    dequantize_frame_v2,
    quantize_frame_v2,
)
from quant_fm.tokenizer.vocab_v2 import (
    BinnedFieldVocab,
    ContinuousNormalizer,
    VocabV2,
)


def _vocab(*, categories: int = 3, clip: float = 5.0) -> VocabV2:
    specs = (
        FieldSpec("kind", "kind", "categorical"),
        FieldSpec("feature", "feature", "continuous"),
    )
    category_values = tuple(f"c{index}" for index in range(categories))
    return VocabV2(
        field_specs=specs,
        categorical={"kind": category_values},
        categorical_occupancy={"kind": tuple(0 for _ in category_values)},
        categorical_unknown_counts={"kind": 0},
        categorical_missing_counts={"kind": 0},
        binned={
            "feature": BinnedFieldVocab(
                requested_n_bins=1,
                occupancy=(0,),
                normalizer=ContinuousNormalizer(clip=clip),
            )
        },
    )


def _frame(values: list[float] | None = None) -> pl.DataFrame:
    scalars = values or [-5.0, -1.25, 0.0, 1.25, 5.0]
    return pl.DataFrame(
        {
            "date": ["2025-01-02"] * len(scalars),
            "tok_kind": np.arange(len(scalars), dtype=np.int64),
            "val_feature": np.asarray(scalars, dtype=np.float32),
        }
    )


def test_q16_round_trip_is_bounded_and_preserves_non_payload_columns() -> None:
    vocab = _vocab(categories=3, clip=5.0)
    source = _frame()

    encoded, metadata = quantize_frame_v2(source, vocab)
    decoded = dequantize_frame_v2(encoded, metadata, vocab=vocab)

    assert encoded.schema["tok_kind"] == pl.UInt8
    assert encoded.schema["val_feature"] == pl.Int16
    assert decoded.schema["val_feature"] == pl.Float32
    assert decoded["date"].to_list() == source["date"].to_list()
    assert decoded["tok_kind"].to_list() == source["tok_kind"].to_list()
    error = np.abs(decoded["val_feature"].to_numpy() - source["val_feature"].to_numpy())
    assert float(error.max()) <= 5.0 / (2 * Q16_MAX) + 2e-7
    assert encoded["val_feature"].to_list()[0] == -Q16_MAX
    assert encoded["val_feature"].to_list()[-1] == Q16_MAX


def test_token_width_uses_complete_vocab_not_observed_shard_maximum() -> None:
    uint8_metadata = build_storage_metadata_v2(_vocab(categories=250))
    uint16_vocab = _vocab(categories=251)
    uint16_metadata = build_storage_metadata_v2(uint16_vocab)

    # Six special ids + 250 categories has largest legal id 255.
    assert uint8_metadata.token_fields[0].vocab_size == 256
    assert uint8_metadata.token_fields[0].storage_dtype == "uint8"
    # One more category forces every shard to UInt16, even if this frame has small ids.
    assert uint16_metadata.token_fields[0].vocab_size == 257
    assert uint16_metadata.token_fields[0].storage_dtype == "uint16"
    encoded, _ = quantize_frame_v2(_frame(), uint16_vocab)
    assert encoded.schema["tok_kind"] == pl.UInt16


@pytest.mark.parametrize("bad_id", [-1, 9])
def test_quantization_rejects_token_ids_outside_vocab_before_unsigned_cast(
    bad_id: int,
) -> None:
    vocab = _vocab(categories=3)
    source = _frame().with_columns(
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(pl.lit(bad_id))
        .otherwise(pl.col("tok_kind"))
        .alias("tok_kind")
    )

    with pytest.raises(ValueError, match="out of vocab range"):
        quantize_frame_v2(source, vocab)


def test_quantization_rejects_nonfinite_and_out_of_clip_scalars() -> None:
    vocab = _vocab()
    with pytest.raises(ValueError, match="non-finite"):
        quantize_frame_v2(_frame([np.nan]), vocab)
    with pytest.raises(ValueError, match="exceeds frozen normalizer clip"):
        quantize_frame_v2(_frame([5.1]), vocab)


def test_metadata_round_trip_is_deterministic_and_detects_tampering() -> None:
    vocab = _vocab()
    metadata = build_storage_metadata_v2(vocab)
    payload = metadata.to_dict()
    restored = StorageEncodingMetadataV2.from_dict(payload)

    assert restored == metadata
    assert restored.metadata_sha256 == metadata.metadata_sha256
    assert len(metadata.vocab_sha256) == 64
    assert len(metadata.metadata_sha256) == 64

    tampered = copy.deepcopy(payload)
    tampered["schema_version"] = "tampered_schema"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        StorageEncodingMetadataV2.from_dict(tampered)

    with_unknown = copy.deepcopy(payload)
    with_unknown["future_scale"] = 1.0
    with pytest.raises(ValueError, match="unknown fields"):
        StorageEncodingMetadataV2.from_dict(with_unknown)


def test_metadata_rejects_a_different_vocab_even_with_same_columns() -> None:
    metadata = build_storage_metadata_v2(_vocab(clip=5.0))

    with pytest.raises(ValueError, match="does not match V2 vocab"):
        assert_storage_metadata_matches_vocab_v2(metadata, _vocab(clip=4.0))


def test_token_contract_rejects_storage_metadata_for_another_vocab(tmp_path) -> None:
    token_path = tmp_path / "tokens.parquet"
    vocab = _vocab(clip=5.0)
    other_metadata = build_storage_metadata_v2(_vocab(clip=4.0))
    write_token_contract(
        token_path,
        vocab,
        storage_encoding=other_metadata.to_dict(),
    )

    with pytest.raises(ValueError, match="disagrees with V2 storage metadata"):
        read_token_contract(token_path)


def test_legacy_float_scalars_are_float32_compatible_without_metadata() -> None:
    vocab = _vocab()
    legacy = _frame().with_columns(pl.col("val_feature").cast(pl.Float64))

    decoded = dequantize_frame_v2(legacy, vocab=vocab)

    assert decoded.schema["val_feature"] == pl.Float32
    np.testing.assert_allclose(
        decoded["val_feature"].to_numpy(),
        legacy["val_feature"].to_numpy(),
        rtol=0.0,
        atol=0.0,
    )


def test_integer_scalar_without_metadata_is_rejected() -> None:
    naked_q16 = _frame().with_columns(pl.col("val_feature").cast(pl.Int16))

    with pytest.raises(ValueError, match="has no Q16 storage metadata"):
        dequantize_frame_v2(naked_q16, vocab=_vocab())


def test_decode_rejects_wrong_encoded_dtypes_and_q16_minimum() -> None:
    vocab = _vocab()
    encoded, metadata = quantize_frame_v2(_frame(), vocab)
    wrong_token = encoded.with_columns(pl.col("tok_kind").cast(pl.UInt16))
    with pytest.raises(TypeError, match="encoded token dtype mismatch"):
        dequantize_frame_v2(wrong_token, metadata, vocab=vocab)

    asymmetric_minimum = encoded.with_columns(
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(pl.lit(-32768, dtype=pl.Int16))
        .otherwise(pl.col("val_feature"))
        .alias("val_feature")
    )
    with pytest.raises(ValueError, match="exceeds symmetric Q16 range"):
        dequantize_frame_v2(asymmetric_minimum, metadata, vocab=vocab)
