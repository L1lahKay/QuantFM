from __future__ import annotations

import json
from hashlib import sha256
from typing import TYPE_CHECKING

import polars as pl
import pytest

from quant_fm.embedding.contract import (
    AFTER_CLOSE_AVAILABILITY,
    CAUSAL_CHUNKED_ENCODER,
    CAUSAL_OVERLAPPING_ENCODER,
    EMBEDDING_CONTRACT_VERSION,
    STOCK_DAY_GRANULARITY,
    EmbeddingContract,
    assert_embedding_contract_compatible,
    embedding_contract_path,
    load_compatible_embedding_contracts,
    load_embedding_contract,
    propagate_embedding_contract,
    validate_strict_causal_representation,
    write_embedding_contract,
)

if TYPE_CHECKING:
    from pathlib import Path


def _contract(
    *,
    checkpoint: str = "a" * 64,
    pooling: str = "mean",
    overlap: bool = False,
    event_ordering_version: str = "exchange_time_sequence_v2",
) -> EmbeddingContract:
    return EmbeddingContract(
        format_version=EMBEDDING_CONTRACT_VERSION,
        fm_checkpoint_sha256=checkpoint,
        vocab_sha256="b" * 64,
        schema_version="cn_l2_v1",
        book_state_timing="none",
        pooling_version="flat_v1",
        granularity=STOCK_DAY_GRANULARITY,
        context=2048,
        chunk_stride=512 if overlap else 2048,
        pooling=pooling,
        last_k=256,
        dtype="bf16",
        encoder_width=2,
        pooling_components=(pooling,),
        pooling_scalar_components=(),
        embedding_columns=("emb_0", "emb_1"),
        embedding_width=2,
        signal_availability=AFTER_CLOSE_AVAILABILITY,
        encoder_semantics=(
            CAUSAL_OVERLAPPING_ENCODER if overlap else CAUSAL_CHUNKED_ENCODER
        ),
        event_ordering_version=event_ordering_version,
        feature_transform_version="ew_vwap_causal_nan_v2",
    )


def _write_frame(path: Path, values: tuple[float, float]) -> None:
    pl.DataFrame(
        {
            "date": ["2025-01-02"],
            "symbol": ["000001"],
            "market": ["SZ"],
            "emb_0": [values[0]],
            "emb_1": [values[1]],
        }
    ).write_parquet(path)


def test_embedding_contract_round_trip_and_optional_legacy(tmp_path: Path) -> None:
    path = tmp_path / "embedding.parquet"
    _write_frame(path, (1.0, 2.0))

    assert load_embedding_contract(path, required=False) is None
    with pytest.raises(ValueError, match="representation contract is missing"):
        load_embedding_contract(path)

    expected = _contract()
    written = write_embedding_contract(path, expected)
    assert written == embedding_contract_path(path)
    sidecar = json.loads(written.read_text(encoding="utf-8"))
    assert sidecar["embedding_file_sha256"] == sha256(path.read_bytes()).hexdigest()
    assert load_embedding_contract(path, require_vocab=True) == expected


def test_embedding_contract_rejects_parquet_byte_tampering(tmp_path: Path) -> None:
    path = tmp_path / "embedding.parquet"
    _write_frame(path, (1.0, 2.0))
    write_embedding_contract(path, _contract())

    _write_frame(path, (9.0, 8.0))

    with pytest.raises(ValueError, match="parquet SHA-256 disagrees"):
        load_embedding_contract(path)


def test_unbound_v2_sidecar_requires_explicit_legacy_loading(tmp_path: Path) -> None:
    path = tmp_path / "embedding.parquet"
    _write_frame(path, (1.0, 2.0))
    embedding_contract_path(path).write_text(
        json.dumps(_contract().to_dict()),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not bound to parquet bytes"):
        load_embedding_contract(path)
    assert load_embedding_contract(path, required=False) == _contract()


@pytest.mark.parametrize(
    "actual",
    [
        _contract(pooling="last"),
        _contract(checkpoint="c" * 64),
    ],
    ids=["same-width-pooling", "same-width-checkpoint"],
)
def test_same_width_representation_changes_are_rejected(
    actual: EmbeddingContract,
) -> None:
    with pytest.raises(ValueError, match="embedding representation mismatch"):
        assert_embedding_contract_compatible(
            _contract(),
            actual,
            context="train vs score",
        )


def test_merge_rejects_inconsistent_part_contracts(tmp_path: Path) -> None:
    first = tmp_path / "part0.parquet"
    second = tmp_path / "part1.parquet"
    _write_frame(first, (1.0, 2.0))
    _write_frame(second, (3.0, 4.0))
    write_embedding_contract(first, _contract())
    write_embedding_contract(second, _contract(pooling="last"))

    with pytest.raises(ValueError, match="embedding representation mismatch"):
        load_compatible_embedding_contracts(
            [first, second],
            context="parts",
        )


def test_merge_propagates_verified_contract(tmp_path: Path) -> None:
    first = tmp_path / "part0.parquet"
    second = tmp_path / "part1.parquet"
    output = tmp_path / "all.parquet"
    _write_frame(first, (1.0, 2.0))
    _write_frame(second, (3.0, 4.0))
    write_embedding_contract(first, _contract())
    write_embedding_contract(second, _contract())
    pl.concat([pl.read_parquet(first), pl.read_parquet(second)]).write_parquet(output)

    actual = propagate_embedding_contract(
        [first, second],
        output,
        context="parts",
    )

    assert actual == _contract()
    assert load_embedding_contract(output) == _contract()


def test_strict_causal_helper_requires_new_token_semantics_and_overlap() -> None:
    validate_strict_causal_representation(_contract(overlap=True))
    with pytest.raises(ValueError, match="encoder_semantics"):
        validate_strict_causal_representation(_contract())
    with pytest.raises(ValueError, match="event_ordering_version"):
        validate_strict_causal_representation(
            _contract(overlap=True, event_ordering_version="local_time_v1")
        )


def test_old_contract_is_diagnostic_only_and_cannot_be_synthesized(
    tmp_path: Path,
) -> None:
    path = tmp_path / "embedding.parquet"
    _write_frame(path, (1.0, 2.0))
    sidecar = embedding_contract_path(path)
    sidecar.write_text('{"format_version":"stock_day_embedding_v1"}\n')

    with pytest.raises(ValueError, match="cannot be safely upgraded"):
        load_embedding_contract(path)
