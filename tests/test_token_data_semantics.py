from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from pylob.event_ordering import (
    CAUSAL_EXCHANGE_TIME_V2,
    LEGACY_LOCAL_TIME_V1,
)
from pylob.orderbook_builder_sh import OrderBookSH
from pylob.orderbook_builder_sz import OrderBookSZ
from pylob.pipeline.events import (
    event_stream_contract_matches,
    read_event_stream_contract,
    write_event_stream_contract,
)
from pylob.pipeline.standardize import standardize_order_frame, standardize_trade_frame

from quant_fm.manifest.build_manifest import Manifest, ShardEntry, build_manifest
from quant_fm.pretrain.train import validate_pretrain_split_contract
from quant_fm.scripts.audit_token_ordering import (
    audit_manifest_token_ordering,
    audit_token_shard_order,
)
from quant_fm.tokenizer.artifact_contract import (
    assert_token_contract_matches,
    read_token_contract,
    token_contract_path,
)
from quant_fm.tokenizer.tokenize_events import tokenize_path
from quant_fm.tokenizer.transforms import (
    FEATURE_TRANSFORM_CAUSAL_V2,
    FEATURE_TRANSFORM_LEGACY_V1,
    ew_vwap_mid,
)
from quant_fm.tokenizer.vocab import Vocab, default_vocab
from quant_fm.tokenizer.vocab_v2 import VocabV2, default_vocab_v2

if TYPE_CHECKING:
    from pathlib import Path


def _market_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    trade = standardize_trade_frame(
        pl.DataFrame(
            {
                "symbol": ["000001"] * 3,
                "int_time": [93_000_000, 93_000_000, 93_000_001],
                "local_time": [40, 10, 30],
                "serial": [1, 2, 3],
                "marker": ["tie-trade", "seq2", "next"],
            }
        )
    )
    order = standardize_order_frame(
        pl.DataFrame(
            {
                "symbol": ["000001"] * 2,
                "int_time": [93_000_000, 93_000_002],
                "local_time": [20, 5],
                "serial": [1, 4],
                "marker": ["tie-order", "last"],
            }
        )
    )
    return trade, order


@pytest.mark.parametrize("builder", [OrderBookSH, OrderBookSZ])
def test_sh_sz_builders_default_to_stable_exchange_order(builder) -> None:
    trade, order = _market_frames()

    causal = builder().prepare_market_data(trade, order, symbol="000001")
    legacy = builder().prepare_market_data(
        trade,
        order,
        symbol="000001",
        event_ordering_version=LEGACY_LOCAL_TIME_V1,
    )

    assert causal["marker"].tolist() == [
        "tie-trade",
        "tie-order",
        "seq2",
        "next",
        "last",
    ]
    assert legacy["marker"].tolist() == [
        "last",
        "seq2",
        "tie-order",
        "next",
        "tie-trade",
    ]


def test_ew_vwap_causal_initialization_is_prefix_invariant() -> None:
    price = np.array([0.0, 0.0, 10.0])
    qty = np.array([0.0, 0.0, 100.0])
    is_trade = np.array([False, False, True])
    elapsed = np.array([0, 1, 2], dtype=np.int64)

    causal = ew_vwap_mid(
        price,
        qty,
        is_trade,
        elapsed,
        transform_version=FEATURE_TRANSFORM_CAUSAL_V2,
    )
    causal_prefix = ew_vwap_mid(
        price[:2],
        qty[:2],
        is_trade[:2],
        elapsed[:2],
        transform_version=FEATURE_TRANSFORM_CAUSAL_V2,
    )
    legacy = ew_vwap_mid(
        price,
        qty,
        is_trade,
        elapsed,
        transform_version=FEATURE_TRANSFORM_LEGACY_V1,
    )

    assert np.isnan(causal[:2]).all()
    assert np.isnan(causal_prefix).all()
    assert causal[2] == pytest.approx(10.0)
    assert legacy.tolist() == pytest.approx([10.0, 10.0, 10.0])


@pytest.mark.parametrize(
    ("factory", "loader"),
    [(default_vocab, Vocab.load), (default_vocab_v2, VocabV2.load)],
)
def test_vocab_without_semantic_fields_remains_readable_as_legacy(
    tmp_path: Path,
    factory,
    loader,
) -> None:
    payload = json.loads(factory().to_json())
    payload.pop("event_ordering_version")
    payload.pop("feature_transform_version")
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = loader(path)

    assert loaded.event_ordering_version == LEGACY_LOCAL_TIME_V1
    assert loaded.feature_transform_version == FEATURE_TRANSFORM_LEGACY_V1
    assert loaded.data_semantics_explicit is False
    roundtrip = json.loads(loaded.to_json())
    assert "event_ordering_version" not in roundtrip
    assert "feature_transform_version" not in roundtrip


def _canonical_events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "security_id": ["000001", "000001"],
            "date": ["2025-01-02", "2025-01-02"],
            "int_time": [93_000_000, 93_000_001],
            "source_seqnum": [1, 2],
            "event_idx": [0, 1],
            "price": [10.0, 10.1],
            "qty": [100.0, 200.0],
            "evt_type": ["ADD", "EXEC"],
            "side": ["B", "S"],
            "session": ["CONT_AM", "CONT_AM"],
            "board": ["MAIN", "MAIN"],
            "order_type": ["LIMIT", "UNKNOWN"],
            "event_source": ["ORDER", "TRADE"],
        }
    )


def test_token_sidecar_and_manifest_bind_vocab_semantics(tmp_path: Path) -> None:
    source = tmp_path / "events.parquet"
    destination = tmp_path / "tokens" / "SZ" / "000001" / "2025-01-02.parquet"
    vocab_path = tmp_path / "vocab.json"
    _canonical_events().write_parquet(source)
    vocab = default_vocab(n_bins=4)
    vocab.save(vocab_path)

    tokenize_path(source, destination, vocab)

    contract = read_token_contract(destination)
    assert len(contract["vocab_sha256"]) == 64
    assert contract["event_ordering_version"] == CAUSAL_EXCHANGE_TIME_V2
    assert contract["feature_transform_version"] == FEATURE_TRANSFORM_CAUSAL_V2
    manifest = build_manifest(
        tmp_path / "tokens",
        train_end="2025-01-02",
        val_end="2025-01-03",
        vocab_path=str(vocab_path),
    )
    assert manifest.event_ordering_version == CAUSAL_EXCHANGE_TIME_V2
    assert manifest.feature_transform_version == FEATURE_TRANSFORM_CAUSAL_V2
    assert manifest.shards[0].data_contract_sha256

    token_contract_path(destination).unlink()
    with pytest.raises(ValueError, match="no explicit data-semantics sidecar"):
        assert_token_contract_matches(destination, vocab)
    with pytest.raises(ValueError, match="no explicit data-semantics sidecar"):
        build_manifest(
            tmp_path / "tokens",
            train_end="2025-01-02",
            val_end="2025-01-03",
            vocab_path=str(vocab_path),
        )


def test_token_sidecar_rejects_same_semantics_with_different_vocab(
    tmp_path: Path,
) -> None:
    source = tmp_path / "events.parquet"
    destination = tmp_path / "tokens.parquet"
    _canonical_events().write_parquet(source)
    original = default_vocab(n_bins=4)
    tokenize_path(source, destination, original)
    changed = default_vocab(n_bins=5)

    assert changed.event_ordering_version == original.event_ordering_version
    assert changed.feature_transform_version == original.feature_transform_version
    with pytest.raises(ValueError, match="vocab_sha256"):
        assert_token_contract_matches(destination, changed)


def test_clean_event_sidecar_infers_old_artifacts_as_legacy(tmp_path: Path) -> None:
    path = tmp_path / "events.parquet"
    path.touch()

    assert read_event_stream_contract(path)["inferred_legacy"] is True
    assert event_stream_contract_matches(path, version=LEGACY_LOCAL_TIME_V1)
    assert not event_stream_contract_matches(path, version=CAUSAL_EXCHANGE_TIME_V2)

    write_event_stream_contract(
        path,
        event_ordering_version=CAUSAL_EXCHANGE_TIME_V2,
    )
    assert event_stream_contract_matches(path, version=CAUSAL_EXCHANGE_TIME_V2)


def test_streaming_token_audit_counts_each_inversion_type(tmp_path: Path) -> None:
    path = tmp_path / "bad.parquet"
    pl.DataFrame(
        {
            "int_time": [100, 99, 99, 99],
            "source_seqnum": [1, 2, 1, 1],
            "event_idx": [0, 1, 2, 1],
        }
    ).write_parquet(path, row_group_size=2)

    report = audit_token_shard_order(path, batch_size=2)

    assert report["int_time_inversions"] == 1
    assert report["same_time_sequence_inversions"] == 1
    assert report["stable_tie_event_idx_inversions"] == 1
    assert report["first_bad_row"] == 1
    assert report["ordered"] is False
    assert report["provenance_ok"] is False
    assert report["passed"] is False


def test_token_audit_fails_sidecarless_v1_provenance_and_documents_blind_spot(
    tmp_path: Path,
) -> None:
    token_path = tmp_path / "tokens.parquet"
    pl.DataFrame({"int_time": [100, 100], "event_idx": [0, 1]}).write_parquet(
        token_path
    )
    manifest_path = tmp_path / "manifest.json"
    Manifest(
        shards=[
            ShardEntry(
                "SZ",
                "000001",
                "2025-01-02",
                str(token_path),
                2,
                "",
                "train",
            )
        ],
        event_ordering_version=LEGACY_LOCAL_TIME_V1,
        feature_transform_version=FEATURE_TRANSFORM_LEGACY_V1,
    ).save(manifest_path)

    report = audit_manifest_token_ordering(manifest_path, full=True)

    assert report["ordered"] is True
    assert report["provenance_ok"] is False
    assert report["passed"] is False
    assert report["provenance_failure_shards"] == 1
    assert "cannot be recovered" in report["shards"][0]["sequence_audit_limit"]


def test_token_audit_fails_manifest_sidecar_semantics_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "events.parquet"
    token_path = tmp_path / "tokens.parquet"
    _canonical_events().write_parquet(source)
    tokenize_path(source, token_path, default_vocab(n_bins=4))
    manifest_path = tmp_path / "manifest.json"
    Manifest(
        shards=[
            ShardEntry(
                "SZ",
                "000001",
                "2025-01-02",
                str(token_path),
                2,
                "",
                "train",
            )
        ],
        event_ordering_version=LEGACY_LOCAL_TIME_V1,
        feature_transform_version=FEATURE_TRANSFORM_LEGACY_V1,
    ).save(manifest_path)

    report = audit_manifest_token_ordering(manifest_path, full=True)

    assert report["ordered"] is True
    assert report["provenance_ok"] is False
    assert report["passed"] is False
    mismatches = report["shards"][0]["manifest_provenance_mismatches"]
    assert "event_ordering_version" in mismatches
    assert "feature_transform_version" in mismatches


def _split_manifest(**semantics) -> Manifest:
    return Manifest(
        shards=[
            ShardEntry("SZ", "000001", "2025-01-02", "a", 1, "", "train"),
            ShardEntry("SZ", "000001", "2025-01-03", "b", 1, "", "val"),
            ShardEntry("SZ", "000001", "2025-01-06", "c", 1, "", "test"),
        ],
        **semantics,
    )


@pytest.mark.parametrize("source", ["manifest", "vocab", "expected"])
def test_pretrain_split_contract_rejects_noncanonical_dates(source: str) -> None:
    manifest = _split_manifest()
    vocab = default_vocab()
    vocab.fit_dates = ("2025-01-02",)
    expected_dates = None
    if source == "manifest":
        manifest.shards[0].date = "2025-1-2"
    elif source == "vocab":
        vocab.fit_dates = ("2025-1-2",)
    else:
        expected_dates = {"train": {"2025-1-2"}}

    with pytest.raises(ValueError, match="canonical YYYY-MM-DD"):
        validate_pretrain_split_contract(
            manifest,
            vocab,
            require_validation=True,
            min_validation_dates=1,
            min_test_dates=1,
            expected_dates=expected_dates,
        )


def test_pretrain_rejects_missing_causal_manifest_semantics(tmp_path: Path) -> None:
    causal_vocab = default_vocab()
    causal_vocab.fit_dates = ("2025-01-02",)
    missing_contract = _split_manifest(
        event_ordering_version=None,
        feature_transform_version=None,
    )

    with pytest.raises(ValueError, match="missing required event_ordering_version"):
        validate_pretrain_split_contract(
            missing_contract,
            causal_vocab,
            require_validation=True,
            min_validation_dates=1,
            min_test_dates=1,
        )

    payload = json.loads(causal_vocab.to_json())
    payload.pop("event_ordering_version")
    payload.pop("feature_transform_version")
    old_vocab_path = tmp_path / "old_vocab.json"
    old_vocab_path.write_text(json.dumps(payload), encoding="utf-8")
    legacy_vocab = Vocab.load(old_vocab_path)
    legacy_vocab.fit_dates = ("2025-01-02",)
    result = validate_pretrain_split_contract(
        missing_contract,
        legacy_vocab,
        require_validation=True,
        min_validation_dates=1,
        min_test_dates=1,
    )
    assert result["event_ordering_version"] == LEGACY_LOCAL_TIME_V1
