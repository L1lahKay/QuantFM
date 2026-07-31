from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
import pytest
import torch
import yaml

from quant_fm.downstream.representation import (
    STRICT_BOOK_STATE_TIMING,
    STRICT_CHUNK_STRIDE,
    STRICT_CONTEXT,
    STRICT_SCHEMA_VERSION,
)
from quant_fm.embedding.contract import (
    AFTER_CLOSE_AVAILABILITY,
    CAUSAL_OVERLAPPING_ENCODER,
    EMBEDDING_CONTRACT_VERSION,
    STOCK_DAY_GRANULARITY,
    STRICT_EVENT_ORDERING_VERSION,
    STRICT_FEATURE_TRANSFORM_VERSION,
    EmbeddingContract,
    write_embedding_contract,
)
from quant_fm.embedding.pooling_spec import DEFAULT_V2_MULTI_SCALE_OUTPUTS
from quant_fm.manifest.build_manifest import Manifest, ShardEntry
from quant_fm.manifest.validation import sha256_file
from quant_fm.monitoring.acceptance import compare_pretrain_evaluations
from quant_fm.pretrain.data_contract import (
    build_pretrain_data_contract,
    write_checkpoint_contract,
)
from quant_fm.scripts.validate_pretrain_lineage import validate_pretrain_lineage
from quant_fm.tokenizer.artifact_contract import stable_vocab_sha256
from quant_fm.tokenizer.vocab import default_vocab
from quant_fm.tokenizer.vocab_v2 import default_vocab_v2

if TYPE_CHECKING:
    from pathlib import Path

    from quant_fm.tokenizer.vocab_v2 import VocabV2


@dataclass(slots=True)
class LineageFixture:
    """Paths and contracts for one internally consistent strict lineage."""

    acceptance: Path
    checkpoint: Path
    config: Path
    manifest: Manifest
    manifest_path: Path
    vocab: VocabV2
    vocab_path: Path
    train_embeddings: Path
    oos_embeddings: Path
    embedding_contract: EmbeddingContract


def _write_evaluation(
    path: Path,
    *,
    loss: float,
    checkpoint: Path | None = None,
    config: Path | None = None,
) -> None:
    payload: dict[str, object] = {
        "split": "val",
        "validation_plan_source_fingerprint": "frozen-validation-plan",
        "validation_windows": 8,
        "total_normalized_ce": loss,
        "per_field_ce": {"tok_evt_type": loss / 2},
    }
    if checkpoint is not None:
        payload["checkpoint"] = str(checkpoint.resolve())
    if config is not None:
        payload["config"] = str(config.resolve())
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _strict_embedding_contract(
    *,
    checkpoint_sha256: str,
    vocab_sha256: str,
) -> EmbeddingContract:
    encoder_width = 1
    width = len(DEFAULT_V2_MULTI_SCALE_OUTPUTS) * encoder_width
    return EmbeddingContract(
        format_version=EMBEDDING_CONTRACT_VERSION,
        fm_checkpoint_sha256=checkpoint_sha256,
        vocab_sha256=vocab_sha256,
        schema_version=STRICT_SCHEMA_VERSION,
        book_state_timing=STRICT_BOOK_STATE_TIMING,
        pooling_version="hierarchical_selected_v2",
        granularity=STOCK_DAY_GRANULARITY,
        context=STRICT_CONTEXT,
        chunk_stride=STRICT_CHUNK_STRIDE,
        pooling="multi_scale",
        last_k=256,
        dtype="bf16",
        encoder_width=encoder_width,
        pooling_components=DEFAULT_V2_MULTI_SCALE_OUTPUTS,
        pooling_scalar_components=(),
        embedding_columns=tuple(f"emb_{index}" for index in range(width)),
        embedding_width=width,
        signal_availability=AFTER_CLOSE_AVAILABILITY,
        encoder_semantics=CAUSAL_OVERLAPPING_ENCODER,
        event_ordering_version=STRICT_EVENT_ORDERING_VERSION,
        feature_transform_version=STRICT_FEATURE_TRANSFORM_VERSION,
    )


def _write_embeddings(
    path: Path,
    *,
    dates: list[str],
    contract: EmbeddingContract,
) -> None:
    payload: dict[str, list[object]] = {
        "date": dates,
        "symbol": [f"{index + 1:06d}" for index in range(len(dates))],
    }
    for column in contract.embedding_columns:
        payload[column] = [float(index) for index in range(len(dates))]
    pl.DataFrame(payload).write_parquet(path)
    write_embedding_contract(path, contract)


def _lineage_fixture(tmp_path: Path) -> LineageFixture:
    tmp_path.mkdir(parents=True, exist_ok=True)
    vocab = default_vocab_v2()
    vocab.fit_dates = ("2025-01-02", "2025-01-03")
    vocab_path = tmp_path / "vocab.json"
    vocab.save(vocab_path)

    manifest = Manifest(
        shards=[
            ShardEntry("SZ", "000001", "2025-01-02", "a", 1, "a", "train"),
            ShardEntry("SZ", "000001", "2025-01-03", "b", 1, "b", "train"),
            ShardEntry("SZ", "000001", "2025-01-06", "c", 1, "c", "val"),
            ShardEntry("SZ", "000001", "2025-01-07", "d", 1, "d", "test"),
        ],
        train_end="2025-01-03",
        val_end="2025-01-06",
        purge_days=0,
        embargo_days=0,
        vocab_path=str(vocab_path.resolve()),
        vocab_sha256=stable_vocab_sha256(vocab),
        schema_version=vocab.schema_version,
        event_ordering_version=vocab.event_ordering_version,
        feature_transform_version=vocab.feature_transform_version,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "manifest": str(manifest_path.resolve()),
                    "vocab": str(vocab_path.resolve()),
                    "min_validation_dates": 1,
                    "min_test_dates": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    data_contract = build_pretrain_data_contract(
        manifest_path=manifest_path,
        manifest=manifest,
        vocab_path=vocab_path,
        vocab=vocab,
    )
    model_contract = {
        "vocab_version": "2.0",
        "vocab_sha256": stable_vocab_sha256(vocab),
        "schema_version": vocab.schema_version,
        "event_ordering_version": vocab.event_ordering_version,
        "feature_transform_version": vocab.feature_transform_version,
        "book_state_timing": STRICT_BOOK_STATE_TIMING,
        "context_horizon": STRICT_CONTEXT,
        "pooling_version": "hierarchical_selected_v2",
        "pooling_method": "multi_scale",
        "pooling_outputs": [*DEFAULT_V2_MULTI_SCALE_OUTPUTS],
        "pooling_stride": STRICT_CHUNK_STRIDE,
        "field_specs": [spec.to_dict() for spec in vocab.field_specs],
    }
    checkpoint = tmp_path / "accepted.pt"
    torch.save(
        {
            "fm_artifact_version": "2.0",
            "config": model_contract,
            "pretrain_data_contract": data_contract,
        },
        checkpoint,
    )
    write_checkpoint_contract(
        checkpoint,
        config=model_contract,
        pretrain_data_contract=data_contract,
    )

    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _write_evaluation(
        candidate,
        loss=1.0,
        checkpoint=checkpoint,
        config=config_path,
    )
    _write_evaluation(baseline, loss=1.0)
    acceptance = tmp_path / "acceptance.json"
    acceptance.write_text(
        json.dumps(compare_pretrain_evaluations(candidate, baseline), sort_keys=True),
        encoding="utf-8",
    )

    embedding_contract = _strict_embedding_contract(
        checkpoint_sha256=sha256_file(checkpoint),
        vocab_sha256=stable_vocab_sha256(vocab),
    )
    train_embeddings = tmp_path / "train.parquet"
    oos_embeddings = tmp_path / "oos.parquet"
    _write_embeddings(
        train_embeddings,
        dates=["2025-01-07", "2025-01-08"],
        contract=embedding_contract,
    )
    _write_embeddings(
        oos_embeddings,
        dates=["2025-01-09"],
        contract=embedding_contract,
    )
    return LineageFixture(
        acceptance=acceptance,
        checkpoint=checkpoint,
        config=config_path,
        manifest=manifest,
        manifest_path=manifest_path,
        vocab=vocab,
        vocab_path=vocab_path,
        train_embeddings=train_embeddings,
        oos_embeddings=oos_embeddings,
        embedding_contract=embedding_contract,
    )


def _validate(fixture: LineageFixture, *, expected: str | None = None):
    return validate_pretrain_lineage(
        acceptance_path=fixture.acceptance,
        train_embeddings=fixture.train_embeddings,
        oos_embeddings=fixture.oos_embeddings,
        expected_training_end=expected,
    )


def test_lineage_derives_training_end_and_accepts_matching_declaration(
    tmp_path: Path,
) -> None:
    fixture = _lineage_fixture(tmp_path)

    derived = _validate(fixture)
    declared = _validate(fixture, expected="2025-01-06")

    assert derived["status"] == "verified"
    assert derived["effective_training_end"] == "2025-01-06"
    assert derived["declared_training_end"] is None
    assert declared["declared_training_end"] == "2025-01-06"
    assert derived["training_embedding_end_date"] == "2025-01-08"
    assert derived["oos_start_date"] == "2025-01-09"
    assert derived["training_embeddings"]["path"] == str(
        fixture.train_embeddings.resolve()
    )
    assert derived["training_embeddings"]["sha256"] == sha256_file(
        fixture.train_embeddings
    )
    assert derived["training_embeddings"]["date_start"] == "2025-01-07"
    assert derived["training_embeddings"]["date_end"] == "2025-01-08"
    assert derived["oos_embeddings"]["path"] == str(fixture.oos_embeddings.resolve())
    assert derived["oos_embeddings"]["sha256"] == sha256_file(fixture.oos_embeddings)
    assert derived["oos_embeddings"]["date_start"] == "2025-01-09"
    assert derived["oos_embeddings"]["date_end"] == "2025-01-09"


def test_lineage_report_hashes_live_embedding_bytes(tmp_path: Path) -> None:
    fixture = _lineage_fixture(tmp_path)
    before = _validate(fixture)
    pl.read_parquet(fixture.train_embeddings).with_columns(
        (pl.col("emb_0") + 1.0).alias("emb_0")
    ).write_parquet(fixture.train_embeddings)

    after = _validate(fixture)

    assert (
        before["training_embeddings"]["sha256"]
        != after["training_embeddings"]["sha256"]
    )
    assert after["training_embeddings"]["sha256"] == sha256_file(
        fixture.train_embeddings
    )


@pytest.mark.parametrize("declared", ["2025-01-02", "2025-1-3", ""])
def test_lineage_rejects_wrong_or_noncanonical_training_end(
    tmp_path: Path,
    declared: str,
) -> None:
    fixture = _lineage_fixture(tmp_path)

    with pytest.raises(ValueError, match=r"training end|YYYY-MM-DD"):
        _validate(fixture, expected=declared)


def test_lineage_rejects_manifest_or_vocab_drift(tmp_path: Path) -> None:
    fixture = _lineage_fixture(tmp_path)
    fixture.manifest.embargo_days = 1
    fixture.manifest.save(fixture.manifest_path)

    with pytest.raises(ValueError, match="does not match current manifest/vocab"):
        _validate(fixture)

    fixture = _lineage_fixture(tmp_path / "vocab-drift")
    fixture.vocab.fit_dates = ("2025-01-02",)
    fixture.vocab.save(fixture.vocab_path)
    with pytest.raises(ValueError, match="manifest/vocab contract mismatch"):
        _validate(fixture)


def test_lineage_rejects_v1_vocab_masquerading_as_v2(tmp_path: Path) -> None:
    fixture = _lineage_fixture(tmp_path)
    disguised_v1 = default_vocab(n_bins=4)
    disguised_v1.schema_version = STRICT_SCHEMA_VERSION
    disguised_v1.fit_dates = ("2025-01-02", "2025-01-03")
    disguised_v1.save(fixture.vocab_path)

    with pytest.raises(TypeError, match="genuine VocabV2"):
        _validate(fixture)


def test_lineage_rejects_noncanonical_or_inconsistent_manifest_boundaries(
    tmp_path: Path,
) -> None:
    fixture = _lineage_fixture(tmp_path)
    fixture.manifest.train_end = "2025-1-3"
    fixture.manifest.save(fixture.manifest_path)

    with pytest.raises(ValueError, match="canonical YYYY-MM-DD"):
        _validate(fixture)

    fixture = _lineage_fixture(tmp_path / "wrong-split")
    fixture.manifest.shards[2].split = "train"
    fixture.manifest.save(fixture.manifest_path)
    with pytest.raises(ValueError, match="split disagrees"):
        _validate(fixture)


def test_lineage_rejects_checkpoint_bytes_or_payload_contract_tamper(
    tmp_path: Path,
) -> None:
    fixture = _lineage_fixture(tmp_path)
    with fixture.checkpoint.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        _validate(fixture)

    fixture = _lineage_fixture(tmp_path / "payload-tamper")
    payload = torch.load(fixture.checkpoint, map_location="cpu", weights_only=False)
    payload["config"] = {**payload["config"], "schema_version": "cn_l2_v1"}
    torch.save(payload, fixture.checkpoint)
    write_checkpoint_contract(
        fixture.checkpoint,
        config={
            "vocab_version": "2.0",
            "vocab_sha256": stable_vocab_sha256(fixture.vocab),
            "schema_version": fixture.vocab.schema_version,
            "event_ordering_version": fixture.vocab.event_ordering_version,
            "feature_transform_version": fixture.vocab.feature_transform_version,
        },
        pretrain_data_contract=payload["pretrain_data_contract"],
    )
    with pytest.raises(ValueError, match="strict V2 Top-K representation"):
        _validate(fixture)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vocab_version", "1.0"),
        ("book_state_timing", "pre_event"),
        ("context_horizon", 4096),
        ("pooling_version", "hierarchical_v1"),
        ("pooling_method", "mean"),
        ("pooling_outputs", ["mean_all"]),
        ("pooling_stride", 2048),
        ("field_specs", []),
    ],
)
def test_lineage_rejects_non_strict_checkpoint_representation(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    fixture = _lineage_fixture(tmp_path)
    payload = torch.load(fixture.checkpoint, map_location="cpu", weights_only=False)
    payload["config"] = {**payload["config"], field: value}
    torch.save(payload, fixture.checkpoint)
    write_checkpoint_contract(
        fixture.checkpoint,
        config=payload["config"],
        pretrain_data_contract=payload["pretrain_data_contract"],
    )

    with pytest.raises(ValueError, match="strict V2 Top-K representation"):
        _validate(fixture)


def test_lineage_rejects_checkpoint_without_v2_artifact_marker(
    tmp_path: Path,
) -> None:
    fixture = _lineage_fixture(tmp_path)
    payload = torch.load(fixture.checkpoint, map_location="cpu", weights_only=False)
    payload["fm_artifact_version"] = "1.0"
    torch.save(payload, fixture.checkpoint)
    write_checkpoint_contract(
        fixture.checkpoint,
        config=payload["config"],
        pretrain_data_contract=payload["pretrain_data_contract"],
    )

    with pytest.raises(ValueError, match=r"fm_artifact_version='2\.0'"):
        _validate(fixture)


def test_lineage_rejects_embedding_identity_schema_or_missing_sidecar(
    tmp_path: Path,
) -> None:
    fixture = _lineage_fixture(tmp_path)
    wrong_identity = EmbeddingContract.from_dict(
        {
            **fixture.embedding_contract.to_dict(),
            "fm_checkpoint_sha256": "0" * 64,
        },
        require_vocab=True,
    )
    write_embedding_contract(fixture.train_embeddings, wrong_identity)
    write_embedding_contract(fixture.oos_embeddings, wrong_identity)
    with pytest.raises(ValueError, match="accepted FM lineage"):
        _validate(fixture)

    fixture = _lineage_fixture(tmp_path / "schema-drift")
    pl.read_parquet(fixture.train_embeddings).drop("emb_3").write_parquet(
        fixture.train_embeddings
    )
    with pytest.raises(ValueError, match="embedding columns"):
        _validate(fixture)

    fixture = _lineage_fixture(tmp_path / "missing-sidecar")
    sidecar = fixture.oos_embeddings.with_name(
        f"{fixture.oos_embeddings.name}.contract.json"
    )
    sidecar.unlink()
    with pytest.raises(ValueError, match="representation contract is missing"):
        _validate(fixture)


def test_lineage_rejects_noncanonical_or_overlapping_embedding_dates(
    tmp_path: Path,
) -> None:
    fixture = _lineage_fixture(tmp_path)
    _write_embeddings(
        fixture.oos_embeddings,
        dates=["2025-1-9"],
        contract=fixture.embedding_contract,
    )
    with pytest.raises(ValueError, match="canonical YYYY-MM-DD"):
        _validate(fixture)

    fixture = _lineage_fixture(tmp_path / "overlap")
    _write_embeddings(
        fixture.oos_embeddings,
        dates=["2025-01-08"],
        contract=fixture.embedding_contract,
    )
    with pytest.raises(ValueError, match="embedding periods overlap"):
        _validate(fixture)

    fixture = _lineage_fixture(tmp_path / "fm-overlap")
    _write_embeddings(
        fixture.oos_embeddings,
        dates=["2025-01-03"],
        contract=fixture.embedding_contract,
    )
    _write_embeddings(
        fixture.train_embeddings,
        dates=["2025-01-02"],
        contract=fixture.embedding_contract,
    )
    with pytest.raises(ValueError, match="training horizon overlaps"):
        _validate(fixture)


def test_lineage_rejects_duplicate_or_blank_embedding_keys(tmp_path: Path) -> None:
    fixture = _lineage_fixture(tmp_path)
    frame = pl.read_parquet(fixture.train_embeddings)
    pl.concat([frame, frame.head(1)]).write_parquet(fixture.train_embeddings)
    with pytest.raises(ValueError, match=r"duplicate \(date, symbol\) keys"):
        _validate(fixture)

    fixture = _lineage_fixture(tmp_path / "blank")
    pl.read_parquet(fixture.oos_embeddings).with_columns(
        pl.lit(" ").alias("symbol")
    ).write_parquet(fixture.oos_embeddings)
    with pytest.raises(ValueError, match="blank date/symbol keys"):
        _validate(fixture)
