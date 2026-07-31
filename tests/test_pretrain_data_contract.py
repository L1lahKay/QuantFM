from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

from quant_fm.manifest.build_manifest import Manifest, ShardEntry
from quant_fm.pretrain.data_contract import (
    build_pretrain_data_contract,
    checkpoint_contract_path,
    load_checkpoint_contract,
)
from quant_fm.pretrain.model import OrderFlowFM
from quant_fm.pretrain.train import (
    _save_checkpoint,
    _validate_resume_metadata,
    load_checkpoint,
)
from quant_fm.tokenizer.artifact_contract import stable_vocab_sha256
from quant_fm.tokenizer.vocab import default_vocab


def _data_contract(tmp_path: Path):
    vocab = default_vocab(n_bins=4)
    vocab.fit_dates = ("2025-01-02", "2025-01-03")
    vocab_path = tmp_path / "vocab.json"
    vocab.save(vocab_path)
    manifest = Manifest(
        shards=[
            ShardEntry("SZ", "000001", "2025-01-02", "a", 1, "a", "train"),
            ShardEntry("SZ", "000001", "2025-01-03", "b", 1, "b", "train"),
            ShardEntry("SZ", "000001", "2025-01-06", "c", 1, "c", "val"),
        ],
        vocab_path=str(vocab_path),
        vocab_sha256=stable_vocab_sha256(vocab),
        schema_version=vocab.schema_version,
        event_ordering_version=vocab.event_ordering_version,
        feature_transform_version=vocab.feature_transform_version,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    contract = build_pretrain_data_contract(
        manifest_path=manifest_path,
        manifest=manifest,
        vocab_path=vocab_path,
        vocab=vocab,
    )
    return vocab, vocab_path, manifest, manifest_path, contract


def test_checkpoint_roundtrips_full_pretrain_data_contract(tmp_path: Path) -> None:
    vocab, vocab_path, _, _, contract = _data_contract(tmp_path)
    model = OrderFlowFM.from_vocab(
        vocab,
        d_model=16,
        n_layers=1,
        n_heads=4,
        max_seq_len=8,
    )
    model.cfg.vocab_sha256 = stable_vocab_sha256(vocab)
    model.cfg.schema_version = vocab.schema_version
    model.cfg.event_ordering_version = vocab.event_ordering_version
    model.cfg.feature_transform_version = vocab.feature_transform_version
    checkpoint = tmp_path / "model.pt"

    _save_checkpoint(
        model,
        model.cfg,
        checkpoint,
        pretrain_data_contract=contract,
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["pretrain_data_contract"] == contract
    assert contract["manifest_train_start"] == "2025-01-02"
    assert contract["manifest_train_end"] == "2025-01-03"
    assert contract["manifest_validation_start"] == "2025-01-06"
    assert contract["manifest_validation_end"] == "2025-01-06"
    assert contract["vocab_fit_start"] == "2025-01-02"
    assert contract["vocab_fit_end"] == "2025-01-03"
    assert contract["effective_training_end"] == "2025-01-06"
    sidecar = load_checkpoint_contract(checkpoint)
    assert sidecar is not None
    assert sidecar["pretrain_data_contract"] == contract
    assert checkpoint_contract_path(checkpoint).is_file()
    restored = load_checkpoint(
        checkpoint,
        torch.device("cpu"),
        vocab_path=vocab_path,
        expected_pretrain_data_contract=contract,
    )
    assert restored.cfg.vocab_sha256 == stable_vocab_sha256(vocab)


def test_checkpoint_rejects_current_manifest_contract_tamper(tmp_path: Path) -> None:
    vocab, vocab_path, manifest, manifest_path, contract = _data_contract(tmp_path)
    model = OrderFlowFM.from_vocab(
        vocab,
        d_model=16,
        n_layers=1,
        n_heads=4,
        max_seq_len=8,
    )
    model.cfg.vocab_sha256 = stable_vocab_sha256(vocab)
    model.cfg.schema_version = vocab.schema_version
    model.cfg.event_ordering_version = vocab.event_ordering_version
    model.cfg.feature_transform_version = vocab.feature_transform_version
    checkpoint = tmp_path / "model.pt"
    _save_checkpoint(
        model,
        model.cfg,
        checkpoint,
        pretrain_data_contract=contract,
    )
    manifest.embargo_days += 1
    manifest.save(manifest_path)
    changed_contract = build_pretrain_data_contract(
        manifest_path=manifest_path,
        manifest=manifest,
        vocab_path=vocab_path,
        vocab=vocab,
    )

    with pytest.raises(ValueError, match="does not match current manifest/vocab"):
        load_checkpoint(
            checkpoint,
            torch.device("cpu"),
            vocab_path=vocab_path,
            expected_pretrain_data_contract=changed_contract,
        )


def test_checkpoint_contract_rejects_live_checkpoint_hash_tamper(
    tmp_path: Path,
) -> None:
    vocab, _, _, _, contract = _data_contract(tmp_path)
    model = OrderFlowFM.from_vocab(
        vocab,
        d_model=16,
        n_layers=1,
        n_heads=4,
        max_seq_len=8,
    )
    model.cfg.vocab_sha256 = stable_vocab_sha256(vocab)
    model.cfg.schema_version = vocab.schema_version
    model.cfg.event_ordering_version = vocab.event_ordering_version
    model.cfg.feature_transform_version = vocab.feature_transform_version
    checkpoint = tmp_path / "model.pt"
    _save_checkpoint(
        model,
        model.cfg,
        checkpoint,
        pretrain_data_contract=contract,
    )
    with checkpoint.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        load_checkpoint_contract(checkpoint)


def test_checkpoint_contract_rejects_stale_precomputed_hash(tmp_path: Path) -> None:
    vocab, _, _, _, contract = _data_contract(tmp_path)
    model = OrderFlowFM.from_vocab(
        vocab,
        d_model=16,
        n_layers=1,
        n_heads=4,
        max_seq_len=8,
    )
    model.cfg.vocab_sha256 = stable_vocab_sha256(vocab)
    model.cfg.schema_version = vocab.schema_version
    model.cfg.event_ordering_version = vocab.event_ordering_version
    model.cfg.feature_transform_version = vocab.feature_transform_version
    checkpoint = tmp_path / "model.pt"
    _save_checkpoint(
        model,
        model.cfg,
        checkpoint,
        pretrain_data_contract=contract,
    )

    with pytest.raises(ValueError, match="stale or incorrect"):
        load_checkpoint_contract(
            checkpoint,
            expected_checkpoint_sha256="0" * 64,
        )


def test_explicit_v1_resume_rejects_vocab_identity_change(tmp_path: Path) -> None:
    vocab, _, _, _, contract = _data_contract(tmp_path)
    model = OrderFlowFM.from_vocab(
        vocab,
        d_model=16,
        n_layers=1,
        n_heads=4,
        max_seq_len=8,
    )
    model.cfg.vocab_sha256 = stable_vocab_sha256(vocab)
    model.cfg.schema_version = vocab.schema_version
    model.cfg.event_ordering_version = vocab.event_ordering_version
    model.cfg.feature_transform_version = vocab.feature_transform_version
    checkpoint = tmp_path / "model.pt"
    _save_checkpoint(
        model,
        model.cfg,
        checkpoint,
        pretrain_data_contract=contract,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    changed = replace(model.cfg, vocab_sha256="0" * 64)

    with pytest.raises(ValueError, match="vocab_sha256"):
        _validate_resume_metadata(
            payload,
            changed,
            target_specs=None,
            require_explicit_data_contract=True,
        )
