from __future__ import annotations

import json
import shutil
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path

from quant_fm.manifest.build_manifest import Manifest, ShardEntry
from quant_fm.pretrain.data_contract import (
    LEGACY_PRETRAIN_DATA_CONTRACT_VERSION,
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
from quant_fm.tokenizer.vocab_v2 import VocabV2
from tests.test_minio_artifact_transfer import _uploadable_v2
from tests.test_pretrain_v2_integration import _model_config as _v2_model_config


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


def _save_v2_checkpoint_with_contract(
    root: Path,
    checkpoint: Path,
) -> tuple[VocabV2, dict[str, object]]:
    vocab_path = root / "data" / "vocab_v2.json"
    manifest_path = root / "data" / "manifest.json"
    vocab = VocabV2.load(vocab_path)
    manifest = Manifest.load(manifest_path)
    contract = build_pretrain_data_contract(
        manifest_path=manifest_path,
        manifest=manifest,
        vocab_path=vocab_path,
        vocab=vocab,
    )
    model = OrderFlowFM(_v2_model_config(vocab, vocab_path))
    _save_checkpoint(
        model,
        model.cfg,
        checkpoint,
        pretrain_data_contract=contract,
    )
    return vocab, contract


def _rebase_v2_generation(source: Path, destination: Path) -> None:
    from quant_fm.scripts.audit_v2_artifacts import audit_v2_artifacts
    from quant_fm.scripts.download_from_minio import _rebase_downloaded_manifest

    shutil.copytree(source, destination)
    _rebase_downloaded_manifest(destination, "vocab_v2.json")
    audit = audit_v2_artifacts(destination, full_path_check=True)
    assert audit["contract_ready"] is True
    (destination / "artifact_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_rebased_v2_generation_strictly_resumes_by_core_identity(
    tmp_path: Path,
) -> None:
    from quant_fm.scripts.upload_to_minio import _validate_local_generation

    source = _uploadable_v2(tmp_path / "producer")
    checkpoint = tmp_path / "checkpoint.pt"
    _, source_contract = _save_v2_checkpoint_with_contract(source, checkpoint)
    committed = _validate_local_generation(
        source,
        run_live_audit=False,
    )
    assert source_contract["core_generation_id"] == committed.core_generation_id

    restored = tmp_path / "different-absolute-workdir"
    _rebase_v2_generation(source, restored)
    restored_vocab_path = restored / "data" / "vocab_v2.json"
    restored_vocab = VocabV2.load(restored_vocab_path)
    restored_manifest_path = restored / "data" / "manifest.json"
    restored_contract = build_pretrain_data_contract(
        manifest_path=restored_manifest_path,
        manifest=Manifest.load(restored_manifest_path),
        vocab_path=restored_vocab_path,
        vocab=restored_vocab,
    )

    assert source_contract["manifest_sha256"] != restored_contract["manifest_sha256"]
    assert (
        source_contract["core_generation_id"] == restored_contract["core_generation_id"]
    )
    assert (
        source_contract["manifest_semantic_sha256"]
        == restored_contract["manifest_semantic_sha256"]
    )
    assert source_contract["coverage_sha256"] == restored_contract["coverage_sha256"]
    loaded = load_checkpoint(
        checkpoint,
        torch.device("cpu"),
        vocab_path=restored_vocab_path,
        expected_pretrain_data_contract=restored_contract,
    )
    assert loaded.cfg.vocab_sha256 == stable_vocab_sha256(restored_vocab)


@pytest.mark.parametrize(
    "mutation",
    ["shard_content", "split", "boundary", "coverage", "vocab"],
)
def test_rebased_v2_resume_rejects_semantic_generation_changes(
    tmp_path: Path,
    mutation: str,
) -> None:
    source = _uploadable_v2(tmp_path / "producer")
    checkpoint = tmp_path / "checkpoint.pt"
    _save_v2_checkpoint_with_contract(source, checkpoint)
    restored = tmp_path / "restored"
    _rebase_v2_generation(source, restored)

    manifest_path = restored / "data" / "manifest.json"
    vocab_path = restored / "data" / "vocab_v2.json"
    manifest = Manifest.load(manifest_path)
    vocab = VocabV2.load(vocab_path)
    if mutation == "shard_content":
        manifest.shards[0].sha256 = "0" * 64
        manifest.save(manifest_path)
    elif mutation == "split":
        manifest.shards[0].split = "val"
        manifest.save(manifest_path)
    elif mutation == "boundary":
        manifest.train_end = manifest.shards[0].date
        manifest.val_end = manifest.shards[0].date
        manifest.save(manifest_path)
    elif mutation == "coverage":
        receipt = next((restored / "data" / "coverage").glob("*.json"))
        receipt.write_text(
            receipt.read_text(encoding="utf-8") + " ",
            encoding="utf-8",
        )
    else:
        vocab.fit_dates = ("2025-01-01",)
        vocab.save(vocab_path)
        manifest.vocab_sha256 = stable_vocab_sha256(vocab)
        manifest.save(manifest_path)

    changed_contract = build_pretrain_data_contract(
        manifest_path=manifest_path,
        manifest=Manifest.load(manifest_path),
        vocab_path=vocab_path,
        vocab=VocabV2.load(vocab_path),
    )
    with pytest.raises(ValueError, match=r"vocab|manifest/vocab generation"):
        load_checkpoint(
            checkpoint,
            torch.device("cpu"),
            vocab_path=vocab_path,
            expected_pretrain_data_contract=changed_contract,
        )


def test_strict_resume_rejects_manifest_shard_reordering(tmp_path: Path) -> None:
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
    checkpoint = tmp_path / "ordered.pt"
    _save_checkpoint(
        model,
        model.cfg,
        checkpoint,
        pretrain_data_contract=contract,
    )

    manifest.shards.reverse()
    manifest.save(manifest_path)
    reordered = build_pretrain_data_contract(
        manifest_path=manifest_path,
        manifest=manifest,
        vocab_path=vocab_path,
        vocab=vocab,
    )
    assert reordered["core_generation_id"] == contract["core_generation_id"]
    assert reordered["manifest_semantic_sha256"] != contract["manifest_semantic_sha256"]

    with pytest.raises(ValueError, match="manifest_semantic_sha256"):
        load_checkpoint(
            checkpoint,
            torch.device("cpu"),
            vocab_path=vocab_path,
            expected_pretrain_data_contract=reordered,
        )


def test_legacy_v2_contract_is_inference_compatible_but_strict_resume_rejected(
    tmp_path: Path,
) -> None:
    vocab, vocab_path, _, _, current_contract = _data_contract(tmp_path)
    legacy_contract = {
        key: value
        for key, value in current_contract.items()
        if key
        not in {
            "manifest_semantic_sha256",
            "core_generation_id",
            "coverage_sha256",
        }
    }
    legacy_contract["format_version"] = LEGACY_PRETRAIN_DATA_CONTRACT_VERSION
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
    checkpoint = tmp_path / "legacy.pt"
    _save_checkpoint(
        model,
        model.cfg,
        checkpoint,
        pretrain_data_contract=legacy_contract,
    )

    assert load_checkpoint(
        checkpoint,
        torch.device("cpu"),
        vocab_path=vocab_path,
    )
    with pytest.raises(ValueError, match=r"legacy.*strict resume.*refused"):
        load_checkpoint(
            checkpoint,
            torch.device("cpu"),
            vocab_path=vocab_path,
            expected_pretrain_data_contract=current_contract,
        )
