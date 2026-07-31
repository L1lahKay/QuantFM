from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from quant_fm.manifest.build_manifest import Manifest, ShardEntry  # noqa: E402
from quant_fm.pretrain.dataset_v2 import (  # noqa: E402
    EventWindowDatasetV2,
    collate_windows_v2,
    field_layout_from_vocab,
)
from quant_fm.pretrain.heads import (  # noqa: E402
    next_event_loss_v2,
    target_specs_from_config,
)
from quant_fm.pretrain.model import OrderFlowFM, OrderFlowFMConfig  # noqa: E402
from quant_fm.pretrain.train import (  # noqa: E402
    _build_dataloaders,
    _save_checkpoint,
    _validate_resume_metadata,
    load_checkpoint,
)
from quant_fm.tokenizer.artifact_contract import stable_vocab_sha256  # noqa: E402
from quant_fm.tokenizer.field_spec import FieldSpec  # noqa: E402
from quant_fm.tokenizer.fit_bins_v2 import fit_vocab_v2  # noqa: E402
from quant_fm.tokenizer.tokenize_events_v2 import tokenize_frame_v2  # noqa: E402
from quant_fm.tokenizer.vocab_v2 import N_SPECIAL, NA_ID  # noqa: E402


def _artifacts(tmp_path):
    frame = pl.DataFrame(
        {
            "date": ["2025-01-02"] * 6,
            "exchange": ["XSHG"] * 6,
            "board": ["MAIN"] * 6,
            "event_idx": np.arange(6),
            "evt_type": ["ADD", "EXEC", "EXEC", "CANCEL", "EXEC", "ADD"],
            "feature": [0.0, 1.0, np.nan, -1.0, 2.0, 0.5],
            "raw": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    specs = (
        FieldSpec("evt_type", "evt_type", "categorical", is_target=True),
        FieldSpec(
            "feature",
            "feature",
            "ordinal",
            n_bins=4,
            applicable_events=("EXEC",),
            is_target=True,
        ),
        FieldSpec("raw", "raw", "continuous"),
    )
    source = tmp_path / "events.parquet"
    frame.write_parquet(source)
    vocab = fit_vocab_v2(
        [source],
        field_specs=specs,
        max_samples_per_field=100,
        categorical_values={"evt_type": ("ADD", "CANCEL", "EXEC")},
    )
    vocab_path = tmp_path / "vocab_v2.json"
    vocab.save(vocab_path)
    token_path = tmp_path / "tokens.parquet"
    tokenize_frame_v2(frame, vocab).write_parquet(token_path)
    shard = ShardEntry(
        market="SH",
        symbol="600000",
        date="2025-01-02",
        path=str(token_path),
        rows=frame.height,
        sha256="test",
    )
    return vocab, vocab_path, shard


def _model_config(vocab, vocab_path) -> OrderFlowFMConfig:
    layout = field_layout_from_vocab(vocab)
    return OrderFlowFMConfig(
        field_sizes=vocab.field_sizes(),
        input_fields=layout.input_fields,
        target_fields=layout.target_fields,
        scalar_fields=layout.scalar_to_token,
        standalone_scalar_fields=layout.standalone_scalar_fields,
        d_model=16,
        n_layers=1,
        n_heads=4,
        ffn_mult=2.0,
        dropout=0.0,
        max_seq_len=16,
        field_fusion="concat_mlp",
        field_dim=4,
        schema_version=vocab.schema_version,
        vocab_version=vocab.VOCAB_VERSION,
        vocab_sha256=stable_vocab_sha256(vocab),
        field_specs=tuple(spec.to_dict() for spec in vocab.field_specs),
        event_ordering_version=vocab.event_ordering_version,
        feature_transform_version=vocab.feature_transform_version,
    )


def _pretrain_contract(vocab) -> dict[str, object]:
    date = str(vocab.fit_dates[0])
    return {
        "format_version": "pretrain_data_contract_v2",
        "manifest_sha256": "a" * 64,
        "vocab_artifact_sha256": "b" * 64,
        "vocab_sha256": stable_vocab_sha256(vocab),
        "schema_version": vocab.schema_version,
        "event_ordering_version": vocab.event_ordering_version,
        "feature_transform_version": vocab.feature_transform_version,
        "manifest_train_start": date,
        "manifest_train_end": date,
        "manifest_validation_start": date,
        "manifest_validation_end": date,
        "vocab_fit_start": date,
        "vocab_fit_end": date,
        "effective_training_end": date,
    }


def test_v2_dataset_model_and_masked_loss_connect_end_to_end(tmp_path) -> None:
    vocab, vocab_path, shard = _artifacts(tmp_path)
    dataset = EventWindowDatasetV2([shard], vocab=vocab, context=3, stride=3, min_len=2)
    batch = collate_windows_v2([dataset[0], dataset[1]])
    layout = field_layout_from_vocab(vocab)

    assert layout.scalar_to_token == {"val_feature": "tok_feature_bin"}
    assert layout.standalone_scalar_fields == ("val_raw",)
    assert batch["val_feature"].dtype == torch.float32
    assert batch["tok_feature_bin"][0, 0].item() == NA_ID
    assert not batch["mask_tok_feature_bin"][0, 0]

    model = OrderFlowFM(_model_config(vocab, vocab_path))
    logits = model(batch)
    loss_config = {
        "targets": {
            field: {"type": "ordinal_ce" if "feature" in field else "ce"}
            for field in layout.target_fields
        },
        "train_entropy": {
            field: max(vocab.train_entropy(field), 1e-6)
            for field in layout.target_fields
        },
    }
    specs = target_specs_from_config(
        layout.target_fields,
        loss_config,
        default_ignore_ids=(0, NA_ID),
        default_ordinal_start_id=N_SPECIAL,
    )
    assert specs is not None
    output = next_event_loss_v2(logits, batch, specs)
    assert torch.isfinite(output.total)


def test_scalar_channel_changes_event_representation(tmp_path) -> None:
    vocab, vocab_path, shard = _artifacts(tmp_path)
    dataset = EventWindowDatasetV2([shard], vocab=vocab, context=6, min_len=2)
    batch = collate_windows_v2([dataset[0]])
    changed = {key: value.clone() for key, value in batch.items()}
    changed["val_feature"] += 1.0
    changed["val_raw"] += 1.0
    model = OrderFlowFM(_model_config(vocab, vocab_path)).eval()

    assert not torch.equal(model.encode(batch), model.encode(changed))


def test_v2_checkpoint_requires_exact_vocab_artifact(tmp_path) -> None:
    vocab, vocab_path, _ = _artifacts(tmp_path)
    config = _model_config(vocab, vocab_path)
    model = OrderFlowFM(config)
    checkpoint = tmp_path / "model.pt"
    _save_checkpoint(
        model,
        config,
        checkpoint,
        pretrain_data_contract=_pretrain_contract(vocab),
    )

    loaded = load_checkpoint(checkpoint, torch.device("cpu"), vocab_path=vocab_path)
    assert loaded.cfg.vocab_version == "2.0"
    with pytest.raises(ValueError, match="requires vocab_path"):
        load_checkpoint(checkpoint, torch.device("cpu"))

    tampered = tmp_path / "tampered_vocab.json"
    tampered_payload = json.loads(vocab.to_json())
    tampered_payload["sampling"] = {"tampered": True}
    tampered.write_text(json.dumps(tampered_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="vocab_sha256"):
        load_checkpoint(checkpoint, torch.device("cpu"), vocab_path=tampered)


def test_v2_checkpoint_rejects_token_semantics_metadata_mismatch(tmp_path) -> None:
    vocab, vocab_path, _ = _artifacts(tmp_path)
    config = _model_config(vocab, vocab_path)
    checkpoint = tmp_path / "model.pt"
    _save_checkpoint(
        OrderFlowFM(config),
        config,
        checkpoint,
        pretrain_data_contract=_pretrain_contract(vocab),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["config"]["event_ordering_version"] = "local_time_v1"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        load_checkpoint(checkpoint, torch.device("cpu"), vocab_path=vocab_path)


def test_explicit_v2_checkpoint_without_data_contract_is_legacy_only(tmp_path) -> None:
    vocab, vocab_path, _ = _artifacts(tmp_path)
    config = _model_config(vocab, vocab_path)
    checkpoint = tmp_path / "model.pt"
    _save_checkpoint(OrderFlowFM(config), config, checkpoint)

    with pytest.raises(ValueError, match="data contract is missing"):
        load_checkpoint(checkpoint, torch.device("cpu"), vocab_path=vocab_path)

    diagnostic = load_checkpoint(
        checkpoint,
        torch.device("cpu"),
        vocab_path=vocab_path,
        allow_missing_pretrain_data_contract=True,
    )
    assert diagnostic.cfg.vocab_sha256 == stable_vocab_sha256(vocab)


def test_v2_resume_rejects_same_shape_configuration_changes(tmp_path) -> None:
    vocab, vocab_path, _ = _artifacts(tmp_path)
    config = _model_config(vocab, vocab_path)
    model = OrderFlowFM(config)
    checkpoint = tmp_path / "model.pt"
    _save_checkpoint(model, config, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    changed = _model_config(vocab, vocab_path)
    changed.pooling_version = "multi_scale_v1"

    with pytest.raises(ValueError, match="pooling_version"):
        _validate_resume_metadata(payload, changed, target_specs=None)


def test_v2_training_loader_persists_fixed_validation_plan(tmp_path) -> None:
    vocab, _, train_shard = _artifacts(tmp_path)
    val_shard = ShardEntry(
        market=train_shard.market,
        symbol="600001",
        date="2025-01-03",
        path=train_shard.path,
        rows=train_shard.rows,
        sha256="val-test",
        split="val",
    )
    manifest = Manifest(shards=[train_shard, val_shard])
    plan_path = tmp_path / "validation.json"
    config = {
        "seed": 7,
        "data": {
            "context": 3,
            "stride": 3,
            "min_len": 2,
            "cache_size": 2,
            "num_workers": 0,
            "validation_plan": str(plan_path),
            "validation_windows": 2,
        },
        "optim": {"batch_size": 1},
        "runtime": {"out_dir": str(tmp_path / "run"), "val_max_batches": 2},
    }

    train_loader, val_loader = _build_dataloaders(
        manifest,
        config,
        vocab=vocab,
        seed=7,
    )

    assert plan_path.is_file()
    assert val_loader is not None
    assert "val_feature" in next(iter(train_loader))
    assert "mask_tok_feature_bin" in next(iter(val_loader))
