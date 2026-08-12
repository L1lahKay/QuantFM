from __future__ import annotations

import hashlib
import importlib

import polars as pl
import pytest
import torch
import yaml

from quant_fm.manifest.build_manifest import Manifest, ShardEntry
from quant_fm.pretrain.dataset import FIELD_ORDER
from quant_fm.pretrain.train import train
from quant_fm.tokenizer.artifact_contract import (
    stable_vocab_sha256,
    write_token_contract,
)
from quant_fm.tokenizer.vocab import default_vocab


def test_train_runs_one_cpu_optimizer_update(tmp_path) -> None:
    vocab = default_vocab(n_bins=4)
    vocab_path = tmp_path / "vocab.json"
    vocab.save(vocab_path)

    token_path = tmp_path / "tokens.parquet"
    pl.DataFrame({field: [1, 1, 1, 1] for field in FIELD_ORDER}).write_parquet(
        token_path
    )
    contract_path = write_token_contract(token_path, vocab)
    validation_token_path = tmp_path / "validation_tokens.parquet"
    pl.DataFrame({field: [1, 1, 1, 1] for field in FIELD_ORDER}).write_parquet(
        validation_token_path
    )
    validation_contract_path = write_token_contract(validation_token_path, vocab)
    shard = ShardEntry(
        market="SH",
        symbol="600000",
        date="2025-01-02",
        path=str(token_path),
        rows=4,
        sha256=hashlib.sha256(token_path.read_bytes()).hexdigest(),
        split="train",
        data_contract_sha256=hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    )
    validation_shard = ShardEntry(
        market="SH",
        symbol="600000",
        date="2025-01-03",
        path=str(validation_token_path),
        rows=4,
        sha256=hashlib.sha256(validation_token_path.read_bytes()).hexdigest(),
        split="val",
        data_contract_sha256=hashlib.sha256(
            validation_contract_path.read_bytes()
        ).hexdigest(),
    )
    manifest_path = tmp_path / "manifest.json"
    Manifest(
        shards=[shard, validation_shard],
        vocab_path=str(vocab_path),
        vocab_sha256=stable_vocab_sha256(vocab),
        schema_version=vocab.schema_version,
        event_ordering_version=vocab.event_ordering_version,
        feature_transform_version=vocab.feature_transform_version,
    ).save(manifest_path)

    out_dir = tmp_path / "run"
    config = {
        "seed": 7,
        "data": {
            "manifest": str(manifest_path),
            "vocab": str(vocab_path),
            "context": 4,
            "stride": 4,
            "min_len": 2,
            "cache_size": 1,
            "num_workers": 0,
            "min_validation_dates": 1,
            "min_test_dates": 1,
        },
        "model": {
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 4,
            "ffn_mult": 2.0,
            "dropout": 0.0,
            "max_seq_len": 4,
            "rope_theta": 10_000.0,
        },
        "optim": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "betas": [0.9, 0.95],
            "grad_clip": 1.0,
            "warmup_steps": 0,
            "max_update_steps": 1,
            "batch_size": 1,
            "grad_accum": 1,
            "precision": "fp32",
        },
        "runtime": {
            "out_dir": str(out_dir),
            "log_every": 1,
            "eval_every": 10,
            "ckpt_every": 10,
            "save_best": True,
            "fsdp": False,
            "device": "cpu",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    train(config_path)

    resumable = torch.load(
        out_dir / "final_resume.pt", map_location="cpu", weights_only=False
    )
    inference = torch.load(out_dir / "final.pt", map_location="cpu", weights_only=False)
    assert resumable["train_state"]["micro_step"] == 1
    assert resumable["train_state"]["update_step"] == 1
    assert resumable["train_state"]["non_pad_tokens_seen"] == 4
    assert resumable["train_state"]["batches_consumed_in_epoch"] == 1
    assert len(resumable["runtime_state_by_rank"]) == 1
    assert set(resumable["runtime_state_by_rank"][0]["rng"]) == {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }
    assert "optimizer_state" in resumable
    assert "optimizer_state" not in inference

    snapshot = (out_dir / "config.snapshot.yaml").read_bytes()
    with pytest.raises(FileExistsError, match="non-empty output directory"):
        train(config_path)
    assert (out_dir / "config.snapshot.yaml").read_bytes() == snapshot

    with pytest.raises(ValueError, match="inference-only"):
        train(config_path, resume=out_dir / "final.pt")
    assert (out_dir / "config.snapshot.yaml").read_bytes() == snapshot

    train(config_path, resume=out_dir / "final_resume.pt")
    assert (out_dir / "config.snapshot.yaml").read_bytes() == snapshot


def test_resume_after_final_validation_is_bitwise_identical_with_dropout(
    tmp_path,
    monkeypatch,
) -> None:
    train_module = importlib.import_module("quant_fm.pretrain.train")
    real_evaluate = train_module.evaluate
    forced_validation_losses: list[float] = []

    def controlled_evaluate(*args, **kwargs) -> float:
        # Preserve the real validation iterator/model-mode path so this remains a
        # regression for validation RNG isolation, while making checkpoint
        # selection deterministic enough to expose resume-boundary drift.
        real_evaluate(*args, **kwargs)
        if not forced_validation_losses:
            msg = "unexpected validation call"
            raise AssertionError(msg)
        return forced_validation_losses.pop(0)

    monkeypatch.setattr(train_module, "evaluate", controlled_evaluate)
    vocab = default_vocab(n_bins=4)
    vocab_path = tmp_path / "bitwise_vocab.json"
    vocab.save(vocab_path)

    shards = []
    for split, date in (("train", "2025-01-02"), ("val", "2025-01-03")):
        token_path = tmp_path / f"bitwise_{split}.parquet"
        pl.DataFrame({field: [1, 1, 1, 1] for field in FIELD_ORDER}).write_parquet(
            token_path
        )
        contract_path = write_token_contract(token_path, vocab)
        shards.append(
            ShardEntry(
                market="SH",
                symbol="600000",
                date=date,
                path=str(token_path),
                rows=4,
                sha256=hashlib.sha256(token_path.read_bytes()).hexdigest(),
                split=split,
                data_contract_sha256=hashlib.sha256(
                    contract_path.read_bytes()
                ).hexdigest(),
            )
        )
    manifest_path = tmp_path / "bitwise_manifest.json"
    Manifest(
        shards=shards,
        vocab_path=str(vocab_path),
        vocab_sha256=stable_vocab_sha256(vocab),
        schema_version=vocab.schema_version,
        event_ordering_version=vocab.event_ordering_version,
        feature_transform_version=vocab.feature_transform_version,
    ).save(manifest_path)

    def config(out_dir, *, max_update_steps: int) -> dict:
        return {
            "seed": 73,
            "data": {
                "manifest": str(manifest_path),
                "vocab": str(vocab_path),
                "context": 4,
                "stride": 4,
                "min_len": 2,
                "cache_size": 1,
                "num_workers": 0,
                "min_validation_dates": 1,
                "min_test_dates": 1,
            },
            "model": {
                "d_model": 16,
                "n_layers": 1,
                "n_heads": 4,
                "ffn_mult": 2.0,
                "dropout": 0.25,
                "max_seq_len": 4,
                "rope_theta": 10_000.0,
            },
            "optim": {
                "lr": 1e-3,
                "weight_decay": 0.0,
                "betas": [0.9, 0.95],
                "grad_clip": 1.0,
                "warmup_steps": 0,
                "max_update_steps": max_update_steps,
                # Keep the LR horizon immutable while extending only the stop budget.
                "lr_schedule_steps": 2,
                "batch_size": 1,
                "grad_accum": 1,
                "precision": "fp32",
            },
            "runtime": {
                "out_dir": str(out_dir),
                "log_every": 10,
                "eval_every": 2,
                "ckpt_every": 10,
                "val_max_batches": 1,
                "save_best": True,
                "fsdp": False,
                "device": "cpu",
            },
        }

    continuous_dir = tmp_path / "continuous"
    continuous_config_path = tmp_path / "continuous.yaml"
    continuous_config_path.write_text(
        yaml.safe_dump(config(continuous_dir, max_update_steps=2)),
        encoding="utf-8",
    )
    forced_validation_losses[:] = [2.0]
    train(continuous_config_path)
    assert not forced_validation_losses

    segmented_dir = tmp_path / "segmented"
    segmented_config_path = tmp_path / "segmented.yaml"
    segmented_config_path.write_text(
        yaml.safe_dump(config(segmented_dir, max_update_steps=1)),
        encoding="utf-8",
    )
    # This lower, non-periodic endpoint diagnostic must not enter best-state.
    forced_validation_losses[:] = [1.0]
    train(segmented_config_path)
    assert not forced_validation_losses
    segmented_config_path.write_text(
        yaml.safe_dump(config(segmented_dir, max_update_steps=2)),
        encoding="utf-8",
    )
    forced_validation_losses[:] = [2.0]
    train(segmented_config_path, resume=segmented_dir / "final_resume.pt")
    assert not forced_validation_losses

    continuous_payload = torch.load(
        continuous_dir / "final_resume.pt",
        map_location="cpu",
        weights_only=False,
    )
    segmented_payload = torch.load(
        segmented_dir / "final_resume.pt",
        map_location="cpu",
        weights_only=False,
    )
    continuous = continuous_payload["model_state"]
    segmented = segmented_payload["model_state"]
    assert continuous.keys() == segmented.keys()
    for name in continuous:
        assert torch.equal(continuous[name], segmented[name]), name
    expected_best_state = {
        "best_val": 2.0,
        "best_update_step": 2,
    }
    for payload in (continuous_payload, segmented_payload):
        for field, expected in expected_best_state.items():
            assert payload["train_state"][field] == expected

    continuous_best = torch.load(
        continuous_dir / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    segmented_best = torch.load(
        segmented_dir / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert continuous_best["step"] == segmented_best["step"] == 2
    for name in continuous_best["model_state"]:
        assert torch.equal(
            continuous_best["model_state"][name],
            segmented_best["model_state"][name],
        ), name


def test_train_can_stop_after_one_complete_data_epoch(tmp_path) -> None:
    vocab = default_vocab(n_bins=4)
    vocab_path = tmp_path / "vocab.json"
    vocab.save(vocab_path)

    token_path = tmp_path / "tokens.parquet"
    pl.DataFrame({field: [1, 1, 1, 1] for field in FIELD_ORDER}).write_parquet(
        token_path
    )
    contract_path = write_token_contract(token_path, vocab)
    manifest_path = tmp_path / "manifest.json"
    Manifest(
        shards=[
            ShardEntry(
                market="SH",
                symbol="600000",
                date="2025-01-02",
                path=str(token_path),
                rows=4,
                sha256=hashlib.sha256(token_path.read_bytes()).hexdigest(),
                split="train",
                data_contract_sha256=hashlib.sha256(
                    contract_path.read_bytes()
                ).hexdigest(),
            )
        ],
        vocab_path=str(vocab_path),
        vocab_sha256=stable_vocab_sha256(vocab),
        schema_version=vocab.schema_version,
        event_ordering_version=vocab.event_ordering_version,
        feature_transform_version=vocab.feature_transform_version,
    ).save(manifest_path)

    out_dir = tmp_path / "epoch_run"
    config = {
        "seed": 7,
        "data": {
            "manifest": str(manifest_path),
            "vocab": str(vocab_path),
            "context": 4,
            "stride": 4,
            "min_len": 2,
            "cache_size": 1,
            "num_workers": 0,
        },
        "model": {
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 4,
            "ffn_mult": 2.0,
            "dropout": 0.0,
            "max_seq_len": 4,
            "rope_theta": 10_000.0,
        },
        "optim": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "betas": [0.9, 0.95],
            "grad_clip": 1.0,
            "warmup_steps": 0,
            "max_data_epochs": 1,
            "lr_schedule_steps": 1,
            "batch_size": 1,
            "grad_accum": 1,
            "precision": "fp32",
        },
        "runtime": {
            "out_dir": str(out_dir),
            "log_every": 1,
            "eval_every": 10,
            "ckpt_every": 10,
            "save_best": False,
            "fsdp": False,
            "device": "cpu",
        },
    }
    config_path = tmp_path / "epoch_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    train(config_path)

    checkpoint = torch.load(
        out_dir / "final_resume.pt", map_location="cpu", weights_only=False
    )
    state = checkpoint["train_state"]
    assert state["update_step"] == 1
    assert state["data_epochs_completed"] == 1
