from __future__ import annotations

import hashlib
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from quant_fm.data_coverage import coverage_set_sha256
from quant_fm.manifest.build_manifest import Manifest, ShardEntry
from quant_fm.manifest.validation import sha256_file
from quant_fm.pretrain.train import (
    TrainState,
    _build_dataloaders,
    _build_grad_scaler,
    _capture_rng_state,
    _data_order_state,
    _immutable_training_config_sha256,
    _remaining_epoch_batches,
    _restore_rng_state,
    _restore_runtime_state,
    _set_data_epoch,
    _training_stop_budgets,
    _validate_config_schema,
    _validate_distributed_runtime,
    _validate_run_directory,
    _validate_training_config,
    _validate_v2_artifact_audit,
)
from quant_fm.pretrain.validation_sampler import (
    ValidationSamplePlan,
    build_validation_sample_plan,
)
from quant_fm.tokenizer.vocab import default_vocab
from quant_fm.tokenizer.vocab_v2 import default_vocab_v2


def test_multi_process_training_requires_fsdp() -> None:
    with pytest.raises(ValueError, match=r"world_size=2.*fsdp=true"):
        _validate_distributed_runtime({"runtime": {"fsdp": False}}, world_size=2)

    assert _validate_distributed_runtime({"runtime": {"fsdp": True}}, world_size=2)
    assert not _validate_distributed_runtime({"runtime": {"fsdp": True}}, world_size=1)


def test_fsdp_fp16_uses_rank_synchronized_grad_scaler() -> None:
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fsdp_scaler = _build_grad_scaler("fp16", use_fsdp=True)
        ordinary_scaler = _build_grad_scaler("fp16", use_fsdp=False)
        bf16_scaler = _build_grad_scaler("bf16", use_fsdp=True)

    assert isinstance(fsdp_scaler, ShardedGradScaler)
    assert not isinstance(ordinary_scaler, ShardedGradScaler)
    assert not isinstance(bf16_scaler, ShardedGradScaler)


def test_fresh_training_rejects_non_empty_run_directory(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    _validate_run_directory(out_dir, resume_path=None)
    sentinel = out_dir / "existing.pt"
    sentinel.write_bytes(b"do-not-overwrite")

    with pytest.raises(FileExistsError, match="refusing fresh training"):
        _validate_run_directory(out_dir, resume_path=None)

    _validate_run_directory(out_dir, resume_path=sentinel)
    assert sentinel.read_bytes() == b"do-not-overwrite"


def test_rng_state_round_trips() -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    saved = _capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    _restore_rng_state(saved)
    actual = (random.random(), np.random.random(), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_mid_epoch_resume_skips_consumed_batches_without_repeating() -> None:
    generator = torch.Generator()
    loader = DataLoader(
        TensorDataset(torch.arange(12)),
        batch_size=2,
        shuffle=True,
        generator=generator,
    )
    _set_data_epoch(loader, 3, seed=11, rank=0)
    complete = [batch[0].clone() for batch in loader]
    state = TrainState(data_epochs_completed=3, batches_consumed_in_epoch=2)
    checkpoint = {
        "runtime_state_by_rank": [
            {
                "rng": _capture_rng_state(),
                "data_order": _data_order_state(
                    loader,
                    next_epoch=3,
                    batches_consumed_in_epoch=2,
                    training_config_sha256="a" * 64,
                    stop_budgets={
                        "max_update_steps": 100,
                        "max_train_tokens": 0,
                        "max_data_epochs": 0,
                    },
                    seed=11,
                    rank=0,
                    world_size=1,
                ),
            }
        ]
    }
    resumed = DataLoader(
        TensorDataset(torch.arange(12)),
        batch_size=2,
        shuffle=True,
        generator=torch.Generator(),
    )
    _restore_runtime_state(
        checkpoint,
        resumed,
        state,
        training_config_sha256="a" * 64,
        stop_budgets={
            "max_update_steps": 100,
            "max_train_tokens": 0,
            "max_data_epochs": 0,
        },
        seed=11,
        rank=0,
        world_size=1,
    )
    _set_data_epoch(resumed, 3, seed=11, rank=0)
    remaining = [
        batch[0].clone()
        for batch in _remaining_epoch_batches(resumed, batches_consumed=2)  # type: ignore[arg-type]
    ]

    assert len(remaining) == len(complete[2:])
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(remaining, complete[2:], strict=True)
    )
    consumed_values = set(torch.cat(complete[:2]).tolist())
    remaining_values = set(torch.cat(remaining).tolist())
    assert consumed_values.isdisjoint(remaining_values)


def _resume_identity_config() -> dict:
    return {
        "schema_version": "cn_l2_v2",
        "seed": 11,
        "data": {
            "manifest": "/generation/data/manifest.json",
            "vocab": "/generation/data/vocab_v2.json",
            "context": 8,
            "stride": 8,
            "min_len": 2,
            "cache_size": 1,
            "num_workers": 0,
            "drop_last": True,
        },
        "model": {"d_model": 16, "n_layers": 1, "n_heads": 4},
        "optim": {
            "lr": 1e-3,
            "micro_batch_size": 2,
            "grad_accum": 1,
            "precision": "fp32",
            "max_update_steps": 100,
        },
        "runtime": {
            "out_dir": "/run-a",
            "log_every": 10,
            "ckpt_every": 20,
            "eval_every": 25,
            "fsdp": False,
            "device": "cpu",
        },
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("log_every", 0),
        ("eval_every", -1),
        ("ckpt_every", False),
        ("ckpt_every", 1.5),
    ],
)
def test_training_config_rejects_invalid_runtime_cadence(key, value) -> None:
    config = _resume_identity_config()
    config["runtime"][key] = value

    with pytest.raises(ValueError, match=rf"runtime\.{key}.*positive integer"):
        _validate_training_config(config)


@pytest.mark.parametrize("precision", [None, "amp", "BF16", 16])
def test_training_config_rejects_unknown_precision(precision) -> None:
    config = _resume_identity_config()
    config["optim"]["precision"] = precision

    with pytest.raises(ValueError, match=r"optim\.precision.*bf16.*fp16.*fp32"):
        _validate_training_config(config)


@pytest.mark.parametrize(
    ("key", "value", "qualifier"),
    [
        ("grad_accum", 0, "positive"),
        ("micro_batch_size", False, "positive"),
        ("lr_schedule_steps", 0, "positive"),
        ("max_update_steps", -1, "non-negative"),
        ("max_train_tokens", False, "non-negative"),
    ],
)
def test_training_config_rejects_invalid_integer_optimizer_controls(
    key,
    value,
    qualifier,
) -> None:
    config = _resume_identity_config()
    config["optim"]["max_train_tokens"] = 10
    config["optim"][key] = value

    with pytest.raises(
        ValueError,
        match=rf"optim\.{key}.*{qualifier} integer",
    ):
        _validate_training_config(config)


def test_training_config_rejects_token_budget_without_schedule_horizon() -> None:
    config = _resume_identity_config()
    del config["optim"]["max_update_steps"]
    config["optim"]["max_train_tokens"] = 10

    with pytest.raises(ValueError, match=r"requires.*optim\.lr_schedule_steps"):
        _validate_training_config(config)


def test_all_shipped_pretraining_configs_pass_value_guards() -> None:
    config_dir = Path(__file__).parents[1] / "quant_fm" / "pretrain"
    for path in sorted(config_dir.glob("config*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert _validate_training_config(config) is config, path


def _runtime_checkpoint_for_config(
    loader: DataLoader,
    config: dict,
) -> dict[str, object]:
    return {
        "runtime_state_by_rank": [
            {
                "rng": _capture_rng_state(),
                "data_order": _data_order_state(
                    loader,
                    next_epoch=0,
                    batches_consumed_in_epoch=0,
                    training_config_sha256=_immutable_training_config_sha256(config),
                    stop_budgets=_training_stop_budgets(config),
                    seed=11,
                    rank=0,
                    world_size=1,
                ),
            }
        ]
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("micro_batch_size", 4),
        ("lr", 2e-3),
        ("grad_accum", 2),
    ],
)
def test_resume_rejects_immutable_optimizer_config_changes(key, value) -> None:
    loader = DataLoader(TensorDataset(torch.arange(8)), batch_size=2, shuffle=False)
    saved_config = _resume_identity_config()
    checkpoint = _runtime_checkpoint_for_config(loader, saved_config)
    changed = json.loads(json.dumps(saved_config))
    changed["optim"][key] = value

    with pytest.raises(ValueError, match="training_config_sha256"):
        _restore_runtime_state(
            checkpoint,
            loader,
            TrainState(),
            training_config_sha256=_immutable_training_config_sha256(changed),
            stop_budgets=_training_stop_budgets(changed),
            seed=11,
            rank=0,
            world_size=1,
            require_runtime_state=True,
        )


def test_resume_allows_only_stop_budget_extension_and_cadence_changes() -> None:
    loader = DataLoader(TensorDataset(torch.arange(8)), batch_size=2, shuffle=False)
    saved_config = _resume_identity_config()
    checkpoint = _runtime_checkpoint_for_config(loader, saved_config)
    extended = json.loads(json.dumps(saved_config))
    extended["optim"]["max_update_steps"] = 150
    extended["runtime"]["log_every"] = 1
    extended["runtime"]["ckpt_every"] = 5

    assert _immutable_training_config_sha256(extended) == (
        _immutable_training_config_sha256(saved_config)
    )
    _restore_runtime_state(
        checkpoint,
        loader,
        TrainState(),
        training_config_sha256=_immutable_training_config_sha256(extended),
        stop_budgets=_training_stop_budgets(extended),
        seed=11,
        rank=0,
        world_size=1,
        require_runtime_state=True,
    )

    shortened = json.loads(json.dumps(saved_config))
    shortened["optim"]["max_update_steps"] = 50
    with pytest.raises(ValueError, match="may only be extended"):
        _restore_runtime_state(
            checkpoint,
            loader,
            TrainState(),
            training_config_sha256=_immutable_training_config_sha256(shortened),
            stop_budgets=_training_stop_budgets(shortened),
            seed=11,
            rank=0,
            world_size=1,
            require_runtime_state=True,
        )


def test_resume_rejects_replaced_fixed_validation_plan(tmp_path: Path) -> None:
    shards = [
        ShardEntry(
            market="SH",
            symbol="600000",
            date="2025-01-02",
            path=str(tmp_path / "same-validation-shard.parquet"),
            rows=64,
            sha256="a" * 64,
            split="val",
        )
    ]
    first = build_validation_sample_plan(
        shards,
        context=8,
        stride=8,
        min_len=2,
        seed=7,
        max_windows=3,
    )
    replacement = build_validation_sample_plan(
        shards,
        context=8,
        stride=8,
        min_len=2,
        seed=8,
        max_windows=3,
    )
    plan_path = tmp_path / "validation-plan.json"
    first.save(plan_path)
    saved_plan_sha256 = ValidationSamplePlan.load(plan_path).sha256
    replacement.save(plan_path)
    current_plan_sha256 = ValidationSamplePlan.load(plan_path).sha256
    assert saved_plan_sha256 != current_plan_sha256

    loader = DataLoader(TensorDataset(torch.arange(8)), batch_size=2, shuffle=False)
    config = _resume_identity_config()
    checkpoint = _runtime_checkpoint_for_config(loader, config)
    checkpoint["runtime_state_by_rank"][0]["data_order"][  # type: ignore[index]
        "validation_plan_sha256"
    ] = saved_plan_sha256

    with pytest.raises(ValueError, match="validation_plan_sha256"):
        _restore_runtime_state(
            checkpoint,
            loader,
            TrainState(),
            training_config_sha256=_immutable_training_config_sha256(config),
            validation_plan_sha256=current_plan_sha256,
            stop_budgets=_training_stop_budgets(config),
            seed=11,
            rank=0,
            world_size=1,
            require_runtime_state=True,
        )


def test_v2_resume_rejects_checkpoint_without_runtime_state() -> None:
    loader = DataLoader(TensorDataset(torch.arange(4)), batch_size=2)
    with pytest.raises(ValueError, match="v2 resume requires"):
        _restore_runtime_state(
            {},
            loader,
            TrainState(),
            training_config_sha256="a" * 64,
            stop_budgets={
                "max_update_steps": 1,
                "max_train_tokens": 0,
                "max_data_epochs": 0,
            },
            seed=11,
            rank=0,
            world_size=1,
            require_runtime_state=True,
        )


def test_v2_25m_pilot_config_uses_fsdp_and_one_day_holdout_gates() -> None:
    path = Path(__file__).parents[1] / "quant_fm" / "pretrain" / "config_v2_25m.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["runtime"]["fsdp"] is True
    assert config["data"]["min_validation_dates"] == 1
    assert config["data"]["min_test_dates"] == 1


def test_training_launcher_starts_tensorboard_after_run_claim() -> None:
    path = Path(__file__).parents[1] / "quant_fm" / "scripts" / "train_medium_8gpu.sh"
    source = path.read_text(encoding="utf-8")

    assert source.index("existing_run_entries") < source.index("launcher_pid=")
    assert source.index('[[ -f "$RUN_DIR/config.snapshot.yaml" ]]') < source.index(
        'bash "$ROOT/quant_fm/scripts/start_tensorboard_medium.sh"'
    )


def test_training_config_rejects_unknown_keys() -> None:
    config = {
        "seed": 1,
        "data": {},
        "model": {},
        "optim": {},
        "runtime": {"fsdp_typo": False},
    }
    with pytest.raises(ValueError, match="unknown training config keys in runtime"):
        _validate_training_config(config)


def test_v2_config_schema_must_match_vocab() -> None:
    vocab = default_vocab_v2()
    with pytest.raises(ValueError, match="does not match vocab schema_version"):
        _validate_config_schema({"schema_version": "cn_l2_v1"}, vocab)
    with pytest.raises(ValueError, match="requires top-level schema_version"):
        _validate_config_schema({}, vocab)


def _write_passing_audit(root: Path) -> tuple[dict, Path, Path, object]:
    data_dir = root / "data"
    coverage_dir = data_dir / "coverage"
    coverage_dir.mkdir(parents=True)
    (coverage_dir / "2025-01-02.json").write_text("{}\n", encoding="utf-8")
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text('{"test": true}\n', encoding="utf-8")
    vocab = default_vocab_v2()
    vocab_path = data_dir / "vocab_v2.json"
    vocab.save(vocab_path)
    identities = {
        "coverage_sha256": coverage_set_sha256(root),
        "manifest_sha256": sha256_file(manifest_path),
        "vocab_file_sha256": sha256_file(vocab_path),
    }
    audit_input_sha256 = hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    audit = {
        "audit_version": "2.0",
        "contract_ready": True,
        "checked_all_paths": True,
        "content_coverage_required": True,
        "schema_version": vocab.schema_version,
        "vocab_version": vocab.VOCAB_VERSION,
        "audit_input_sha256": audit_input_sha256,
        **identities,
    }
    (root / "artifact_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    return audit, manifest_path, vocab_path, vocab


def test_v2_training_requires_full_identity_bound_artifact_audit(
    tmp_path: Path,
) -> None:
    audit, manifest_path, vocab_path, vocab = _write_passing_audit(tmp_path)
    config = {"data": {}}

    result = _validate_v2_artifact_audit(
        config,
        manifest_path=manifest_path,
        vocab_path=vocab_path,
        vocab=vocab,  # type: ignore[arg-type]
    )
    assert result is not None
    assert result["manifest_sha256"] == audit["manifest_sha256"]

    audit["checked_all_paths"] = False
    (tmp_path / "artifact_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match current artifacts"):
        _validate_v2_artifact_audit(
            config,
            manifest_path=manifest_path,
            vocab_path=vocab_path,
            vocab=vocab,  # type: ignore[arg-type]
        )

    audit["checked_all_paths"] = True
    (tmp_path / "artifact_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    manifest_path.write_text('{"test": "tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_sha256"):
        _validate_v2_artifact_audit(
            config,
            manifest_path=manifest_path,
            vocab_path=vocab_path,
            vocab=vocab,  # type: ignore[arg-type]
        )


def test_v2_training_rejects_missing_artifact_audit(tmp_path: Path) -> None:
    vocab = default_vocab_v2()
    manifest_path = tmp_path / "data" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text("{}", encoding="utf-8")
    vocab_path = manifest_path.parent / "vocab_v2.json"
    vocab.save(vocab_path)

    with pytest.raises(FileNotFoundError, match="requires a full artifact audit"):
        _validate_v2_artifact_audit(
            {"data": {}},
            manifest_path=manifest_path,
            vocab_path=vocab_path,
            vocab=vocab,
        )


def _loader_config(tmp_path: Path, *, batch_size: int) -> dict:
    return {
        "seed": 7,
        "data": {
            "context": 4,
            "stride": 4,
            "min_len": 2,
            "cache_size": 1,
            "num_workers": 0,
            "drop_last": True,
        },
        "optim": {"batch_size": batch_size, "grad_accum": 1},
        "runtime": {"out_dir": str(tmp_path / "run")},
    }


@pytest.mark.parametrize(
    ("rows", "batch_size", "message"),
    [
        (1, 1, "training dataset contains no windows"),
        (2, 2, "training DataLoader contains no batches"),
    ],
)
def test_training_loader_rejects_zero_progress_inputs(
    tmp_path: Path,
    rows: int,
    batch_size: int,
    message: str,
) -> None:
    manifest = Manifest(
        shards=[
            ShardEntry(
                market="SH",
                symbol="600000",
                date="2025-01-02",
                path=str(tmp_path / "unused.parquet"),
                rows=rows,
                sha256="unused",
                split="train",
            )
        ]
    )

    with pytest.raises(ValueError, match=message):
        _build_dataloaders(
            manifest,
            _loader_config(tmp_path, batch_size=batch_size),
            vocab=default_vocab(n_bins=4),
            seed=7,
        )
