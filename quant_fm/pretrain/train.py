"""
OrderFlow FM 的可复现预训练循环。

特性：配置驱动、固定种子、bf16/fp16 自动混合精度、AdamW、warmup+余弦
学习率、梯度裁剪与累积、可选单节点 FSDP、TensorBoard 日志、
定期验证、自动保存最优 ``best.pt``（按 val_loss）与定期/最终检查点。
完整配置快照于每个检查点旁，便于字节级复现。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import date as calendar_date
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from quant_fm.data_coverage import coverage_set_sha256
from quant_fm.manifest.build_manifest import Manifest
from quant_fm.manifest.validation import (
    sha256_file,
    validate_manifest_shards,
    validate_manifest_vocab_contract,
)
from quant_fm.moe.config import BackboneMoEConfig
from quant_fm.pretrain.data_contract import (
    build_pretrain_data_contract,
    load_checkpoint_contract,
    validate_checkpoint_data_contract,
    write_checkpoint_contract,
)
from quant_fm.pretrain.dataset import (
    DEFAULT_TARGET_FIELDS,
    FIELD_ORDER,
    EventWindowDataset,
    collate_windows,
)
from quant_fm.pretrain.dataset_v2 import (
    EventWindowDatasetV2,
    collate_windows_v2,
    field_layout_from_vocab,
)
from quant_fm.pretrain.heads import (
    next_event_loss,
    next_event_loss_v2,
    target_specs_from_config,
)
from quant_fm.pretrain.model import (
    OrderFlowFM,
    OrderFlowFMConfig,
    field_sizes_from_vocab,
)
from quant_fm.pretrain.sampler import ShardAwareDistributedSampler
from quant_fm.pretrain.validation_sampler import (
    FixedValidationSampler,
    ValidationSamplePlan,
    build_validation_sample_plan,
)
from quant_fm.tokenizer.artifact_contract import stable_vocab_sha256
from quant_fm.tokenizer.vocab import Vocab
from quant_fm.tokenizer.vocab_v2 import N_SPECIAL as V2_N_SPECIAL
from quant_fm.tokenizer.vocab_v2 import NA_ID as V2_NA_ID
from quant_fm.tokenizer.vocab_v2 import PAD_ID as V2_PAD_ID
from quant_fm.tokenizer.vocab_v2 import VocabV2

if TYPE_CHECKING:
    from collections.abc import Iterator

    from quant_fm.pretrain.heads import LossOutput, TargetSpec

logger = logging.getLogger(__name__)

_CONFIG_KEYS: dict[str, frozenset[str]] = {
    "data": frozenset(
        {
            "artifact_audit",
            "cache_size",
            "context",
            "drop_last",
            "manifest",
            "min_len",
            "min_test_dates",
            "min_validation_dates",
            "num_workers",
            "persistent_workers",
            "prefetch_factor",
            "shard_aware_sampler",
            "stride",
            "test_dates_file",
            "train_dates_file",
            "validation_dates_file",
            "validation_plan",
            "validation_seed",
            "validation_windows",
            "vocab",
        }
    ),
    "model": frozenset(
        {
            "backbone_moe",
            "d_model",
            "dropout",
            "ffn_hidden",
            "ffn_mult",
            "field_dim",
            "field_dropout",
            "field_fusion",
            "field_input_norm",
            "input_fields",
            "max_seq_len",
            "n_heads",
            "n_layers",
            "rope_theta",
            "target_fields",
        }
    ),
    "loss": frozenset({"normalize_by_train_entropy", "targets", "train_entropy"}),
    "book": frozenset({"state_timing"}),
    "pooling": frozenset({"method", "outputs", "stride", "version"}),
    "optim": frozenset(
        {
            "batch_size",
            "betas",
            "grad_accum",
            "grad_clip",
            "lr",
            "lr_schedule_steps",
            "max_data_epochs",
            "max_steps",
            "max_train_tokens",
            "max_update_steps",
            "micro_batch_size",
            "precision",
            "warmup_steps",
            "weight_decay",
        }
    ),
    "runtime": frozenset(
        {
            "ckpt_every",
            "device",
            "eval_every",
            "fsdp",
            "fsdp_no_sync",
            "log_every",
            "out_dir",
            "save_best",
            "val_max_batches",
        }
    ),
}


def _validate_integer_config_value(
    section: dict,
    qualified_key: str,
    *,
    minimum: int,
    required: bool = False,
) -> None:
    """Reject bool/coerced numeric controls before training mutates state."""
    key = qualified_key.rsplit(".", 1)[-1]
    if key not in section:
        if not required:
            return
        value = None
    else:
        value = section[key]
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        msg = f"{qualified_key} must be a {qualifier} integer"
        raise ValueError(msg)


def _validate_training_config(cfg: object) -> dict:
    """拒绝未知/非法训练值，避免错误在训练中途才触发。"""
    if not isinstance(cfg, dict):
        msg = "training config must be a mapping"
        raise TypeError(msg)
    allowed_top = {"schema_version", "seed", *_CONFIG_KEYS}
    unknown_top = sorted(set(cfg) - allowed_top)
    if unknown_top:
        msg = f"unknown training config keys: {unknown_top}"
        raise ValueError(msg)
    missing = sorted({"seed", "data", "model", "optim", "runtime"} - set(cfg))
    if missing:
        msg = f"training config is missing sections: {missing}"
        raise ValueError(msg)
    for section, allowed in _CONFIG_KEYS.items():
        if section not in cfg and section in {"loss", "book", "pooling"}:
            continue
        value = cfg.get(section)
        if not isinstance(value, dict):
            msg = f"training config section {section!r} must be a mapping"
            raise TypeError(msg)
        unknown = sorted(set(value) - allowed)
        if unknown:
            msg = f"unknown training config keys in {section}: {unknown}"
            raise ValueError(msg)
    fusion = cfg["model"].get("field_fusion")
    if isinstance(fusion, dict):
        allowed_fusion = {
            "categorical_dim",
            "field_dim",
            "field_dropout",
            "input_norm",
            "method",
        }
        unknown_fusion = sorted(set(fusion) - allowed_fusion)
        if unknown_fusion:
            msg = f"unknown model.field_fusion keys: {unknown_fusion}"
            raise ValueError(msg)

    runtime = cfg["runtime"]
    for key in ("log_every", "eval_every", "ckpt_every"):
        _validate_integer_config_value(
            runtime,
            f"runtime.{key}",
            minimum=1,
            required=True,
        )
    if "val_max_batches" in runtime:
        _validate_integer_config_value(
            runtime,
            "runtime.val_max_batches",
            minimum=0,
        )
    for key in ("fsdp", "fsdp_no_sync", "save_best"):
        if key in runtime and not isinstance(runtime[key], bool):
            msg = f"runtime.{key} must be a boolean"
            raise TypeError(msg)

    optim = cfg["optim"]
    precision = optim.get("precision")
    if not isinstance(precision, str) or precision not in {"bf16", "fp16", "fp32"}:
        msg = "optim.precision must be one of: bf16, fp16, fp32"
        raise ValueError(msg)
    _validate_integer_config_value(
        optim,
        "optim.grad_accum",
        minimum=1,
        required=True,
    )
    for key in ("batch_size", "micro_batch_size", "lr_schedule_steps"):
        if key in optim:
            _validate_integer_config_value(optim, f"optim.{key}", minimum=1)
    if "warmup_steps" in optim:
        _validate_integer_config_value(optim, "optim.warmup_steps", minimum=0)
    for key in (
        "max_steps",
        "max_update_steps",
        "max_train_tokens",
        "max_data_epochs",
    ):
        if key in optim:
            _validate_integer_config_value(optim, f"optim.{key}", minimum=0)
    max_update_steps = optim.get("max_update_steps", optim.get("max_steps", 0))
    if not any(
        value > 0
        for value in (
            max_update_steps,
            optim.get("max_train_tokens", 0),
            optim.get("max_data_epochs", 0),
        )
    ):
        msg = (
            "optim requires a positive max_update_steps/max_steps, "
            "max_train_tokens, or max_data_epochs"
        )
        raise ValueError(msg)
    if max_update_steps == 0 and "lr_schedule_steps" not in optim:
        msg = "token/epoch-budget training requires a positive optim.lr_schedule_steps"
        raise ValueError(msg)
    return cfg


def _immutable_training_config_sha256(cfg: dict) -> str:
    """哈希所有会改变优化语义的配置，排除路径、停止预算和纯日志频率。"""
    excluded: dict[str, set[str]] = {
        "data": {
            "artifact_audit",
            "manifest",
            "test_dates_file",
            "train_dates_file",
            "validation_dates_file",
            "validation_plan",
            "vocab",
        },
        "optim": {
            "max_data_epochs",
            "max_steps",
            "max_train_tokens",
            "max_update_steps",
        },
        "runtime": {"ckpt_every", "log_every", "out_dir"},
    }
    identity: dict[str, object] = {}
    for section, value in cfg.items():
        if isinstance(value, dict):
            ignored = excluded.get(section, set())
            identity[section] = {
                key: item for key, item in value.items() if key not in ignored
            }
        else:
            identity[section] = value
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _training_stop_budgets(cfg: dict) -> dict[str, int]:
    """规范化三种可独立启用的停止预算；零表示该维度不设上限。"""
    optim = cfg["optim"]
    return {
        "max_update_steps": max(
            0,
            int(optim.get("max_update_steps", optim.get("max_steps", 0))),
        ),
        "max_train_tokens": max(0, int(optim.get("max_train_tokens", 0))),
        "max_data_epochs": max(0, int(optim.get("max_data_epochs", 0))),
    }


def _validate_stop_budget_extension(
    saved: object,
    current: dict[str, int],
) -> None:
    """续训只能延长既有停止预算；不得新增更早终止条件。"""
    if not isinstance(saved, dict):
        msg = "checkpoint stop-budget state is invalid"
        raise TypeError(msg)
    mismatches: dict[str, tuple[object, int]] = {}
    for name, current_value in current.items():
        saved_value = saved.get(name)
        if isinstance(saved_value, bool) or not isinstance(saved_value, int):
            mismatches[name] = (saved_value, current_value)
            continue
        is_extension = (
            current_value == saved_value
            or (saved_value > 0 and current_value == 0)
            or (saved_value > 0 and current_value >= saved_value)
        )
        if not is_extension:
            mismatches[name] = (saved_value, current_value)
    if mismatches:
        msg = f"resume stop budgets may only be extended: {mismatches}"
        raise ValueError(msg)


def _validate_config_schema(cfg: dict, vocab: Vocab | VocabV2) -> None:
    """将 YAML 顶层 schema 声明绑定到实际加载的 vocab。"""
    configured = cfg.get("schema_version")
    if isinstance(vocab, VocabV2) and configured is None:
        msg = "Tokenizer v2 training requires top-level schema_version"
        raise ValueError(msg)
    if configured is not None and configured != vocab.schema_version:
        msg = (
            f"training config schema_version={configured!r} does not match "
            f"vocab schema_version={vocab.schema_version!r}"
        )
        raise ValueError(msg)


def _validate_v2_artifact_audit(
    cfg: dict,
    *,
    manifest_path: Path,
    vocab_path: Path,
    vocab: Vocab | VocabV2,
) -> dict[str, object] | None:
    """正式 cn_l2_v2 训练只接受绑定当前 generation 的全路径 PASS audit。"""
    if not isinstance(vocab, VocabV2) or vocab.schema_version != "cn_l2_v2":
        return None
    configured = cfg["data"].get("artifact_audit")
    audit_path = (
        Path(configured)
        if configured is not None
        else manifest_path.parent.parent / "artifact_audit.json"
    )
    if not audit_path.is_file():
        msg = f"cn_l2_v2 training requires a full artifact audit: missing {audit_path}"
        raise FileNotFoundError(msg)
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"invalid V2 artifact audit {audit_path}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(audit, dict):
        msg = f"V2 artifact audit must contain a JSON object: {audit_path}"
        raise TypeError(msg)
    root = audit_path.parent
    identities = {
        "coverage_sha256": coverage_set_sha256(root),
        "manifest_sha256": sha256_file(manifest_path),
        "vocab_file_sha256": sha256_file(vocab_path),
    }
    audit_input_sha256 = hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected: dict[str, object] = {
        "audit_version": "2.0",
        "contract_ready": True,
        "checked_all_paths": True,
        "content_coverage_required": True,
        "schema_version": vocab.schema_version,
        "vocab_version": vocab.VOCAB_VERSION,
        "audit_input_sha256": audit_input_sha256,
        **identities,
    }
    mismatches = {
        key: (audit.get(key), value)
        for key, value in expected.items()
        if audit.get(key) != value
    }
    if mismatches:
        msg = f"V2 artifact audit does not match current artifacts: {mismatches}"
        raise ValueError(msg)
    return {"path": str(audit_path), **expected}


def _load_vocab(path: Path) -> Vocab | VocabV2:
    """根据显式 artifact 版本选择 v1/v2 loader，拒绝静默混用。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("vocab_version") == VocabV2.VOCAB_VERSION:
        return VocabV2.load(path)
    return Vocab.load(path)


def _canonical_split_date(value: object, *, context: str) -> str:
    """Reject non-ISO and non-zero-padded dates before lexical comparisons."""
    if not isinstance(value, str) or not value:
        msg = f"{context} must be a canonical YYYY-MM-DD string"
        raise ValueError(msg)
    try:
        parsed = calendar_date.fromisoformat(value)
    except ValueError as exc:
        msg = f"{context} must be a canonical YYYY-MM-DD date: {value!r}"
        raise ValueError(msg) from exc
    if parsed.isoformat() != value:
        msg = f"{context} must be a canonical YYYY-MM-DD date: {value!r}"
        raise ValueError(msg)
    return value


def validate_pretrain_split_contract(
    manifest: Manifest,
    vocab: Vocab | VocabV2,
    *,
    require_validation: bool,
    min_validation_dates: int = 5,
    min_test_dates: int = 5,
    expected_dates: dict[str, set[str]] | None = None,
) -> dict[str, object]:
    """在训练开始前校验整日时间切分与 vocab 拟合边界。"""
    if min_validation_dates < 1 or min_test_dates < 1:
        msg = "minimum validation/test date counts must be positive"
        raise ValueError(msg)
    issues: list[str] = []
    dates: dict[str, set[str]] = {split: set() for split in ("train", "val", "test")}
    for index, shard in enumerate(manifest.shards):
        canonical = _canonical_split_date(
            shard.date,
            context=f"manifest shard[{index}] date",
        )
        if shard.split not in dates:
            issues.append(f"unknown manifest split: {shard.split}")
            continue
        dates[shard.split].add(canonical)
    for field in ("train_end", "val_end"):
        boundary = getattr(manifest, field)
        if boundary is not None:
            _canonical_split_date(boundary, context=f"manifest {field}")
    canonical_expected_dates: dict[str, set[str]] = {}
    for split, expected in (expected_dates or {}).items():
        canonical_expected_dates[split] = {
            _canonical_split_date(value, context=f"expected {split} date")
            for value in expected
        }
    fit_dates = {
        _canonical_split_date(value, context=f"vocab fit_dates[{index}]")
        for index, value in enumerate(vocab.fit_dates)
    }
    if not dates["train"]:
        issues.append("train split is empty")
    overlap = (
        (dates["train"] & dates["val"])
        | (dates["train"] & dates["test"])
        | (dates["val"] & dates["test"])
    )
    if overlap:
        issues.append(f"dates occur in multiple splits: {sorted(overlap)}")
    if dates["train"] and dates["val"] and max(dates["train"]) >= min(dates["val"]):
        issues.append("validation dates are not strictly after training dates")
    prior = dates["val"] or dates["train"]
    if prior and dates["test"] and max(prior) >= min(dates["test"]):
        issues.append("test dates are not strictly after train/validation dates")
    if require_validation and len(dates["val"]) < min_validation_dates:
        issues.append(
            "validation split is too short for checkpoint selection: "
            f"have={len(dates['val'])}, need={min_validation_dates}"
        )
    if dates["test"] and len(dates["test"]) < min_test_dates:
        issues.append(
            f"test split is too short: have={len(dates['test'])}, need={min_test_dates}"
        )
    for split, canonical_expected in canonical_expected_dates.items():
        if split not in dates:
            issues.append(f"unknown expected split: {split}")
            continue
        missing = sorted(canonical_expected - dates[split])
        extra = sorted(dates[split] - canonical_expected)
        if missing or extra:
            issues.append(
                f"{split} dates differ from frozen plan: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
    leaked = fit_dates & (dates["val"] | dates["test"])
    if leaked:
        issues.append(f"vocab was fitted on validation/test dates: {sorted(leaked)}")
    outside_training_manifest = fit_dates - dates["train"]
    if outside_training_manifest:
        issues.append(
            "vocab fit dates are not contained in the manifest training split: "
            f"{sorted(outside_training_manifest)}"
        )
    semantic_pairs = {
        "event_ordering_version": (
            manifest.event_ordering_version,
            vocab.event_ordering_version,
        ),
        "feature_transform_version": (
            manifest.feature_transform_version,
            vocab.feature_transform_version,
        ),
    }
    for name, (manifest_value, vocab_value) in semantic_pairs.items():
        if vocab.data_semantics_explicit and manifest_value is None:
            issues.append(f"manifest is missing required {name}")
        elif manifest_value is not None and manifest_value != vocab_value:
            issues.append(
                f"manifest {name}={manifest_value!r} does not match "
                f"vocab={vocab_value!r}"
            )
    if issues:
        msg = "invalid pretraining split contract: " + "; ".join(issues)
        raise ValueError(msg)
    return {
        "train_dates": len(dates["train"]),
        "validation_dates": len(dates["val"]),
        "test_dates": len(dates["test"]),
        "train_start": min(dates["train"]) if dates["train"] else None,
        "train_end": max(dates["train"]) if dates["train"] else None,
        "validation_start": min(dates["val"]) if dates["val"] else None,
        "validation_end": max(dates["val"]) if dates["val"] else None,
        "test_start": min(dates["test"]) if dates["test"] else None,
        "test_end": max(dates["test"]) if dates["test"] else None,
        "vocab_fit_dates": len(fit_dates),
        "event_ordering_version": vocab.event_ordering_version,
        "feature_transform_version": vocab.feature_transform_version,
    }


def _validate_training_shards(
    manifest: Manifest,
    vocab: Vocab | VocabV2,
    *,
    expected_tokens_root: Path,
    rank: int,
    world_size: int,
) -> dict[str, int] | None:
    """Validate live train/validation bytes once and broadcast any failure."""
    result: dict[str, int] | None = None
    error: str | None = None
    if rank == 0:
        try:
            selected = (
                list(manifest.shards)
                if isinstance(vocab, VocabV2)
                else [*manifest.split("train"), *manifest.split("val")]
            )
            result = validate_manifest_shards(
                manifest,
                vocab,
                shards=selected,
                context="pretraining",
                expected_tokens_root=expected_tokens_root,
            )
        except (OSError, TypeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    if world_size > 1:
        import torch.distributed as dist

        payload: list[object] = [error, result]
        dist.broadcast_object_list(payload, src=0)
        error = payload[0] if isinstance(payload[0], str) else None
        result = payload[1] if isinstance(payload[1], dict) else None
    if error is not None:
        msg = f"pretraining token provenance validation failed: {error}"
        raise ValueError(msg)
    return result


def set_seed(seed: int) -> None:
    """为可复现性设置 python、numpy 与 torch 种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str, *, local_rank: int | None = None) -> torch.device:
    """解析 ``auto|cuda|cpu`` 设备规格；分布式时绑定 ``cuda:{local_rank}``。"""
    if local_rank is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def setup_distributed() -> tuple[int, int, int]:
    """若由 ``torchrun`` 启动则初始化进程组，否则返回单进程上下文。"""
    if "LOCAL_RANK" not in os.environ:
        return 0, 0, 1
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return local_rank, dist.get_rank(), dist.get_world_size()


def _validate_distributed_runtime(cfg: dict, *, world_size: int) -> bool:
    """多 rank 只能进入已实现且会同步梯度的 FSDP 路径。"""
    requested = cfg["runtime"].get("fsdp", False)
    if not isinstance(requested, bool):
        msg = "runtime.fsdp must be a boolean"
        raise TypeError(msg)
    if world_size > 1 and not requested:
        msg = (
            f"world_size={world_size} requires runtime.fsdp=true; "
            "unwrapped multi-process training would create divergent models"
        )
        raise ValueError(msg)
    return requested and world_size > 1


def _validate_run_directory(out_dir: Path, *, resume_path: Path | None) -> None:
    """新训练拒绝复用非空目录；显式且已解析的 resume 才可复用。"""
    if resume_path is not None or not out_dir.exists():
        return
    if not out_dir.is_dir():
        msg = f"training output path exists and is not a directory: {out_dir}"
        raise NotADirectoryError(msg)
    try:
        first_entry = next(out_dir.iterdir())
    except StopIteration:
        return
    msg = (
        f"refusing fresh training in non-empty output directory {out_dir}; "
        f"found {first_entry.name!r}. Use --resume with a resumable checkpoint "
        "or choose a new runtime.out_dir"
    )
    raise FileExistsError(msg)


def _prepare_output_directory(
    out_dir: Path,
    config_path: Path,
    *,
    resume_path: Path | None,
) -> None:
    """在所有 resume 校验完成后创建目录，并保持原始配置快照不可变。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = out_dir / "config.snapshot.yaml"
    if resume_path is not None and snapshot.exists():
        logger.info("preserving original config snapshot during resume: %s", snapshot)
        return
    temporary = snapshot.with_name(f".{snapshot.name}.tmp")
    shutil.copyfile(config_path, temporary)
    temporary.replace(snapshot)


def _capture_rng_state() -> dict[str, object]:
    """捕获当前 rank 的 Python/NumPy/Torch RNG，供断点续训恢复。"""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng_state(state: dict[str, object]) -> None:
    """恢复由 :func:`_capture_rng_state` 保存的 RNG 状态。"""
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required - set(state))
    if missing:
        msg = f"checkpoint RNG state is incomplete: missing={missing}"
        raise ValueError(msg)
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(state["torch_cpu"])  # type: ignore[arg-type]
    cuda_states = state["torch_cuda"]
    if torch.cuda.is_available():
        if not isinstance(cuda_states, list):
            msg = "checkpoint CUDA RNG state must be a list"
            raise TypeError(msg)
        if len(cuda_states) != torch.cuda.device_count():
            msg = (
                "checkpoint CUDA RNG topology mismatch: "
                f"saved={len(cuda_states)}, current={torch.cuda.device_count()}"
            )
            raise ValueError(msg)
        torch.cuda.set_rng_state_all(cuda_states)


def _set_data_epoch(
    train_dl: DataLoader,
    epoch: int,
    *,
    seed: int,
    rank: int,
) -> None:
    """使 shard-aware 和普通随机 sampler 都按 epoch 确定性重放。"""
    loader_generator = getattr(train_dl, "generator", None)
    if isinstance(loader_generator, torch.Generator):
        loader_generator.manual_seed(seed + rank + epoch)
    sampler = train_dl.sampler
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
        return
    generator = getattr(sampler, "generator", None)
    if isinstance(generator, torch.Generator):
        generator.manual_seed(seed + rank + epoch)


def _remaining_epoch_batches(
    train_dl: DataLoader,
    batches_consumed: int,
) -> Iterator[dict[str, torch.Tensor]]:
    """确定性跳过 checkpoint 已完成的 epoch 内 batch。"""
    if batches_consumed < 0 or batches_consumed > len(train_dl):
        msg = (
            "checkpoint batches_consumed_in_epoch is outside the current loader: "
            f"consumed={batches_consumed}, batches={len(train_dl)}"
        )
        raise ValueError(msg)
    for batch_index, batch in enumerate(train_dl):
        if batch_index < batches_consumed:
            continue
        yield batch


def _data_order_state(
    train_dl: DataLoader,
    *,
    next_epoch: int,
    batches_consumed_in_epoch: int,
    training_config_sha256: str,
    validation_plan_sha256: str | None = None,
    stop_budgets: dict[str, int],
    seed: int,
    rank: int,
    world_size: int,
) -> dict[str, object]:
    """记录可安全恢复的数据顺序身份；未完成 epoch 会从头确定性重放。"""
    return {
        "sampler_type": type(train_dl.sampler).__name__,
        "next_epoch": int(next_epoch),
        "batches_consumed_in_epoch": int(batches_consumed_in_epoch),
        "training_config_sha256": training_config_sha256,
        "validation_plan_sha256": validation_plan_sha256,
        "stop_budgets": dict(stop_budgets),
        "seed": int(seed),
        "rank": int(rank),
        "world_size": int(world_size),
    }


def _restore_runtime_state(
    checkpoint: dict[str, object],
    train_dl: DataLoader,
    state: TrainState,
    *,
    training_config_sha256: str,
    validation_plan_sha256: str | None = None,
    stop_budgets: dict[str, int],
    seed: int,
    rank: int,
    world_size: int,
    require_runtime_state: bool = False,
) -> None:
    """恢复当前 rank RNG，并校验/恢复确定性 sampler epoch。"""
    saved_by_rank = checkpoint.get("runtime_state_by_rank")
    if saved_by_rank is None:
        if require_runtime_state:
            msg = "Tokenizer v2 resume requires checkpoint RNG/sampler/config state"
            raise ValueError(msg)
        logger.warning(
            "resume checkpoint has no RNG/sampler state; continuation is not "
            "bitwise reproducible"
        )
        return
    if not isinstance(saved_by_rank, list) or len(saved_by_rank) != world_size:
        msg = (
            "checkpoint runtime-state topology mismatch: "
            f"saved={len(saved_by_rank) if isinstance(saved_by_rank, list) else 'invalid'}, "
            f"current={world_size}"
        )
        raise ValueError(msg)
    saved = saved_by_rank[rank]
    if not isinstance(saved, dict):
        msg = f"checkpoint runtime state for rank {rank} is invalid"
        raise TypeError(msg)
    order = saved.get("data_order")
    if not isinstance(order, dict):
        msg = f"checkpoint data-order state for rank {rank} is invalid"
        raise TypeError(msg)
    expected = {
        "sampler_type": type(train_dl.sampler).__name__,
        "next_epoch": state.data_epochs_completed,
        "batches_consumed_in_epoch": state.batches_consumed_in_epoch,
        "training_config_sha256": training_config_sha256,
        "validation_plan_sha256": validation_plan_sha256,
        "seed": seed,
        "rank": rank,
        "world_size": world_size,
    }
    mismatches = {
        key: (order.get(key), value)
        for key, value in expected.items()
        if order.get(key) != value
    }
    if mismatches:
        msg = f"checkpoint sampler state does not match current loader: {mismatches}"
        raise ValueError(msg)
    _validate_stop_budget_extension(order.get("stop_budgets"), stop_budgets)
    rng = saved.get("rng")
    if not isinstance(rng, dict):
        msg = f"checkpoint RNG state for rank {rank} is invalid"
        raise TypeError(msg)
    _set_data_epoch(
        train_dl,
        state.data_epochs_completed,
        seed=seed,
        rank=rank,
    )
    _restore_rng_state(rng)


def _validation_plan_sha256(val_dl: DataLoader | None) -> str | None:
    """Return the canonical identity of the fixed validation window plan."""
    if val_dl is None:
        return None
    sampler = val_dl.sampler
    if not isinstance(sampler, FixedValidationSampler):
        msg = "validation DataLoader must use FixedValidationSampler"
        raise TypeError(msg)
    return sampler.plan.sha256


def _require_exact_validation_windows(
    plan: ValidationSamplePlan,
    *,
    requested_windows: int,
) -> None:
    """Reject configured validation plans that silently select fewer than N windows."""
    if plan.max_windows != requested_windows or len(plan.windows) != requested_windows:
        msg = (
            f"data.validation_windows requires exactly {requested_windows} windows, "
            f"got selection.max_windows={plan.max_windows!r}, "
            f"selected={len(plan.windows)}, candidates={plan.total_candidate_windows}"
        )
        raise ValueError(msg)


def is_main_process(rank: int) -> bool:
    """是否为主进程（负责日志、存盘）。"""
    return rank == 0


def cosine_lr(step: int, *, warmup: int, max_steps: int, base_lr: float) -> float:
    """线性 warmup 后余弦衰减至基础学习率的 10%。"""
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    progress = min(progress, 1.0)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


@dataclass(slots=True)
class TrainState:
    """可变的训练进度。"""

    micro_step: int = 0
    update_step: int = 0
    samples_seen: int = 0
    non_pad_tokens_seen: int = 0
    data_epochs_completed: int = 0
    batches_consumed_in_epoch: int = 0
    best_val: float = float("inf")
    best_update_step: int = -1

    @property
    def step(self) -> int:
        """兼容旧调用：step 现在严格表示 optimizer update。"""
        return self.update_step

    @property
    def best_step(self) -> int:
        """旧日志兼容别名。"""
        return self.best_update_step


def _restore_train_state(
    checkpoint: dict[str, object], *, grad_accum: int
) -> TrainState:
    """恢复新版计数；旧 checkpoint 的 step 被视为 micro-batch 计数。"""
    saved = checkpoint.get("train_state", {})
    if not isinstance(saved, dict):
        saved = {}
    if "update_step" in saved:
        return TrainState(
            micro_step=int(saved.get("micro_step", 0)),
            update_step=int(saved.get("update_step", 0)),
            samples_seen=int(saved.get("samples_seen", 0)),
            non_pad_tokens_seen=int(saved.get("non_pad_tokens_seen", 0)),
            data_epochs_completed=int(saved.get("data_epochs_completed", 0)),
            batches_consumed_in_epoch=int(saved.get("batches_consumed_in_epoch", 0)),
            best_val=float(saved.get("best_val", float("inf"))),
            best_update_step=int(saved.get("best_update_step", -1)),
        )
    legacy_step = int(saved.get("step", checkpoint.get("step", 0)))
    legacy_best = int(saved.get("best_step", -1))
    return TrainState(
        micro_step=legacy_step,
        update_step=legacy_step // max(grad_accum, 1),
        best_val=float(saved.get("best_val", checkpoint.get("val_loss", float("inf")))),
        best_update_step=(
            legacy_best // max(grad_accum, 1) if legacy_best >= 0 else -1
        ),
    )


def _amp_dtype(precision: str) -> torch.dtype | None:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]


def _build_grad_scaler(
    precision: str,
    *,
    use_fsdp: bool,
) -> torch.amp.GradScaler:
    """Build an FP16 scaler whose overflow decision is identical on every rank."""
    if precision == "fp16" and use_fsdp:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        # The ordinary GradScaler only observes this rank's gradient shards.  A
        # rank-local overflow decision could make some ranks skip optimizer.step
        # while peers enter the following collectives.  ShardedGradScaler reduces
        # found_inf across the FSDP process group before making that decision.
        return ShardedGradScaler()
    return torch.amp.GradScaler(enabled=precision == "fp16")


def _fusion_options(model_config: dict) -> dict[str, object]:
    """同时接受 v1 扁平配置和 v2 ``model.field_fusion`` 映射。"""
    raw = model_config.get("field_fusion", "legacy_sum")
    if isinstance(raw, str):
        return {
            "field_fusion": raw,
            "field_dim": int(model_config.get("field_dim", 32)),
            "field_dropout": float(model_config.get("field_dropout", 0.0)),
            "field_input_norm": bool(model_config.get("field_input_norm", True)),
        }
    if not isinstance(raw, dict):
        msg = "model.field_fusion must be a method string or mapping"
        raise TypeError(msg)
    return {
        "field_fusion": str(raw.get("method", "scaled_sum")),
        "field_dim": int(raw.get("field_dim", raw.get("categorical_dim", 32))),
        "field_dropout": float(raw.get("field_dropout", 0.0)),
        "field_input_norm": bool(raw.get("input_norm", True)),
    }


def _compute_loss(
    logits: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    target_fields: tuple[str, ...],
    target_specs: tuple[TargetSpec, ...] | None,
) -> LossOutput:
    """在无 v2 配置时保持旧损失逐位一致。"""
    if target_specs is None:
        return next_event_loss(logits, batch, target_fields)
    return next_event_loss_v2(logits, batch, target_specs)


def _validate_resume_metadata(
    checkpoint: dict,
    expected: OrderFlowFMConfig,
    target_specs: tuple[TargetSpec, ...] | None,
    *,
    require_explicit_data_contract: bool = False,
) -> None:
    """禁止在 resume 时混用 v1/v2 vocab、schema 或字段顺序。"""
    saved = checkpoint.get("config", {})
    saved_vocab_version = str(saved.get("vocab_version", "1.0"))
    if saved_vocab_version != expected.vocab_version:
        msg = (
            f"checkpoint vocab_version={saved_vocab_version} does not match "
            f"current vocab_version={expected.vocab_version}"
        )
        raise ValueError(msg)
    if (
        require_explicit_data_contract
        or expected.vocab_version == VocabV2.VOCAB_VERSION
    ):
        common_checks = {
            "schema_version": expected.schema_version,
            "vocab_sha256": expected.vocab_sha256,
            "event_ordering_version": expected.event_ordering_version,
            "feature_transform_version": expected.feature_transform_version,
        }
        for key, expected_value in common_checks.items():
            legacy_default = {
                "schema_version": "cn_l2_v1",
                "vocab_sha256": "",
                "event_ordering_version": "local_time_v1",
                "feature_transform_version": "ew_vwap_future_backfill_v1",
            }[key]
            if saved.get(key, legacy_default) != expected_value:
                msg = f"checkpoint metadata mismatch for {key}"
                raise ValueError(msg)
    if expected.vocab_version == VocabV2.VOCAB_VERSION:
        if checkpoint.get("fm_artifact_version") != "2.0":
            msg = "v2 checkpoint is missing fm_artifact_version=2.0"
            raise ValueError(msg)
        checks = {
            "field_sizes": expected.field_sizes,
            "input_fields": list(expected.input_fields),
            "target_fields": list(expected.target_fields),
            "field_specs": list(expected.field_specs),
            "d_model": expected.d_model,
            "n_layers": expected.n_layers,
            "n_heads": expected.n_heads,
            "ffn_mult": expected.ffn_mult,
            "ffn_hidden": expected.ffn_hidden,
            "dropout": expected.dropout,
            "max_seq_len": expected.max_seq_len,
            "rope_theta": expected.rope_theta,
            "field_fusion": expected.field_fusion,
            "field_dim": expected.field_dim,
            "field_dropout": expected.field_dropout,
            "field_input_norm": expected.field_input_norm,
            "scalar_fields": expected.scalar_fields,
            "standalone_scalar_fields": list(expected.standalone_scalar_fields),
            "continuous_normalizers": expected.continuous_normalizers,
            "book_state_timing": expected.book_state_timing,
            "context_horizon": expected.context_horizon,
            "pooling_version": expected.pooling_version,
            "pooling_method": expected.pooling_method,
            "pooling_outputs": list(expected.pooling_outputs),
            "pooling_stride": expected.pooling_stride,
            "backbone_moe": expected.backbone_moe.to_dict(),
        }
        for key, expected_value in checks.items():
            if saved.get(key) != expected_value:
                msg = f"v2 checkpoint metadata mismatch for {key}"
                raise ValueError(msg)
        expected_targets = (
            None if target_specs is None else [asdict(spec) for spec in target_specs]
        )
        if checkpoint.get("target_specs") != expected_targets:
            msg = "v2 checkpoint target_specs do not match the current loss config"
            raise ValueError(msg)


def _build_dataloaders(
    manifest: Manifest,
    cfg: dict,
    *,
    vocab: Vocab | VocabV2,
    seed: int,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader | None]:
    data = cfg["data"]
    batch_size = int(
        cfg["optim"].get("micro_batch_size", cfg["optim"].get("batch_size", 1))
    )
    if batch_size < 1:
        msg = f"training batch size must be positive, got {batch_size}"
        raise ValueError(msg)
    grad_accum = int(cfg["optim"].get("grad_accum", 1))
    if grad_accum < 1:
        msg = f"optim.grad_accum must be positive, got {grad_accum}"
        raise ValueError(msg)
    num_workers = int(data["num_workers"])
    if num_workers < 0:
        msg = f"data.num_workers must be non-negative, got {num_workers}"
        raise ValueError(msg)
    dataset_options = {
        "context": data["context"],
        "stride": data["stride"],
        "min_len": data["min_len"],
        "cache_size": data["cache_size"],
    }
    if isinstance(vocab, VocabV2):
        train_ds = EventWindowDatasetV2.from_manifest(
            manifest,
            "train",
            vocab=vocab,
            **dataset_options,
        )
        collate_fn = collate_windows_v2
    else:
        train_ds = EventWindowDataset.from_manifest(
            manifest,
            "train",
            **dataset_options,
        )
        collate_fn = collate_windows
    if train_ds.context < 2 or train_ds.min_len < 2:
        msg = "next-event training requires data.context and data.min_len >= 2"
        raise ValueError(msg)
    if len(train_ds) == 0:
        msg = (
            "training dataset contains no windows; check the train split and "
            "data.context/stride/min_len"
        )
        raise ValueError(msg)
    drop_last = bool(data.get("drop_last", True))
    train_sampler = None
    shuffle = True
    if world_size > 1 or bool(data.get("shard_aware_sampler", False)):
        train_sampler = ShardAwareDistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=drop_last,
            samples_per_rank_multiple=(1 if drop_last else batch_size * grad_accum),
        )
        shuffle = False
    generator = torch.Generator()
    generator.manual_seed(seed + rank)
    worker_options: dict[str, object] = {}
    if num_workers > 0:
        prefetch_factor = int(data.get("prefetch_factor", 2))
        if prefetch_factor < 1:
            msg = f"data.prefetch_factor must be positive, got {prefetch_factor}"
            raise ValueError(msg)
        worker_options = {
            "persistent_workers": bool(data.get("persistent_workers", True)),
            "prefetch_factor": prefetch_factor,
        }
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        **worker_options,
    )
    if len(train_dl) == 0:
        msg = (
            "training DataLoader contains no batches on this rank; reduce the "
            "batch size, set data.drop_last=false, or provide more windows"
        )
        raise ValueError(msg)
    val_dl = None
    val_shards = manifest.split("val")
    if val_shards:
        if isinstance(vocab, VocabV2):
            val_ds = EventWindowDatasetV2(
                val_shards,
                vocab=vocab,
                **dataset_options,
            )
        else:
            val_ds = EventWindowDataset(val_shards, **dataset_options)
        if len(val_ds) == 0:
            msg = (
                "validation dataset contains no windows; check the validation "
                "split and data.context/stride/min_len"
            )
            raise ValueError(msg)
        plan_path = Path(
            data.get(
                "validation_plan",
                Path(cfg["runtime"]["out_dir"]) / "validation_windows.json",
            )
        )
        configured_validation_windows = data.get("validation_windows")
        if configured_validation_windows is not None and (
            type(configured_validation_windows) is not int
            or configured_validation_windows < 1
        ):
            msg = "data.validation_windows must be a positive integer"
            raise ValueError(msg)
        if not plan_path.exists():
            max_batches = int(cfg["runtime"].get("val_max_batches", 50))
            plan = build_validation_sample_plan(
                val_shards,
                context=data["context"],
                stride=data["stride"],
                min_len=data["min_len"],
                seed=int(data.get("validation_seed", cfg["seed"])),
                max_windows=(
                    configured_validation_windows
                    if configured_validation_windows is not None
                    else max_batches * batch_size
                ),
            )
            if configured_validation_windows is not None:
                _require_exact_validation_windows(
                    plan,
                    requested_windows=configured_validation_windows,
                )
            if rank == 0:
                plan.save(plan_path)
                logger.info("created fixed validation plan %s", plan_path)
        if world_size > 1:
            import torch.distributed as dist

            dist.barrier()
        plan = ValidationSamplePlan.load(plan_path)
        plan.validate(
            val_shards,
            context=data["context"],
            stride=data["stride"],
            min_len=data["min_len"],
        )
        if configured_validation_windows is not None:
            _require_exact_validation_windows(
                plan,
                requested_windows=configured_validation_windows,
            )
        val_sampler = FixedValidationSampler(plan)
        if len(val_sampler) == 0:
            msg = "fixed validation plan contains no windows"
            raise ValueError(msg)
        # DataLoader creates an iterator base seed even with num_workers=0.  Without
        # a private generator, every validation pass advances the global torch RNG
        # and changes the next training dropout mask (including across resume).
        val_generator = torch.Generator()
        val_generator.manual_seed(seed + rank + 1_000_003)
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            generator=val_generator,
            pin_memory=torch.cuda.is_available(),
            **worker_options,
        )
    return train_dl, val_dl


@torch.no_grad()
def evaluate(
    model: OrderFlowFM,
    val_dl: DataLoader,
    device: torch.device,
    target_fields: tuple[str, ...],
    *,
    target_specs: tuple[TargetSpec, ...] | None = None,
    max_batches: int | None = None,
    world_size: int = 1,
) -> float:
    """在完整固定计划上返回平均总验证损失（多卡时 all-reduce 平均）。"""
    if max_batches is not None and max_batches < 1:
        msg = "max_batches must be positive when provided"
        raise ValueError(msg)
    if isinstance(getattr(val_dl, "sampler", None), FixedValidationSampler):
        # A persisted plan is the validation identity.  Never reinterpret the
        # legacy batch budget as permission to evaluate only its prefix.
        max_batches = None
    model.eval()
    total, n = 0.0, 0
    raw_sums = dict.fromkeys(target_fields, 0.0)
    ordinal_sums = dict.fromkeys(target_fields, 0.0)
    valid_counts = dict.fromkeys(target_fields, 0.0)
    for i, batch in enumerate(val_dl):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        loss = _compute_loss(logits, batch, target_fields, target_specs)
        total += float(loss.total.item())
        n += 1
        if target_specs is not None:
            for field, count in loss.valid_counts.items():
                raw_sums[field] += float(loss.per_field[field].item()) * count
                ordinal_sums[field] += (
                    float(loss.ordinal_per_field[field].item()) * count
                )
                valid_counts[field] += count
    model.train()
    if target_specs is not None:
        stats = torch.tensor(
            [
                *(raw_sums[field] for field in target_fields),
                *(ordinal_sums[field] for field in target_fields),
                *(valid_counts[field] for field in target_fields),
            ],
            device=device,
            dtype=torch.float64,
        )
        if world_size > 1:
            import torch.distributed as dist

            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        width = len(target_fields)
        raw_values = stats[:width]
        ordinal_values = stats[width : 2 * width]
        count_values = stats[2 * width :]
        by_name = {spec.name: spec for spec in target_specs}
        result = 0.0
        for index, field in enumerate(target_fields):
            count = float(count_values[index].item())
            if count <= 0:
                continue
            spec = by_name[field]
            raw = float(raw_values[index].item()) / count
            ordinal = float(ordinal_values[index].item()) / count
            result += spec.weight * (raw / spec.entropy + spec.ordinal_weight * ordinal)
        return result
    if world_size > 1:
        import torch.distributed as dist

        stats = torch.tensor([total, float(n)], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total, n = float(stats[0].item()), float(stats[1].item())
    return total / max(n, 1.0)


def train(config_path: Path, *, resume: Path | str | None = None) -> None:
    """从 YAML 配置运行预训练。"""
    # [导读] yaml.safe_load 把 config.yaml 读成 Python 字典 cfg
    config_path = Path(config_path)
    cfg = _validate_training_config(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    training_config_sha256 = _immutable_training_config_sha256(cfg)
    stop_budgets = _training_stop_budgets(cfg)
    local_rank, rank, world_size = setup_distributed()
    use_fsdp = _validate_distributed_runtime(cfg, world_size=world_size)
    set_seed(cfg["seed"] + rank)  # 固定随机种子，各 rank 略有偏移
    device = resolve_device(cfg["runtime"]["device"], local_rank=local_rank)

    out_dir = Path(cfg["runtime"]["out_dir"])
    resume_path = _resolve_resume_path(out_dir, resume)
    _validate_run_directory(out_dir, resume_path=resume_path)

    # [导读] manifest 告诉 DataLoader 读哪些 parquet；vocab 告诉模型每个字段词表多大
    manifest_path = Path(cfg["data"]["manifest"])
    manifest = Manifest.load(manifest_path)
    vocab_path = Path(cfg["data"]["vocab"])
    vocab = _load_vocab(vocab_path)
    _validate_config_schema(cfg, vocab)
    validate_manifest_vocab_contract(
        manifest,
        vocab,
        context="pretraining",
    )
    artifact_audit = _validate_v2_artifact_audit(
        cfg,
        manifest_path=manifest_path,
        vocab_path=vocab_path,
        vocab=vocab,
    )
    expected_dates: dict[str, set[str]] = {}
    for split, key in (
        ("train", "train_dates_file"),
        ("val", "validation_dates_file"),
        ("test", "test_dates_file"),
    ):
        if configured_path := cfg["data"].get(key):
            expected_dates[split] = {
                value.strip()
                for value in Path(configured_path)
                .read_text(encoding="utf-8")
                .splitlines()
                if value.strip()
            }
    split_contract = validate_pretrain_split_contract(
        manifest,
        vocab,
        require_validation=bool(cfg["runtime"].get("save_best", True)),
        min_validation_dates=int(cfg["data"].get("min_validation_dates", 5)),
        min_test_dates=int(cfg["data"].get("min_test_dates", 5)),
        expected_dates=expected_dates,
    )
    if is_main_process(rank):
        logger.info("pretraining split contract: %s", split_contract)
    shard_contract = _validate_training_shards(
        manifest,
        vocab,
        expected_tokens_root=(
            manifest_path.parent.parent / "tokens"
            if manifest_path.parent.name == "data"
            else manifest_path.parent / "tokens"
        ),
        rank=rank,
        world_size=world_size,
    )
    pretrain_data_contract = build_pretrain_data_contract(
        manifest_path=manifest_path,
        manifest=manifest,
        vocab_path=vocab_path,
        vocab=vocab,
    )
    if is_main_process(rank):
        logger.info("pretraining shard contract: %s", shard_contract)
        logger.info("pretraining data contract: %s", pretrain_data_contract)
        if artifact_audit is not None:
            logger.info("pretraining V2 artifact audit: %s", artifact_audit)

    mcfg = cfg["model"]
    if isinstance(vocab, VocabV2):
        layout = field_layout_from_vocab(vocab)
        input_fields = tuple(mcfg.get("input_fields", layout.input_fields))
        target_fields = tuple(mcfg.get("target_fields", layout.target_fields))
        scalar_fields = dict(layout.scalar_to_token)
        standalone_scalar_fields = layout.standalone_scalar_fields
        field_sizes = {
            str(spec.token_column): vocab.size(spec.name)
            for spec in vocab.field_specs
            if spec.token_column is not None
        }
        loss_config = dict(cfg.get("loss", {}))
        if loss_config.get("normalize_by_train_entropy", True):
            configured_entropy = dict(loss_config.get("train_entropy", {}))
            for field in target_fields:
                configured_entropy.setdefault(
                    field,
                    max(vocab.train_entropy(field), 1e-6),
                )
            loss_config["train_entropy"] = configured_entropy
        target_specs = target_specs_from_config(
            target_fields,
            loss_config,
            default_ignore_ids=(V2_PAD_ID, V2_NA_ID),
            default_ordinal_start_id=V2_N_SPECIAL,
        )
        if target_specs is None:
            msg = "Tokenizer v2 training requires explicit loss.targets"
            raise ValueError(msg)
        schema_version = vocab.schema_version
        vocab_version = vocab.VOCAB_VERSION
        frozen_field_specs = tuple(spec.to_dict() for spec in vocab.field_specs)
        continuous_normalizers = {
            str(spec.value_column): vocab.binned[spec.name].normalizer.to_dict()
            for spec in vocab.field_specs
            if spec.value_column is not None and spec.name in vocab.binned
        }
    else:
        input_fields = tuple(mcfg.get("input_fields", FIELD_ORDER))
        target_fields = tuple(mcfg.get("target_fields", DEFAULT_TARGET_FIELDS))
        scalar_fields = {}
        standalone_scalar_fields = ()
        field_sizes = field_sizes_from_vocab(vocab)
        target_specs = target_specs_from_config(target_fields, cfg.get("loss"))
        schema_version = vocab.schema_version
        vocab_version = "1.0"
        frozen_field_specs = ()
        continuous_normalizers = {}
    pooling_cfg = cfg.get("pooling", {})
    model_cfg = OrderFlowFMConfig(
        field_sizes=field_sizes,
        input_fields=input_fields,
        target_fields=target_fields,
        d_model=mcfg["d_model"],
        n_layers=mcfg["n_layers"],
        n_heads=mcfg["n_heads"],
        ffn_mult=float(mcfg.get("ffn_mult", 4.0)),
        ffn_hidden=(
            None if mcfg.get("ffn_hidden") is None else int(mcfg["ffn_hidden"])
        ),
        dropout=mcfg["dropout"],
        max_seq_len=mcfg["max_seq_len"],
        rope_theta=mcfg["rope_theta"],
        scalar_fields=scalar_fields,
        standalone_scalar_fields=standalone_scalar_fields,
        schema_version=schema_version,
        vocab_version=vocab_version,
        vocab_sha256=stable_vocab_sha256(vocab),
        field_specs=frozen_field_specs,
        continuous_normalizers=continuous_normalizers,
        book_state_timing=str(cfg.get("book", {}).get("state_timing", "none")),
        context_horizon=int(cfg["data"]["context"]),
        pooling_version=str(pooling_cfg.get("version", "flat_v1")),
        pooling_method=str(pooling_cfg.get("method", "mean")),
        pooling_outputs=tuple(str(value) for value in pooling_cfg.get("outputs", ())),
        pooling_stride=int(pooling_cfg.get("stride", cfg["data"]["context"])),
        event_ordering_version=str(
            getattr(vocab, "event_ordering_version", "local_time_v1")
        ),
        feature_transform_version=str(
            getattr(
                vocab,
                "feature_transform_version",
                "ew_vwap_future_backfill_v1",
            )
        ),
        backbone_moe=BackboneMoEConfig.from_dict(mcfg.get("backbone_moe")),
        **_fusion_options(mcfg),
    )
    model = OrderFlowFM(model_cfg).to(device)  # .to(device) 把模型放到 CPU/GPU
    resume_ckpt = None
    if resume_path is not None:
        resume_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        if "optimizer_state" not in resume_ckpt:
            msg = (
                f"checkpoint {resume_path} is inference-only and cannot resume "
                "training; use step*.pt or final_resume.pt"
            )
            raise ValueError(msg)
        resume_contract = load_checkpoint_contract(
            resume_path,
            required=vocab.data_semantics_explicit,
        )
        validate_checkpoint_data_contract(
            resume_contract,
            checkpoint_payload=resume_ckpt,
            vocab=vocab,
            expected_pretrain_data_contract=pretrain_data_contract,
        )
        _validate_resume_metadata(
            resume_ckpt,
            model_cfg,
            target_specs,
            require_explicit_data_contract=vocab.data_semantics_explicit,
        )
        model.load_state_dict(resume_ckpt["model_state"])
        if is_main_process(rank):
            logger.info("loaded model checkpoint %s", resume_path)

    if is_main_process(rank):
        logger.info("model parameters: %.2fM", model.num_parameters() / 1e6)
        if use_fsdp:
            logger.info("FSDP enabled across %d ranks", world_size)

    if use_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        model = FSDP(model, device_id=local_rank)

    opt = cfg["optim"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt["lr"],
        weight_decay=opt["weight_decay"],
        betas=tuple(opt["betas"]),
    )
    amp_dtype = _amp_dtype(opt["precision"])
    scaler = _build_grad_scaler(opt["precision"], use_fsdp=use_fsdp)

    state = TrainState()
    accum = int(opt["grad_accum"])
    if accum < 1:
        msg = f"optim.grad_accum must be positive, got {accum}"
        raise ValueError(msg)
    if resume_ckpt is not None:
        optimizer_state = resume_ckpt.get("optimizer_state")
        if optimizer_state is not None:
            if use_fsdp:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                full_state = optimizer_state if is_main_process(rank) else None
                optimizer_state = FSDP.scatter_full_optim_state_dict(
                    full_state, model, optim=optimizer
                )
            optimizer.load_state_dict(optimizer_state)
        if "scaler_state" in resume_ckpt:
            scaler.load_state_dict(resume_ckpt["scaler_state"])
        state = _restore_train_state(resume_ckpt, grad_accum=accum)
        if state.micro_step % accum:
            msg = (
                "resume checkpoint was saved mid gradient-accumulation cycle and "
                "does not contain pending gradients"
            )
            raise ValueError(msg)
        if is_main_process(rank):
            logger.info(
                "resumed training at update %d / micro %d (best %.4f @ %d)",
                state.update_step,
                state.micro_step,
                state.best_val,
                state.best_update_step,
            )

    max_update_steps = int(opt.get("max_update_steps", opt.get("max_steps", 0)))
    max_train_tokens = int(opt.get("max_train_tokens", 0))
    max_data_epochs = int(opt.get("max_data_epochs", 0))
    if max_update_steps < 1 and max_train_tokens < 1 and max_data_epochs < 1:
        msg_0 = (
            "optim requires max_update_steps/max_steps, max_train_tokens, "
            "or max_data_epochs"
        )
        raise ValueError(msg_0)
    schedule_update_steps = int(opt.get("lr_schedule_steps", max_update_steps))
    if schedule_update_steps < 1:
        msg_1 = (
            "token-budget-only training requires optim.lr_schedule_steps so the "
            "cosine schedule has a fixed horizon"
        )
        raise ValueError(msg_1)

    train_dl, val_dl = _build_dataloaders(
        manifest,
        cfg,
        vocab=vocab,
        seed=cfg["seed"],
        rank=rank,
        world_size=world_size,
    )
    validation_plan_sha256 = _validation_plan_sha256(val_dl)
    if max_data_epochs > 0 and len(train_dl) % accum:
        msg = (
            "epoch-budget training requires the per-rank DataLoader batch count "
            f"to be divisible by optim.grad_accum; batches={len(train_dl)}, "
            f"grad_accum={accum}"
        )
        raise ValueError(msg)

    if resume_ckpt is not None:
        _restore_runtime_state(
            resume_ckpt,
            train_dl,
            state,
            training_config_sha256=training_config_sha256,
            validation_plan_sha256=validation_plan_sha256,
            stop_budgets=stop_budgets,
            seed=cfg["seed"],
            rank=rank,
            world_size=world_size,
            require_runtime_state=isinstance(vocab, VocabV2),
        )
        del resume_ckpt

    output_error: str | None = None
    if is_main_process(rank):
        try:
            _prepare_output_directory(
                out_dir,
                config_path,
                resume_path=resume_path,
            )
        except OSError as exc:
            output_error = f"{type(exc).__name__}: {exc}"
    if world_size > 1:
        import torch.distributed as dist

        payload: list[object] = [output_error]
        dist.broadcast_object_list(payload, src=0)
        output_error = payload[0] if isinstance(payload[0], str) else None
    if output_error is not None:
        msg = f"failed to prepare training output directory: {output_error}"
        raise OSError(msg)

    writer = None
    if is_main_process(rank):
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(str(out_dir / "tb"))
        except ImportError:
            logger.warning("tensorboard unavailable; skipping scalar logging")

    def checkpoint_data_order_state() -> dict[str, object]:
        return _data_order_state(
            train_dl,
            next_epoch=state.data_epochs_completed,
            batches_consumed_in_epoch=state.batches_consumed_in_epoch,
            training_config_sha256=training_config_sha256,
            validation_plan_sha256=validation_plan_sha256,
            stop_budgets=stop_budgets,
            seed=cfg["seed"],
            rank=rank,
            world_size=world_size,
        )

    model.train()  # 训练模式（启用 dropout 等）
    optimizer.zero_grad(set_to_none=True)
    pending_samples = 0
    pending_tokens = 0

    def training_complete() -> bool:
        updates_done = max_update_steps > 0 and state.update_step >= max_update_steps
        tokens_done = (
            max_train_tokens > 0 and state.non_pad_tokens_seen >= max_train_tokens
        )
        epochs_done = (
            max_data_epochs > 0 and state.data_epochs_completed >= max_data_epochs
        )
        return updates_done or tokens_done or epochs_done

    # 调度、日志、验证与存盘统一使用 optimizer update 计数。
    while not training_complete():
        _set_data_epoch(
            train_dl,
            state.data_epochs_completed,
            seed=cfg["seed"],
            rank=rank,
        )
        for batch in _remaining_epoch_batches(
            train_dl,
            state.batches_consumed_in_epoch,
        ):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            lr = cosine_lr(
                state.update_step,
                warmup=opt["warmup_steps"],
                max_steps=schedule_update_steps,
                base_lr=opt["lr"],
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            should_update = (state.micro_step + 1) % accum == 0
            sync_context = nullcontext()
            if (
                use_fsdp
                and not should_update
                and bool(cfg["runtime"].get("fsdp_no_sync", True))
                and hasattr(model, "no_sync")
            ):
                sync_context = model.no_sync()
            use_amp = amp_dtype is not None and device.type == "cuda"
            with sync_context:
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    logits = model(batch)
                    loss_out = _compute_loss(logits, batch, target_fields, target_specs)
                    raw_model = model.module if hasattr(model, "module") else model
                    moe_aux = raw_model.moe_auxiliary_loss()
                    auxiliary = (
                        loss_out.total.new_zeros(()) if moe_aux is None else moe_aux
                    )
                    loss = (loss_out.total + auxiliary) / accum
                scaler.scale(loss).backward()

            state.micro_step += 1
            state.batches_consumed_in_epoch += 1
            pending_samples += int(batch["attention_mask"].size(0))
            pending_tokens += int(batch["attention_mask"].sum().item())

            if should_update:
                scaler.unscale_(optimizer)
                if use_fsdp:
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                    FSDP.clip_grad_norm_(model, opt["grad_clip"])
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt["grad_clip"])
                scale_before = scaler.get_scale()
                scaler.step(optimizer)  # 用梯度更新权重
                scaler.update()
                step_succeeded = (
                    not scaler.is_enabled() or scaler.get_scale() >= scale_before
                )
                optimizer.zero_grad(set_to_none=True)  # 清空梯度，准备下一步
                attempted_samples = pending_samples
                attempted_tokens = pending_tokens
                pending_samples = pending_tokens = 0
                if not step_succeeded:
                    if is_main_process(rank):
                        logger.warning(
                            "skipped fp16 optimizer update after overflow at micro %d",
                            state.micro_step,
                        )
                    continue
                state.update_step += 1
                counts = torch.tensor(
                    [attempted_samples, attempted_tokens],
                    dtype=torch.long,
                    device=device,
                )
                if world_size > 1:
                    import torch.distributed as dist

                    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                state.samples_seen += int(counts[0].item())
                state.non_pad_tokens_seen += int(counts[1].item())
            else:
                continue

            if (
                is_main_process(rank)
                and state.update_step % cfg["runtime"]["log_every"] == 0
            ):
                logger.info(
                    "update %d micro %d tokens %d lr %.2e loss %.4f aux %.4f",
                    state.update_step,
                    state.micro_step,
                    state.non_pad_tokens_seen,
                    lr,
                    float(loss_out.total.item()),
                    float(auxiliary.detach().item()),
                )
                if writer is not None:
                    writer.add_scalar(
                        "train/loss", loss_out.total.item(), state.update_step
                    )
                    writer.add_scalar("train/lr", lr, state.update_step)
                    writer.add_scalar(
                        "train/moe_aux", auxiliary.detach().item(), state.update_step
                    )
                    for f, fl in loss_out.per_field.items():
                        writer.add_scalar(f"train/ce_{f}", fl.item(), state.update_step)
                    for field, value in loss_out.normalized_per_field.items():
                        writer.add_scalar(
                            f"train/normalized_ce_{field}",
                            value.item(),
                            state.update_step,
                        )
                    for field, value in loss_out.ordinal_per_field.items():
                        writer.add_scalar(
                            f"train/ordinal_{field}",
                            value.item(),
                            state.update_step,
                        )
                    for field, count in loss_out.valid_counts.items():
                        writer.add_scalar(
                            f"train/valid_targets_{field}",
                            count,
                            state.update_step,
                        )

            if (
                val_dl is not None
                and state.update_step % cfg["runtime"]["eval_every"] == 0
            ):
                val_loss = evaluate(
                    model,
                    val_dl,
                    device,
                    target_fields,
                    target_specs=target_specs,
                    max_batches=None,
                    world_size=world_size,
                )
                if is_main_process(rank):
                    logger.info("update %d val_loss %.4f", state.update_step, val_loss)
                    if writer is not None:
                        writer.add_scalar("val/loss", val_loss, state.update_step)
                if cfg["runtime"].get("save_best", True):
                    _maybe_save_best(
                        model,
                        model_cfg,
                        out_dir,
                        state,
                        val_loss=val_loss,
                        target_specs=target_specs,
                        pretrain_data_contract=pretrain_data_contract,
                        data_order_state=checkpoint_data_order_state(),
                        rank=rank,
                        writer=writer,
                    )

            if state.update_step % cfg["runtime"]["ckpt_every"] == 0:
                _save_checkpoint(
                    model,
                    model_cfg,
                    out_dir / f"step{state.update_step}.pt",
                    optimizer=optimizer,
                    scaler=scaler,
                    train_state=state,
                    target_specs=target_specs,
                    pretrain_data_contract=pretrain_data_contract,
                    data_order_state=checkpoint_data_order_state(),
                    rank=rank,
                    step=state.update_step,
                )

            if training_complete():
                break
        else:
            # 只有自然耗尽整个 DataLoader 才记为完整 epoch。中途按 token/update
            # 预算停止时不会伪造全量覆盖；checkpoint 会记录 epoch 内 batch offset，
            # 恢复时按同一 sampler 顺序精确跳过已完成 batch。
            state.data_epochs_completed += 1
            state.batches_consumed_in_epoch = 0
            if is_main_process(rank):
                logger.info(
                    "completed data epoch %d/%s",
                    state.data_epochs_completed,
                    max_data_epochs if max_data_epochs > 0 else "unbounded",
                )

    # 非周期的终点验证只是诊断，不参与 best checkpoint 选择。否则
    # 先跑短预算、再延长预算会额外让中间终点参与选模，使分段续训
    # 与一次性连续训练产生不同的 best.pt。刚刚完成周期验证时也无需重复。
    final_needs_diagnostic = (
        val_dl is not None
        and cfg["runtime"].get("save_best", True)
        and state.update_step % int(cfg["runtime"]["eval_every"]) != 0
    )
    if final_needs_diagnostic:
        val_loss = evaluate(
            model,
            val_dl,
            device,
            target_fields,
            target_specs=target_specs,
            max_batches=None,
            world_size=world_size,
        )
        if is_main_process(rank):
            logger.info(
                "final diagnostic val_loss %.4f (scheduled best %.4f @ step %d)",
                val_loss,
                state.best_val,
                state.best_update_step,
            )
            if writer is not None:
                writer.add_scalar(
                    "val/final_diagnostic_loss",
                    val_loss,
                    state.update_step,
                )

    _save_checkpoint(
        model,
        model_cfg,
        out_dir / "final_resume.pt",
        optimizer=optimizer,
        scaler=scaler,
        train_state=state,
        target_specs=target_specs,
        pretrain_data_contract=pretrain_data_contract,
        data_order_state=checkpoint_data_order_state(),
        rank=rank,
        step=state.update_step,
        val_loss=state.best_val if state.best_update_step >= 0 else None,
    )
    _save_checkpoint(
        model,
        model_cfg,
        out_dir / "final.pt",
        train_state=state,
        target_specs=target_specs,
        pretrain_data_contract=pretrain_data_contract,
        data_order_state=checkpoint_data_order_state(),
        rank=rank,
        step=state.update_step,
        val_loss=state.best_val if state.best_update_step >= 0 else None,
    )
    if is_main_process(rank):
        if writer is not None:
            writer.close()
        if state.best_update_step >= 0:
            logger.info(
                "training complete: %d updates; best.pt from update %d (val_loss=%.4f)",
                state.update_step,
                state.best_update_step,
                state.best_val,
            )
        else:
            logger.info(
                "training complete: %d updates; no val set → final artifacts saved",
                state.update_step,
            )

    if world_size > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


def _model_state_dict(model: OrderFlowFM) -> dict[str, torch.Tensor]:
    """从裸模型或 FSDP 包装中提取完整 state_dict。"""
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(model, FSDP):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            return model.state_dict()
    raw = model.module if hasattr(model, "module") else model
    return raw.state_dict()


def _maybe_save_best(
    model: OrderFlowFM,
    model_cfg: OrderFlowFMConfig,
    out_dir: Path,
    state: TrainState,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    target_specs: tuple[TargetSpec, ...] | None = None,
    pretrain_data_contract: dict[str, object] | None = None,
    data_order_state: dict[str, object] | None = None,
    val_loss: float,
    rank: int,
    writer=None,
) -> bool:
    """若 ``val_loss`` 刷新最优，则自动覆盖写入 ``best.pt``。返回是否保存。"""
    improved = val_loss < state.best_val
    # FSDP 需要所有 rank 一起 gather state_dict；真正写盘只在 rank0
    if improved:
        prev = state.best_val
        state.best_val = val_loss
        state.best_update_step = state.update_step
        _save_checkpoint(
            model,
            model_cfg,
            out_dir / "best.pt",
            train_state=state,
            target_specs=target_specs,
            pretrain_data_contract=pretrain_data_contract,
            data_order_state=data_order_state,
            rank=rank,
            step=state.update_step,
            val_loss=val_loss,
            is_best=True,
        )
        if is_main_process(rank):
            logger.info(
                "new best checkpoint → %s (step %d, val_loss %.4f ← %.4f)",
                out_dir / "best.pt",
                state.update_step,
                val_loss,
                prev,
            )
            if writer is not None:
                writer.add_scalar("val/best_loss", val_loss, state.update_step)
        return True
    return False


def _save_checkpoint(
    model: OrderFlowFM,
    cfg: OrderFlowFMConfig,
    path: Path,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    train_state: TrainState | None = None,
    target_specs: tuple[TargetSpec, ...] | None = None,
    pretrain_data_contract: dict[str, object] | None = None,
    data_order_state: dict[str, object] | None = None,
    rank: int = 0,
    step: int | None = None,
    val_loss: float | None = None,
    is_best: bool = False,
) -> None:
    """
    保存模型权重与配置，便于可复现加载。

    FSDP 下所有 rank 都必须进入 state_dict 收集；只有 rank0 写文件。
    """
    local_runtime_state = {
        "rng": _capture_rng_state(),
        "data_order": dict(data_order_state or {}),
    }
    runtime_state_by_rank: list[object]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        runtime_state_by_rank = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(
            runtime_state_by_rank,
            local_runtime_state,
        )
    else:
        runtime_state_by_rank = [local_runtime_state]
    state = _model_state_dict(model)
    optimizer_state = None
    if optimizer is not None:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(model, FSDP):
            optimizer_state = FSDP.full_optim_state_dict(
                model, optimizer, rank0_only=True
            )
        elif is_main_process(rank):
            optimizer_state = optimizer.state_dict()
    if not is_main_process(rank):
        return
    payload: dict = {
        "model_state": state,
        "fm_artifact_version": (
            "2.0" if cfg.vocab_version == VocabV2.VOCAB_VERSION else "1.0"
        ),
        "config": {
            "field_sizes": cfg.field_sizes,
            "input_fields": list(cfg.input_fields),
            "target_fields": list(cfg.target_fields),
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "ffn_mult": cfg.ffn_mult,
            "ffn_hidden": cfg.ffn_hidden,
            "dropout": cfg.dropout,
            "max_seq_len": cfg.max_seq_len,
            "rope_theta": cfg.rope_theta,
            "field_fusion": cfg.field_fusion,
            "field_dim": cfg.field_dim,
            "field_dropout": cfg.field_dropout,
            "field_input_norm": cfg.field_input_norm,
            "scalar_fields": cfg.scalar_fields,
            "standalone_scalar_fields": list(cfg.standalone_scalar_fields),
            "schema_version": cfg.schema_version,
            "vocab_version": cfg.vocab_version,
            "vocab_sha256": cfg.vocab_sha256,
            "field_specs": list(cfg.field_specs),
            "continuous_normalizers": cfg.continuous_normalizers,
            "book_state_timing": cfg.book_state_timing,
            "context_horizon": cfg.context_horizon,
            "pooling_version": cfg.pooling_version,
            "pooling_method": cfg.pooling_method,
            "pooling_outputs": list(cfg.pooling_outputs),
            "pooling_stride": cfg.pooling_stride,
            "event_ordering_version": cfg.event_ordering_version,
            "feature_transform_version": cfg.feature_transform_version,
            "backbone_moe": cfg.backbone_moe.to_dict(),
        },
        "runtime_state_by_rank": runtime_state_by_rank,
    }
    if target_specs is not None:
        payload["target_specs"] = [asdict(spec) for spec in target_specs]
    if pretrain_data_contract is not None:
        payload["pretrain_data_contract"] = dict(pretrain_data_contract)
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    if train_state is not None:
        payload["train_state"] = {
            "micro_step": train_state.micro_step,
            "update_step": train_state.update_step,
            "samples_seen": train_state.samples_seen,
            "non_pad_tokens_seen": train_state.non_pad_tokens_seen,
            "data_epochs_completed": train_state.data_epochs_completed,
            "batches_consumed_in_epoch": train_state.batches_consumed_in_epoch,
            "best_val": train_state.best_val,
            "best_update_step": train_state.best_update_step,
        }
    if step is not None:
        payload["step"] = step
    if val_loss is not None:
        payload["val_loss"] = float(val_loss)
    if is_best:
        payload["is_best"] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    if pretrain_data_contract is not None:
        write_checkpoint_contract(
            path,
            config=payload["config"],
            pretrain_data_contract=pretrain_data_contract,
        )
    logger.info("saved checkpoint %s", path)


def load_checkpoint(
    path: Path,
    device: torch.device,
    *,
    vocab_path: Path | None = None,
    expected_pretrain_data_contract: dict[str, object] | None = None,
    checkpoint_sha256: str | None = None,
    allow_missing_pretrain_data_contract: bool = False,
) -> OrderFlowFM:
    """重建模型；v2 必须提供词表并验证 hash/schema/字段顺序。"""
    provided_vocab = _load_vocab(vocab_path) if vocab_path is not None else None
    checkpoint_contract = load_checkpoint_contract(
        path,
        expected_checkpoint_sha256=checkpoint_sha256,
        required=(
            provided_vocab is not None
            and provided_vocab.data_semantics_explicit
            and not allow_missing_pretrain_data_contract
        ),
    )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    c = ckpt["config"]
    if str(c.get("vocab_version", "1.0")) == VocabV2.VOCAB_VERSION:
        if ckpt.get("fm_artifact_version") != "2.0":
            msg = "v2 checkpoint is missing fm_artifact_version=2.0"
            raise ValueError(msg)
        if vocab_path is None:
            msg = "loading a v2 checkpoint requires vocab_path for compatibility checks"
            raise ValueError(msg)
        vocab = provided_vocab
        if not isinstance(vocab, VocabV2):
            msg = "a v2 checkpoint cannot be loaded with a v1 vocab"
            raise ValueError(msg)
        vocab_hash = stable_vocab_sha256(vocab)
        layout = field_layout_from_vocab(vocab)
        checks = {
            "schema_version": vocab.schema_version,
            "vocab_sha256": vocab_hash,
            "field_specs": [spec.to_dict() for spec in vocab.field_specs],
            "event_ordering_version": vocab.event_ordering_version,
            "feature_transform_version": vocab.feature_transform_version,
        }
        for key, expected_value in checks.items():
            legacy_default = {
                "event_ordering_version": "local_time_v1",
                "feature_transform_version": "ew_vwap_future_backfill_v1",
            }.get(key)
            actual_value = c.get(key, legacy_default)
            if actual_value != expected_value:
                msg = f"v2 checkpoint/vocab mismatch for {key}"
                raise ValueError(msg)
        for key, available in (
            ("input_fields", layout.input_fields),
            ("target_fields", layout.target_fields),
        ):
            selected = tuple(c.get(key, ()))
            selected_set = set(selected)
            expected_order = tuple(
                field for field in available if field in selected_set
            )
            if selected != expected_order:
                msg = f"v2 checkpoint has invalid {key} order or unknown fields"
                raise ValueError(msg)
    elif provided_vocab is not None:
        if not isinstance(provided_vocab, Vocab):
            msg = "a v1 checkpoint cannot be loaded with a v2 vocab"
            raise ValueError(msg)
        if provided_vocab.data_semantics_explicit:
            v1_checks = {
                "schema_version": provided_vocab.schema_version,
                "vocab_sha256": stable_vocab_sha256(provided_vocab),
                "event_ordering_version": provided_vocab.event_ordering_version,
                "feature_transform_version": provided_vocab.feature_transform_version,
            }
            for key, expected_value in v1_checks.items():
                if c.get(key) != expected_value:
                    msg = f"v1 checkpoint/vocab mismatch for {key}"
                    raise ValueError(msg)
    if provided_vocab is not None:
        if checkpoint_contract is None and allow_missing_pretrain_data_contract:
            logger.warning(
                "loading checkpoint without pretrain data contract for explicit "
                "legacy/diagnostic use only: %s",
                path,
            )
        else:
            validate_checkpoint_data_contract(
                checkpoint_contract,
                checkpoint_payload=ckpt,
                vocab=provided_vocab,
                expected_pretrain_data_contract=expected_pretrain_data_contract,
            )
    pooling_version = str(c.get("pooling_version", "flat_v1"))
    cfg = OrderFlowFMConfig(
        field_sizes=c["field_sizes"],
        input_fields=tuple(c["input_fields"]),
        target_fields=tuple(c["target_fields"]),
        d_model=c["d_model"],
        n_layers=c["n_layers"],
        n_heads=c["n_heads"],
        ffn_mult=c["ffn_mult"],
        ffn_hidden=c.get("ffn_hidden"),
        dropout=c["dropout"],
        max_seq_len=c["max_seq_len"],
        rope_theta=c["rope_theta"],
        field_fusion=c.get("field_fusion", "legacy_sum"),
        field_dim=int(c.get("field_dim", 32)),
        field_dropout=float(c.get("field_dropout", 0.0)),
        field_input_norm=bool(c.get("field_input_norm", True)),
        scalar_fields={
            str(field): str(token)
            for field, token in c.get("scalar_fields", {}).items()
        },
        standalone_scalar_fields=tuple(c.get("standalone_scalar_fields", ())),
        schema_version=str(c.get("schema_version", "cn_l2_v1")),
        vocab_version=str(c.get("vocab_version", "1.0")),
        vocab_sha256=str(c.get("vocab_sha256", "")),
        field_specs=tuple(c.get("field_specs", ())),
        continuous_normalizers=dict(c.get("continuous_normalizers", {})),
        book_state_timing=str(c.get("book_state_timing", "none")),
        context_horizon=int(c.get("context_horizon", 0)),
        pooling_version=pooling_version,
        pooling_method=str(
            c.get(
                "pooling_method",
                "multi_scale" if pooling_version == "hierarchical_v1" else "mean",
            )
        ),
        pooling_outputs=tuple(c.get("pooling_outputs", ())),
        pooling_stride=int(c.get("pooling_stride", c.get("context_horizon", 0))),
        event_ordering_version=str(c.get("event_ordering_version", "local_time_v1")),
        feature_transform_version=str(
            c.get(
                "feature_transform_version",
                "ew_vwap_future_backfill_v1",
            )
        ),
        backbone_moe=BackboneMoEConfig.from_dict(c.get("backbone_moe")),
    )
    model = OrderFlowFM(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def _resolve_resume_path(out_dir: Path, resume: Path | str | None) -> Path | None:
    """解析显式 checkpoint，或在 ``auto`` 模式下选择最新定期 checkpoint。"""
    if resume is None:
        return None
    if str(resume) != "auto":
        path = Path(resume)
        if not path.is_file():
            msg = f"resume checkpoint not found: {path}"
            raise FileNotFoundError(msg)
        return path

    periodic = sorted(
        out_dir.glob("step*.pt"),
        key=lambda p: int(p.stem.removeprefix("step")),
    )
    if periodic:
        return periodic[-1]
    for name in ("final_resume.pt",):
        path = out_dir / name
        if path.is_file():
            return path
    logger.info(
        "resume=auto: no resumable step*.pt/final_resume.pt under %s; starting fresh",
        out_dir,
    )
    return None


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="checkpoint 路径，或 auto（自动选择 out_dir 下最新 checkpoint）",
    )
    args = parser.parse_args()
    train(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
