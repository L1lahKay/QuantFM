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
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from quant_fm.manifest.build_manifest import Manifest
from quant_fm.moe.config import BackboneMoEConfig
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
from quant_fm.tokenizer.vocab import Vocab
from quant_fm.tokenizer.vocab_v2 import N_SPECIAL as V2_N_SPECIAL
from quant_fm.tokenizer.vocab_v2 import NA_ID as V2_NA_ID
from quant_fm.tokenizer.vocab_v2 import PAD_ID as V2_PAD_ID
from quant_fm.tokenizer.vocab_v2 import VocabV2

if TYPE_CHECKING:
    from quant_fm.pretrain.heads import LossOutput, TargetSpec

logger = logging.getLogger(__name__)


def _load_vocab(path: Path) -> Vocab | VocabV2:
    """根据显式 artifact 版本选择 v1/v2 loader，拒绝静默混用。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("vocab_version") == VocabV2.VOCAB_VERSION:
        return VocabV2.load(path)
    return Vocab.load(path)


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
    if expected.vocab_version == VocabV2.VOCAB_VERSION:
        if checkpoint.get("fm_artifact_version") != "2.0":
            msg = "v2 checkpoint is missing fm_artifact_version=2.0"
            raise ValueError(msg)
        checks = {
            "field_sizes": expected.field_sizes,
            "schema_version": expected.schema_version,
            "vocab_sha256": expected.vocab_sha256,
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
    train_sampler = None
    shuffle = True
    if world_size > 1 or bool(data.get("shard_aware_sampler", False)):
        train_sampler = ShardAwareDistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
        shuffle = False
    generator = torch.Generator()
    generator.manual_seed(seed + rank)
    worker_options: dict[str, object] = {}
    if int(data["num_workers"]) > 0:
        worker_options = {
            "persistent_workers": bool(data.get("persistent_workers", True)),
            "prefetch_factor": int(data.get("prefetch_factor", 2)),
        }
    batch_size = int(
        cfg["optim"].get("micro_batch_size", cfg["optim"].get("batch_size", 1))
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=data["num_workers"],
        drop_last=True,
        generator=generator if train_sampler is None else None,
        pin_memory=torch.cuda.is_available(),
        **worker_options,
    )
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
        plan_path = Path(
            data.get(
                "validation_plan",
                Path(cfg["runtime"]["out_dir"]) / "validation_windows.json",
            )
        )
        if rank == 0 and not plan_path.exists():
            max_batches = int(cfg["runtime"].get("val_max_batches", 50))
            plan = build_validation_sample_plan(
                val_shards,
                context=data["context"],
                stride=data["stride"],
                min_len=data["min_len"],
                seed=int(data.get("validation_seed", cfg["seed"])),
                max_windows=int(
                    data.get(
                        "validation_windows",
                        max_batches * batch_size,
                    )
                ),
            )
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
        val_sampler = FixedValidationSampler(plan)
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=data["num_workers"],
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
    max_batches: int = 50,
    world_size: int = 1,
) -> float:
    """在若干验证批上返回平均总验证损失（多卡时 all-reduce 平均）。"""
    model.eval()
    total, n = 0.0, 0
    raw_sums = dict.fromkeys(target_fields, 0.0)
    ordinal_sums = dict.fromkeys(target_fields, 0.0)
    valid_counts = dict.fromkeys(target_fields, 0.0)
    for i, batch in enumerate(val_dl):
        if i >= max_batches:
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
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    local_rank, rank, world_size = setup_distributed()
    set_seed(cfg["seed"] + rank)  # 固定随机种子，各 rank 略有偏移
    device = resolve_device(cfg["runtime"]["device"], local_rank=local_rank)

    out_dir = Path(cfg["runtime"]["out_dir"])
    if is_main_process(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(config_path, out_dir / "config.snapshot.yaml")  # 存档配置

    if world_size > 1:
        import torch.distributed as dist

        dist.barrier()

    writer = None
    if is_main_process(rank):
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(str(out_dir / "tb"))
        except ImportError:
            logger.warning("tensorboard unavailable; skipping scalar logging")

    # [导读] manifest 告诉 DataLoader 读哪些 parquet；vocab 告诉模型每个字段词表多大
    manifest = Manifest.load(Path(cfg["data"]["manifest"]))
    vocab_path = Path(cfg["data"]["vocab"])
    vocab = _load_vocab(vocab_path)

    train_dl, val_dl = _build_dataloaders(
        manifest,
        cfg,
        vocab=vocab,
        seed=cfg["seed"],
        rank=rank,
        world_size=world_size,
    )

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
        vocab_sha256=hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
        field_specs=frozen_field_specs,
        continuous_normalizers=continuous_normalizers,
        book_state_timing=str(cfg.get("book", {}).get("state_timing", "none")),
        context_horizon=int(cfg["data"]["context"]),
        pooling_version=str(cfg.get("pooling", {}).get("version", "flat_v1")),
        backbone_moe=BackboneMoEConfig.from_dict(mcfg.get("backbone_moe")),
        **_fusion_options(mcfg),
    )
    model = OrderFlowFM(model_cfg).to(device)  # .to(device) 把模型放到 CPU/GPU
    resume_path = _resolve_resume_path(out_dir, resume)
    resume_ckpt = None
    if resume_path is not None:
        resume_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        if "optimizer_state" not in resume_ckpt:
            msg = (
                f"checkpoint {resume_path} is inference-only and cannot resume "
                "training; use step*.pt or final_resume.pt"
            )
            raise ValueError(msg)
        _validate_resume_metadata(resume_ckpt, model_cfg, target_specs)
        model.load_state_dict(resume_ckpt["model_state"])
        if is_main_process(rank):
            logger.info("loaded model checkpoint %s", resume_path)

    use_fsdp = cfg["runtime"].get("fsdp") and world_size > 1
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
    scaler = torch.amp.GradScaler(enabled=opt["precision"] == "fp16")

    state = TrainState()
    accum = int(opt["grad_accum"])
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
        if is_main_process(rank):
            logger.info(
                "resumed training at update %d / micro %d (best %.4f @ %d)",
                state.update_step,
                state.micro_step,
                state.best_val,
                state.best_update_step,
            )
        del resume_ckpt

    max_update_steps = int(opt.get("max_update_steps", opt.get("max_steps", 0)))
    max_train_tokens = int(opt.get("max_train_tokens", 0))
    if max_update_steps < 1 and max_train_tokens < 1:
        msg_0 = "optim requires max_update_steps/max_steps or max_train_tokens"
        raise ValueError(msg_0)
    schedule_update_steps = int(opt.get("lr_schedule_steps", max_update_steps))
    if schedule_update_steps < 1:
        msg_1 = (
            "token-budget-only training requires optim.lr_schedule_steps so the "
            "cosine schedule has a fixed horizon"
        )
        raise ValueError(msg_1)
    model.train()  # 训练模式（启用 dropout 等）
    optimizer.zero_grad(set_to_none=True)
    data_epoch = 0
    pending_samples = 0
    pending_tokens = 0

    def training_complete() -> bool:
        updates_done = max_update_steps > 0 and state.update_step >= max_update_steps
        tokens_done = (
            max_train_tokens > 0 and state.non_pad_tokens_seen >= max_train_tokens
        )
        return updates_done or tokens_done

    # 调度、日志、验证与存盘统一使用 optimizer update 计数。
    while not training_complete():
        if hasattr(train_dl.sampler, "set_epoch"):
            train_dl.sampler.set_epoch(data_epoch)
        data_epoch += 1
        for batch in train_dl:
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
                    max_batches=int(cfg["runtime"].get("val_max_batches", 50)),
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
                    rank=rank,
                    step=state.update_step,
                )

            if training_complete():
                break

    # 训练结束：再跑一轮验证，保证 best.pt 覆盖「最后一段未踩到 eval_every」的改进
    if val_dl is not None and cfg["runtime"].get("save_best", True):
        val_loss = evaluate(
            model,
            val_dl,
            device,
            target_fields,
            target_specs=target_specs,
            max_batches=int(cfg["runtime"].get("val_max_batches", 50)),
            world_size=world_size,
        )
        if is_main_process(rank):
            logger.info(
                "final val_loss %.4f (best so far %.4f @ step %d)",
                val_loss,
                state.best_val,
                state.best_update_step,
            )
            if writer is not None:
                writer.add_scalar("val/loss", val_loss, state.update_step)
        _maybe_save_best(
            model,
            model_cfg,
            out_dir,
            state,
            val_loss=val_loss,
            target_specs=target_specs,
            rank=rank,
            writer=writer,
        )

    _save_checkpoint(
        model,
        model_cfg,
        out_dir / "final_resume.pt",
        optimizer=optimizer,
        scaler=scaler,
        train_state=state,
        target_specs=target_specs,
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
    rank: int = 0,
    step: int | None = None,
    val_loss: float | None = None,
    is_best: bool = False,
) -> None:
    """
    保存模型权重与配置，便于可复现加载。

    FSDP 下所有 rank 都必须进入 state_dict 收集；只有 rank0 写文件。
    """
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
            "backbone_moe": cfg.backbone_moe.to_dict(),
        },
    }
    if target_specs is not None:
        payload["target_specs"] = [asdict(spec) for spec in target_specs]
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
    torch.save(payload, path)
    logger.info("saved checkpoint %s", path)


def load_checkpoint(
    path: Path,
    device: torch.device,
    *,
    vocab_path: Path | None = None,
) -> OrderFlowFM:
    """重建模型；v2 必须提供词表并验证 hash/schema/字段顺序。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    c = ckpt["config"]
    if str(c.get("vocab_version", "1.0")) == VocabV2.VOCAB_VERSION:
        if ckpt.get("fm_artifact_version") != "2.0":
            msg = "v2 checkpoint is missing fm_artifact_version=2.0"
            raise ValueError(msg)
        if vocab_path is None:
            msg = "loading a v2 checkpoint requires vocab_path for compatibility checks"
            raise ValueError(msg)
        vocab = _load_vocab(vocab_path)
        if not isinstance(vocab, VocabV2):
            msg = "a v2 checkpoint cannot be loaded with a v1 vocab"
            raise ValueError(msg)
        vocab_hash = hashlib.sha256(Path(vocab_path).read_bytes()).hexdigest()
        layout = field_layout_from_vocab(vocab)
        checks = {
            "schema_version": vocab.schema_version,
            "vocab_sha256": vocab_hash,
            "field_specs": [spec.to_dict() for spec in vocab.field_specs],
        }
        for key, expected_value in checks.items():
            if c.get(key) != expected_value:
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
        pooling_version=str(c.get("pooling_version", "flat_v1")),
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
