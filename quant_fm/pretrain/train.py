"""
OrderFlow FM 的可复现预训练循环。

特性：配置驱动、固定种子、bf16/fp16 自动混合精度、AdamW、warmup+余弦
学习率、梯度裁剪与累积、可选单节点 FSDP、TensorBoard 日志、
定期验证、自动保存最优 ``best.pt``（按 val_loss）与定期/最终检查点。
完整配置快照于每个检查点旁，便于字节级复现。
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from quant_fm.manifest.build_manifest import Manifest
from quant_fm.pretrain.dataset import (
    DEFAULT_TARGET_FIELDS,
    EventWindowDataset,
    collate_windows,
)
from quant_fm.pretrain.heads import next_event_loss
from quant_fm.pretrain.model import (
    OrderFlowFM,
    OrderFlowFMConfig,
    field_sizes_from_vocab,
)
from quant_fm.tokenizer.vocab import Vocab

logger = logging.getLogger(__name__)


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

    step: int = 0
    best_val: float = float("inf")
    best_step: int = -1


def _amp_dtype(precision: str) -> torch.dtype | None:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]


def _build_dataloaders(
    manifest: Manifest,
    cfg: dict,
    *,
    seed: int,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader | None]:
    data = cfg["data"]
    train_ds = EventWindowDataset.from_manifest(
        manifest,
        "train",
        context=data["context"],
        stride=data["stride"],
        min_len=data["min_len"],
        cache_size=data["cache_size"],
    )
    train_sampler = None
    shuffle = True
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        shuffle = False
    generator = torch.Generator()
    generator.manual_seed(seed + rank)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["optim"]["batch_size"],
        shuffle=shuffle,
        sampler=train_sampler,
        collate_fn=collate_windows,
        num_workers=data["num_workers"],
        drop_last=True,
        generator=generator if train_sampler is None else None,
        pin_memory=torch.cuda.is_available(),
    )
    val_dl = None
    val_shards = manifest.split("val")
    if val_shards:
        val_ds = EventWindowDataset(
            val_shards,
            context=data["context"],
            stride=data["stride"],
            min_len=data["min_len"],
            cache_size=data["cache_size"],
        )
        val_sampler = None
        if world_size > 1:
            val_sampler = DistributedSampler(
                val_ds, num_replicas=world_size, rank=rank, shuffle=False
            )
        val_dl = DataLoader(
            val_ds,
            batch_size=cfg["optim"]["batch_size"],
            shuffle=False,
            sampler=val_sampler,
            collate_fn=collate_windows,
            num_workers=data["num_workers"],
            pin_memory=torch.cuda.is_available(),
        )
    return train_dl, val_dl


@torch.no_grad()
def evaluate(
    model: OrderFlowFM,
    val_dl: DataLoader,
    device: torch.device,
    target_fields: tuple[str, ...],
    *,
    max_batches: int = 50,
    world_size: int = 1,
) -> float:
    """在若干验证批上返回平均总验证损失（多卡时 all-reduce 平均）。"""
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(val_dl):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        loss = next_event_loss(logits, batch, target_fields)
        total += float(loss.total.item())
        n += 1
    model.train()
    if world_size > 1:
        import torch.distributed as dist

        stats = torch.tensor([total, float(n)], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total, n = float(stats[0].item()), float(stats[1].item())
    return total / max(n, 1.0)


def train(config_path: Path) -> None:
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
    vocab = Vocab.load(Path(cfg["data"]["vocab"]))

    train_dl, val_dl = _build_dataloaders(
        manifest, cfg, seed=cfg["seed"], rank=rank, world_size=world_size
    )

    mcfg = cfg["model"]
    target_fields = tuple(mcfg.get("target_fields", DEFAULT_TARGET_FIELDS))
    model_cfg = OrderFlowFMConfig(
        field_sizes=field_sizes_from_vocab(vocab),
        target_fields=target_fields,
        d_model=mcfg["d_model"],
        n_layers=mcfg["n_layers"],
        n_heads=mcfg["n_heads"],
        ffn_mult=mcfg["ffn_mult"],
        dropout=mcfg["dropout"],
        max_seq_len=mcfg["max_seq_len"],
        rope_theta=mcfg["rope_theta"],
    )
    model = OrderFlowFM(model_cfg).to(device)  # .to(device) 把模型放到 CPU/GPU
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
    accum = opt["grad_accum"]  # [导读] 梯度累积：每 accum 步才真正 optimizer.step 一次
    model.train()  # 训练模式（启用 dropout 等）

    # [导读] 外层 while 控制总步数；内层 for 遍历 DataLoader 的每个 batch
    while state.step < opt["max_steps"]:
        if hasattr(train_dl.sampler, "set_epoch"):
            train_dl.sampler.set_epoch(state.step)
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}  # 数据也搬到 GPU
            lr = cosine_lr(
                state.step,
                warmup=opt["warmup_steps"],
                max_steps=opt["max_steps"],
                base_lr=opt["lr"],
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            use_amp = amp_dtype is not None and device.type == "cuda"
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                logits = model(batch)  # 前向：得到各字段的预测分数
                loss_out = next_event_loss(logits, batch, target_fields)
                loss = loss_out.total / accum  # 累积时先除以 accum

            scaler.scale(loss).backward()  # 反向传播，计算梯度

            if (state.step + 1) % accum == 0:
                scaler.unscale_(optimizer)
                if use_fsdp:
                    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                    FSDP.clip_grad_norm_(model, opt["grad_clip"])
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt["grad_clip"])
                scaler.step(optimizer)  # 用梯度更新权重
                scaler.update()
                optimizer.zero_grad(set_to_none=True)  # 清空梯度，准备下一步

            if is_main_process(rank) and state.step % cfg["runtime"]["log_every"] == 0:
                logger.info(
                    "step %d lr %.2e loss %.4f",
                    state.step,
                    lr,
                    float(loss_out.total.item()),
                )
                if writer is not None:
                    writer.add_scalar("train/loss", loss_out.total.item(), state.step)
                    writer.add_scalar("train/lr", lr, state.step)
                    for f, fl in loss_out.per_field.items():
                        writer.add_scalar(f"train/ce_{f}", fl.item(), state.step)

            if (
                val_dl is not None
                and state.step > 0
                and state.step % cfg["runtime"]["eval_every"] == 0
            ):
                val_loss = evaluate(
                    model, val_dl, device, target_fields, world_size=world_size
                )
                if is_main_process(rank):
                    logger.info("step %d val_loss %.4f", state.step, val_loss)
                    if writer is not None:
                        writer.add_scalar("val/loss", val_loss, state.step)
                if cfg["runtime"].get("save_best", True):
                    _maybe_save_best(
                        model,
                        model_cfg,
                        out_dir,
                        state,
                        val_loss=val_loss,
                        rank=rank,
                        writer=writer,
                    )

            if state.step > 0 and state.step % cfg["runtime"]["ckpt_every"] == 0:
                _save_checkpoint(
                    model,
                    model_cfg,
                    out_dir / f"step{state.step}.pt",
                    rank=rank,
                    step=state.step,
                )

            state.step += 1
            if state.step >= opt["max_steps"]:
                break

    # 训练结束：再跑一轮验证，保证 best.pt 覆盖「最后一段未踩到 eval_every」的改进
    if val_dl is not None and cfg["runtime"].get("save_best", True):
        val_loss = evaluate(model, val_dl, device, target_fields, world_size=world_size)
        if is_main_process(rank):
            logger.info(
                "final val_loss %.4f (best so far %.4f @ step %d)",
                val_loss,
                state.best_val,
                state.best_step,
            )
            if writer is not None:
                writer.add_scalar("val/loss", val_loss, state.step)
        _maybe_save_best(
            model,
            model_cfg,
            out_dir,
            state,
            val_loss=val_loss,
            rank=rank,
            writer=writer,
        )

    _save_checkpoint(
        model,
        model_cfg,
        out_dir / "final.pt",
        rank=rank,
        step=state.step,
        val_loss=state.best_val if state.best_step >= 0 else None,
    )
    if is_main_process(rank):
        if writer is not None:
            writer.close()
        if state.best_step >= 0:
            logger.info(
                "training complete: %d steps; best.pt from step %d (val_loss=%.4f)",
                state.step,
                state.best_step,
                state.best_val,
            )
        else:
            logger.info(
                "training complete: %d steps; no val set → only final.pt saved",
                state.step,
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
        state.best_step = state.step
        _save_checkpoint(
            model,
            model_cfg,
            out_dir / "best.pt",
            rank=rank,
            step=state.step,
            val_loss=val_loss,
            is_best=True,
        )
        if is_main_process(rank):
            logger.info(
                "new best checkpoint → %s (step %d, val_loss %.4f ← %.4f)",
                out_dir / "best.pt",
                state.step,
                val_loss,
                prev,
            )
            if writer is not None:
                writer.add_scalar("val/best_loss", val_loss, state.step)
        return True
    return False


def _save_checkpoint(
    model: OrderFlowFM,
    cfg: OrderFlowFMConfig,
    path: Path,
    *,
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
    if not is_main_process(rank):
        return
    payload: dict = {
        "model_state": state,
        "config": {
            "field_sizes": cfg.field_sizes,
            "input_fields": list(cfg.input_fields),
            "target_fields": list(cfg.target_fields),
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "ffn_mult": cfg.ffn_mult,
            "dropout": cfg.dropout,
            "max_seq_len": cfg.max_seq_len,
            "rope_theta": cfg.rope_theta,
        },
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


def load_checkpoint(path: Path, device: torch.device) -> OrderFlowFM:
    """从 :func:`_save_checkpoint` 保存的检查点重建模型。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    c = ckpt["config"]
    cfg = OrderFlowFMConfig(
        field_sizes=c["field_sizes"],
        input_fields=tuple(c["input_fields"]),
        target_fields=tuple(c["target_fields"]),
        d_model=c["d_model"],
        n_layers=c["n_layers"],
        n_heads=c["n_heads"],
        ffn_mult=c["ffn_mult"],
        dropout=c["dropout"],
        max_seq_len=c["max_seq_len"],
        rope_theta=c["rope_theta"],
    )
    model = OrderFlowFM(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
