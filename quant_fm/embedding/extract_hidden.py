"""
用冻结 FM 对 token 分片编码并输出股日嵌入。

每个股日分片按 context 大小分块编码事件流；逐事件隐状态经
:mod:`pool_stock_day` 池化为每个 (date, symbol) 一向量。输出为整洁的嵌入 parquet，
便于下游与手工因子拼接。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch

from quant_fm.embedding.pool_stock_day import pool_hidden
from quant_fm.manifest.build_manifest import Manifest
from quant_fm.pretrain.dataset import FIELD_ORDER
from quant_fm.pretrain.train import load_checkpoint, resolve_device

if TYPE_CHECKING:
    from quant_fm.manifest.build_manifest import ShardEntry
    from quant_fm.pretrain.model import OrderFlowFM

logger = logging.getLogger(__name__)

DTYPE_MAP: dict[str, torch.dtype | None] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": None,
}


def _balanced_shard_partition(
    shards: list[ShardEntry], num_parts: int, part_index: int
) -> list[ShardEntry]:
    """Greedy row-balanced partition; file-count stride badly skews active stocks."""
    parts: list[list[ShardEntry]] = [[] for _ in range(num_parts)]
    loads = [0] * num_parts
    for shard in sorted(shards, key=lambda item: (-item.rows, item.path)):
        target = min(range(num_parts), key=lambda index: (loads[index], index))
        parts[target].append(shard)
        loads[target] += shard.rows
    logger.info("part token-row loads: %s", loads)
    return sorted(parts[part_index], key=lambda item: item.path)


def _autocast(device: torch.device, amp_dtype: torch.dtype | None):
    """与训练一致的自动混合精度上下文；``amp_dtype`` 为 None 时不启用。"""
    return torch.autocast(
        device_type=device.type,
        dtype=amp_dtype if amp_dtype is not None else torch.float32,
        enabled=amp_dtype is not None,
    )


@torch.no_grad()
def _embed_shard(
    model: OrderFlowFM,
    shard: ShardEntry,
    device: torch.device,
    *,
    context: int,
    pooling: str,
    amp_dtype: torch.dtype | None = None,
) -> np.ndarray:
    """分块编码单个分片并返回池化嵌入向量。"""
    df = pl.read_parquet(shard.path, columns=list(FIELD_ORDER))
    n = df.height
    fields = {f: df[f].to_numpy().astype(np.int64) for f in FIELD_ORDER}

    chunk_embs: list[torch.Tensor] = []
    chunk_weights: list[int] = []
    for start in range(0, n, context):
        end = min(start + context, n)
        length = end - start
        batch = {
            f: torch.from_numpy(fields[f][start:end].copy()).unsqueeze(0).to(device)
            for f in FIELD_ORDER
        }
        batch["attention_mask"] = torch.ones(
            (1, length), dtype=torch.bool, device=device
        )
        with _autocast(device, amp_dtype):
            hidden = model.encode(batch)[0]  # [L, d]
        pooled = pool_hidden(hidden, batch["attention_mask"][0], method=pooling)
        chunk_embs.append(pooled.float().cpu())
        chunk_weights.append(length)

    if not chunk_embs:
        return np.zeros(model.cfg.d_model, dtype=np.float32)
    weights = torch.tensor(chunk_weights, dtype=torch.float32).unsqueeze(1)
    stacked = torch.stack(chunk_embs)
    day_emb = (stacked * weights).sum(dim=0) / weights.sum()
    return day_emb.numpy().astype(np.float32)


def extract_stock_day_embeddings(
    model: OrderFlowFM,
    shards: list[ShardEntry],
    device: torch.device,
    *,
    context: int = 2048,
    pooling: str = "mean",
    amp_dtype: torch.dtype | None = None,
) -> pl.DataFrame:
    """生成 ``(date, symbol, market, emb_*)`` 嵌入数据帧。"""
    rows: list[dict[str, object]] = []
    d = model.cfg.d_model
    for shard in shards:
        vec = _embed_shard(
            model, shard, device, context=context, pooling=pooling, amp_dtype=amp_dtype
        )
        row: dict[str, object] = {
            "date": shard.date,
            "symbol": shard.symbol,
            "market": shard.market,
        }
        row.update({f"emb_{i}": float(vec[i]) for i in range(d)})
        rows.append(row)
        logger.info("embedded %s %s", shard.date, shard.symbol)
    return pl.DataFrame(rows)


@torch.no_grad()
def extract_stock_day_embeddings_batched(
    model: OrderFlowFM,
    shards: list[ShardEntry],
    device: torch.device,
    *,
    context: int = 2048,
    pooling: str = "mean",
    batch_size: int = 16,
    log_every: int = 200,
    amp_dtype: torch.dtype | None = None,
) -> pl.DataFrame:
    """
    批处理版：把多个（股日）分片的 context 分块打包进同一 forward，显著提高 GPU 吞吐。

    每个分片按 ``context`` 切块；不同分片的块右侧 padding 到同一长度后组 batch。
    逐块池化后按块事件数加权，重建每个分片的股日向量（与逐块加权平均一致）。
    """
    d = model.cfg.d_model

    # 预扫描出所有 (shard_idx, 起止) 块，块内数据延迟读取以省内存。
    meta: list[dict] = [
        {"date": s.date, "symbol": s.symbol, "market": s.market} for s in shards
    ]
    acc = [torch.zeros(d, dtype=torch.float32) for _ in shards]
    wsum = [0.0 for _ in shards]

    pending: list[tuple[int, dict[str, np.ndarray], int]] = []

    def flush(batch_chunks: list[tuple[int, dict[str, np.ndarray], int]]) -> None:
        if not batch_chunks:
            return
        max_len = max(length for _, _, length in batch_chunks)
        bsz = len(batch_chunks)
        fields_t = {
            f: torch.zeros((bsz, max_len), dtype=torch.long) for f in FIELD_ORDER
        }
        mask = torch.zeros((bsz, max_len), dtype=torch.bool)
        for bi, (_, fields, length) in enumerate(batch_chunks):
            for f in FIELD_ORDER:
                fields_t[f][bi, :length] = torch.from_numpy(fields[f])
            mask[bi, :length] = True
        batch = {f: fields_t[f].to(device) for f in FIELD_ORDER}
        batch["attention_mask"] = mask.to(device)
        with _autocast(device, amp_dtype):
            hidden = model.encode(batch)  # [B, Lmax, d]
        for bi, (sidx, _, length) in enumerate(batch_chunks):
            pooled = pool_hidden(hidden[bi], mask[bi], method=pooling).float().cpu()
            acc[sidx] += pooled * float(length)
            wsum[sidx] += float(length)

    for sidx, shard in enumerate(shards):
        df = pl.read_parquet(shard.path, columns=list(FIELD_ORDER))
        n = df.height
        cols = {f: df[f].to_numpy().astype(np.int64) for f in FIELD_ORDER}
        for start in range(0, n, context):
            end = min(start + context, n)
            chunk = {f: cols[f][start:end].copy() for f in FIELD_ORDER}
            pending.append((sidx, chunk, end - start))
            if len(pending) >= batch_size:
                flush(pending)
                pending = []
        if (sidx + 1) % log_every == 0:
            logger.info("embedded %d/%d shards", sidx + 1, len(shards))
    flush(pending)

    rows: list[dict[str, object]] = []
    for sidx, m in enumerate(meta):
        if wsum[sidx] > 0:
            vec = (acc[sidx] / wsum[sidx]).numpy().astype(np.float32)
        else:
            vec = np.zeros(d, dtype=np.float32)
        row: dict[str, object] = dict(m)
        row.update({f"emb_{i}": float(vec[i]) for i in range(d)})
        rows.append(row)
    logger.info("embedded %d/%d shards (done)", len(shards), len(shards))
    return pl.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="推理精度；默认 bf16（与训练一致，约 2× 提速且省显存）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=">1 时启用批处理编码（多股日打包一次 forward），大幅提速",
    )
    parser.add_argument(
        "--num-parts",
        type=int,
        default=1,
        help="将该 split 的分片按 stride 均分为 N 份（多卡并行时用）",
    )
    parser.add_argument(
        "--part-index",
        type=int,
        default=0,
        help="本进程处理第 part-index 份（0..num-parts-1）",
    )
    return parser.parse_args()


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype = DTYPE_MAP[args.dtype] if device.type == "cuda" else None
    model = load_checkpoint(args.checkpoint, device)
    manifest = Manifest.load(args.manifest)
    shards = manifest.split(args.split)
    if args.num_parts > 1:
        shards = _balanced_shard_partition(shards, args.num_parts, args.part_index)
        logger.info(
            "part %d/%d: %d shards, %d token rows (dtype=%s)",
            args.part_index,
            args.num_parts,
            len(shards),
            sum(shard.rows for shard in shards),
            args.dtype,
        )
    if args.batch_size > 1:
        embeddings = extract_stock_day_embeddings_batched(
            model,
            shards,
            device,
            context=args.context,
            pooling=args.pooling,
            batch_size=args.batch_size,
            amp_dtype=amp_dtype,
        )
    else:
        embeddings = extract_stock_day_embeddings(
            model,
            shards,
            device,
            context=args.context,
            pooling=args.pooling,
            amp_dtype=amp_dtype,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    embeddings.write_parquet(args.out)
    logger.info("wrote %d embeddings to %s", embeddings.height, args.out)


if __name__ == "__main__":
    main()
