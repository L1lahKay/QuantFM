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

from quant_fm.embedding.pool_stock_day import (
    MultiScaleStockDayPoolAccumulator,
    StockDayPoolAccumulator,
)
from quant_fm.manifest.build_manifest import Manifest
from quant_fm.pretrain.train import load_checkpoint, resolve_device
from quant_fm.tokenizer.transforms import int_time_to_ms

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
    last_k: int = 256,
    amp_dtype: torch.dtype | None = None,
) -> np.ndarray:
    """分块编码单个分片并返回池化嵌入向量。"""
    token_fields = model.cfg.input_fields
    scalar_fields = (
        *model.cfg.scalar_fields,
        *model.cfg.standalone_scalar_fields,
    )
    input_columns = (*token_fields, *scalar_fields)
    read_columns = (
        (*input_columns, "int_time") if pooling == "multi_scale" else input_columns
    )
    df = pl.read_parquet(shard.path, columns=list(read_columns))
    n = df.height
    fields = {field: df[field].to_numpy().astype(np.int64) for field in token_fields}
    fields.update(
        {field: df[field].to_numpy().astype(np.float32) for field in scalar_fields}
    )

    accumulator = (
        MultiScaleStockDayPoolAccumulator(model.cfg.d_model)
        if pooling == "multi_scale"
        else StockDayPoolAccumulator(model.cfg.d_model, method=pooling, last_k=last_k)
    )
    elapsed = (
        int_time_to_ms(df["int_time"].to_numpy()) if pooling == "multi_scale" else None
    )
    for start in range(0, n, context):
        end = min(start + context, n)
        length = end - start
        batch = {
            field: torch.from_numpy(fields[field][start:end].copy())
            .unsqueeze(0)
            .to(device)
            for field in input_columns
        }
        batch["attention_mask"] = torch.ones(
            (1, length), dtype=torch.bool, device=device
        )
        with _autocast(device, amp_dtype):
            hidden = model.encode(batch)[0]  # [L, d]
        if isinstance(accumulator, MultiScaleStockDayPoolAccumulator):
            if elapsed is None:  # pragma: no cover - construction invariant
                msg = "missing intraday time array"
                raise RuntimeError(msg)
            time_tensor = torch.from_numpy(elapsed[start:end].copy()).to(device)
            accumulator.update(hidden, batch["attention_mask"][0], time_tensor)
        else:
            accumulator.update(hidden, batch["attention_mask"][0])

    value = (
        accumulator.concatenate()
        if isinstance(accumulator, MultiScaleStockDayPoolAccumulator)
        else accumulator.value()
    )
    return value.numpy().astype(np.float32)


def extract_stock_day_embeddings(
    model: OrderFlowFM,
    shards: list[ShardEntry],
    device: torch.device,
    *,
    context: int = 2048,
    pooling: str = "mean",
    last_k: int = 256,
    amp_dtype: torch.dtype | None = None,
) -> pl.DataFrame:
    """生成 ``(date, symbol, market, emb_*)`` 嵌入数据帧。"""
    rows: list[dict[str, object]] = []
    for shard in shards:
        vec = _embed_shard(
            model,
            shard,
            device,
            context=context,
            pooling=pooling,
            last_k=last_k,
            amp_dtype=amp_dtype,
        )
        row: dict[str, object] = {
            "date": shard.date,
            "symbol": shard.symbol,
            "market": shard.market,
        }
        row.update({f"emb_{i}": float(value) for i, value in enumerate(vec)})
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
    last_k: int = 256,
    batch_size: int = 16,
    log_every: int = 200,
    amp_dtype: torch.dtype | None = None,
) -> pl.DataFrame:
    """
    批处理版：把多个（股日）分片的 context 分块打包进同一 forward，显著提高 GPU 吞吐。

    每个分片按 ``context`` 切块；不同分片的块右侧 padding 到同一长度后组 batch。
    流式累加每个股日：mean 按真实事件严格平均，last/lastk 跨 chunk 保留尾部。
    """
    d = model.cfg.d_model
    token_fields = model.cfg.input_fields
    scalar_fields = (
        *model.cfg.scalar_fields,
        *model.cfg.standalone_scalar_fields,
    )
    input_columns = (*token_fields, *scalar_fields)
    read_columns = (
        (*input_columns, "int_time") if pooling == "multi_scale" else input_columns
    )

    # 预扫描出所有 (shard_idx, 起止) 块，块内数据延迟读取以省内存。
    meta: list[dict] = [
        {"date": s.date, "symbol": s.symbol, "market": s.market} for s in shards
    ]
    accumulators = (
        [MultiScaleStockDayPoolAccumulator(d) for _ in shards]
        if pooling == "multi_scale"
        else [StockDayPoolAccumulator(d, method=pooling, last_k=last_k) for _ in shards]
    )

    pending: list[tuple[int, dict[str, np.ndarray], int]] = []

    def flush(batch_chunks: list[tuple[int, dict[str, np.ndarray], int]]) -> None:
        if not batch_chunks:
            return
        max_len = max(length for _, _, length in batch_chunks)
        bsz = len(batch_chunks)
        fields_t: dict[str, torch.Tensor] = {
            field: torch.zeros((bsz, max_len), dtype=torch.long)
            for field in token_fields
        }
        fields_t.update(
            {
                field: torch.zeros((bsz, max_len), dtype=torch.float32)
                for field in scalar_fields
            }
        )
        mask = torch.zeros((bsz, max_len), dtype=torch.bool)
        for bi, (_, fields, length) in enumerate(batch_chunks):
            for field in input_columns:
                fields_t[field][bi, :length] = torch.from_numpy(fields[field])
            mask[bi, :length] = True
        batch = {field: fields_t[field].to(device) for field in input_columns}
        batch["attention_mask"] = mask.to(device)
        with _autocast(device, amp_dtype):
            hidden = model.encode(batch)  # [B, Lmax, d]
        for bi, (sidx, fields, length) in enumerate(batch_chunks):
            accumulator = accumulators[sidx]
            # Keep all three streams aligned to the real chunk length.  In a
            # mixed-length batch ``hidden`` and ``attention_mask`` are padded
            # to ``max_len`` while the intraday timestamps are not.
            device_mask = batch["attention_mask"][bi, :length]
            chunk_hidden = hidden[bi, :length]
            if isinstance(accumulator, MultiScaleStockDayPoolAccumulator):
                times = torch.from_numpy(fields["__time_of_day_ms"]).to(device)
                accumulator.update(chunk_hidden, device_mask, times)
            else:
                accumulator.update(chunk_hidden, device_mask)

    for sidx, shard in enumerate(shards):
        df = pl.read_parquet(shard.path, columns=list(read_columns))
        n = df.height
        cols = {field: df[field].to_numpy().astype(np.int64) for field in token_fields}
        cols.update(
            {field: df[field].to_numpy().astype(np.float32) for field in scalar_fields}
        )
        if pooling == "multi_scale":
            cols["__time_of_day_ms"] = int_time_to_ms(df["int_time"].to_numpy())
        for start in range(0, n, context):
            end = min(start + context, n)
            chunk = {
                field: cols[field][start:end].copy()
                for field in (
                    (*input_columns, "__time_of_day_ms")
                    if pooling == "multi_scale"
                    else input_columns
                )
            }
            pending.append((sidx, chunk, end - start))
            if len(pending) >= batch_size:
                flush(pending)
                pending = []
        if (sidx + 1) % log_every == 0:
            logger.info("embedded %d/%d shards", sidx + 1, len(shards))
    flush(pending)

    rows: list[dict[str, object]] = []
    for sidx, m in enumerate(meta):
        accumulator = accumulators[sidx]
        vec_tensor = (
            accumulator.concatenate()
            if isinstance(accumulator, MultiScaleStockDayPoolAccumulator)
            else accumulator.value()
        )
        vec = vec_tensor.numpy().astype(np.float32)
        row: dict[str, object] = dict(m)
        row.update({f"emb_{i}": float(value) for i, value in enumerate(vec)})
        rows.append(row)
    logger.info("embedded %d/%d shards (done)", len(shards), len(shards))
    return pl.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--vocab",
        type=Path,
        help="v2 checkpoint 必填，用于校验 schema/字段顺序/vocab hash",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument(
        "--last-k",
        type=int,
        default=256,
        help="lastk_mean 跨整个股日保留的最后事件数",
    )
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
    model = load_checkpoint(args.checkpoint, device, vocab_path=args.vocab)
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
            last_k=args.last_k,
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
            last_k=args.last_k,
            amp_dtype=amp_dtype,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    embeddings.write_parquet(args.out)
    logger.info("wrote %d embeddings to %s", embeddings.height, args.out)


if __name__ == "__main__":
    main()
