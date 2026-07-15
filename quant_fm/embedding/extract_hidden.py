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


@torch.no_grad()
def _embed_shard(
    model: OrderFlowFM,
    shard: ShardEntry,
    device: torch.device,
    *,
    context: int,
    pooling: str,
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
) -> pl.DataFrame:
    """生成 ``(date, symbol, market, emb_*)`` 嵌入数据帧。"""
    rows: list[dict[str, object]] = []
    d = model.cfg.d_model
    for shard in shards:
        vec = _embed_shard(model, shard, device, context=context, pooling=pooling)
        row: dict[str, object] = {
            "date": shard.date,
            "symbol": shard.symbol,
            "market": shard.market,
        }
        row.update({f"emb_{i}": float(vec[i]) for i in range(d)})
        rows.append(row)
        logger.info("embedded %s %s", shard.date, shard.symbol)
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
    return parser.parse_args()


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    device = resolve_device(args.device)
    model = load_checkpoint(args.checkpoint, device)
    manifest = Manifest.load(args.manifest)
    shards = manifest.split(args.split)
    embeddings = extract_stock_day_embeddings(
        model, shards, device, context=args.context, pooling=args.pooling
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    embeddings.write_parquet(args.out)
    logger.info("wrote %d embeddings to %s", embeddings.height, args.out)


if __name__ == "__main__":
    main()
