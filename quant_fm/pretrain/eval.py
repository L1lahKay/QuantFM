"""
预训练评估：各字段困惑度与风格化事实检验。

困惑度为过程指标（真正验收在下游 RankIC 提升）。
风格化事实（厚尾收益、波动率聚集）用于检验模型/数据是否再现已知微观结构规律。

CLI 用法::

    python -m quant_fm.pretrain.eval \\
      --checkpoint quant_fm/runs/medium_300m/run/best.pt \\
      --config quant_fm/pretrain/config_medium_300m_8gpu.yaml \\
      --split val \\
      --max-batches 100 \\
      --out quant_fm/runs/medium_300m/run/eval_val.json \\
      --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from quant_fm.manifest.build_manifest import Manifest
from quant_fm.pretrain.dataset import (
    DEFAULT_TARGET_FIELDS,
    EventWindowDataset,
    collate_windows,
)
from quant_fm.pretrain.heads import next_event_loss
from quant_fm.pretrain.train import load_checkpoint, resolve_device

if TYPE_CHECKING:
    from quant_fm.pretrain.model import OrderFlowFM

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PerplexityReport:
    """各字段交叉熵与困惑度。"""

    ce: dict[str, float]

    @property
    def perplexity(self) -> dict[str, float]:
        """各字段 CE 的指数。"""
        return {f: math.exp(v) for f, v in self.ce.items()}

    @property
    def total_ce(self) -> float:
        """各字段 CE 之和（训练目标）。"""
        return float(sum(self.ce.values()))


@torch.no_grad()
def field_perplexity(
    model: OrderFlowFM,
    loader: DataLoader,
    device: torch.device,
    target_fields: tuple[str, ...],
    *,
    max_batches: int = 200,
) -> PerplexityReport:
    """在 DataLoader 上平均各字段交叉熵。"""
    model.eval()
    sums = dict.fromkeys(target_fields, 0.0)
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch)
        out = next_event_loss(logits, batch, target_fields)
        for f, fl in out.per_field.items():
            sums[f] += float(fl.item())
        n += 1
    ce = {f: sums[f] / max(n, 1) for f in target_fields}
    return PerplexityReport(ce=ce)


def stylized_facts(returns: np.ndarray, *, max_lag: int = 50) -> dict[str, float]:
    """
    计算超额峰度与波动率聚集自相关。

    参数
    ----------

    Returns
    -------
        一维（对数）收益序列，如中间价变化。
    max_lag
        |收益| 自相关平均的最大滞后。

    返回
    -------
    dict
        ``excess_kurtosis``（厚尾 > 0）与 ``vol_clustering_acf``（|收益| 自相关均值；
        > 0 表示聚集）。
    """
    r = returns[np.isfinite(returns)]
    if r.size < max_lag + 2:
        return {"excess_kurtosis": float("nan"), "vol_clustering_acf": float("nan")}
    r = r - r.mean()
    var = r.var()
    kurt = (np.mean(r**4) / (var**2)) - 3.0 if var > 0 else float("nan")

    absr = np.abs(r)
    absr = absr - absr.mean()
    denom = np.sum(absr**2)
    acfs = []
    for lag in range(1, max_lag + 1):
        num = np.sum(absr[:-lag] * absr[lag:])
        acfs.append(num / denom if denom > 0 else 0.0)
    return {
        "excess_kurtosis": float(kurt),
        "vol_clustering_acf": float(np.mean(acfs)),
    }


def main() -> None:
    """对 checkpoint 跑字段 CE / perplexity 并落盘 JSON。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--device", default="cpu", help="训练占满 GPU 时用 cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    model = load_checkpoint(args.checkpoint, device)
    target_fields = tuple(
        cfg["model"].get(
            "target_fields", model.cfg.target_fields or DEFAULT_TARGET_FIELDS
        )
    )

    manifest = Manifest.load(Path(cfg["data"]["manifest"]))
    shards = manifest.split(args.split)
    if not shards:
        raise SystemExit(f"no shards for split={args.split}")
    data = cfg["data"]
    ds = EventWindowDataset(
        shards,
        context=data["context"],
        stride=data["stride"],
        min_len=data["min_len"],
        cache_size=min(8, int(data.get("cache_size", 8))),
    )
    loader = DataLoader(
        ds,
        batch_size=1 if str(device).startswith("cpu") else int(cfg["optim"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_windows,
        num_workers=0,
    )
    report = field_perplexity(
        model, loader, device, target_fields, max_batches=args.max_batches
    )
    payload = {
        "created_utc": datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ"),
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "split": args.split,
        "max_batches": args.max_batches,
        "device": str(device),
        "n_shards": len(shards),
        "ce": report.ce,
        "perplexity": report.perplexity,
        "total_ce": report.total_ce,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "wrote %s total_ce=%.4f perplexity=%s",
        args.out,
        report.total_ce,
        {k: round(v, 3) for k, v in report.perplexity.items()},
    )


if __name__ == "__main__":
    main()
