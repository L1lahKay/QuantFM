r"""
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
from dataclasses import asdict, dataclass
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
from quant_fm.pretrain.dataset_v2 import EventWindowDatasetV2, collate_windows_v2
from quant_fm.pretrain.train import _load_vocab, load_checkpoint, resolve_device
from quant_fm.pretrain.validation_sampler import (
    FixedValidationSampler,
    ValidationSamplePlan,
    build_validation_sample_plan,
)
from quant_fm.tokenizer.vocab import PAD_ID
from quant_fm.tokenizer.vocab_v2 import VocabV2

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from quant_fm.pretrain.model import OrderFlowFM

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PerplexityReport:
    """各字段交叉熵与困惑度。"""

    ce: dict[str, float]

    @property
    def perplexity(self) -> dict[str, float]:
        """各字段 CE 的指数。"""
        return {
            field: math.exp(value) if value < 709 else float("inf")
            for field, value in self.ce.items()
        }

    @property
    def total_ce(self) -> float:
        """各字段 CE 之和（训练目标）。"""
        return float(sum(self.ce.values()))


@dataclass(frozen=True, slots=True)
class FieldEval:
    """Diagnostics for one next-event prediction head."""

    ce: float
    perplexity: float
    top1_accuracy: float
    balanced_accuracy: float
    unigram_entropy: float
    normalized_ce: float
    copy_baseline_ce: float


@dataclass(frozen=True, slots=True)
class FieldEvaluationReport:
    """Token-weighted diagnostics for all requested prediction heads."""

    fields: dict[str, FieldEval]
    n_predictions: dict[str, int]
    unigram_source: str

    @property
    def total_ce(self) -> float:
        """Return the legacy sum of raw per-field CE values."""
        return float(sum(field.ce for field in self.fields.values()))

    @property
    def total_normalized_ce(self) -> float:
        """Return the sum over finite entropy-normalized CE values."""
        values = [
            field.normalized_ce
            for field in self.fields.values()
            if math.isfinite(field.normalized_ce)
        ]
        return float(sum(values)) if values else float("nan")

    def to_dict(self) -> dict[str, object]:
        """Serialize both nested diagnostics and convenient per-metric maps."""
        nested = {name: asdict(field) for name, field in self.fields.items()}
        ce = {name: field.ce for name, field in self.fields.items()}
        perplexity = {name: field.perplexity for name, field in self.fields.items()}
        return {
            "fields": nested,
            "n_predictions": self.n_predictions,
            "unigram_source": self.unigram_source,
            # Legacy aliases stay available for existing dashboard readers.
            "ce": ce,
            "perplexity": perplexity,
            "per_field_ce": ce,
            "per_field_perplexity": perplexity,
            "per_field_top1_accuracy": {
                name: field.top1_accuracy for name, field in self.fields.items()
            },
            "per_field_balanced_accuracy": {
                name: field.balanced_accuracy for name, field in self.fields.items()
            },
            "train_unigram_entropy": {
                name: field.unigram_entropy for name, field in self.fields.items()
            },
            "ce_over_unigram_entropy": {
                name: field.normalized_ce for name, field in self.fields.items()
            },
            "copy_previous_event_ce": {
                name: field.copy_baseline_ce for name, field in self.fields.items()
            },
            "total_ce": self.total_ce,
            "total_normalized_ce": self.total_normalized_ce,
        }


def unigram_entropy(counts: Sequence[int] | np.ndarray) -> float:
    """Return empirical categorical entropy in nats, ignoring zero counts."""
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 1 or np.any(values < 0):
        msg = "unigram counts must be a one-dimensional non-negative array"
        raise ValueError(msg)
    total = float(values.sum())
    if total <= 0:
        return float("nan")
    probs = values[values > 0] / total
    return float(-np.sum(probs * np.log(probs)))


def _valid_targets(
    batch: Mapping[str, torch.Tensor], field: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return current/next field tokens at valid next-event positions."""
    mask = batch["attention_mask"]
    valid = mask[:, :-1] & mask[:, 1:]
    current = batch[field][:, :-1]
    target = batch[field][:, 1:]
    valid = valid & target.ne(PAD_ID)
    explicit = batch.get(f"mask_{field}")
    if explicit is not None:
        valid = valid & explicit[:, 1:].bool()
    return current[valid], target[valid], valid


@torch.no_grad()
def collect_unigram_counts(
    loader: DataLoader,
    target_fields: tuple[str, ...],
    *,
    field_sizes: Mapping[str, int] | None = None,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    """
    Count next-event targets for entropy estimation.

    This function deliberately does not inspect model predictions.  Use it on the
    training split, then pass the frozen counts to :func:`field_diagnostics` when
    evaluating validation or test checkpoints.
    """
    if max_batches is not None and max_batches < 1:
        msg = "max_batches must be >= 1 when provided"
        raise ValueError(msg)
    counts = {
        field: np.zeros(int(field_sizes.get(field, 0)), dtype=np.int64)
        if field_sizes is not None
        else np.zeros(0, dtype=np.int64)
        for field in target_fields
    }
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        for field in target_fields:
            _, target, _ = _valid_targets(batch, field)
            if target.numel() == 0:
                continue
            batch_counts = torch.bincount(target.detach().cpu()).numpy()
            if batch_counts.size > counts[field].size:
                counts[field] = np.pad(
                    counts[field], (0, batch_counts.size - counts[field].size)
                )
            counts[field][: batch_counts.size] += batch_counts
    return counts


def per_field_gradient_norms(
    model: OrderFlowFM,
    loader: DataLoader,
    device: torch.device,
    target_fields: tuple[str, ...],
    *,
    max_batches: int = 1,
) -> dict[str, float]:
    """返回各任务 CE 对共享 event hidden 的 RMS 梯度范数。"""
    if max_batches < 1:
        return {}
    sums = dict.fromkeys(target_fields, 0.0)
    counts = dict.fromkeys(target_fields, 0)
    was_training = model.training
    model.eval()
    try:
        for batch_index, cpu_batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = {key: value.to(device) for key, value in cpu_batch.items()}
            hidden = model.encode(batch)
            logits = model.head(hidden)
            for field in target_fields:
                _, target, valid = _valid_targets(batch, field)
                if target.numel() == 0:
                    continue
                prediction = logits[field][:, :-1, :][valid]
                loss = torch.nn.functional.cross_entropy(prediction, target)
                gradient = torch.autograd.grad(
                    loss,
                    hidden,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                selected = gradient[:, :-1, :][valid]
                rms = selected.square().mean().sqrt()
                sums[field] += float(rms.item())
                counts[field] += 1
    finally:
        if was_training:
            model.train()
    return {
        field: sums[field] / counts[field] if counts[field] else float("nan")
        for field in target_fields
    }


def _copy_baseline_loss(
    current: torch.Tensor,
    target: torch.Tensor,
    *,
    vocab_size: int,
    epsilon: float,
) -> float:
    """Return summed CE for a smoothed ``next == current`` baseline."""
    if current.numel() == 0:
        return 0.0
    usable_classes = vocab_size - 1  # PAD is never a valid next-event class.
    if usable_classes <= 1:
        return 0.0 if torch.equal(current, target) else float("inf")
    matches = int(current.eq(target).sum().item())
    misses = target.numel() - matches
    match_loss = -math.log1p(-epsilon)
    miss_loss = -math.log(epsilon / (usable_classes - 1))
    return matches * match_loss + misses * miss_loss


@torch.no_grad()
def field_diagnostics(
    model: OrderFlowFM,
    loader: DataLoader,
    device: torch.device,
    target_fields: tuple[str, ...],
    *,
    train_unigram_counts: Mapping[str, Sequence[int] | np.ndarray] | None,
    max_batches: int = 200,
    copy_epsilon: float = 1e-6,
) -> FieldEvaluationReport:
    """
    Evaluate CE, accuracy, entropy normalization and a copy baseline.

    All model metrics are aggregated per valid token, not as an unweighted mean of
    batches.  Balanced accuracy is macro recall over target classes that occur in
    the evaluated sample.  ``copy_baseline_ce`` uses a deterministic copy predictor
    with ``copy_epsilon`` smoothing so a non-copy transition has finite CE.
    """
    if max_batches < 1:
        msg = "max_batches must be >= 1"
        raise ValueError(msg)
    if not 0.0 < copy_epsilon < 1.0:
        msg = "copy_epsilon must be between zero and one"
        raise ValueError(msg)

    ce_sums = dict.fromkeys(target_fields, 0.0)
    copy_sums = dict.fromkeys(target_fields, 0.0)
    n_predictions = dict.fromkeys(target_fields, 0)
    supports: dict[str, np.ndarray] = {
        field: np.zeros(0, dtype=np.int64) for field in target_fields
    }
    correct_by_class: dict[str, np.ndarray] = {
        field: np.zeros(0, dtype=np.int64) for field in target_fields
    }

    was_training = model.training
    model.eval()
    try:
        for batch_index, cpu_batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            batch = {key: value.to(device) for key, value in cpu_batch.items()}
            logits = model(batch)
            for field in target_fields:
                if field not in logits:
                    msg = f"model output is missing target field {field!r}"
                    raise KeyError(msg)
                current, target, valid = _valid_targets(batch, field)
                if target.numel() == 0:
                    continue
                pred_logits = logits[field][:, :-1, :][valid]
                vocab_size = pred_logits.size(-1)
                ce_sums[field] += float(
                    torch.nn.functional.cross_entropy(
                        pred_logits, target, reduction="sum"
                    ).item()
                )
                predicted = pred_logits.argmax(dim=-1)
                support = torch.bincount(target, minlength=vocab_size).cpu().numpy()
                correct = (
                    torch.bincount(target[predicted.eq(target)], minlength=vocab_size)
                    .cpu()
                    .numpy()
                )
                if supports[field].size < vocab_size:
                    supports[field] = np.pad(
                        supports[field], (0, vocab_size - supports[field].size)
                    )
                    correct_by_class[field] = np.pad(
                        correct_by_class[field],
                        (0, vocab_size - correct_by_class[field].size),
                    )
                supports[field][:vocab_size] += support
                correct_by_class[field][:vocab_size] += correct
                copy_sums[field] += _copy_baseline_loss(
                    current,
                    target,
                    vocab_size=vocab_size,
                    epsilon=copy_epsilon,
                )
                n_predictions[field] += int(target.numel())
    finally:
        if was_training:
            model.train()

    if not any(n_predictions.values()):
        msg = "evaluation loader produced no valid next-event targets"
        raise ValueError(msg)

    fields: dict[str, FieldEval] = {}
    source = "train" if train_unigram_counts is not None else "evaluation_fallback"
    for field in target_fields:
        count = n_predictions[field]
        if count == 0:
            fields[field] = FieldEval(*([float("nan")] * 7))
            continue
        ce = ce_sums[field] / count
        support = supports[field]
        present = support > 0
        recalls = correct_by_class[field][present] / support[present]
        top1 = float(correct_by_class[field].sum() / count)
        balanced = float(recalls.mean()) if recalls.size else float("nan")
        entropy_counts = (
            np.asarray(train_unigram_counts[field], dtype=np.int64)
            if train_unigram_counts is not None
            else support
        )
        entropy = unigram_entropy(entropy_counts)
        normalized = ce / entropy if entropy > 0 else float("nan")
        fields[field] = FieldEval(
            ce=ce,
            perplexity=math.exp(ce) if ce < 709 else float("inf"),
            top1_accuracy=top1,
            balanced_accuracy=balanced,
            unigram_entropy=entropy,
            normalized_ce=normalized,
            copy_baseline_ce=copy_sums[field] / count,
        )
    return FieldEvaluationReport(
        fields=fields,
        n_predictions=n_predictions,
        unigram_source=source,
    )


@torch.no_grad()
def field_perplexity(
    model: OrderFlowFM,
    loader: DataLoader,
    device: torch.device,
    target_fields: tuple[str, ...],
    *,
    max_batches: int = 200,
) -> PerplexityReport:
    """Return legacy per-field CE/perplexity using token-weighted aggregation."""
    report = field_diagnostics(
        model,
        loader,
        device,
        target_fields,
        train_unigram_counts=None,
        max_batches=max_batches,
    )
    return PerplexityReport(
        ce={field: metrics.ce for field, metrics in report.fields.items()}
    )


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
    """对 checkpoint 跑完整字段诊断并落盘 JSON。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument(
        "--unigram-max-batches",
        type=int,
        default=200,
        help="用于估计训练集 unigram entropy 的分层窗口批数",
    )
    parser.add_argument(
        "--validation-plan",
        type=Path,
        help="固定验证窗口 JSON；不存在时按分层规则创建",
    )
    parser.add_argument(
        "--validation-windows",
        type=int,
        help="新建固定计划时的窗口数；默认 max_batches * batch_size",
    )
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--copy-epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--gradient-norm-batches",
        type=int,
        default=1,
        help="计算各字段 hidden gradient norm 的验证批数；0 表示跳过",
    )
    parser.add_argument("--device", default="cpu", help="训练占满 GPU 时用 cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    vocab_path = Path(cfg["data"]["vocab"])
    vocab = _load_vocab(vocab_path)
    model = load_checkpoint(
        args.checkpoint,
        device,
        vocab_path=vocab_path,
    )
    target_fields = tuple(
        cfg["model"].get(
            "target_fields", model.cfg.target_fields or DEFAULT_TARGET_FIELDS
        )
    )

    manifest = Manifest.load(Path(cfg["data"]["manifest"]))
    shards = manifest.split(args.split)
    if not shards:
        msg = f"no shards for split={args.split}"
        raise SystemExit(msg)
    data = cfg["data"]
    dataset_options = {
        "context": data["context"],
        "stride": data["stride"],
        "min_len": data["min_len"],
        "cache_size": min(8, int(data.get("cache_size", 8))),
    }
    if isinstance(vocab, VocabV2):
        ds = EventWindowDatasetV2(shards, vocab=vocab, **dataset_options)
        collate_fn = collate_windows_v2
    else:
        ds = EventWindowDataset(shards, **dataset_options)
        collate_fn = collate_windows
    batch_size = 1 if str(device).startswith("cpu") else int(cfg["optim"]["batch_size"])
    plan_path = args.validation_plan
    if plan_path is None:
        plan_path = args.out.with_suffix(".windows.json")
    if plan_path.exists():
        plan = ValidationSamplePlan.load(plan_path)
        plan.validate(
            shards,
            context=data["context"],
            stride=data["stride"],
            min_len=data["min_len"],
        )
    else:
        plan = build_validation_sample_plan(
            shards,
            context=data["context"],
            stride=data["stride"],
            min_len=data["min_len"],
            seed=args.sampling_seed,
            max_windows=args.validation_windows or args.max_batches * batch_size,
        )
        plan.save(plan_path)
        logger.info("created fixed validation plan: %s", plan_path)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=FixedValidationSampler(plan),
        collate_fn=collate_fn,
        num_workers=0,
    )

    train_shards = manifest.split("train")
    if not train_shards:
        msg = "no shards for split=train; cannot estimate unigram entropy"
        raise SystemExit(msg)
    if isinstance(vocab, VocabV2):
        train_ds = EventWindowDatasetV2(
            train_shards,
            vocab=vocab,
            **dataset_options,
        )
    else:
        train_ds = EventWindowDataset(train_shards, **dataset_options)
    train_plan = build_validation_sample_plan(
        train_shards,
        context=data["context"],
        stride=data["stride"],
        min_len=data["min_len"],
        seed=args.sampling_seed,
        max_windows=args.unigram_max_batches * batch_size,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=FixedValidationSampler(train_plan),
        collate_fn=collate_fn,
        num_workers=0,
    )
    unigram_counts = collect_unigram_counts(
        train_loader,
        target_fields,
        field_sizes=model.cfg.field_sizes,
        max_batches=args.unigram_max_batches,
    )
    report = field_diagnostics(
        model,
        loader,
        device,
        target_fields,
        train_unigram_counts=unigram_counts,
        max_batches=args.max_batches,
        copy_epsilon=args.copy_epsilon,
    )
    gradient_norms = per_field_gradient_norms(
        model,
        loader,
        device,
        target_fields,
        max_batches=args.gradient_norm_batches,
    )
    payload = {
        "created_utc": datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ"),
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "split": args.split,
        "max_batches": args.max_batches,
        "device": str(device),
        "n_shards": len(shards),
        "validation_plan": str(plan_path.resolve()),
        "validation_plan_source_fingerprint": plan.source_fingerprint,
        "validation_windows": len(plan.windows),
        "unigram_windows": len(train_plan.windows),
        "per_field_gradient_norm": gradient_norms,
        **report.to_dict(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "wrote %s total_ce=%.4f normalized_ce=%.4f top1=%s",
        args.out,
        report.total_ce,
        report.total_normalized_ce,
        {name: round(field.top1_accuracy, 3) for name, field in report.fields.items()},
    )


if __name__ == "__main__":
    main()
