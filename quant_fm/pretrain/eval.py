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
import hashlib
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from quant_fm.manifest.build_manifest import Manifest
from quant_fm.manifest.validation import (
    sha256_file,
    validate_manifest_shards,
    validate_manifest_vocab_contract,
)
from quant_fm.pretrain.data_contract import build_pretrain_data_contract
from quant_fm.pretrain.dataset import (
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
    from collections.abc import Sequence

    from quant_fm.pretrain.model import OrderFlowFM

logger = logging.getLogger(__name__)

_NORMALIZATION_CONTRACT_VERSION = "train_unigram_normalization_v3"


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _normalization_contract(
    *,
    target_fields: tuple[str, ...],
    train_plan: ValidationSamplePlan,
    unigram_counts: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Bind normalized CE to the exact train sample, counts, and denominators."""
    if set(unigram_counts) != set(target_fields):
        msg = "train unigram count fields must exactly match target_fields"
        raise ValueError(msg)
    counts_payload: dict[str, list[int]] = {}
    for field in target_fields:
        values = np.asarray(unigram_counts[field])
        if (
            values.ndim != 1
            or values.size == 0
            or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0)
        ):
            msg = f"train unigram counts for {field!r} must be non-negative integers"
            raise ValueError(msg)
        counts_payload[field] = [int(value) for value in values.tolist()]
    entropy_payload = {
        field: float(unigram_entropy(unigram_counts[field])) for field in target_fields
    }
    if any(
        not math.isfinite(entropy) or entropy <= 0
        for entropy in entropy_payload.values()
    ):
        msg = "train unigram entropy must be finite and positive for every target"
        raise ValueError(msg)
    contract = {
        "format_version": _NORMALIZATION_CONTRACT_VERSION,
        "target_fields": list(target_fields),
        "train_unigram_plan_sha256": train_plan.sha256,
        "train_unigram_plan_source_fingerprint": train_plan.source_fingerprint,
        "train_unigram_windows": len(train_plan.windows),
        "train_unigram_counts": counts_payload,
        "train_unigram_counts_sha256": _canonical_sha256(counts_payload),
        "train_unigram_entropy": entropy_payload,
    }
    return {**contract, "sha256": _canonical_sha256(contract)}


def evaluation_batch_size(cfg: Mapping[str, object], device: torch.device) -> int:
    """解析 v1 ``batch_size`` 或 v2 ``micro_batch_size`` 的评估 batch。"""
    if device.type == "cpu":
        return 1
    optim = cfg.get("optim")
    if not isinstance(optim, Mapping):
        msg = "config.optim must be a mapping"
        raise TypeError(msg)
    value = optim.get("micro_batch_size", optim.get("batch_size", 1))
    batch_size = int(value)  # type: ignore[arg-type]
    if batch_size < 1:
        msg = "evaluation batch size must be positive"
        raise ValueError(msg)
    return batch_size


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
    evaluated_windows: int
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
            "evaluated_windows": self.evaluated_windows,
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
    counts, _, _ = _collect_unigram_counts_with_stats(
        loader,
        target_fields,
        field_sizes=field_sizes,
        max_batches=max_batches,
    )
    return counts


@torch.no_grad()
def _collect_unigram_counts_with_stats(
    loader: DataLoader,
    target_fields: tuple[str, ...],
    *,
    field_sizes: Mapping[str, int] | None = None,
    max_batches: int | None = None,
) -> tuple[dict[str, np.ndarray], int, dict[str, int]]:
    """Count targets and return exact loader-window and per-field consumption."""
    if max_batches is not None and max_batches < 1:
        msg = "max_batches must be >= 1 when provided"
        raise ValueError(msg)
    counts = {
        field: np.zeros(int(field_sizes.get(field, 0)), dtype=np.int64)
        if field_sizes is not None
        else np.zeros(0, dtype=np.int64)
        for field in target_fields
    }
    evaluated_windows = 0
    prediction_counts = dict.fromkeys(target_fields, 0)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        evaluated_windows += int(batch["attention_mask"].shape[0])
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
            prediction_counts[field] += int(target.numel())
    return counts, evaluated_windows, prediction_counts


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
    max_batches: int | None = None,
    copy_epsilon: float = 1e-6,
) -> FieldEvaluationReport:
    """
    Evaluate CE, accuracy, entropy normalization and a copy baseline.

    All model metrics are aggregated per valid token, not as an unweighted mean of
    batches.  Balanced accuracy is macro recall over target classes that occur in
    the evaluated sample.  ``copy_baseline_ce`` uses a deterministic copy predictor
    with ``copy_epsilon`` smoothing so a non-copy transition has finite CE.
    """
    if max_batches is not None and max_batches < 1:
        msg = "max_batches must be >= 1 when provided"
        raise ValueError(msg)
    if not 0.0 < copy_epsilon < 1.0:
        msg = "copy_epsilon must be between zero and one"
        raise ValueError(msg)

    ce_sums = dict.fromkeys(target_fields, 0.0)
    copy_sums = dict.fromkeys(target_fields, 0.0)
    n_predictions = dict.fromkeys(target_fields, 0)
    evaluated_windows = 0
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
            if max_batches is not None and batch_index >= max_batches:
                break
            evaluated_windows += int(cpu_batch["attention_mask"].shape[0])
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
        evaluated_windows=evaluated_windows,
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


def resolve_unigram_windows(
    *,
    unigram_windows: int | None,
    legacy_unigram_max_batches: int | None,
    default: int = 200,
) -> int:
    """Resolve a device-independent exact train-unigram window count."""
    if unigram_windows is not None and legacy_unigram_max_batches is not None:
        if unigram_windows != legacy_unigram_max_batches:
            msg = (
                "--unigram-windows and deprecated --unigram-max-batches "
                "must agree when both are provided"
            )
            raise ValueError(msg)
    value = (
        unigram_windows
        if unigram_windows is not None
        else legacy_unigram_max_batches
        if legacy_unigram_max_batches is not None
        else default
    )
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = "unigram window count must be a positive integer"
        raise ValueError(msg)
    return value


def require_exact_plan_windows(
    plan: ValidationSamplePlan,
    *,
    requested_windows: int,
    context: str,
) -> None:
    """Reject a plan that cannot prove it selected exactly the requested N windows."""
    if plan.max_windows != requested_windows or len(plan.windows) != requested_windows:
        msg = (
            f"{context} requires exactly {requested_windows} windows, got "
            f"selection.max_windows={plan.max_windows!r}, "
            f"selected={len(plan.windows)}, candidates={plan.total_candidate_windows}"
        )
        raise ValueError(msg)


def resolve_checkpoint_target_fields(
    checkpoint_target_fields: Sequence[str],
    configured_target_fields: object,
) -> tuple[str, ...]:
    """Use checkpoint heads as truth and reject YAML target-field substitution."""
    checkpoint_fields = tuple(checkpoint_target_fields)
    if (
        not checkpoint_fields
        or not all(isinstance(field, str) and field for field in checkpoint_fields)
        or len(set(checkpoint_fields)) != len(checkpoint_fields)
    ):
        msg = "checkpoint target_fields must be unique non-empty strings"
        raise ValueError(msg)
    if configured_target_fields is not None:
        if not isinstance(configured_target_fields, list | tuple):
            msg = "config model.target_fields must be a sequence"
            raise TypeError(msg)
        configured = tuple(configured_target_fields)
        if configured != checkpoint_fields:
            msg = (
                "config model.target_fields must exactly match checkpoint target_fields: "
                f"config={configured}, checkpoint={checkpoint_fields}"
            )
            raise ValueError(msg)
    return checkpoint_fields


def stylized_facts(returns: np.ndarray, *, max_lag: int = 50) -> dict[str, float]:
    """
    计算超额峰度与波动率聚集自相关。

    Parameters
    ----------
    returns : np.ndarray
        一维（对数）收益序列，如中间价变化。
    max_lag : int
        |收益| 自相关平均的最大滞后。

    Returns
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
    parser.add_argument(
        "--max-batches",
        type=int,
        default=100,
        help=(
            "deprecated CE batch cap; only supplies the exact window limit when "
            "creating a plan without --validation-windows"
        ),
    )
    parser.add_argument(
        "--unigram-windows",
        type=int,
        help="训练 split unigram entropy 计划的精确窗口数（默认 200）",
    )
    parser.add_argument(
        "--unigram-max-batches",
        type=int,
        help=(
            "deprecated alias for --unigram-windows; interpreted as an exact "
            "window count, independent of device batch size"
        ),
    )
    parser.add_argument(
        "--validation-plan",
        type=Path,
        help="固定验证窗口 JSON；不存在时按分层规则创建",
    )
    parser.add_argument(
        "--validation-windows",
        type=int,
        help="新建固定计划时的精确窗口数；默认采用 --max-batches 的数值",
    )
    parser.add_argument(
        "--train-unigram-plan",
        type=Path,
        help="冻结训练 split unigram 分母的窗口计划 JSON",
    )
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument(
        "--liquidity-json",
        type=Path,
        help="构建/验证分层计划所用的时点流动性输入",
    )
    parser.add_argument("--liquidity-buckets", type=int, default=3)
    parser.add_argument("--activity-buckets", type=int, default=3)
    parser.add_argument("--windows-per-stratum", type=int)
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

    if args.max_batches < 1:
        msg = "--max-batches must be a positive plan window limit"
        raise ValueError(msg)
    if args.validation_windows is not None and args.validation_windows < 1:
        msg = "--validation-windows must be a positive integer"
        raise ValueError(msg)
    unigram_window_limit = resolve_unigram_windows(
        unigram_windows=args.unigram_windows,
        legacy_unigram_max_batches=args.unigram_max_batches,
    )
    if args.unigram_max_batches is not None:
        logger.warning(
            "--unigram-max-batches is deprecated and now denotes an exact window "
            "count; use --unigram-windows"
        )

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    vocab_path = Path(cfg["data"]["vocab"])
    vocab = _load_vocab(vocab_path)
    manifest_path = Path(cfg["data"]["manifest"])
    manifest = Manifest.load(manifest_path)
    validate_manifest_vocab_contract(manifest, vocab, context="pretrain evaluation")
    shards = manifest.split(args.split)
    if not shards:
        msg = f"no shards for split={args.split}"
        raise SystemExit(msg)
    train_shards = manifest.split("train")
    if not train_shards:
        msg = "no shards for split=train; cannot estimate unigram entropy"
        raise SystemExit(msg)
    selected_by_path = {shard.path: shard for shard in [*shards, *train_shards]}
    validate_manifest_shards(
        manifest,
        vocab,
        shards=list(selected_by_path.values()),
        context="pretrain evaluation",
        expected_tokens_root=(
            manifest_path.parent.parent / "tokens"
            if manifest_path.parent.name == "data"
            else manifest_path.parent / "tokens"
        ),
    )
    expected_pretrain_data_contract = build_pretrain_data_contract(
        manifest_path=manifest_path,
        manifest=manifest,
        vocab_path=vocab_path,
        vocab=vocab,
    )
    checkpoint_sha256 = sha256_file(args.checkpoint)
    model = load_checkpoint(
        args.checkpoint,
        device,
        vocab_path=vocab_path,
        checkpoint_sha256=checkpoint_sha256,
        expected_pretrain_data_contract=expected_pretrain_data_contract,
    )
    configured_target_fields = cfg["model"].get("target_fields")
    target_fields = resolve_checkpoint_target_fields(
        model.cfg.target_fields,
        configured_target_fields,
    )
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
    batch_size = evaluation_batch_size(cfg, device)
    plan_path = args.validation_plan
    if plan_path is None:
        plan_path = args.out.with_suffix(".windows.json")
    liquidity_values = None
    if args.liquidity_json is not None:
        raw_liquidity = json.loads(args.liquidity_json.read_text(encoding="utf-8"))
        if not isinstance(raw_liquidity, list):
            msg = "liquidity JSON must be a list of date/market/symbol/liquidity rows"
            raise TypeError(msg)
        liquidity_values = {
            (
                str(row["date"]),
                str(row["market"]).upper(),
                str(row["symbol"]).zfill(6),
            ): float(row["liquidity"])
            for row in raw_liquidity
        }
    if plan_path.exists():
        plan = ValidationSamplePlan.load(plan_path)
        plan.validate(
            shards,
            context=data["context"],
            stride=data["stride"],
            min_len=data["min_len"],
            liquidity_values=liquidity_values,
            n_liquidity_buckets=args.liquidity_buckets,
            n_activity_buckets=args.activity_buckets,
        )
        if args.validation_windows is not None:
            require_exact_plan_windows(
                plan,
                requested_windows=args.validation_windows,
                context="validation plan",
            )
    else:
        plan = build_validation_sample_plan(
            shards,
            context=data["context"],
            stride=data["stride"],
            min_len=data["min_len"],
            seed=args.sampling_seed,
            max_windows=args.validation_windows or args.max_batches,
            windows_per_stratum=args.windows_per_stratum,
            liquidity_values=liquidity_values,
            n_liquidity_buckets=args.liquidity_buckets,
            n_activity_buckets=args.activity_buckets,
        )
        if args.validation_windows is not None:
            require_exact_plan_windows(
                plan,
                requested_windows=args.validation_windows,
                context="validation plan",
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

    if isinstance(vocab, VocabV2):
        train_ds = EventWindowDatasetV2(
            train_shards,
            vocab=vocab,
            **dataset_options,
        )
    else:
        train_ds = EventWindowDataset(train_shards, **dataset_options)
    train_plan_path = args.train_unigram_plan
    if train_plan_path is None:
        train_plan_path = args.out.with_suffix(".train-unigram.windows.json")
    if train_plan_path.resolve() == plan_path.resolve():
        msg = "validation and train-unigram plans must use different files"
        raise ValueError(msg)
    expected_train_plan = build_validation_sample_plan(
        train_shards,
        context=data["context"],
        stride=data["stride"],
        min_len=data["min_len"],
        seed=args.sampling_seed,
        max_windows=unigram_window_limit,
    )
    require_exact_plan_windows(
        expected_train_plan,
        requested_windows=unigram_window_limit,
        context="train-unigram plan",
    )
    if train_plan_path.exists():
        train_plan = ValidationSamplePlan.load(train_plan_path)
        train_plan.validate(
            train_shards,
            context=data["context"],
            stride=data["stride"],
            min_len=data["min_len"],
        )
        if train_plan.sha256 != expected_train_plan.sha256:
            msg = (
                "train-unigram plan does not match the requested seed/window count; "
                "use the frozen evaluation settings or a new plan path"
            )
            raise ValueError(msg)
    else:
        train_plan = expected_train_plan
        train_plan.save(train_plan_path)
        logger.info("created fixed train-unigram plan: %s", train_plan_path)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=FixedValidationSampler(train_plan),
        collate_fn=collate_fn,
        num_workers=0,
    )
    (
        unigram_counts,
        train_unigram_evaluated_windows,
        train_unigram_prediction_counts,
    ) = _collect_unigram_counts_with_stats(
        train_loader,
        target_fields,
        field_sizes=model.cfg.field_sizes,
        max_batches=None,
    )
    if train_unigram_evaluated_windows != len(train_plan.windows):
        msg = "train-unigram counting did not consume the complete frozen plan"
        raise RuntimeError(msg)
    max_train_predictions = sum(
        max(window.length - 1, 0) for window in train_plan.windows
    )
    for field in target_fields:
        count_sum = int(np.asarray(unigram_counts[field], dtype=np.int64).sum())
        if (
            train_unigram_prediction_counts[field] != count_sum
            or count_sum <= 0
            or count_sum > max_train_predictions
        ):
            msg = f"train-unigram target count is inconsistent for {field!r}"
            raise RuntimeError(msg)
    normalization = _normalization_contract(
        target_fields=target_fields,
        train_plan=train_plan,
        unigram_counts=unigram_counts,
    )
    report = field_diagnostics(
        model,
        loader,
        device,
        target_fields,
        train_unigram_counts=unigram_counts,
        max_batches=None,
        copy_epsilon=args.copy_epsilon,
    )
    if report.evaluated_windows != len(plan.windows):
        msg = (
            "pretraining CE diagnostics did not consume the complete frozen "
            "validation plan"
        )
        raise RuntimeError(msg)
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
        "checkpoint_sha256": checkpoint_sha256,
        "config": str(args.config.resolve()),
        "split": args.split,
        "max_batches": None,
        "deprecated_max_batches_plan_window_limit": args.max_batches,
        "evaluation_scope": "full_validation_plan",
        "validation_plan_window_limit": args.validation_windows or args.max_batches,
        "validation_plan_window_mode": (
            "exact" if args.validation_windows is not None else "legacy_cap"
        ),
        "train_unigram_window_limit": unigram_window_limit,
        "device": str(device),
        "n_shards": len(shards),
        "validation_plan": str(plan_path.resolve()),
        "validation_plan_sha256": plan.sha256,
        "validation_plan_source_fingerprint": plan.source_fingerprint,
        "validation_windows": len(plan.windows),
        "train_unigram_plan": str(train_plan_path.resolve()),
        "train_unigram_plan_sha256": train_plan.sha256,
        "train_unigram_plan_source_fingerprint": train_plan.source_fingerprint,
        "train_unigram_windows": len(train_plan.windows),
        "unigram_windows": len(train_plan.windows),
        "normalization_target_fields": normalization["target_fields"],
        "checkpoint_target_fields": list(target_fields),
        "train_unigram_counts": normalization["train_unigram_counts"],
        "train_unigram_counts_sha256": normalization["train_unigram_counts_sha256"],
        "train_unigram_evaluated_windows": train_unigram_evaluated_windows,
        "train_unigram_prediction_counts": train_unigram_prediction_counts,
        "normalization_contract_sha256": normalization["sha256"],
        "per_field_gradient_norm": gradient_norms,
        **report.to_dict(),
    }
    if sha256_file(args.checkpoint) != checkpoint_sha256:
        msg = "checkpoint changed while pretraining evaluation was running"
        raise RuntimeError(msg)
    for identity_path, expected_sha, context in (
        (plan_path, plan.sha256, "validation"),
        (train_plan_path, train_plan.sha256, "train-unigram"),
    ):
        if ValidationSamplePlan.load(identity_path).sha256 != expected_sha:
            msg = f"{context} plan changed while pretraining evaluation was running"
            raise RuntimeError(msg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    logger.info(
        "wrote %s total_ce=%.4f normalized_ce=%.4f top1=%s",
        args.out,
        report.total_ce,
        report.total_normalized_ce,
        {name: round(field.top1_accuracy, 3) for name, field in report.fields.items()},
    )


if __name__ == "__main__":
    main()
