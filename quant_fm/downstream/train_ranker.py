"""
置换等变的横截面排序器与 Top-K 选股目标。

排序主头使用多截断 sampled LambdaNDCG，全截面 IC 作为稳定项，独立辅助头
用 SmoothL1 学习稳健标准化超额收益。所有损失按交易日单独计算，避免大横截面
对训练目标产生额外权重。
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RankerConfig:
    """:class:`CrossSectionalRanker` 的超参数。"""

    in_dim: int
    hidden: int = 128
    depth: int = 2
    n_heads: int = 4
    dropout: float = 0.3
    use_attention: bool = True


@dataclass(frozen=True, slots=True)
class RankerObjectiveConfig:
    """Top-K 选股联合目标。"""

    ndcg_ks: tuple[int, ...] = (50, 300, 350)
    ndcg_k_weights: tuple[float, ...] = (0.20, 0.60, 0.20)
    head_weight: float = 1.0
    global_ic_weight: float = 0.30
    aux_huber_weight: float = 0.05
    aux_huber_beta: float = 0.5
    pair_samples_per_day: int = 8192
    hard_pair_fraction: float = 0.75
    min_label_rank_gap: float = 0.02
    score_temperature: float = 1.0

    def validate(self) -> None:
        """校验多截断、采样与损失权重。"""
        if not self.ndcg_ks or len(self.ndcg_ks) != len(self.ndcg_k_weights):
            msg = "ndcg_ks and ndcg_k_weights must have the same non-zero length"
            raise ValueError(msg)
        if min(self.ndcg_ks) < 1:
            msg = "all ndcg_ks must be positive"
            raise ValueError(msg)
        if min(self.ndcg_k_weights) < 0 or sum(self.ndcg_k_weights) <= 0:
            msg = "ndcg_k_weights must be non-negative and sum to a positive value"
            raise ValueError(msg)
        if min(self.head_weight, self.global_ic_weight, self.aux_huber_weight) < 0:
            msg = "objective component weights must be non-negative"
            raise ValueError(msg)
        if self.aux_huber_beta <= 0 or self.score_temperature <= 0:
            msg = "aux_huber_beta and score_temperature must be positive"
            raise ValueError(msg)
        if self.pair_samples_per_day < 1:
            msg = "pair_samples_per_day must be positive"
            raise ValueError(msg)
        if not 0 <= self.hard_pair_fraction <= 1:
            msg = "hard_pair_fraction must be in [0, 1]"
            raise ValueError(msg)
        if not 0 <= self.min_label_rank_gap < 1:
            msg = "min_label_rank_gap must be in [0, 1)"
            raise ValueError(msg)

    @property
    def primary_k(self) -> int:
        """返回权重最高的正式持仓截断。"""
        return self.ndcg_ks[int(np.argmax(self.ndcg_k_weights))]


@dataclass(slots=True)
class RankerTrainingResult:
    """包含最佳验证模型与逐 epoch 轨迹的训练结果。"""

    model: CrossSectionalRanker
    history: list[dict[str, float | int | None]]
    best_epoch: int
    best_val_ic: float | None
    best_val_ndcg: float | None
    best_val_top_spread: float | None
    best_selection_score: float
    stopped_early: bool
    objective: RankerObjectiveConfig


class _RowMLP(nn.Module):
    """逐行 MLP（对各股票相同变换 -> 等变）。"""

    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """带残差的逐行变换。"""
        return self.norm(x + self.net(x))


class _CrossSectionAttention(nn.Module):
    """横截面上的无位置多头自注意力。"""

    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """跨股票注意力（无位置编码 => 等变）。"""
        out, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + out)


class CrossSectionalRanker(nn.Module):
    """浅层、置换等变的日度横截面打分器。"""

    def __init__(self, cfg: RankerConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.in_dim, cfg.hidden)
        layers: list[nn.Module] = []
        for _ in range(cfg.depth):
            layers.append(_RowMLP(cfg.hidden, cfg.hidden * 2, cfg.dropout))
            if cfg.use_attention:
                layers.append(
                    _CrossSectionAttention(cfg.hidden, cfg.n_heads, cfg.dropout)
                )
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(cfg.hidden, 1)
        self.aux_out = nn.Linear(cfg.hidden, 1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """对 ``[N, F]`` 横截面编码，保持股票维度置换等变。"""
        h = self.proj(x).unsqueeze(0)  # [1, N, H]
        for layer in self.layers:
            h = layer(h)
        return h.squeeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对横截面 ``[N, F]`` 打分 -> ``[N]``。"""
        return self.out(self._encode(x)).squeeze(-1)

    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回排序分数和仅用于训练的超额收益辅助预测。"""
        hidden = self._encode(x)
        return self.out(hidden).squeeze(-1), self.aux_out(hidden).squeeze(-1)


def _pearson(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """可微 Pearson 相关。"""
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = pred.norm() * target.norm() + 1e-8
    return (pred * target).sum() / denom


def rank_ic_loss(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """横截面相关负值（最大化 RankIC）。"""
    return -_pearson(pred, label)


def normalize_daily_scores(pred: torch.Tensor) -> torch.Tensor:
    """日内标准化分数，使 pairwise 梯度不受原始输出尺度影响。"""
    centered = pred - pred.mean()
    scale = centered.square().mean().sqrt().clamp_min(1e-6)
    return centered / scale


def _canonical_order(
    pred: torch.Tensor,
    label: torch.Tensor,
    head_gain: torch.Tensor,
) -> np.ndarray:
    """以数值而非输入行号建立采样顺序，保持置换不变性。"""
    pred_np = pred.detach().to(dtype=torch.float64, device="cpu").numpy()
    label_np = label.detach().to(dtype=torch.float64, device="cpu").numpy()
    gain_np = head_gain.detach().to(dtype=torch.float64, device="cpu").numpy()
    # np.lexsort uses the final key as primary: label, then gain, then score.
    return np.lexsort((-pred_np, -gain_np, -label_np))


def _draw_pair_group(
    *,
    first_pool: np.ndarray,
    second_pool: np.ndarray,
    label: np.ndarray,
    limit: int,
    min_gap: float,
    generator: torch.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """有界有放回采样，返回按真实标签高低定向的 pair。"""
    if limit <= 0 or first_pool.size == 0 or second_pool.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    draw_count = max(8, limit * 4)
    generator_device = generator.device
    first_draw = (
        torch.randint(
            first_pool.size,
            (draw_count,),
            generator=generator,
            device=generator_device,
        )
        .cpu()
        .numpy()
    )
    second_draw = (
        torch.randint(
            second_pool.size,
            (draw_count,),
            generator=generator,
            device=generator_device,
        )
        .cpu()
        .numpy()
    )
    left = first_pool[first_draw]
    right = second_pool[second_draw]
    left_label = label[left]
    right_label = label[right]
    high = np.where(left_label >= right_label, left, right)
    low = np.where(left_label >= right_label, right, left)
    valid = (high != low) & (np.abs(left_label - right_label) >= min_gap)
    return high[valid][:limit], low[valid][:limit]


def _sample_lambda_pairs(
    *,
    pred: torch.Tensor,
    label: torch.Tensor,
    head_gain: torch.Tensor,
    cfg: RankerObjectiveConfig,
    generator: torch.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """在真实/预测头部与边界优先采样，并保留四分之一全局 pair。"""
    order = _canonical_order(pred, label, head_gain)
    canonical_pred = pred.detach().to(dtype=torch.float64, device="cpu").numpy()[order]
    canonical_label = (
        label.detach().to(dtype=torch.float64, device="cpu").numpy()[order]
    )
    canonical_gain = (
        head_gain.detach().to(dtype=torch.float64, device="cpu").numpy()[order]
    )
    n_names = canonical_label.size
    if n_names < 2:
        empty = np.empty(0, dtype=np.int64)
        return order, canonical_gain, empty, empty, empty.astype(np.float64)

    # Ties in score are resolved by the true label only for the detached metric
    # weight. The differentiable pairwise term itself never uses this tie-break.
    pred_order = np.lexsort((-canonical_label, -canonical_pred))
    pred_rank = np.empty(n_names, dtype=np.int64)
    pred_rank[pred_order] = np.arange(1, n_names + 1)
    max_k = min(max(cfg.ndcg_ks), n_names)
    true_head = np.arange(max_k, dtype=np.int64)
    focus = pred_rank <= max_k
    true_rank = np.arange(1, n_names + 1)
    for raw_k in cfg.ndcg_ks:
        cutoff = min(raw_k, n_names)
        width = max(5, round(cutoff * 0.20))
        focus |= np.abs(pred_rank - cutoff) <= width
        focus |= np.abs(true_rank - cutoff) <= width
    predicted_head_and_boundary = np.flatnonzero(focus).astype(np.int64)
    all_names = np.arange(n_names, dtype=np.int64)

    daily_budget = min(cfg.pair_samples_per_day, n_names * (n_names - 1) // 2)
    hard_limit = round(daily_budget * cfg.hard_pair_fraction)
    random_limit = daily_budget - hard_limit
    hard_high, hard_low = _draw_pair_group(
        first_pool=true_head,
        second_pool=predicted_head_and_boundary,
        label=canonical_label,
        limit=hard_limit,
        min_gap=cfg.min_label_rank_gap,
        generator=generator,
    )
    random_high, random_low = _draw_pair_group(
        first_pool=all_names,
        second_pool=all_names,
        label=canonical_label,
        limit=random_limit,
        min_gap=cfg.min_label_rank_gap,
        generator=generator,
    )
    high = np.concatenate((hard_high, random_high))
    low = np.concatenate((hard_low, random_low))
    if high.size == 0:
        return order, canonical_gain, high, low, np.empty(0, dtype=np.float64)

    normalized_k_weights = np.asarray(cfg.ndcg_k_weights, dtype=np.float64)
    normalized_k_weights /= normalized_k_weights.sum()
    pair_weight = np.zeros(high.size, dtype=np.float64)
    ideal_gain = np.sort(canonical_gain)[::-1]
    for raw_k, cutoff_weight in zip(cfg.ndcg_ks, normalized_k_weights, strict=True):
        cutoff = min(raw_k, n_names)
        ideal_discount = 1.0 / np.log2(np.arange(2, cutoff + 2))
        ideal_dcg = float(np.dot(ideal_gain[:cutoff], ideal_discount))
        if ideal_dcg <= 1e-12:
            continue
        discount = np.zeros(n_names, dtype=np.float64)
        inside = pred_rank <= cutoff
        discount[inside] = 1.0 / np.log2(pred_rank[inside] + 1.0)
        delta = np.abs(
            (canonical_gain[high] - canonical_gain[low])
            * (discount[high] - discount[low])
        )
        pair_weight += cutoff_weight * delta / ideal_dcg
    positive = pair_weight > 0
    return (
        order,
        canonical_gain,
        high[positive],
        low[positive],
        pair_weight[positive],
    )


def sampled_lambda_ndcg_loss(
    pred: torch.Tensor,
    label: torch.Tensor,
    head_gain: torch.Tensor,
    *,
    objective: RankerObjectiveConfig | None = None,
    generator: torch.Generator | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """
    计算多截断 sampled LambdaNDCG pairwise loss。

    权重由当前预测排名下交换两股造成的 ``delta NDCG@K``
    确定，并在当日的已采样 pair 上归一化。
    """
    cfg = objective or RankerObjectiveConfig()
    cfg.validate()
    if pred.ndim != 1 or label.shape != pred.shape or head_gain.shape != pred.shape:
        msg = "pred, label and head_gain must be one-dimensional tensors of equal size"
        raise ValueError(msg)
    if not (
        torch.isfinite(pred).all()
        and torch.isfinite(label).all()
        and torch.isfinite(head_gain).all()
    ):
        msg = "pred, label and head_gain must be finite"
        raise ValueError(msg)
    if (head_gain < 0).any():
        msg = "head_gain must be non-negative"
        raise ValueError(msg)
    if pred.numel() < 2:
        return pred.sum() * 0.0
    pair_generator = generator
    if pair_generator is None:
        pair_generator = torch.Generator(device="cpu")
        pair_generator.manual_seed(seed)
    normalized = normalize_daily_scores(pred)
    order, _gain, high, low, pair_weight = _sample_lambda_pairs(
        pred=normalized,
        label=label,
        head_gain=head_gain,
        cfg=cfg,
        generator=pair_generator,
    )
    if high.size == 0:
        return pred.sum() * 0.0
    device = pred.device
    canonical_index = torch.as_tensor(order, dtype=torch.long, device=device)
    high_index = torch.as_tensor(high, dtype=torch.long, device=device)
    low_index = torch.as_tensor(low, dtype=torch.long, device=device)
    weights = torch.as_tensor(pair_weight, dtype=pred.dtype, device=device)
    canonical_scores = normalized[canonical_index]
    margin = (canonical_scores[high_index] - canonical_scores[low_index]) / (
        cfg.score_temperature
    )
    pair_loss = F.softplus(-margin)
    return (weights * pair_loss).sum() / weights.sum().clamp_min(1e-12)


def ranker_objective_loss(
    pred: torch.Tensor,
    label: torch.Tensor,
    head_gain: torch.Tensor,
    *,
    aux_pred: torch.Tensor | None = None,
    aux_target: torch.Tensor | None = None,
    objective: RankerObjectiveConfig | None = None,
    generator: torch.Generator | None = None,
    seed: int = 0,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """组合 LambdaNDCG、全局 IC 与独立辅助回归损失。"""
    cfg = objective or RankerObjectiveConfig()
    cfg.validate()
    normalized = normalize_daily_scores(pred)
    ic_loss = 1.0 - _pearson(normalized, label)
    head_loss = sampled_lambda_ndcg_loss(
        pred,
        label,
        head_gain,
        objective=cfg,
        generator=generator,
        seed=seed,
    )
    if (aux_pred is None) != (aux_target is None):
        msg = "aux_pred and aux_target must either both be provided or both be omitted"
        raise ValueError(msg)
    if aux_pred is None or aux_target is None:
        if cfg.aux_huber_weight > 0:
            msg = (
                "aux_pred and aux_target are required when aux_huber_weight is positive"
            )
            raise ValueError(msg)
        aux_loss = pred.sum() * 0.0
    else:
        if aux_pred.shape != pred.shape or aux_target.shape != pred.shape:
            msg = "aux_pred and aux_target must match pred shape"
            raise ValueError(msg)
        if not torch.isfinite(aux_pred).all() or not torch.isfinite(aux_target).all():
            msg = "aux_pred and aux_target must be finite"
            raise ValueError(msg)
        aux_loss = F.smooth_l1_loss(
            aux_pred,
            aux_target,
            beta=cfg.aux_huber_beta,
        )
    total = (
        cfg.head_weight * head_loss
        + cfg.global_ic_weight * ic_loss
        + cfg.aux_huber_weight * aux_loss
    )
    if not return_components:
        return total
    return total, {
        "lambda_ndcg": head_loss,
        "global_ic": ic_loss,
        "aux_huber": aux_loss,
    }


def chronological_ranker_split(
    features: pl.DataFrame,
    *,
    val_days: int = 10,
    purge_days: int = 2,
    min_train_days: int = 20,
    require_validation: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame | None, dict[str, int | str | bool | None]]:
    """按日期尾部留出验证集，并在训练/验证之间执行 purge。"""
    if val_days < 1 or purge_days < 0 or min_train_days < 1:
        msg = "val_days/min_train_days must be positive and purge_days non-negative"
        raise ValueError(msg)
    dates = sorted(str(value) for value in features["date"].unique())
    required_days = min_train_days + purge_days + val_days
    if len(dates) < required_days:
        if require_validation:
            msg = (
                "insufficient ranker dates for chronological validation: "
                f"have={len(dates)}, need={required_days}"
            )
            raise ValueError(msg)
        metadata: dict[str, int | str | bool | None] = {
            "validation_enabled": False,
            "available_days": len(dates),
            "train_days": len(dates),
            "purge_days": 0,
            "val_days": 0,
            "train_end": dates[-1] if dates else None,
            "val_start": None,
            "val_end": None,
        }
        return features, None, metadata

    val_dates = dates[-val_days:]
    train_dates = dates[: -(val_days + purge_days)]
    train = features.filter(pl.col("date").is_in(train_dates))
    validation = features.filter(pl.col("date").is_in(val_dates))
    metadata = {
        "validation_enabled": True,
        "available_days": len(dates),
        "train_days": len(train_dates),
        "purge_days": purge_days,
        "val_days": len(val_dates),
        "train_end": train_dates[-1],
        "val_start": val_dates[0],
        "val_end": val_dates[-1],
    }
    return train, validation, metadata


@dataclass(frozen=True, slots=True)
class _RankerDay:
    """一个日度横截面的训练张量数组。"""

    date: str
    x: np.ndarray
    label: np.ndarray
    head_gain: np.ndarray
    aux_target: np.ndarray


def _fallback_head_gain(label: np.ndarray) -> np.ndarray:
    """由收益百分位构造只强调上半区的连续 gain。"""
    return np.square(np.clip((label.astype(np.float64) - 0.5) / 0.5, 0.0, None))


def _days_from_frame(
    features: pl.DataFrame,
    *,
    objective: RankerObjectiveConfig,
) -> list[_RankerDay]:
    """将特征表按日拆分；旧数据仅兼容派生 ``head_gain``。"""
    feat_cols = [c for c in features.columns if c.startswith(("emb_", "factor_"))]
    if not feat_cols:
        msg = "no emb_* or factor_* columns available for ranker training"
        raise ValueError(msg)
    if "label" not in features.columns:
        msg = "ranker training features must contain label"
        raise ValueError(msg)
    days: list[_RankerDay] = []
    for (date,), sub in features.group_by(["date"], maintain_order=True):
        x = sub.select(feat_cols).to_numpy().astype(np.float32)
        label = sub["label"].to_numpy().astype(np.float32)
        head_gain = (
            sub["head_gain"].to_numpy().astype(np.float32)
            if "head_gain" in sub.columns
            else _fallback_head_gain(label).astype(np.float32)
        )
        if "aux_target" in sub.columns:
            aux_target = sub["aux_target"].to_numpy().astype(np.float32)
        elif objective.aux_huber_weight > 0:
            msg = (
                "ranker training features must contain aux_target when "
                "aux_huber_weight is positive"
            )
            raise ValueError(msg)
        else:
            aux_target = np.zeros_like(label, dtype=np.float32)
        arrays = {
            "features": x,
            "label": label,
            "head_gain": head_gain,
            "aux_target": aux_target,
        }
        invalid = [
            name for name, value in arrays.items() if not np.isfinite(value).all()
        ]
        if invalid:
            msg = f"ranker day {date} contains non-finite arrays: {invalid}"
            raise ValueError(msg)
        if np.any(head_gain < 0):
            msg = f"ranker day {date} contains negative head_gain"
            raise ValueError(msg)
        days.append(
            _RankerDay(
                date=str(date),
                x=x,
                label=label,
                head_gain=head_gain,
                aux_target=aux_target,
            )
        )
    return days


def feature_columns(features: pl.DataFrame) -> list[str]:
    """返回 Ranker 使用的有序特征列。"""
    return [c for c in features.columns if c.startswith(("emb_", "factor_"))]


def _scoring_days(
    features: pl.DataFrame,
    expected_columns: list[str] | None = None,
) -> list[tuple[str, pl.DataFrame, np.ndarray]]:
    """将无标签推理表按日拆为日期、主键和特征矩阵。"""
    columns = feature_columns(features)
    if expected_columns is not None and columns != expected_columns:
        msg = (
            "ranker feature columns mismatch: "
            f"expected={expected_columns}, actual={columns}"
        )
        raise ValueError(msg)
    if not columns:
        msg = "no emb_* or factor_* columns available for scoring"
        raise ValueError(msg)
    days: list[tuple[str, pl.DataFrame, np.ndarray]] = []
    for (date,), sub in features.group_by(["date"], maintain_order=True):
        keys = sub.select(["date", "symbol"])
        x = sub.select(columns).to_numpy().astype(np.float32)
        days.append((str(date), keys, x))
    return days


def train_ranker(
    features: pl.DataFrame,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    hidden: int = 128,
    depth: int = 2,
    dropout: float = 0.3,
    use_attention: bool = True,
    device: str = "cpu",
    seed: int = 0,
    objective: RankerObjectiveConfig | None = None,
) -> tuple[CrossSectionalRanker, list[float]]:
    """兼容训练入口；无验证集时返回各 epoch 训练 RankIC。"""
    result = fit_ranker(
        features,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        hidden=hidden,
        depth=depth,
        dropout=dropout,
        use_attention=use_attention,
        device=device,
        seed=seed,
        objective=objective,
    )
    history = [float(row["train_ic"] or 0.0) for row in result.history]
    return result.model, history


def _evaluate_days(
    model: CrossSectionalRanker,
    days: list[_RankerDay],
    device: torch.device,
) -> float:
    """关闭 dropout 后计算整段日均截面相关。"""
    if not days:
        return float("nan")
    was_training = model.training
    model.eval()
    values: list[float] = []
    with torch.inference_mode():
        for day in days:
            pred = model(torch.from_numpy(day.x).to(device))
            target = torch.from_numpy(day.label).to(device)
            values.append(float(_pearson(pred, target).item()))
    model.train(was_training)
    return float(np.mean(values))


def _exact_ndcg(pred: np.ndarray, gain: np.ndarray, cutoff: int) -> float:
    """计算一日 exact NDCG@K；``K>N`` 自动截断。"""
    effective_k = min(cutoff, pred.size)
    if effective_k < 1:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, effective_k + 2))
    predicted_order = np.argsort(-pred, kind="stable")[:effective_k]
    ideal_order = np.argsort(-gain, kind="stable")[:effective_k]
    dcg = float(np.dot(gain[predicted_order], discounts))
    ideal_dcg = float(np.dot(gain[ideal_order], discounts))
    return dcg / ideal_dcg if ideal_dcg > 1e-12 else 0.0


def _evaluate_ndcg(
    model: CrossSectionalRanker,
    days: list[_RankerDay],
    device: torch.device,
    *,
    objective: RankerObjectiveConfig,
) -> tuple[float, dict[int, float]]:
    """计算日等权 exact multi-K NDCG。"""
    if not days:
        return float("nan"), {cutoff: float("nan") for cutoff in objective.ndcg_ks}
    was_training = model.training
    model.eval()
    values = {cutoff: [] for cutoff in objective.ndcg_ks}
    with torch.inference_mode():
        for day in days:
            pred = model(torch.from_numpy(day.x).to(device)).cpu().numpy()
            for cutoff in objective.ndcg_ks:
                values[cutoff].append(_exact_ndcg(pred, day.head_gain, cutoff))
    model.train(was_training)
    per_k = {cutoff: float(np.mean(rows)) for cutoff, rows in values.items()}
    weights = np.asarray(objective.ndcg_k_weights, dtype=np.float64)
    weights /= weights.sum()
    weighted = float(
        sum(
            weight * per_k[cutoff]
            for cutoff, weight in zip(objective.ndcg_ks, weights, strict=True)
        )
    )
    return weighted, per_k


def _evaluate_top_spread(
    model: CrossSectionalRanker,
    features: pl.DataFrame | None,
    device: torch.device,
    *,
    top_k: int,
) -> float:
    """计算预测最高尾部相对当日横截面均值的真实超额收益。"""
    if features is None or features.is_empty() or "target_return" not in features:
        return float("nan")
    columns = feature_columns(features)
    was_training = model.training
    model.eval()
    spreads: list[float] = []
    with torch.inference_mode():
        for _, daily in features.group_by("date", maintain_order=True):
            x = daily.select(columns).to_numpy().astype(np.float32)
            pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
            realized = daily["target_return"].to_numpy().astype(np.float64)
            selected = np.argsort(pred)[-min(top_k, realized.size) :]
            spreads.append(float(realized[selected].mean() - realized.mean()))
    model.train(was_training)
    return float(np.mean(spreads)) if spreads else float("nan")


def fit_ranker(
    train_features: pl.DataFrame,
    *,
    val_features: pl.DataFrame | None = None,
    epochs: int = 100,
    patience: int = 8,
    min_delta: float = 1e-4,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    hidden: int = 128,
    depth: int = 2,
    dropout: float = 0.3,
    use_attention: bool = True,
    device: str = "cpu",
    seed: int = 0,
    shuffle_days: bool = True,
    objective: RankerObjectiveConfig | None = None,
) -> RankerTrainingResult:
    """训练并按 exact multi-K NDCG + IC early-stop。"""
    torch.manual_seed(seed)
    objective_cfg = objective or RankerObjectiveConfig()
    objective_cfg.validate()
    train_days = _days_from_frame(train_features, objective=objective_cfg)
    val_days = (
        _days_from_frame(val_features, objective=objective_cfg)
        if val_features is not None
        else []
    )
    if not train_days:
        msg = "no days available to train the ranker"
        raise ValueError(msg)
    if val_days and val_days[0].x.shape[1] != train_days[0].x.shape[1]:
        msg = "train and validation feature dimensions differ"
        raise ValueError(msg)
    in_dim = train_days[0].x.shape[1]
    dev = torch.device(device)
    model = CrossSectionalRanker(
        RankerConfig(
            in_dim=in_dim,
            hidden=hidden,
            depth=depth,
            dropout=dropout,
            use_attention=use_attention,
        )
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[dict[str, float | int | None]] = []
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = -float("inf")
    best_val_ic: float | None = None
    best_val_ndcg: float | None = None
    best_val_top_spread: float | None = None
    best_epoch = -1
    stale_epochs = 0
    stopped_early = False
    model.train()
    for epoch in range(epochs):
        epoch_total_losses: list[float] = []
        epoch_head_losses: list[float] = []
        epoch_ic_losses: list[float] = []
        epoch_aux_losses: list[float] = []
        order = (
            rng.permutation(len(train_days))
            if shuffle_days
            else np.arange(len(train_days))
        )
        for day_index in order:
            day = train_days[int(day_index)]
            xt = torch.from_numpy(day.x).to(dev)
            yt = torch.from_numpy(day.label).to(dev)
            gain = torch.from_numpy(day.head_gain).to(dev)
            aux_target = torch.from_numpy(day.aux_target).to(dev)
            optimizer.zero_grad(set_to_none=True)
            pred, aux_pred = model.forward_with_aux(xt)
            pair_seed = (
                seed + epoch * 1_000_003 + zlib.crc32(day.date.encode("utf-8"))
            ) % (2**63 - 1)
            loss_result = ranker_objective_loss(
                pred,
                yt,
                gain,
                aux_pred=aux_pred,
                aux_target=aux_target,
                objective=objective_cfg,
                seed=pair_seed,
                return_components=True,
            )
            loss, components = loss_result
            loss.backward()
            optimizer.step()
            epoch_total_losses.append(float(loss.detach().item()))
            epoch_head_losses.append(float(components["lambda_ndcg"].detach().item()))
            epoch_ic_losses.append(float(components["global_ic"].detach().item()))
            epoch_aux_losses.append(float(components["aux_huber"].detach().item()))
        train_ic = _evaluate_days(model, train_days, dev)
        val_ic = _evaluate_days(model, val_days, dev) if val_days else None
        train_ndcg, train_ndcg_by_k = _evaluate_ndcg(
            model,
            train_days,
            dev,
            objective=objective_cfg,
        )
        if val_days:
            val_ndcg, val_ndcg_by_k = _evaluate_ndcg(
                model,
                val_days,
                dev,
                objective=objective_cfg,
            )
        else:
            val_ndcg = None
            val_ndcg_by_k = dict.fromkeys(objective_cfg.ndcg_ks)
        train_top_spread = _evaluate_top_spread(
            model,
            train_features,
            dev,
            top_k=objective_cfg.primary_k,
        )
        val_top_spread = _evaluate_top_spread(
            model,
            val_features,
            dev,
            top_k=objective_cfg.primary_k,
        )
        history_row: dict[str, float | int | None] = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_total_losses)),
            "train_lambda_ndcg_loss": float(np.mean(epoch_head_losses)),
            "train_global_ic_loss": float(np.mean(epoch_ic_losses)),
            "train_aux_huber_loss": float(np.mean(epoch_aux_losses)),
            "train_ic": train_ic,
            "val_ic": val_ic,
            "train_ndcg": train_ndcg,
            "val_ndcg": val_ndcg,
            "train_top_spread": train_top_spread,
            "val_top_spread": val_top_spread,
        }
        for cutoff in objective_cfg.ndcg_ks:
            history_row[f"train_ndcg_{cutoff}"] = train_ndcg_by_k[cutoff]
            history_row[f"val_ndcg_{cutoff}"] = val_ndcg_by_k[cutoff]
        history.append(history_row)
        logger.info(
            "epoch %d train RankIC %.4f NDCG %.4f val RankIC %s NDCG %s",
            epoch,
            train_ic,
            train_ndcg,
            f"{val_ic:.4f}" if val_ic is not None else "n/a",
            f"{val_ndcg:.4f}" if val_ndcg is not None else "n/a",
        )
        selection_ic = val_ic if val_ic is not None else train_ic
        selection_ndcg = val_ndcg if val_ndcg is not None else train_ndcg
        selection_top_spread = val_top_spread if val_days else train_top_spread
        finite_top_spread = (
            selection_top_spread if np.isfinite(selection_top_spread) else 0.0
        )
        selection_score = (
            objective_cfg.head_weight * selection_ndcg
            + objective_cfg.global_ic_weight * selection_ic
        )
        if selection_score > best_val + min_delta:
            best_val = selection_score
            best_val_ic = float(selection_ic)
            best_val_ndcg = float(selection_ndcg)
            best_val_top_spread = float(finite_top_spread)
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if val_days and stale_epochs >= patience:
            stopped_early = True
            logger.info("early stop at epoch %d; best epoch=%d", epoch, best_epoch)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return RankerTrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_ic=best_val_ic if val_days else None,
        best_val_ndcg=best_val_ndcg if val_days else None,
        best_val_top_spread=best_val_top_spread if val_days else None,
        best_selection_score=float(best_val),
        stopped_early=stopped_early,
        objective=objective_cfg,
    )


@torch.inference_mode()
def predict(
    model: CrossSectionalRanker,
    features: pl.DataFrame,
    *,
    device: str = "cpu",
    expected_columns: list[str] | None = None,
) -> pl.DataFrame:
    """对无标签特征打分；返回严格的 ``date, symbol, score``。"""
    model.eval()
    dev = torch.device(device)
    out_rows: list[pl.DataFrame] = []
    for _date, keys, x in _scoring_days(features, expected_columns):
        pred = model(torch.from_numpy(x).to(dev)).cpu().numpy()
        out_rows.append(keys.with_columns(pl.Series("score", pred)))
    if not out_rows:
        msg = "no days available to score"
        raise ValueError(msg)
    return pl.concat(out_rows)
