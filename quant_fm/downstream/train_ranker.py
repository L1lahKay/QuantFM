"""
置换等变的横截面排序器与 RankIC 目标。

按报告设计：整日横截面作为集合输入（无位置编码），打乱输入顺序不改变各预测。
网络刻意浅、窄、强正则（低信噪比环境）。损失为预测与标签的日度横截面相关负值
（RankIC），再对日平均。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch import nn

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


@dataclass(slots=True)
class RankerTrainingResult:
    """包含最佳验证模型与逐 epoch 轨迹的训练结果。"""

    model: CrossSectionalRanker
    history: list[dict[str, float | int | None]]
    best_epoch: int
    best_val_ic: float | None
    stopped_early: bool


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对横截面 ``[N, F]`` 打分 -> ``[N]``。"""
        h = self.proj(x).unsqueeze(0)  # [1, N, H]
        for layer in self.layers:
            h = layer(h)
        return self.out(h).squeeze(-1).squeeze(0)  # [N]


def _pearson(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """可微 Pearson 相关。"""
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = pred.norm() * target.norm() + 1e-8
    return (pred * target).sum() / denom


def rank_ic_loss(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """横截面相关负值（最大化 RankIC）。"""
    return -_pearson(pred, label)


def _days_from_frame(
    features: pl.DataFrame,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """将特征表按日拆为 (date, X, y) 元组列表。"""
    feat_cols = [c for c in features.columns if c.startswith(("emb_", "factor_"))]
    days: list[tuple[str, np.ndarray, np.ndarray]] = []
    for (date,), sub in features.group_by(["date"], maintain_order=True):
        x = sub.select(feat_cols).to_numpy().astype(np.float32)
        y = sub["label"].to_numpy().astype(np.float32)
        days.append((str(date), x, y))
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
    )
    history = [float(row["train_ic"] or 0.0) for row in result.history]
    return result.model, history


def _evaluate_days(
    model: CrossSectionalRanker,
    days: list[tuple[str, np.ndarray, np.ndarray]],
    device: torch.device,
) -> float:
    """关闭 dropout 后计算整段日均截面相关。"""
    if not days:
        return float("nan")
    was_training = model.training
    model.eval()
    values: list[float] = []
    with torch.inference_mode():
        for _date, x, y in days:
            pred = model(torch.from_numpy(x).to(device))
            target = torch.from_numpy(y).to(device)
            values.append(float(_pearson(pred, target).item()))
    model.train(was_training)
    return float(np.mean(values))


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
) -> RankerTrainingResult:
    """训练并按验证 IC early-stop，最终恢复最佳 epoch 权重。"""
    torch.manual_seed(seed)
    train_days = _days_from_frame(train_features)
    val_days = _days_from_frame(val_features) if val_features is not None else []
    if not train_days:
        msg = "no days available to train the ranker"
        raise ValueError(msg)
    if val_days and val_days[0][1].shape[1] != train_days[0][1].shape[1]:
        msg = "train and validation feature dimensions differ"
        raise ValueError(msg)
    in_dim = train_days[0][1].shape[1]
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
    best_epoch = -1
    stale_epochs = 0
    stopped_early = False
    model.train()
    for epoch in range(epochs):
        ics = []
        order = (
            rng.permutation(len(train_days))
            if shuffle_days
            else np.arange(len(train_days))
        )
        for day_index in order:
            _date, x, y = train_days[int(day_index)]
            xt = torch.from_numpy(x).to(dev)
            yt = torch.from_numpy(y).to(dev)
            pred = model(xt)
            loss = rank_ic_loss(pred, yt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ics.append(-float(loss.item()))
        train_ic = _evaluate_days(model, train_days, dev)
        val_ic = _evaluate_days(model, val_days, dev) if val_days else None
        history.append({"epoch": epoch, "train_ic": train_ic, "val_ic": val_ic})
        logger.info(
            "epoch %d train RankIC %.4f val RankIC %s",
            epoch,
            train_ic,
            f"{val_ic:.4f}" if val_ic is not None else "n/a",
        )
        selection_ic = val_ic if val_ic is not None else train_ic
        if selection_ic > best_val + min_delta:
            best_val = selection_ic
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
        best_val_ic=float(best_val) if val_days else None,
        stopped_early=stopped_early,
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
