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
    """训练排序器；返回模型与各 epoch 平均训练 RankIC。"""
    torch.manual_seed(seed)
    days = _days_from_frame(features)
    if not days:
        msg = "no days available to train the ranker"
        raise ValueError(msg)
    in_dim = days[0][1].shape[1]
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

    history: list[float] = []
    model.train()
    for epoch in range(epochs):
        ics = []
        for _date, x, y in days:
            xt = torch.from_numpy(x).to(dev)
            yt = torch.from_numpy(y).to(dev)
            pred = model(xt)
            loss = rank_ic_loss(pred, yt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ics.append(-float(loss.item()))
        mean_ic = float(np.mean(ics))
        history.append(mean_ic)
        logger.info("epoch %d train RankIC %.4f", epoch, mean_ic)
    return model, history


@torch.no_grad()
def predict(
    model: CrossSectionalRanker,
    features: pl.DataFrame,
    *,
    device: str = "cpu",
) -> pl.DataFrame:
    """对每个 (date, symbol) 行打分；返回 ``date, symbol, score``。"""
    model.eval()
    dev = torch.device(device)
    out_rows: list[pl.DataFrame] = []
    for date, x, _y in _days_from_frame(features):
        pred = model(torch.from_numpy(x).to(dev)).cpu().numpy()
        sub = features.filter(pl.col("date") == date).select(["date", "symbol"])
        out_rows.append(sub.with_columns(pl.Series("score", pred)))
    return pl.concat(out_rows)
