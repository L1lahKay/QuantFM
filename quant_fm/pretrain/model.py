"""
解码器订单流基础模型，多字段输入输出。

架构：各字段 token 嵌入求和为 ``d_model``，再经 pre-norm（RMSNorm）Transformer 块堆叠，
含旋转位置编码（RoPE）、因果注意力与 SwiGLU 前馈，最后每个预测字段一个分类头。
作为*序列*模型保留位置信息（RoPE）；置换等变、无位置的设计留给下游横截面排序器，而非此处。

参数量随 ``d_model``/``n_layers`` 缩放，同一套代码可覆盖 5M 试点与 80–120M 全市场配置。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from quant_fm.pretrain.dataset import DEFAULT_TARGET_FIELDS, FIELD_ORDER
from quant_fm.pretrain.heads import MultiHead
from quant_fm.tokenizer.tokenize_events import TOKEN_FIELDS
from quant_fm.tokenizer.vocab import PAD_ID

if TYPE_CHECKING:
    from quant_fm.tokenizer.vocab import Vocab


def field_sizes_from_vocab(vocab: Vocab) -> dict[str, int]:
    """将 token 列名（``tok_*``）映射到其 id 空间大小。"""
    return {tok: vocab.size(src) for tok, (_, src) in TOKEN_FIELDS.items()}


@dataclass(slots=True)
class OrderFlowFMConfig:
    """:class:`OrderFlowFM` 的超参数。"""

    field_sizes: dict[str, int]
    input_fields: tuple[str, ...] = FIELD_ORDER
    target_fields: tuple[str, ...] = DEFAULT_TARGET_FIELDS
    d_model: int = 512
    n_layers: int = 10
    n_heads: int = 8
    ffn_mult: float = 4.0
    dropout: float = 0.1
    max_seq_len: int = 4096
    rope_theta: float = 10_000.0
    tie_none: bool = True

    def target_sizes(self) -> dict[str, int]:
        """返回预测（头）字段的 id 空间大小。"""
        return {f: self.field_sizes[f] for f in self.target_fields}


class RMSNorm(nn.Module):
    """均方根层归一化。"""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """按最后一维的 RMS 归一化。"""
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def _rope_cache(
    seq_len: int, head_dim: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算旋转嵌入的余弦/正弦表。"""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """旋转最后一维的两半。"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """对形状 ``[B, H, L, D]`` 的 query/key 应用 RoPE。"""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class CausalAttention(nn.Module):
    """带 RoPE 与 key 填充掩码的多头因果自注意力。"""

    def __init__(self, cfg: OrderFlowFMConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        """运行注意力。``key_mask`` 为 ``[B, L]`` 布尔（True = 有效）。"""
        b, length, _ = x.shape
        qkv = self.qkv(x).view(b, length, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = _apply_rope(q, k, cos, sin)

        # 合并因果掩码与 key 填充为布尔掩码（True = 可 attend）。
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        attn_mask = causal[None, None] & key_mask[:, None, None, :]
        out = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(b, length, -1)
        return self.proj(out)


class SwiGLU(nn.Module):
    """SwiGLU 前馈网络。"""

    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """门控前馈变换。"""
        return self.w3(nn.functional.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    """Pre-norm Transformer 块。"""

    def __init__(self, cfg: OrderFlowFMConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        hidden = int(cfg.d_model * cfg.ffn_mult)
        self.ffn = SwiGLU(cfg.d_model, hidden)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        """带残差连接的注意力与前馈。"""
        x = x + self.dropout(self.attn(self.norm1(x), cos, sin, key_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class OrderFlowFM(nn.Module):
    """解码器多字段订单流基础模型。"""

    def __init__(self, cfg: OrderFlowFMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # [导读] 每个字段一张 Embedding 表：整数 id → d_model 维向量
        self.embeddings = nn.ModuleDict(
            {
                f: nn.Embedding(cfg.field_sizes[f], cfg.d_model, padding_idx=PAD_ID)
                for f in cfg.input_fields
            }
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = MultiHead(cfg.d_model, cfg.target_sizes())  # 6 个分类头
        self._rope: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            with torch.no_grad():
                module.weight[PAD_ID].zero_()

    def num_parameters(self) -> int:
        """可训练参数总数。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _get_rope(
        self, length: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_dim = self.cfg.d_model // self.cfg.n_heads
        cos, sin = _rope_cache(
            max(length, self.cfg.max_seq_len), head_dim, self.cfg.rope_theta, device
        )
        return cos[:length], sin[:length]

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """返回批数据的最终隐状态 ``[B, L, d_model]``。"""
        key_mask = batch["attention_mask"]
        length = key_mask.size(1)
        device = key_mask.device

        # [导读] 步骤1：各字段 embedding 相加，得到「每个事件的向量表示」
        x = None
        for f in self.cfg.input_fields:
            emb = self.embeddings[f](batch[f])
            x = emb if x is None else x + emb
        x = self.drop(x)

        # [导读] 步骤2：过 N 层 Transformer（因果注意力 = 只能看过去的事件）
        cos, sin = self._get_rope(length, device)
        for block in self.blocks:
            x = block(x, cos, sin, key_mask)
        return self.norm(x)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """返回序列上各字段 logits（未做 softmax 的原始分数）。"""
        hidden = self.encode(batch)
        return self.head(hidden)  # 每个字段一个 [B, L, 词表大小] 的张量

    @classmethod
    def from_vocab(cls, vocab: Vocab, **overrides: object) -> OrderFlowFM:
        """根据 ``vocab`` 的字段规模构建模型。"""
        cfg = OrderFlowFMConfig(field_sizes=field_sizes_from_vocab(vocab), **overrides)  # type: ignore[arg-type]
        return cls(cfg)
