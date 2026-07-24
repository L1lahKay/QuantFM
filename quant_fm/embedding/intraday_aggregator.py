"""轻量、因果的股日日内 chunk 聚合器。"""

from __future__ import annotations

import math

import torch
from torch import nn


class IntradayAggregator(nn.Module):
    """
    按时间顺序将 event-transformer 的 chunk 摘要聚合为股日表示。

    参数
    ----------
    input_dim
        单个 ``chunk_summary`` 的宽度。
    d_model
        聚合器隐状态及四个输出摘要的宽度；默认等于 ``input_dim``。
    n_layers
        因果 GRUCell 层数，只允许 1 或 2 层。
    num_sessions
        ``chunk_session`` 有效 id 的数量，有效值范围为
        ``[0, num_sessions)``。
    session_dim
        session embedding 宽度。
    time_frequencies
        真实日内时间与有效 chunk 序号所用 Fourier 频率数量。
    time_scale
        ``chunk_time`` 的单位缩放。例如输入为日内毫秒时可设为
        ``86_400_000``；输入已归一化到一天时保持默认值 ``1``。
    position_scale
        有效 chunk 序号的固定缩放。它不能由当日 chunk 总数推导，否则早期
        hidden 会间接看到未来序列长度。
    dropout
        两层聚合器之间的 dropout；单层时不生效。

    说明
    ----
    padding 位置不会更新 recurrent state。顺序位置使用有效 chunk 的累计
    序号，而非张量的绝对下标，因此在有效 chunk 之间插入 padding 不会改变
    结果。``encode_chunks`` 可用于检查逐位置因果 hidden。
    """

    OUTPUT_KEYS = (
        "full_day_summary",
        "close_summary",
        "intraday_trend_summary",
        "activity_summary",
    )

    def __init__(
        self,
        input_dim: int,
        *,
        d_model: int | None = None,
        n_layers: int = 1,
        num_sessions: int = 8,
        session_dim: int = 8,
        time_frequencies: int = 4,
        time_scale: float = 1.0,
        position_scale: float = 128.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            msg = "input_dim must be positive"
            raise ValueError(msg)
        if d_model is not None and d_model < 1:
            msg = "d_model must be positive"
            raise ValueError(msg)
        if n_layers not in {1, 2}:
            msg = "n_layers must be 1 or 2"
            raise ValueError(msg)
        if num_sessions < 1:
            msg = "num_sessions must be positive"
            raise ValueError(msg)
        if session_dim < 1:
            msg = "session_dim must be positive"
            raise ValueError(msg)
        if time_frequencies < 1:
            msg = "time_frequencies must be positive"
            raise ValueError(msg)
        if not math.isfinite(time_scale) or time_scale <= 0:
            msg = "time_scale must be finite and positive"
            raise ValueError(msg)
        if not math.isfinite(position_scale) or position_scale <= 0:
            msg = "position_scale must be finite and positive"
            raise ValueError(msg)
        if not 0.0 <= dropout < 1.0:
            msg = "dropout must be in [0, 1)"
            raise ValueError(msg)

        self.input_dim = input_dim
        self.d_model = d_model or input_dim
        self.n_layers = n_layers
        self.num_sessions = num_sessions
        self.time_frequencies = time_frequencies
        self.time_scale = float(time_scale)
        self.position_scale = float(position_scale)

        # id=0 专用于 padding；有效 session id 整体右移一位。
        self.session_embedding = nn.Embedding(
            num_sessions + 1,
            session_dim,
            padding_idx=0,
        )
        temporal_dim = 4 * time_frequencies
        self.input_projection = nn.Linear(
            input_dim + session_dim + temporal_dim,
            self.d_model,
        )
        self.input_norm = nn.LayerNorm(self.d_model)
        self.cells = nn.ModuleList(
            nn.GRUCell(self.d_model, self.d_model) for _ in range(n_layers)
        )
        self.inter_layer_dropout = nn.Dropout(dropout)

        # trend 为有符号的早晚变化；activity 为可学习的掩码注意力池化。
        self.trend_projection = nn.Linear(self.d_model, self.d_model, bias=False)
        self.activity_score = nn.Linear(self.d_model, 1, bias=False)

        frequencies = 2.0 ** torch.arange(time_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def _validate_inputs(
        self,
        chunk_summaries: torch.Tensor,
        chunk_time: torch.Tensor,
        chunk_session: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> torch.Tensor:
        """校验输入并返回布尔 mask。"""
        if chunk_summaries.ndim != 3:
            msg = "chunk_summaries must have shape [batch, chunks, input_dim]"
            raise ValueError(msg)
        batch, chunks, width = chunk_summaries.shape
        if chunks < 1:
            msg = "at least one chunk position is required"
            raise ValueError(msg)
        if width != self.input_dim:
            msg = f"chunk_summaries width {width} != input_dim {self.input_dim}"
            raise ValueError(msg)
        expected = (batch, chunks)
        for name, value in (
            ("chunk_time", chunk_time),
            ("chunk_session", chunk_session),
            ("chunk_mask", chunk_mask),
        ):
            if value.shape != expected:
                msg = f"{name} must have shape {expected}, got {tuple(value.shape)}"
                raise ValueError(msg)

        mask = chunk_mask.bool()
        valid_time = chunk_time[mask]
        if valid_time.numel() and not bool(torch.isfinite(valid_time).all()):
            msg = "valid chunk_time values must be finite"
            raise ValueError(msg)
        valid_session = chunk_session[mask]
        if valid_session.numel() and (
            bool((valid_session < 0).any())
            or bool((valid_session >= self.num_sessions).any())
        ):
            msg = f"valid chunk_session ids must be in [0, {self.num_sessions})"
            raise ValueError(msg)
        return mask

    def _fourier_features(self, value: torch.Tensor) -> torch.Tensor:
        """对无量纲标量构造固定 Fourier 特征。"""
        angle = 2.0 * math.pi * value.unsqueeze(-1) * self.frequencies
        return torch.cat((angle.sin(), angle.cos()), dim=-1)

    def _encode_inputs(
        self,
        chunk_summaries: torch.Tensor,
        chunk_time: torch.Tensor,
        chunk_session: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """融合 chunk、真实时间、session 和有效顺序特征。"""
        dtype = chunk_summaries.dtype
        safe_summaries = torch.where(
            mask.unsqueeze(-1),
            chunk_summaries,
            torch.zeros_like(chunk_summaries),
        )
        safe_time = torch.where(mask, chunk_time, torch.zeros_like(chunk_time))
        scaled_time = safe_time.to(dtype=dtype) / self.time_scale

        # rank 对 padding 不增量，避免 padding 布局改变有效 chunk 的位置编码。
        valid_rank = mask.long().cumsum(dim=1) - 1
        safe_rank = torch.where(mask, valid_rank, torch.zeros_like(valid_rank))
        normalized_rank = safe_rank.to(dtype=dtype) / self.position_scale

        safe_session = torch.where(mask, chunk_session.long() + 1, 0)
        session = self.session_embedding(safe_session)
        temporal = torch.cat(
            (
                self._fourier_features(scaled_time),
                self._fourier_features(normalized_rank),
            ),
            dim=-1,
        )
        encoded = torch.cat((safe_summaries, session, temporal), dim=-1)
        encoded = self.input_norm(self.input_projection(encoded))
        return encoded.masked_fill(~mask.unsqueeze(-1), 0.0)

    def encode_chunks(
        self,
        chunk_summaries: torch.Tensor,
        chunk_time: torch.Tensor,
        chunk_session: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        返回逐 chunk 的严格因果 hidden，形状为 ``[B, C, d_model]``。

        任一位置只依赖该位置及更早的有效 chunk；padding 位置输出零。
        """
        mask = self._validate_inputs(
            chunk_summaries,
            chunk_time,
            chunk_session,
            chunk_mask,
        )
        layer_input = self._encode_inputs(
            chunk_summaries,
            chunk_time,
            chunk_session,
            mask,
        )
        batch, chunks, _ = layer_input.shape

        for layer_idx, cell in enumerate(self.cells):
            state = layer_input.new_zeros((batch, self.d_model))
            outputs: list[torch.Tensor] = []
            for chunk_idx in range(chunks):
                valid = mask[:, chunk_idx].unsqueeze(-1)
                candidate = cell(layer_input[:, chunk_idx], state)
                state = torch.where(valid, candidate, state)
                outputs.append(torch.where(valid, state, torch.zeros_like(state)))
            layer_output = torch.stack(outputs, dim=1)
            if layer_idx + 1 < self.n_layers:
                layer_input = self.inter_layer_dropout(layer_output)
            else:
                layer_input = layer_output
        return layer_input

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """计算 padding 无关的序列均值。"""
        weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
        count = weights.sum(dim=1).clamp_min(1.0)
        return (hidden * weights).sum(dim=1) / count

    @staticmethod
    def _last_valid(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """取最后一个有效 chunk 的因果 hidden；全 padding 时返回零。"""
        batch = hidden.size(0)
        index = torch.arange(mask.size(1), device=mask.device).expand(batch, -1)
        last_index = index.masked_fill(~mask, -1).amax(dim=1)
        safe_index = last_index.clamp_min(0)
        result = hidden[torch.arange(batch, device=hidden.device), safe_index]
        return result.masked_fill((last_index < 0).unsqueeze(-1), 0.0)

    def _trend_summary(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """用按有效顺序中心化的线性权重提取早晚日内变化。"""
        rank = mask.long().cumsum(dim=1) - 1
        count = mask.sum(dim=1, keepdim=True)
        denominator = (count - 1).clamp_min(1)
        weights = 2.0 * rank.to(hidden.dtype) / denominator.to(hidden.dtype) - 1.0
        weights = torch.where(mask & (count > 1), weights, 0.0)
        norm = weights.abs().sum(dim=1, keepdim=True).clamp_min(1.0)
        raw_trend = (hidden * weights.unsqueeze(-1)).sum(dim=1) / norm
        return self.trend_projection(raw_trend)

    def _activity_summary(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """通过掩码注意力突出信息量较高的日内 chunk。"""
        score = self.activity_score(hidden).squeeze(-1)
        score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
        has_valid = mask.any(dim=1, keepdim=True)
        score = torch.where(has_valid, score, torch.zeros_like(score))
        weight = torch.softmax(score, dim=1) * mask.to(dtype=score.dtype)
        weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (hidden * weight.unsqueeze(-1)).sum(dim=1)

    def forward(
        self,
        chunk_summaries: torch.Tensor,
        chunk_time: torch.Tensor,
        chunk_session: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """返回全日、尾盘、日内趋势和活跃度四种摘要。"""
        mask = chunk_mask.bool()
        hidden = self.encode_chunks(
            chunk_summaries,
            chunk_time,
            chunk_session,
            mask,
        )
        return {
            "full_day_summary": self._masked_mean(hidden, mask),
            "close_summary": self._last_valid(hidden, mask),
            "intraday_trend_summary": self._trend_summary(hidden, mask),
            "activity_summary": self._activity_summary(hidden, mask),
        }
