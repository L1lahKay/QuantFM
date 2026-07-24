"""将逐事件隐状态序列池化为单向量股日嵌入。"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

MULTISCALE_VECTOR_NAMES: tuple[str, ...] = (
    "mean_all",
    "last_256",
    "last_1024",
    "open_call",
    "continuous_am",
    "continuous_pm",
    "close_call",
    "close_30m",
)


def pool_hidden(
    hidden: torch.Tensor,
    mask: torch.Tensor,
    *,
    method: str = "mean",
    last_k: int = 256,
) -> torch.Tensor:
    """
    将 ``[L, d]`` 隐状态约简为单个 ``[d]`` 嵌入。

    参数
    ----------
    hidden
        逐事件隐状态，形状 ``[L, d]``。
    mask
        布尔有效性掩码，形状 ``[L]``（True = 真实事件）。
    method
        ``"mean"``（掩码平均）、``"last"``（最后有效事件）或
        ``"lastk_mean"``（最后 ``last_k`` 个有效事件的均值）。
    last_k
        ``"lastk_mean"`` 的窗口大小。

    返回
    -------
    torch.Tensor
        形状 ``[d]`` 的池化嵌入。
    """
    valid = mask.bool()
    if valid.sum() == 0:
        return torch.zeros(hidden.size(-1), device=hidden.device)

    if method == "mean":
        h = hidden[valid]
        return h.mean(dim=0)
    if method == "last":
        idx = torch.nonzero(valid, as_tuple=False).max()
        return hidden[idx]
    if method == "lastk_mean":
        h = hidden[valid]
        return h[-last_k:].mean(dim=0)
    msg = f"unknown pooling method {method!r}"
    raise ValueError(msg)


@dataclass(slots=True)
class StockDayPoolAccumulator:
    """跨任意数量 chunk 保持正确股日池化语义的流式累加器。"""

    d_model: int
    method: str = "mean"
    last_k: int = 256
    _sum: torch.Tensor = field(init=False, repr=False)
    _count: int = field(init=False, default=0, repr=False)
    _tail: torch.Tensor | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.method not in {"mean", "last", "lastk_mean"}:
            msg = f"unknown pooling method {self.method!r}"
            raise ValueError(msg)
        if self.last_k < 1:
            msg = "last_k must be positive"
            raise ValueError(msg)
        self._sum = torch.zeros(self.d_model, dtype=torch.float32)

    def update(self, hidden: torch.Tensor, mask: torch.Tensor) -> None:
        """按时间顺序加入一个 chunk；内部状态固定留在 CPU float32。"""
        valid_hidden = hidden[mask.bool()]
        if valid_hidden.numel() == 0:
            return
        if valid_hidden.size(-1) != self.d_model:
            msg = f"hidden width {valid_hidden.size(-1)} != d_model {self.d_model}"
            raise ValueError(msg)

        if self.method == "mean":
            self._sum += valid_hidden.float().sum(dim=0).cpu()
            self._count += int(valid_hidden.size(0))
            return
        if self.method == "last":
            self._tail = valid_hidden[-1:].float().cpu()
            self._count = 1
            return

        new_tail = valid_hidden[-self.last_k :].float().cpu()
        if self._tail is not None:
            new_tail = torch.cat((self._tail, new_tail), dim=0)[-self.last_k :]
        self._tail = new_tail
        self._count = int(new_tail.size(0))

    def value(self) -> torch.Tensor:
        """返回当前股日向量；无事件时返回零向量。"""
        if self.method == "mean":
            return self._sum / self._count if self._count else self._sum.clone()
        if self._tail is None:
            return self._sum.clone()
        if self.method == "last":
            return self._tail[-1]
        return self._tail.mean(dim=0)


def pool_hidden_chunks(
    chunks: list[torch.Tensor],
    *,
    method: str = "mean",
    last_k: int = 256,
) -> torch.Tensor:
    """测试和离线工具共用的跨 chunk 池化入口。"""
    if not chunks:
        msg = "at least one hidden chunk is required"
        raise ValueError(msg)
    accumulator = StockDayPoolAccumulator(
        chunks[0].size(-1), method=method, last_k=last_k
    )
    for chunk in chunks:
        mask = torch.ones(chunk.size(0), dtype=torch.bool, device=chunk.device)
        accumulator.update(chunk, mask)
    return accumulator.value()


@dataclass(slots=True)
class MultiScaleStockDayPoolAccumulator:
    """流式生成全日、尾部及 A 股交易阶段的固定多尺度表示。"""

    d_model: int
    _sums: dict[str, torch.Tensor] = field(init=False, repr=False)
    _counts: dict[str, int] = field(init=False, repr=False)
    _tail: torch.Tensor | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        window_names = MULTISCALE_VECTOR_NAMES[3:]
        self._sums = {
            name: torch.zeros(self.d_model, dtype=torch.float32)
            for name in ("mean_all", *window_names)
        }
        self._counts = dict.fromkeys(self._sums, 0)

    def _accumulate(self, name: str, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        self._sums[name] += values.float().sum(dim=0).cpu()
        self._counts[name] += int(values.size(0))

    def update(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
        time_of_day_ms: torch.Tensor,
    ) -> None:
        """加入按时间排序的 chunk，时间为自午夜毫秒。"""
        valid = mask.bool()
        if time_of_day_ms.shape != mask.shape:
            msg = "time_of_day_ms and mask must have the same shape"
            raise ValueError(msg)
        values = hidden[valid]
        times = time_of_day_ms[valid]
        if values.numel() == 0:
            return
        if values.size(-1) != self.d_model:
            msg = f"hidden width {values.size(-1)} != d_model {self.d_model}"
            raise ValueError(msg)

        self._accumulate("mean_all", values)
        tail = values[-1024:].float().cpu()
        if self._tail is not None:
            tail = torch.cat((self._tail, tail), dim=0)[-1024:]
        self._tail = tail

        minute = 60_000
        ranges = {
            "open_call": ((9 * 60 + 15) * minute, (9 * 60 + 30) * minute),
            "continuous_am": ((9 * 60 + 30) * minute, (11 * 60 + 31) * minute),
            "continuous_pm": (13 * 60 * minute, (14 * 60 + 57) * minute),
            "close_call": ((14 * 60 + 57) * minute, (15 * 60 + 1) * minute),
            "close_30m": ((14 * 60 + 30) * minute, (15 * 60 + 1) * minute),
        }
        for name, (start, end) in ranges.items():
            in_window = times.ge(start) & times.lt(end)
            self._accumulate(name, values[in_window])

    def value_dict(self) -> dict[str, torch.Tensor]:
        """返回固定顺序的八个 d_model 向量。"""
        output: dict[str, torch.Tensor] = {}
        output["mean_all"] = (
            self._sums["mean_all"] / self._counts["mean_all"]
            if self._counts["mean_all"]
            else self._sums["mean_all"].clone()
        )
        zero = torch.zeros(self.d_model, dtype=torch.float32)
        output["last_256"] = (
            self._tail[-256:].mean(dim=0) if self._tail is not None else zero.clone()
        )
        output["last_1024"] = (
            self._tail.mean(dim=0) if self._tail is not None else zero.clone()
        )
        for name in MULTISCALE_VECTOR_NAMES[3:]:
            output[name] = (
                self._sums[name] / self._counts[name]
                if self._counts[name]
                else zero.clone()
            )
        return output

    def concatenate(self) -> torch.Tensor:
        """拼接八个向量和原始 event_count 标量，供现有 Ranker 直接消费。"""
        values = self.value_dict()
        event_count = torch.tensor(
            [float(self._counts["mean_all"])], dtype=torch.float32
        )
        return torch.cat(
            [*(values[name] for name in MULTISCALE_VECTOR_NAMES), event_count]
        )
