"""
将 token 化股日分片切为定长窗口的 Torch 数据集。

【新手】PyTorch 训练的标准套路：
  Dataset.__getitem__(i) 返回一条样本
  DataLoader 负责打乱、组 batch
  collate_fn 把多条样本拼成统一 shape 的 batch

每个分片（一股一日）按 ``context`` 个事件切窗。各字段为独立整数序列；
collator 以 ``PAD_ID = 0`` 填充至批内最大长度并返回有效性掩码。
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from quant_fm.tokenizer.tokenize_events import TOKEN_FIELDS
from quant_fm.tokenizer.vocab import PAD_ID

if TYPE_CHECKING:
    from quant_fm.manifest.build_manifest import Manifest, ShardEntry

logger = logging.getLogger(__name__)

FIELD_ORDER: tuple[str, ...] = tuple(TOKEN_FIELDS.keys())

# 配备预测头的字段（多任务 CE 目标）。
DEFAULT_TARGET_FIELDS: tuple[str, ...] = (
    "tok_evt_type",
    "tok_side",
    "tok_session",
    "tok_price_bin",
    "tok_volume_bin",
    "tok_delta_t_bin",
)


def validate_window_geometry(
    *,
    context: int,
    stride: int | None,
    min_len: int,
    cache_size: int,
) -> tuple[int, int, int, int]:
    """校验滑窗参数并返回规范化后的正整数配置。"""
    values = {
        "context": context,
        "min_len": min_len,
        "cache_size": cache_size,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            msg = f"{name} must be a positive integer, got {value!r}"
            raise ValueError(msg)
    effective_stride = context if stride is None else stride
    if (
        isinstance(effective_stride, bool)
        or not isinstance(effective_stride, int)
        or effective_stride < 1
    ):
        msg = f"stride must be a positive integer, got {effective_stride!r}"
        raise ValueError(msg)
    if min_len > context:
        msg = f"min_len ({min_len}) cannot exceed context ({context})"
        raise ValueError(msg)
    return context, effective_stride, min_len, cache_size


@dataclass(slots=True)
class _Window:
    shard_idx: int
    start: int
    length: int


class _ShardCache:
    """已加载分片字段数组的小型 LRU 缓存。"""

    def __init__(self, capacity: int = 8) -> None:
        self.capacity = capacity
        self._store: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, path: str) -> dict[str, np.ndarray]:
        """返回 ``path`` 的字段数组，未命中时加载并缓存。"""
        if path in self._store:
            self._store.move_to_end(path)
            return self._store[path]
        df = pl.read_parquet(path, columns=list(FIELD_ORDER))
        arrays = {f: df[f].to_numpy().astype(np.int64) for f in FIELD_ORDER}
        self._store[path] = arrays
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)
        return arrays


class EventWindowDataset(Dataset):
    """在 token 化股日分片上的滑动窗口数据集。"""

    def __init__(
        self,
        shards: list[
            ShardEntry
        ],  # [导读] 来自 manifest，每个元素是一个 parquet 文件信息
        *,
        context: int = 2048,  # 一个训练窗口最多包含多少个事件
        stride: int | None = None,  # 滑窗步长；默认=context 表示不重叠
        min_len: int = 16,
        cache_size: int = 8,
    ) -> None:
        context, effective_stride, min_len, cache_size = validate_window_geometry(
            context=context,
            stride=stride,
            min_len=min_len,
            cache_size=cache_size,
        )
        self.shards = shards
        self.context = context
        self.stride = effective_stride
        self.min_len = min_len
        self._cache = _ShardCache(capacity=cache_size)
        self._windows: list[_Window] = self._index_windows()
        logger.info(
            "dataset: %d shards -> %d windows (context=%d, stride=%d)",
            len(shards),
            len(self._windows),
            context,
            self.stride,
        )

    def _index_windows(self) -> list[_Window]:
        windows: list[_Window] = []
        for i, shard in enumerate(self.shards):
            n = shard.rows
            if n < self.min_len:
                continue
            start = 0
            while start < n:
                length = min(self.context, n - start)
                if length >= self.min_len:
                    windows.append(_Window(i, start, length))
                start += self.stride
        return windows

    @classmethod
    def from_manifest(
        cls,
        manifest: Manifest,
        split: str,
        **kwargs: object,
    ) -> EventWindowDataset:
        """从清单切分构建数据集。"""
        return cls(manifest.split(split), **kwargs)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self._windows)  # [导读] DataLoader 用这个决定一共有多少条样本

    def window_shard_index(self, index: int) -> int:
        """供 shard-aware sampler 使用，不暴露内部窗口实现。"""
        return self._windows[index].shard_idx

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """返回第 idx 个窗口：各字段的一段整数序列 + length。"""
        win = self._windows[idx]
        arrays = self._cache.get(self.shards[win.shard_idx].path)
        sl = slice(win.start, win.start + win.length)  # [导读] Python 切片：取连续一段
        sample = {f: torch.from_numpy(arrays[f][sl].copy()) for f in FIELD_ORDER}
        sample["length"] = torch.tensor(win.length, dtype=torch.long)
        return sample


def collate_windows(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """将变长窗口列表填充为批张量。"""
    # [导读] batch 是 list，长度 = batch_size；每项是 __getitem__ 返回的 dict
    lengths = torch.stack([b["length"] for b in batch])
    max_len = int(lengths.max().item())  # 这一批里最长的序列
    bsz = len(batch)

    out: dict[str, torch.Tensor] = {}
    for f in FIELD_ORDER:
        buf = torch.full((bsz, max_len), PAD_ID, dtype=torch.long)  # 先全填 0（PAD）
        for i, b in enumerate(batch):
            seq = b[f]
            buf[i, : seq.numel()] = seq  # 把真实数据拷到前面
        out[f] = buf

    mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    for i, ln in enumerate(lengths):
        mask[i, : int(ln.item())] = True  # True = 真实事件；False = padding
    out["attention_mask"] = mask
    out["length"] = lengths
    return out
