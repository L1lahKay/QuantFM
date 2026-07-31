"""FieldSpec 驱动的 Tokenizer v2 训练窗口数据集。"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

from quant_fm.tokenizer.storage_encoding_v2 import read_token_frame_v2
from quant_fm.tokenizer.vocab_v2 import NA_ID, PAD_ID

if TYPE_CHECKING:
    from quant_fm.manifest.build_manifest import Manifest, ShardEntry
    from quant_fm.tokenizer.vocab_v2 import VocabV2

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class V2FieldLayout:
    """由冻结 FieldSpec 派生出的 parquet/batch 字段契约。"""

    token_fields: tuple[str, ...]
    input_fields: tuple[str, ...]
    target_fields: tuple[str, ...]
    value_fields: tuple[str, ...]
    scalar_to_token: dict[str, str]
    standalone_scalar_fields: tuple[str, ...]


def field_layout_from_vocab(vocab: VocabV2) -> V2FieldLayout:
    """只从 artifact 中的有序 FieldSpec 生成模型字段，禁止隐式全局 tuple。"""
    token_fields: list[str] = []
    input_fields: list[str] = []
    target_fields: list[str] = []
    value_fields: list[str] = []
    scalar_to_token: dict[str, str] = {}
    standalone_scalar_fields: list[str] = []
    for spec in vocab.field_specs:
        token = spec.token_column
        value = spec.value_column
        if token is not None and (spec.is_input or spec.is_target):
            token_fields.append(token)
        if spec.is_input and token is not None:
            input_fields.append(token)
        if spec.is_target and token is not None:
            target_fields.append(token)
        if spec.is_input and value is not None:
            value_fields.append(value)
            if token is None:
                standalone_scalar_fields.append(value)
            else:
                scalar_to_token[value] = token
    return V2FieldLayout(
        token_fields=tuple(token_fields),
        input_fields=tuple(input_fields),
        target_fields=tuple(target_fields),
        value_fields=tuple(value_fields),
        scalar_to_token=scalar_to_token,
        standalone_scalar_fields=tuple(standalone_scalar_fields),
    )


@dataclass(frozen=True, slots=True)
class _Window:
    shard_idx: int
    start: int
    length: int


class _V2ShardCache:
    """按冻结字段读取 token/scalar parquet 的小型 LRU。"""

    def __init__(
        self,
        capacity: int,
        layout: V2FieldLayout,
        vocab: VocabV2,
    ) -> None:
        self.capacity = capacity
        self.layout = layout
        self.vocab = vocab
        self._store: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, path: str) -> dict[str, np.ndarray]:
        """加载一个 v2 shard，并保持 token/int64 与 scalar/float32 类型。"""
        if path in self._store:
            self._store.move_to_end(path)
            return self._store[path]
        columns = [*self.layout.token_fields, *self.layout.value_fields]
        frame = read_token_frame_v2(path, columns=columns, vocab=self.vocab)
        arrays = {
            field: frame[field].to_numpy().astype(np.int64)
            for field in self.layout.token_fields
        }
        arrays.update(
            {
                field: frame[field].to_numpy().astype(np.float32)
                for field in self.layout.value_fields
            }
        )
        self._store[path] = arrays
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)
        return arrays


class EventWindowDatasetV2(Dataset):
    """在 v2 token/scalar 分片上的滑动窗口数据集。"""

    def __init__(
        self,
        shards: list[ShardEntry],
        *,
        vocab: VocabV2,
        context: int = 2048,
        stride: int | None = None,
        min_len: int = 16,
        cache_size: int = 8,
    ) -> None:
        self.shards = shards
        self.layout = field_layout_from_vocab(vocab)
        self.context = context
        self.stride = stride or context
        self.min_len = min_len
        self._cache = _V2ShardCache(cache_size, self.layout, vocab)
        self._windows = self._index_windows()

    def _index_windows(self) -> list[_Window]:
        windows: list[_Window] = []
        for shard_index, shard in enumerate(self.shards):
            if shard.rows < self.min_len:
                continue
            for start in range(0, shard.rows, self.stride):
                length = min(self.context, shard.rows - start)
                if length >= self.min_len:
                    windows.append(_Window(shard_index, start, length))
        return windows

    @classmethod
    def from_manifest(
        cls,
        manifest: Manifest,
        split: str,
        **kwargs: object,
    ) -> EventWindowDatasetV2:
        """从 manifest split 构建数据集。"""
        return cls(manifest.split(split), **kwargs)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self._windows)

    def window_shard_index(self, index: int) -> int:
        """供 shard-aware sampler 使用，不触发 parquet 读取。"""
        return self._windows[index].shard_idx

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """返回 token、连续值和由 NA 生成的 target applicability mask。"""
        window = self._windows[index]
        arrays = self._cache.get(self.shards[window.shard_idx].path)
        selection = slice(window.start, window.start + window.length)
        sample = {
            field: torch.from_numpy(arrays[field][selection].copy())
            for field in (*self.layout.token_fields, *self.layout.value_fields)
        }
        for field in self.layout.target_fields:
            sample[f"mask_{field}"] = sample[field].ne(NA_ID)
        sample["length"] = torch.tensor(window.length, dtype=torch.long)
        return sample


def collate_windows_v2(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """按 dtype 填充 v2 token/scalar/mask，并生成统一 attention mask。"""
    if not batch:
        msg = "cannot collate an empty batch"
        raise ValueError(msg)
    lengths = torch.stack([sample["length"] for sample in batch])
    max_len = int(lengths.max().item())
    batch_size = len(batch)
    output: dict[str, torch.Tensor] = {}
    for field, example in batch[0].items():
        if field == "length":
            continue
        if example.dtype == torch.bool:
            buffer = torch.zeros((batch_size, max_len), dtype=torch.bool)
        elif example.is_floating_point():
            buffer = torch.zeros((batch_size, max_len), dtype=torch.float32)
        else:
            buffer = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long)
        for index, sample in enumerate(batch):
            sequence = sample[field]
            buffer[index, : sequence.numel()] = sequence
        output[field] = buffer

    attention = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for index, length in enumerate(lengths):
        attention[index, : int(length.item())] = True
    output["attention_mask"] = attention
    output["length"] = lengths
    return output
