"""按 parquet shard 聚簇、再按 rank 分片的训练采样器。"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Sampler

if TYPE_CHECKING:
    from collections.abc import Iterator, Sized


class ShardAwareDistributedSampler(Sampler[int]):
    """减少随机 parquet 打开，同时保持 epoch 级 shard/window 打乱。"""

    def __init__(
        self,
        dataset: Sized,
        *,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if not hasattr(dataset, "window_shard_index"):
            msg = "dataset must expose window_shard_index(index)"
            raise TypeError(msg)
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            msg = "invalid num_replicas/rank"
            raise ValueError(msg)
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        if drop_last:
            self.num_samples = len(dataset) // num_replicas
        else:
            self.num_samples = math.ceil(len(dataset) / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """改变确定性随机种子，使下一 epoch 重新打乱。"""
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        by_shard: dict[int, list[int]] = defaultdict(list)
        shard_at = self.dataset.window_shard_index  # type: ignore[attr-defined]
        for index in range(len(self.dataset)):
            by_shard[int(shard_at(index))].append(index)
        shard_ids = sorted(by_shard)
        if self.shuffle:
            order = torch.randperm(len(shard_ids), generator=generator).tolist()
            shard_ids = [shard_ids[index] for index in order]
        indices: list[int] = []
        for shard_id in shard_ids:
            shard_indices = by_shard[shard_id]
            if self.shuffle:
                order = torch.randperm(len(shard_indices), generator=generator).tolist()
                shard_indices = [shard_indices[index] for index in order]
            indices.extend(shard_indices)
        if self.drop_last:
            indices = indices[: self.total_size]
        elif indices:
            repeats = math.ceil(self.total_size / len(indices))
            indices = (indices * repeats)[: self.total_size]
        start = self.rank * self.num_samples
        return iter(indices[start : start + self.num_samples])
