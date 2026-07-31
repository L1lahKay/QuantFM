from quant_fm.pretrain.sampler import ShardAwareDistributedSampler


class _Dataset:
    shards = (0, 0, 0, 1, 1, 2, 2, 2)

    def __len__(self) -> int:
        return len(self.shards)

    def window_shard_index(self, index: int) -> int:
        return self.shards[index]


def test_sampler_is_complete_disjoint_and_deterministic() -> None:
    dataset = _Dataset()
    rank0 = ShardAwareDistributedSampler(dataset, num_replicas=2, rank=0, seed=4)
    rank1 = ShardAwareDistributedSampler(dataset, num_replicas=2, rank=1, seed=4)
    left, right = list(rank0), list(rank1)
    assert not set(left) & set(right)
    assert set(left + right) == set(range(len(dataset)))
    assert left == list(rank0)
    rank0.set_epoch(1)
    assert left != list(rank0)


def test_sampler_cyclically_pads_when_dataset_is_smaller_than_world() -> None:
    class _TinyDataset:
        def __len__(self) -> int:
            return 1

        def window_shard_index(self, index: int) -> int:
            return 0

    samplers = [
        ShardAwareDistributedSampler(
            _TinyDataset(),
            num_replicas=4,
            rank=rank,
            shuffle=False,
        )
        for rank in range(4)
    ]

    assert [len(sampler) for sampler in samplers] == [1, 1, 1, 1]
    assert [list(sampler) for sampler in samplers] == [[0], [0], [0], [0]]


def test_sampler_can_pad_each_rank_to_complete_optimizer_updates() -> None:
    class _TenWindows:
        def __len__(self) -> int:
            return 10

        def window_shard_index(self, index: int) -> int:
            return index // 2

    samplers = [
        ShardAwareDistributedSampler(
            _TenWindows(),
            num_replicas=2,
            rank=rank,
            shuffle=False,
            drop_last=False,
            samples_per_rank_multiple=4,
        )
        for rank in range(2)
    ]
    values = [list(sampler) for sampler in samplers]
    assert [len(items) for items in values] == [8, 8]
    assert set(values[0] + values[1]) == set(range(len(_TenWindows())))
