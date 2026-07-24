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
