from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest

torch = pytest.importorskip("torch")

from quant_fm.embedding.extract_hidden import (  # noqa: E402
    _embed_shard,
    extract_stock_day_embeddings_batched,
)
from quant_fm.embedding.pool_stock_day import (  # noqa: E402
    MULTISCALE_VECTOR_NAMES,
    MultiScaleStockDayPoolAccumulator,
    StockDayPoolAccumulator,
    pool_hidden_chunks,
)
from quant_fm.manifest.build_manifest import ShardEntry  # noqa: E402


def _chunks() -> list[torch.Tensor]:
    return [
        torch.tensor([[1.0, 10.0], [2.0, 20.0]]),
        torch.tensor([[3.0, 30.0]]),
        torch.tensor([[4.0, 40.0], [5.0, 50.0]]),
    ]


def test_multichunk_mean_matches_all_event_mean() -> None:
    chunks = _chunks()
    expected = torch.cat(chunks).mean(dim=0)
    assert torch.equal(pool_hidden_chunks(chunks, method="mean"), expected)


def test_multichunk_last_is_actual_day_tail() -> None:
    chunks = _chunks()
    assert torch.equal(
        pool_hidden_chunks(chunks, method="last"),
        torch.tensor([5.0, 50.0]),
    )


def test_last_k_crosses_chunk_boundaries() -> None:
    chunks = _chunks()
    expected = torch.tensor([[3.0, 30.0], [4.0, 40.0], [5.0, 50.0]]).mean(0)
    actual = pool_hidden_chunks(chunks, method="lastk_mean", last_k=3)
    assert torch.equal(actual, expected)


def test_padded_rows_do_not_enter_accumulator() -> None:
    accumulator = StockDayPoolAccumulator(2, method="mean")
    accumulator.update(
        torch.tensor([[1.0, 2.0], [999.0, 999.0]]),
        torch.tensor([True, False]),
    )
    assert torch.equal(accumulator.value(), torch.tensor([1.0, 2.0]))


def test_chunk_order_changes_temporal_tail_summary() -> None:
    chunks = _chunks()
    original = pool_hidden_chunks(chunks, method="last")
    reordered = pool_hidden_chunks(list(reversed(chunks)), method="last")
    assert not torch.equal(original, reordered)


def _ms(hour: int, minute: int) -> int:
    return (hour * 60 + minute) * 60_000


def test_multiscale_pooling_respects_am_pm_close_masks_across_chunks() -> None:
    accumulator = MultiScaleStockDayPoolAccumulator(1)
    accumulator.update(
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.ones(3, dtype=torch.bool),
        torch.tensor([_ms(9, 20), _ms(9, 35), _ms(11, 30)]),
    )
    accumulator.update(
        torch.tensor([[4.0], [5.0], [6.0]]),
        torch.ones(3, dtype=torch.bool),
        torch.tensor([_ms(13, 5), _ms(14, 45), _ms(14, 58)]),
    )
    values = accumulator.value_dict()

    assert values["mean_all"].item() == pytest.approx(3.5)
    assert values["open_call"].item() == 1.0
    assert values["continuous_am"].item() == pytest.approx(2.5)
    assert values["continuous_pm"].item() == pytest.approx(4.5)
    assert values["close_call"].item() == 6.0
    assert values["close_30m"].item() == pytest.approx(5.5)
    assert values["last_256"].item() == pytest.approx(3.5)
    assert accumulator.concatenate().shape == (len(MULTISCALE_VECTOR_NAMES) + 1,)
    assert accumulator.concatenate()[-1].item() == 6.0


class _IdentityModel:
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(
            input_fields=("tok_value",),
            scalar_fields={},
            standalone_scalar_fields=(),
            d_model=1,
        )

    def encode(self, batch):
        return batch["tok_value"].float().unsqueeze(-1)


def test_multiscale_extract_path_preserves_intraday_time(tmp_path) -> None:
    path = tmp_path / "tokens.parquet"
    pl.DataFrame(
        {
            "tok_value": [1, 2, 3, 4],
            "int_time": [92_000_000, 93_500_000, 130_500_000, 145_800_000],
        }
    ).write_parquet(path)
    shard = ShardEntry(
        market="SH",
        symbol="600000",
        date="2025-01-02",
        path=str(path),
        rows=4,
        sha256="test",
    )
    model = _IdentityModel()

    single = _embed_shard(
        model,
        shard,
        torch.device("cpu"),
        context=2,
        pooling="multi_scale",
    )
    batched = extract_stock_day_embeddings_batched(
        model,
        [shard],
        torch.device("cpu"),
        context=2,
        pooling="multi_scale",
        batch_size=2,
    )

    assert single.shape == (len(MULTISCALE_VECTOR_NAMES) + 1,)
    assert single[-1] == 4.0
    actual = batched.select(pl.exclude("date", "symbol", "market")).row(0)
    assert actual == pytest.approx(tuple(single.tolist()))


def test_multiscale_batched_extract_handles_mixed_chunk_lengths(tmp_path) -> None:
    shards = []
    for index, values in enumerate(([1, 2, 3, 4], [10, 20, 30])):
        path = tmp_path / f"tokens-{index}.parquet"
        pl.DataFrame(
            {
                "tok_value": values,
                "int_time": [92_000_000, 93_500_000, 130_500_000, 145_800_000][
                    : len(values)
                ],
            }
        ).write_parquet(path)
        shards.append(
            ShardEntry(
                market="SH",
                symbol=f"60000{index}",
                date="2025-01-02",
                path=str(path),
                rows=len(values),
                sha256="test",
            )
        )

    model = _IdentityModel()
    batched = extract_stock_day_embeddings_batched(
        model,
        shards,
        torch.device("cpu"),
        context=4,
        pooling="multi_scale",
        batch_size=2,
    )
    expected = [
        _embed_shard(
            model,
            shard,
            torch.device("cpu"),
            context=4,
            pooling="multi_scale",
        )
        for shard in shards
    ]

    for index, vector in enumerate(expected):
        actual = batched.select(pl.exclude("date", "symbol", "market")).row(index)
        assert actual == pytest.approx(tuple(vector.tolist()))
