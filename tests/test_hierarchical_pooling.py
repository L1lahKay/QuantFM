from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest

torch = pytest.importorskip("torch")

from quant_fm.embedding.extract_hidden import (  # noqa: E402
    _causal_windows,
    _embed_shard,
    _parse_args,
    extract_stock_day_embeddings_batched,
)
from quant_fm.embedding.pool_stock_day import (  # noqa: E402
    MULTISCALE_VECTOR_NAMES,
    MultiScaleStockDayPoolAccumulator,
    StockDayPoolAccumulator,
    pool_hidden_chunks,
)
from quant_fm.embedding.pooling_spec import (  # noqa: E402
    DEFAULT_V2_MULTI_SCALE_OUTPUTS,
)
from quant_fm.manifest.build_manifest import ShardEntry  # noqa: E402
from quant_fm.tokenizer.artifact_contract import write_token_contract  # noqa: E402
from quant_fm.tokenizer.field_spec import FieldSpec  # noqa: E402
from quant_fm.tokenizer.storage_encoding_v2 import quantize_frame_v2  # noqa: E402
from quant_fm.tokenizer.vocab_v2 import default_vocab_v2  # noqa: E402


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


class _TrackingIdentityModel(_IdentityModel):
    def __init__(self) -> None:
        super().__init__()
        self.all_valid_hints: list[bool] = []

    def encode(self, batch):
        self.all_valid_hints.append(bool(batch.get("_all_tokens_valid", False)))
        return super().encode(batch)


class _CumulativeModel(_IdentityModel):
    def encode(self, batch):
        return batch["tok_value"].float().cumsum(dim=1).unsqueeze(-1)


class _ScalarIdentityModel(_IdentityModel):
    def __init__(self) -> None:
        self.cfg = SimpleNamespace(
            input_fields=("tok_value",),
            scalar_fields={},
            standalone_scalar_fields=("val_feature",),
            d_model=1,
        )

    def encode(self, batch):
        return batch["val_feature"].unsqueeze(-1)


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

    assert single.shape == (len(DEFAULT_V2_MULTI_SCALE_OUTPUTS),)
    actual = batched.select(pl.exclude("date", "symbol", "market")).row(0)
    assert actual == pytest.approx(tuple(single.tolist()))


def test_embedding_single_and_batched_paths_decode_q16_like_legacy(tmp_path) -> None:
    vocab = default_vocab_v2(
        (
            FieldSpec("value", "value", "categorical"),
            FieldSpec("feature", "feature", "continuous"),
        ),
        categorical={"value": ("x",)},
    )
    semantic = pl.DataFrame(
        {
            "tok_value": pl.Series([6, 6, 6, 6], dtype=pl.Int64),
            "val_feature": pl.Series([-5.0, -1.25, 0.25, 5.0], dtype=pl.Float32),
        }
    )
    legacy_path = tmp_path / "legacy.parquet"
    q16_path = tmp_path / "q16.parquet"
    semantic.write_parquet(legacy_path)
    encoded, metadata = quantize_frame_v2(semantic, vocab)
    encoded.write_parquet(q16_path)
    write_token_contract(
        q16_path,
        vocab,
        storage_encoding=metadata.to_dict(),
    )
    legacy = ShardEntry("SH", "600000", "2025-01-02", str(legacy_path), 4, "a")
    q16 = ShardEntry("SH", "600000", "2025-01-02", str(q16_path), 4, "b")
    model = _ScalarIdentityModel()

    expected = _embed_shard(
        model,
        legacy,
        torch.device("cpu"),
        context=2,
        pooling="mean",
    )
    actual_single = _embed_shard(
        model,
        q16,
        torch.device("cpu"),
        context=2,
        pooling="mean",
    )
    actual_batched = extract_stock_day_embeddings_batched(
        model,
        [q16],
        torch.device("cpu"),
        context=2,
        pooling="mean",
        batch_size=2,
    )["emb_0"][0]

    assert actual_single == pytest.approx(expected, abs=5.0 / 32767.0)
    assert actual_batched == pytest.approx(expected.item(), abs=5.0 / 32767.0)


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


def test_mean_batched_extract_buckets_tail_lengths_without_padding(tmp_path) -> None:
    shards = []
    for index, values in enumerate(([1, 2, 3, 4, 5], [10, 20, 30])):
        path = tmp_path / f"mean-{index}.parquet"
        pl.DataFrame({"tok_value": values}).write_parquet(path)
        shards.append(
            ShardEntry(
                market="SH",
                symbol=f"60000{index}",
                date="2025-01-02",
                path=str(path),
                rows=len(values),
                sha256=f"test-{index}",
            )
        )

    model = _TrackingIdentityModel()
    result = extract_stock_day_embeddings_batched(
        model,
        shards,
        torch.device("cpu"),
        context=4,
        pooling="mean",
        batch_size=2,
    )

    assert result["emb_0"].to_list() == pytest.approx([3.0, 20.0])
    assert model.all_valid_hints
    assert all(model.all_valid_hints)


def test_order_sensitive_pooling_keeps_full_chunk_before_its_tail(tmp_path) -> None:
    shards = []
    for index, values in enumerate(([1, 2, 3, 4, 5], [10])):
        path = tmp_path / f"last-{index}.parquet"
        pl.DataFrame({"tok_value": values}).write_parquet(path)
        shards.append(
            ShardEntry(
                market="SH",
                symbol=f"60000{index}",
                date="2025-01-02",
                path=str(path),
                rows=len(values),
                sha256=f"last-{index}",
            )
        )

    result = extract_stock_day_embeddings_batched(
        _IdentityModel(),
        shards,
        torch.device("cpu"),
        context=4,
        pooling="last",
        batch_size=2,
    )

    assert result["emb_0"].to_list() == pytest.approx([5.0, 10.0])


def test_overlap_windows_have_bounded_prefix_and_unique_emission() -> None:
    windows = _causal_windows(9, context=4, stride=2)

    assert windows == [(0, 4, 0), (2, 6, 2), (4, 8, 2), (6, 9, 2)]
    emitted = [
        global_index
        for start, end, emit_start in windows
        for global_index in range(start + emit_start, end)
    ]
    assert emitted == list(range(9))


def test_overlap_extraction_adds_history_without_double_pooling(tmp_path) -> None:
    path = tmp_path / "overlap.parquet"
    pl.DataFrame({"tok_value": [1, 2, 3, 4, 5, 6]}).write_parquet(path)
    shard = ShardEntry(
        market="SH",
        symbol="600000",
        date="2025-01-02",
        path=str(path),
        rows=6,
        sha256="overlap",
    )

    independent = _embed_shard(
        _CumulativeModel(),
        shard,
        torch.device("cpu"),
        context=4,
        stride=4,
        pooling="mean",
    )
    overlapping = _embed_shard(
        _CumulativeModel(),
        shard,
        torch.device("cpu"),
        context=4,
        stride=2,
        pooling="mean",
    )

    assert independent.item() == pytest.approx(6.0)
    assert overlapping.item() == pytest.approx(50.0 / 6.0)


def test_extract_hidden_parser_has_one_context_and_supports_stride() -> None:
    args = _parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--manifest",
            "manifest.json",
            "--out",
            "embeddings.parquet",
            "--context",
            "2048",
            "--stride",
            "512",
        ]
    )

    assert args.context == 2048
    assert args.stride == 512
    assert args.pooling is None


def test_extract_hidden_parser_defers_representation_to_checkpoint() -> None:
    args = _parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--manifest",
            "manifest.json",
            "--out",
            "embeddings.parquet",
        ]
    )

    assert args.context is None
    assert args.pooling is None
    assert args.stride is None
