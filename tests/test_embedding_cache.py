from argparse import Namespace
from pathlib import Path

import polars as pl
import pytest

import quant_fm.embedding.extract_hidden as extract_hidden
from quant_fm.embedding.extract_hidden import (
    _atomic_write_json,
    _cache_metadata_path,
    _embedding_cache_hit,
    _embedding_cache_spec,
    _select_evenly_spaced_dates,
)
from quant_fm.manifest.build_manifest import ShardEntry, build_manifest
from quant_fm.tokenizer.artifact_contract import token_contract_path
from quant_fm.tokenizer.tokenize_events import tokenize_path
from quant_fm.tokenizer.vocab import default_vocab


def _shards() -> list[ShardEntry]:
    return [
        ShardEntry(
            market="SZ",
            symbol=f"{symbol:06d}",
            date=f"2025-01-{day:02d}",
            path=f"/tokens/{day}/{symbol}.parquet",
            rows=day * 100 + symbol,
            sha256=f"sha-{day}-{symbol}",
            split="train",
        )
        for day in range(1, 11)
        for symbol in range(2)
    ]


def _args() -> Namespace:
    return Namespace(
        split="train",
        context=2048,
        pooling="mean",
        last_k=256,
        dtype="bf16",
        batch_size=16,
        num_parts=1,
        part_index=0,
    )


def test_representative_date_subset_keeps_complete_cross_sections() -> None:
    selected = _select_evenly_spaced_dates(_shards(), 3)

    assert sorted({shard.date for shard in selected}) == [
        "2025-01-01",
        "2025-01-05",
        "2025-01-10",
    ]
    assert len(selected) == 6


def test_embedding_cache_requires_matching_spec_and_readable_output(tmp_path) -> None:
    selected = _select_evenly_spaced_dates(_shards(), 2)
    spec = _embedding_cache_spec(_args(), selected, checkpoint_id="checkpoint-a")
    output = tmp_path / "train.parquet"
    pl.DataFrame(
        {
            "date": [shard.date for shard in selected],
            "symbol": [shard.symbol for shard in selected],
            "market": [shard.market for shard in selected],
            "emb_0": [0.0] * len(selected),
        }
    ).write_parquet(output)
    metadata = _cache_metadata_path(output)
    _atomic_write_json(metadata, spec)

    assert _embedding_cache_hit(output, metadata, spec)

    changed = dict(spec)
    changed["checkpoint_sha256"] = "checkpoint-b"
    assert not _embedding_cache_hit(output, metadata, changed)

    output.write_text("not parquet", encoding="utf-8")
    assert not _embedding_cache_hit(output, metadata, spec)


def test_embedding_cache_fingerprint_includes_token_contract() -> None:
    original = _shards()
    changed_contract = _shards()
    original[0].data_contract_sha256 = "contract-a"
    changed_contract[0].data_contract_sha256 = "contract-b"

    first = _embedding_cache_spec(_args(), original, checkpoint_id="checkpoint-a")
    second = _embedding_cache_spec(
        _args(), changed_contract, checkpoint_id="checkpoint-a"
    )

    assert first["format_version"] == 3
    assert first["shards_sha256"] != second["shards_sha256"]


def test_embedding_cache_hashes_sidecar_for_legacy_manifest_entries(tmp_path) -> None:
    token_path = tmp_path / "tokens.parquet"
    shard = ShardEntry("SZ", "000001", "2025-01-02", str(token_path), 3, "")
    sidecar = token_contract_path(token_path)
    sidecar.write_text("contract-a", encoding="utf-8")
    first = _embedding_cache_spec(_args(), [shard], checkpoint_id="checkpoint-a")

    sidecar.write_text("contract-b", encoding="utf-8")
    second = _embedding_cache_spec(_args(), [shard], checkpoint_id="checkpoint-a")

    assert first["shards_sha256"] != second["shards_sha256"]


def test_embedding_main_validates_live_shards_before_cache_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "events.parquet"
    token = tmp_path / "tokens" / "SZ" / "000001" / "2025-01-02.parquet"
    vocab_path = tmp_path / "vocab.json"
    pl.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "security_id": ["000001"],
            "date": ["2025-01-02"],
            "int_time": [93_000_000],
            "source_seqnum": [1],
            "event_idx": [0],
            "price": [10.0],
            "qty": [100.0],
            "evt_type": ["ADD"],
            "side": ["B"],
            "session": ["CONT_AM"],
            "board": ["MAIN"],
            "order_type": ["LIMIT"],
            "event_source": ["ORDER"],
        }
    ).write_parquet(source)
    vocab = default_vocab(n_bins=4)
    vocab.fit_dates = ("2025-01-02",)
    vocab.save(vocab_path)
    tokenize_path(source, token, vocab)
    manifest = build_manifest(
        tmp_path / "tokens",
        train_end="2025-01-02",
        val_end="2025-01-03",
        vocab_path=str(vocab_path),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    pl.read_parquet(token).with_columns(
        (pl.col("tok_evt_type") + 1).alias("tok_evt_type")
    ).write_parquet(token)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"not reached")
    args = Namespace(
        checkpoint=checkpoint,
        checkpoint_id=None,
        manifest=manifest_path,
        vocab=vocab_path,
        split="train",
        out=tmp_path / "embedding.parquet",
        context=8,
        stride=4,
        pooling="mean",
        last_k=4,
        dtype="fp32",
        batch_size=1,
        num_parts=1,
        part_index=0,
        max_dates=0,
        resume=True,
        device="cpu",
    )
    cache_called = False

    def _unexpected_cache(*_args, **_kwargs):
        nonlocal cache_called
        cache_called = True
        return True

    monkeypatch.setattr(extract_hidden, "_parse_args", lambda: args)
    monkeypatch.setattr(extract_hidden, "_embedding_cache_hit", _unexpected_cache)

    with pytest.raises(ValueError, match="parquet SHA-256 mismatch"):
        extract_hidden.main()
    assert cache_called is False
