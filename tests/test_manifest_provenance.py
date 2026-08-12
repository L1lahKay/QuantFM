from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import polars as pl
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from quant_fm.manifest.build_manifest import Manifest, ShardEntry, build_manifest
from quant_fm.manifest.validation import (
    sha256_file,
    validate_manifest_shard_paths,
    validate_manifest_shards,
)
from quant_fm.scripts.make_adhoc_manifest import build_adhoc_manifest
from quant_fm.tokenizer.artifact_contract import token_contract_path
from quant_fm.tokenizer.tokenize_events import tokenize_path
from quant_fm.tokenizer.vocab import default_vocab


def _events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "security_id": ["000001", "000001"],
            "date": ["2025-01-02", "2025-01-02"],
            "int_time": [93_000_000, 93_000_001],
            "source_seqnum": [1, 2],
            "event_idx": [0, 1],
            "price": [10.0, 10.1],
            "qty": [100.0, 200.0],
            "evt_type": ["ADD", "EXEC"],
            "side": ["B", "S"],
            "session": ["CONT_AM", "CONT_AM"],
            "board": ["MAIN", "MAIN"],
            "order_type": ["LIMIT", "UNKNOWN"],
            "event_source": ["ORDER", "TRADE"],
        }
    )


def _artifacts(tmp_path: Path):
    source = tmp_path / "events.parquet"
    token = tmp_path / "tokens" / "SZ" / "000001" / "2025-01-02.parquet"
    vocab_path = tmp_path / "vocab.json"
    _events().write_parquet(source)
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
    return vocab, token, manifest


def test_manifest_rejects_split_that_disagrees_with_declared_boundaries(
    tmp_path: Path,
) -> None:
    manifest = Manifest(
        shards=[
            ShardEntry(
                "SZ",
                "000001",
                "2025-01-02",
                "unused.parquet",
                1,
                "hash",
                "test",
            )
        ],
        train_end="2025-01-02",
        val_end="2025-01-03",
    )

    with pytest.raises(ValueError, match="disagrees with declared boundaries"):
        manifest.save(tmp_path / "manifest.json")


@pytest.mark.parametrize(
    ("train_end", "val_end", "message"),
    [
        ("2025-1-2", "2025-01-03", "canonical YYYY-MM-DD"),
        ("2025-01-04", "2025-01-03", "train_end <= val_end"),
        ("2025-01-02", None, "declare train_end and val_end together"),
    ],
)
def test_manifest_rejects_invalid_declared_boundaries(
    tmp_path: Path,
    train_end: str | None,
    val_end: str | None,
    message: str,
) -> None:
    manifest = Manifest(train_end=train_end, val_end=val_end)

    with pytest.raises(ValueError, match=message):
        manifest.save(tmp_path / "manifest.json")


def test_manifest_runtime_validation_rejects_live_parquet_hash_tamper(
    tmp_path: Path,
) -> None:
    vocab, token, manifest = _artifacts(tmp_path)
    frame = pl.read_parquet(token).with_columns(
        (pl.col("tok_evt_type") + 1).alias("tok_evt_type")
    )
    frame.write_parquet(token)

    with pytest.raises(ValueError, match="parquet SHA-256 mismatch"):
        validate_manifest_shards(
            manifest,
            vocab,
            context="test",
        )


def test_manifest_runtime_validation_rejects_sidecar_hash_tamper(
    tmp_path: Path,
) -> None:
    vocab, token, manifest = _artifacts(tmp_path)
    sidecar = token_contract_path(token)
    sidecar.write_text(sidecar.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="sidecar SHA-256 mismatch"):
        validate_manifest_shards(
            manifest,
            vocab,
            context="test",
        )


def test_manifest_runtime_validation_rejects_swapped_logical_paths(
    tmp_path: Path,
) -> None:
    _vocab, train_token, manifest = _artifacts(tmp_path)
    validation_token = tmp_path / "tokens" / "SZ" / "000001" / "2025-01-03.parquet"
    shutil.copy2(train_token, validation_token)
    shutil.copy2(
        token_contract_path(train_token),
        token_contract_path(validation_token),
    )
    train = manifest.shards[0]
    train.path = str(validation_token.resolve())
    validation = ShardEntry(
        market="SZ",
        symbol="000001",
        date="2025-01-03",
        path=str(train_token.resolve()),
        rows=train.rows,
        sha256=sha256_file(train_token),
        split="val",
        data_contract_sha256=sha256_file(token_contract_path(train_token)),
    )
    manifest.shards.append(validation)

    with pytest.raises(ValueError, match="path does not match its logical identity"):
        validate_manifest_shard_paths(manifest, context="test")


def test_manifest_path_contract_requires_one_root_and_unique_logical_keys(
    tmp_path: Path,
) -> None:
    first = ShardEntry(
        "SZ",
        "000001",
        "2025-01-02",
        str(tmp_path / "first" / "tokens" / "SZ" / "000001" / "2025-01-02.parquet"),
        1,
        "a",
    )
    second = ShardEntry(
        "SH",
        "600000",
        "2025-01-03",
        str(tmp_path / "second" / "tokens" / "SH" / "600000" / "2025-01-03.parquet"),
        1,
        "b",
    )
    with pytest.raises(ValueError, match="do not share one tokens root"):
        validate_manifest_shard_paths(
            Manifest(shards=[first, second]),
            context="test",
        )

    duplicate = ShardEntry(
        first.market,
        first.symbol,
        first.date,
        first.path,
        first.rows,
        first.sha256,
    )
    with pytest.raises(ValueError, match="duplicate logical shard"):
        validate_manifest_shard_paths(
            Manifest(shards=[first, duplicate]),
            context="test",
        )


def test_manifest_path_contract_rejects_external_tokens_root(tmp_path: Path) -> None:
    shard = ShardEntry(
        "SZ",
        "000001",
        "2025-01-02",
        str(tmp_path / "external" / "tokens" / "SZ" / "000001" / "2025-01-02.parquet"),
        1,
        "hash",
    )

    with pytest.raises(ValueError, match="escape the expected tokens root"):
        validate_manifest_shard_paths(
            Manifest(shards=[shard]),
            context="test",
            expected_tokens_root=tmp_path / "generation" / "tokens",
        )


def test_explicit_shard_requires_parquet_hash_even_with_recorded_sidecar(
    tmp_path: Path,
) -> None:
    vocab, _, manifest = _artifacts(tmp_path)
    shard = manifest.shards[0]
    shard.sha256 = ""

    with pytest.raises(ValueError, match="full manifest-recorded parquet SHA-256"):
        validate_manifest_shards(manifest, vocab, context="test")

    shard.data_contract_sha256 = None
    with pytest.raises(ValueError, match="full manifest-recorded parquet SHA-256"):
        validate_manifest_shards(manifest, vocab, context="test")


def test_adhoc_manifest_records_full_parquet_and_sidecar_hashes(
    tmp_path: Path,
) -> None:
    vocab, token, _ = _artifacts(tmp_path)
    destination = tmp_path / "adhoc.json"

    build_adhoc_manifest(
        tokens_dir=tmp_path / "tokens",
        out=destination,
        vocab_path=tmp_path / "vocab.json",
    )

    manifest = Manifest.load(destination)
    assert len(manifest.shards) == 1
    shard = manifest.shards[0]
    assert shard.sha256 == sha256_file(token)
    assert shard.data_contract_sha256 == sha256_file(token_contract_path(token))
    validate_manifest_shards(manifest, vocab, context="test")
