from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from quant_fm.embedding.extract_hidden import _balanced_shard_partition
from quant_fm.manifest.build_manifest import ShardEntry
from quant_fm.scripts.build_oos_delivery import (
    _prune_consumed_tokens,
    build_oos_delivery,
)
from quant_fm.scripts.make_adhoc_manifest import scan_dates, write_day_index

if TYPE_CHECKING:
    from pathlib import Path


def _embedding_rows(dates: list[str], symbols: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "date": date,
                "symbol": symbol,
                "market": "SZ",
                "emb_0": float(day_index + symbol_index),
                "emb_1": float(day_index - symbol_index),
            }
            for day_index, date in enumerate(dates)
            for symbol_index, symbol in enumerate(symbols)
        ]
    )


def _panel_rows(dates: list[str], symbols: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"date": date, "symbol": symbol, "fwd_ret": symbol_index / 100.0}
            for date in dates
            for symbol_index, symbol in enumerate(symbols)
        ]
    )


def test_ranker_cache_and_incremental_scores(tmp_path: Path) -> None:
    symbols = ["000001", "000002", "000003"]
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    _embedding_rows(["2025-01-02", "2025-01-03"], symbols).write_parquet(
        train_dir / "all.parquet"
    )
    train_panel = tmp_path / "train_panel.parquet"
    _panel_rows(["2025-01-02", "2025-01-03"], symbols).write_parquet(train_panel)
    test_emb = tmp_path / "test.parquet"
    first = _embedding_rows(["2026-01-05"], symbols)
    first.write_parquet(test_emb)
    out = tmp_path / "out"
    kwargs = {
        "train_emb_dir": train_dir,
        "train_panel": train_panel,
        "test_emb": test_emb,
        "out_dir": out,
        "epochs": 1,
        "device": "cpu",
        "min_names_per_day": 2,
    }

    build_oos_delivery(**kwargs)
    checkpoint_mtime = (out / "ranker_checkpoint.pt").stat().st_mtime_ns
    build_oos_delivery(**kwargs)
    assert (out / "ranker_checkpoint.pt").stat().st_mtime_ns == checkpoint_mtime

    second = _embedding_rows(["2026-01-06"], symbols)
    pl.concat([first, second]).write_parquet(test_emb)
    build_oos_delivery(**kwargs)
    scores = pl.read_parquet(out / "scores.parquet")
    assert scores.height == 6
    assert scores["date"].n_unique() == 2
    assert (out / "ranker_checkpoint.pt").stat().st_mtime_ns == checkpoint_mtime


def test_day_index_and_safe_pruning(tmp_path: Path) -> None:
    shard = tmp_path / "tokens" / "SZ" / "000001" / "2026-01-05.parquet"
    shard.parent.mkdir(parents=True)
    pl.DataFrame({"tok_event_type": [1, 2]}).write_parquet(shard)
    write_day_index(tmp_path / "tokens", "2026-01-05")
    entries = scan_dates(
        tmp_path / "tokens", include_dates={"2026-01-05"}, split="test"
    )
    assert len(entries) == 1
    assert entries[0].rows == 2

    embedding = tmp_path / "embeddings" / "incr" / "oos_all.parquet"
    embedding.parent.mkdir(parents=True)
    _embedding_rows(["2026-01-05"], ["000001"]).write_parquet(embedding)
    (tmp_path / "data" / ".prune_embedded_tokens").write_text("enabled")
    _prune_consumed_tokens(embedding, ["2026-01-05"])
    assert not shard.exists()
    assert (tmp_path / "data/pruned_token_receipts/2026-01-05.json").exists()


def test_embedding_partition_balances_rows() -> None:
    shards = [
        ShardEntry("SZ", str(index), "2026-01-05", str(index), rows, "")
        for index, rows in enumerate([100, 90, 20, 10])
    ]
    left = _balanced_shard_partition(shards, 2, 0)
    right = _balanced_shard_partition(shards, 2, 1)
    assert {item.path for item in left + right} == {item.path for item in shards}
    assert abs(sum(item.rows for item in left) - sum(item.rows for item in right)) <= 20


def test_score_lock_is_removed_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant_fm.scripts.build_oos_delivery as delivery

    def fail(**_kwargs: object) -> Path:
        message = "synthetic failure"
        raise RuntimeError(message)

    monkeypatch.setattr(delivery, "_build_oos_delivery_locked", fail)
    out_dir = tmp_path / "delivery"
    with pytest.raises(RuntimeError, match="synthetic failure"):
        delivery.build_oos_delivery(
            train_emb_dir=tmp_path / "train_emb",
            train_panel=tmp_path / "train_panel.parquet",
            test_emb=tmp_path / "test_emb.parquet",
            out_dir=out_dir,
            device="cpu",
        )
    assert not (out_dir / ".score.lock").exists()
