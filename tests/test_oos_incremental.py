from __future__ import annotations

import json
from typing import TYPE_CHECKING

import polars as pl
import pytest

from quant_fm.downstream.train_ranker import RankerObjectiveConfig
from quant_fm.embedding.contract import (
    AFTER_CLOSE_AVAILABILITY,
    CAUSAL_CHUNKED_ENCODER,
    EMBEDDING_CONTRACT_VERSION,
    STOCK_DAY_GRANULARITY,
    EmbeddingContract,
    write_embedding_contract,
)
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


def _embedding_contract(*, pooling: str = "mean") -> EmbeddingContract:
    return EmbeddingContract(
        format_version=EMBEDDING_CONTRACT_VERSION,
        fm_checkpoint_sha256="a" * 64,
        vocab_sha256="b" * 64,
        schema_version="cn_l2_v1",
        book_state_timing="none",
        pooling_version="flat_v1",
        granularity=STOCK_DAY_GRANULARITY,
        context=2048,
        chunk_stride=2048,
        pooling=pooling,
        last_k=256,
        dtype="bf16",
        encoder_width=2,
        pooling_components=(pooling,),
        pooling_scalar_components=(),
        embedding_columns=("emb_0", "emb_1"),
        embedding_width=2,
        signal_availability=AFTER_CLOSE_AVAILABILITY,
        encoder_semantics=CAUSAL_CHUNKED_ENCODER,
        event_ordering_version="local_time_v1",
        feature_transform_version="ew_vwap_future_backfill_v1",
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
        "require_ranker_validation": False,
        "allow_legacy_training_panel": True,
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


def test_ranker_objective_change_forces_new_cache_generation(tmp_path: Path) -> None:
    symbols = ["000001", "000002", "000003"]
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    _embedding_rows(["2025-01-02", "2025-01-03"], symbols).write_parquet(
        train_dir / "all.parquet"
    )
    train_panel = tmp_path / "train_panel.parquet"
    _panel_rows(["2025-01-02", "2025-01-03"], symbols).write_parquet(train_panel)
    test_emb = tmp_path / "test.parquet"
    _embedding_rows(["2026-01-05"], symbols).write_parquet(test_emb)
    out = tmp_path / "out"
    kwargs = {
        "train_emb_dir": train_dir,
        "train_panel": train_panel,
        "test_emb": test_emb,
        "out_dir": out,
        "epochs": 1,
        "device": "cpu",
        "min_names_per_day": 2,
        "require_ranker_validation": False,
        "allow_legacy_training_panel": True,
    }

    build_oos_delivery(**kwargs)
    first = json.loads((out / "ranker_metadata.json").read_text())
    assert first["delivery_policy"]["mode"] == "legacy_diagnostic"
    assert first["delivery_policy"]["production_eligible"] is False
    changed = RankerObjectiveConfig(
        global_ic_weight=0.31,
        pair_samples_per_day=32,
    )
    build_oos_delivery(
        **kwargs,
        objective=changed,
        allow_research_objective_return_spec_override=True,
    )
    second = json.loads((out / "ranker_metadata.json").read_text())

    assert first["cache_key"] != second["cache_key"]
    assert second["checkpoint_version"] == "multi_lambda_ndcg_v1"
    assert second["objective"]["global_ic_weight"] == 0.31
    assert second["delivery_policy"]["mode"] == "legacy_research_override"
    assert second["delivery_policy"]["production_eligible"] is False


def test_oos_delivery_persists_verified_embedding_representation(
    tmp_path: Path,
) -> None:
    symbols = ["000001", "000002", "000003"]
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    train_embedding = train_dir / "all.parquet"
    _embedding_rows(["2025-01-02", "2025-01-03"], symbols).write_parquet(
        train_embedding
    )
    write_embedding_contract(train_embedding, _embedding_contract())
    train_panel = tmp_path / "train_panel.parquet"
    _panel_rows(["2025-01-02", "2025-01-03"], symbols).write_parquet(train_panel)
    test_embedding = tmp_path / "test.parquet"
    _embedding_rows(["2026-01-05"], symbols).write_parquet(test_embedding)
    write_embedding_contract(test_embedding, _embedding_contract())
    out = tmp_path / "out"

    build_oos_delivery(
        train_emb_dir=train_dir,
        train_panel=train_panel,
        test_emb=test_embedding,
        out_dir=out,
        epochs=1,
        device="cpu",
        min_names_per_day=2,
        require_ranker_validation=False,
        allow_legacy_training_panel=True,
    )

    metadata = json.loads((out / "ranker_metadata.json").read_text())
    representation = metadata["embedding_representation"]
    assert representation["mode"] == "verified"
    assert representation["contract"]["pooling"] == "mean"
    assert metadata["cache_spec"]["embedding_representation"] == representation
    assert metadata["foundation_model"]["lineage_mode"] == "legacy_manual_unverified"
    assert metadata["cache_spec"]["foundation_model"] == metadata["foundation_model"]
    signal_manifest = json.loads((out / "signal_manifest.json").read_text())
    assert signal_manifest["foundation_model"] == metadata["foundation_model"]


def test_oos_delivery_rejects_same_width_pooling_mismatch(tmp_path: Path) -> None:
    symbols = ["000001", "000002", "000003"]
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    train_embedding = train_dir / "all.parquet"
    _embedding_rows(["2025-01-02"], symbols).write_parquet(train_embedding)
    write_embedding_contract(train_embedding, _embedding_contract())
    train_panel = tmp_path / "train_panel.parquet"
    _panel_rows(["2025-01-02"], symbols).write_parquet(train_panel)
    test_embedding = tmp_path / "test.parquet"
    _embedding_rows(["2026-01-05"], symbols).write_parquet(test_embedding)
    write_embedding_contract(test_embedding, _embedding_contract(pooling="last"))

    with pytest.raises(ValueError, match="embedding representation mismatch"):
        build_oos_delivery(
            train_emb_dir=train_dir,
            train_panel=train_panel,
            test_emb=test_embedding,
            out_dir=tmp_path / "out",
            epochs=1,
            device="cpu",
            min_names_per_day=2,
            require_ranker_validation=False,
            allow_legacy_training_panel=True,
        )


def test_ranker_excludes_labels_available_on_oos_start_and_rekeys_cache(
    tmp_path: Path,
) -> None:
    symbols = ["000001", "000002", "000003"]
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    _embedding_rows(["2025-12-30", "2025-12-31"], symbols).write_parquet(
        train_dir / "all.parquet"
    )
    train_panel = tmp_path / "train_panel.parquet"
    pl.DataFrame(
        [
            {
                "date": date,
                "next_date": next_date,
                "label_availability_timestamp": f"{next_date}T16:00:00",
                "symbol": symbol,
                "fwd_ret": symbol_index / 100.0,
            }
            for date, next_date in (
                ("2025-12-30", "2025-12-31"),
                ("2025-12-31", "2026-01-05"),
            )
            for symbol_index, symbol in enumerate(symbols)
        ]
    ).write_parquet(train_panel)
    test_emb = tmp_path / "test.parquet"
    out = tmp_path / "out"
    kwargs = {
        "train_emb_dir": train_dir,
        "train_panel": train_panel,
        "test_emb": test_emb,
        "out_dir": out,
        "epochs": 1,
        "device": "cpu",
        "min_names_per_day": 2,
        "require_ranker_validation": False,
        "allow_legacy_training_panel": True,
    }

    _embedding_rows(["2026-01-06"], symbols).write_parquet(test_emb)
    build_oos_delivery(**kwargs)
    first_metadata = json.loads((out / "ranker_metadata.json").read_text())
    assert first_metadata["training_end_date"] == "2025-12-31"

    _embedding_rows(["2026-01-05"], symbols).write_parquet(test_emb)
    build_oos_delivery(**kwargs)
    second_metadata = json.loads((out / "ranker_metadata.json").read_text())

    assert second_metadata["cache_key"] != first_metadata["cache_key"]
    assert second_metadata["training_end_date"] == "2025-12-30"
    assert second_metadata["label_cutoff"]["oos_start_date"] == "2026-01-05"
    assert second_metadata["label_cutoff"]["label_horizon_columns"] == [
        "next_date",
        "label_availability_timestamp",
    ]
    assert second_metadata["label_cutoff"]["retained_label_rows"] == 3
    assert second_metadata["label_cutoff"]["excluded_label_rows"] == 3
    assert (
        second_metadata["cache_spec"]["label_cutoff"]["oos_start_date"] == "2026-01-05"
    )


def test_ranker_fails_when_no_label_is_available_before_oos(tmp_path: Path) -> None:
    symbols = ["000001", "000002", "000003"]
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    _embedding_rows(["2025-12-31"], symbols).write_parquet(train_dir / "all.parquet")
    train_panel = tmp_path / "train_panel.parquet"
    pl.DataFrame(
        [
            {
                "date": "2025-12-31",
                "next_date": "2026-01-05",
                "symbol": symbol,
                "fwd_ret": symbol_index / 100.0,
            }
            for symbol_index, symbol in enumerate(symbols)
        ]
    ).write_parquet(train_panel)
    test_emb = tmp_path / "test.parquet"
    _embedding_rows(["2026-01-05"], symbols).write_parquet(test_emb)
    out = tmp_path / "out"

    with pytest.raises(RuntimeError, match="no finite training labels"):
        build_oos_delivery(
            train_emb_dir=train_dir,
            train_panel=train_panel,
            test_emb=test_emb,
            out_dir=out,
            epochs=1,
            device="cpu",
            min_names_per_day=2,
            require_ranker_validation=False,
            allow_legacy_training_panel=True,
        )
    assert not (out / "ranker_checkpoint.pt").exists()
    assert not (out / ".score.lock").exists()


def test_ranker_rejects_legacy_training_panel_by_default(tmp_path: Path) -> None:
    symbols = ["000001", "000002", "000003"]
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    _embedding_rows(["2025-01-02"], symbols).write_parquet(train_dir / "all.parquet")
    train_panel = tmp_path / "train_panel.parquet"
    _panel_rows(["2025-01-02"], symbols).write_parquet(train_panel)
    test_emb = tmp_path / "test.parquet"
    _embedding_rows(["2026-01-05"], symbols).write_parquet(test_emb)

    with pytest.raises(ValueError, match="strict tradable evaluation"):
        build_oos_delivery(
            train_emb_dir=train_dir,
            train_panel=train_panel,
            test_emb=test_emb,
            out_dir=tmp_path / "out",
            epochs=1,
            device="cpu",
            min_names_per_day=2,
            require_ranker_validation=False,
        )


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
    assert len(entries[0].sha256) == 64

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
