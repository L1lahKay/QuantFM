from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from pylob.pipeline.standardize import standardize_order_frame, standardize_trade_frame
from pylob.pipeline.workflow import _clean_one_symbol

from quant_fm.manifest.build_manifest import Manifest, ShardEntry
from quant_fm.scripts.download_from_minio import _rebase_downloaded_manifest
from quant_fm.scripts.run_medium import run as run_medium
from quant_fm.scripts.run_pilot import run
from quant_fm.tokenizer.artifact_contract import read_token_contract
from quant_fm.tokenizer.field_spec import BOOK_FIELD_SPECS_V2
from quant_fm.tokenizer.vocab_v2 import VocabV2


def test_download_rebases_manifest_to_local_v2_root(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest_path = data_dir / "manifest.json"
    Manifest(
        shards=[
            ShardEntry(
                market="SZ",
                symbol="000001",
                date="2025-01-02",
                path="/producer/root/tokens/SZ/000001/2025-01-02.parquet",
                rows=2,
                sha256="hash",
            )
        ],
        vocab_path="/producer/root/data/vocab_v2.json",
        schema_version="cn_l2_v2",
    ).save(manifest_path)

    _rebase_downloaded_manifest(tmp_path, "vocab_v2.json")

    rebased = Manifest.load(manifest_path)
    assert rebased.vocab_path == str((tmp_path / "data" / "vocab_v2.json").resolve())
    assert rebased.shards[0].path == str(
        (tmp_path / "tokens" / "SZ" / "000001" / "2025-01-02.parquet").resolve()
    )


def test_clean_replay_persists_real_aligned_book_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pylob.book_state as book_state_module

    original_snapshot = book_state_module.snapshot_book_state
    snapshot_calls = 0

    def counted_snapshot(*args, **kwargs):
        nonlocal snapshot_calls
        snapshot_calls += 1
        return original_snapshot(*args, **kwargs)

    monkeypatch.setattr(book_state_module, "snapshot_book_state", counted_snapshot)
    orders = standardize_order_frame(
        pl.DataFrame(
            {
                "symbol": ["000001", "000001"],
                "int_time": [93_000_000, 93_000_001],
                "local_time": [93_000_100, 93_000_101],
                "serial": [1, 2],
                "bsflag": ["B", "S"],
                "orderorino": [1, 2],
                "order_price": [10_000, 10_200],
                "order_volume": [100, 300],
                "order_type": ["0", "0"],
            }
        )
    )
    trades = standardize_trade_frame(pl.DataFrame(schema={"symbol": pl.String}))

    status = _clean_one_symbol(
        symbol="000001",
        trade_df=trades,
        order_df=orders,
        market="SZ",
        output_dir=tmp_path,
        cut_time=151_000_000,
        cut_serial=None,
        write_debug_artifacts=False,
        capture_book_state=True,
        timeout_s=0,
    )

    assert status == "written"
    features = pl.read_parquet(tmp_path / "SZ" / "000001" / "book_features.parquet")
    assert features.height == 2
    assert features["book_valid_post"].to_list() == [False, True]
    assert features["spread_ticks_post"].to_list() == [None, 2]
    assert features["imbalance_l1_post"][1] == pytest.approx(-0.5)
    assert snapshot_calls == orders.height + 1


def _clean_events(date_index: int) -> pl.DataFrame:
    base = 10_000 + date_index * 100
    return pl.DataFrame(
        {
            "symbol": ["000001"] * 4,
            "market": ["SZ"] * 4,
            "event_idx": [0, 1, 2, 3],
            "int_time": [93_000_000, 93_000_001, 93_000_002, 93_000_003],
            "local_time": [93_000_100, 93_000_101, 93_000_102, 93_000_103],
            "serial": [1, 2, 3, 4],
            "delta_t": [0, 1, 1, 1],
            "session_phase": ["CONTINUOUS_AM"] * 4,
            "event_type": ["ADD", "ADD", "TRADE", "CANCEL"],
            "side": ["BUY", "SELL", "BUY", "SELL"],
            "price": [base, base + 200, base + 100, base + 200],
            "volume": [100, 200, 50, 100],
            "log_volume": [0.0] * 4,
            "orderorino": [1, 2, 0, 2],
            "buy_id": [0, 0, 1, 0],
            "sell_id": [0, 0, 2, 2],
        }
    )


def _book_features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "book_valid_post": [False, True, True, False],
            "spread_ticks_post": [None, 2, 2, None],
            "microprice_delta_ticks_post": [None, -0.5, 0.0, None],
            "imbalance_l1_post": [1.0, -1 / 3, 0.0, 1.0],
            "imbalance_l5_post": [1.0, -1 / 3, 0.0, 1.0],
            "imbalance_l10_post": [1.0, -1 / 3, 0.0, 1.0],
            "log_bid_depth_l5_post": [4.615, 4.615, 4.394, 4.394],
            "log_ask_depth_l5_post": [0.0, 5.303, 4.615, 0.0],
            "event_price_distance_ticks_pre": [None, None, 0.0, 1.0],
        }
    )


def _write_clean_day(root: Path, date: str, date_index: int) -> None:
    symbol_dir = root / "clean" / date / "SZ" / "000001"
    symbol_dir.mkdir(parents=True)
    _clean_events(date_index).write_parquet(symbol_dir / "events.parquet")
    _book_features().write_parquet(symbol_dir / "book_features.parquet")


def test_medium_events_only_defers_global_vocab_and_tokens(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dates = ["2025-01-02", "2025-01-03", "2025-01-06"]
    for index, date in enumerate(dates):
        _write_clean_day(tmp_path, date, index)
    monkeypatch.setenv("CANON_WORKERS", "1")

    run_medium(
        dates=dates,
        symbols_sz=("000001",),
        symbols_sh=(),
        workdir=tmp_path,
        train_end=dates[0],
        val_end=dates[1],
        n_bins=8,
        skip_clean=True,
        drop_clean=True,
        drop_events=False,
        fit_sample_days=None,
        resume=False,
        estimate_only=False,
        events_only=True,
    )

    assert len(list((tmp_path / "events").rglob("*.parquet"))) == len(dates)
    assert not (tmp_path / "data" / "vocab_v2.json").exists()
    assert not (tmp_path / "tokens").exists()
    assert not (tmp_path / "data" / "manifest.json").exists()
    for date in dates:
        marker = tmp_path / "data" / ".done" / date
        assert marker.read_text(encoding="utf-8") == "canonicalized:cn_l2_v2\n"


def test_pilot_builds_audited_v2_artifacts_from_captured_book_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dates = ["2025-01-02", "2025-01-03", "2025-01-06"]
    for index, date in enumerate(dates):
        _write_clean_day(tmp_path, date, index)
    monkeypatch.setenv("CANON_WORKERS", "1")

    run(
        dates=dates,
        symbols=("000001",),
        market="SZ",
        workdir=tmp_path,
        train_end=dates[0],
        val_end=dates[1],
        n_bins=8,
        skip_clean=True,
        data_version="v2",
        v2_max_samples_per_field=100,
    )

    vocab = VocabV2.load(tmp_path / "data" / "vocab_v2.json")
    manifest = Manifest.load(tmp_path / "data" / "manifest.json")
    audit = json.loads((tmp_path / "artifact_audit.json").read_text())

    assert manifest.schema_version == "cn_l2_v2"
    assert {shard.split for shard in manifest.shards} == {"train", "val", "test"}
    assert {spec.name for spec in BOOK_FIELD_SPECS_V2} <= {
        spec.name for spec in vocab.field_specs
    }
    assert audit["contract_ready"] is True

    token_path = Path(manifest.shards[0].path)
    physical = pl.read_parquet(token_path)
    contract = read_token_contract(token_path)
    assert physical.schema["val_spread_ticks_post"] == pl.Int16
    assert physical.schema["tok_spread_ticks_post_bin"] == pl.UInt8
    assert contract["schema_version"] == "cn_l2_v2"
    assert contract["storage_encoding"]["format_version"] == "token_uint_scalar_q16_v1"
