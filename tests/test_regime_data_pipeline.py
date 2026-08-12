from __future__ import annotations

import json
import math
from datetime import date, timedelta
from typing import TYPE_CHECKING

import polars as pl
import pytest
from pylob.book_state import BookState, BookStateTransition

from quant_fm.data_coverage import write_coverage_receipt
from quant_fm.regime.archive import archive_atomic_day
from quant_fm.regime.atomic import (
    CONTINUOUS_SESSION_SECONDS,
    build_stock_day_atomic,
)
from quant_fm.regime.contract import (
    L2_FEATURE_COLUMNS,
    atomic_manifest_path,
    regime_manifest_path,
)
from quant_fm.regime.finalize import (
    build_l2_regime_features,
    finalize_l2_regime_features,
)

if TYPE_CHECKING:
    from pathlib import Path


def _state(
    *,
    bid: int = 10_000,
    ask: int = 10_200,
    bid_qty: int = 100,
    ask_qty: int = 100,
) -> BookState:
    total = bid_qty + ask_qty
    return BookState(
        valid=True,
        bid1=bid,
        ask1=ask,
        bid_qty_1=bid_qty,
        ask_qty_1=ask_qty,
        bid_depth_5=bid_qty,
        ask_depth_5=ask_qty,
        bid_depth_10=bid_qty,
        ask_depth_10=ask_qty,
        spread_ticks=2,
        imbalance_1=(bid_qty - ask_qty) / total,
        imbalance_5=(bid_qty - ask_qty) / total,
        imbalance_10=(bid_qty - ask_qty) / total,
        microprice_delta_ticks=0.0,
    )


def _atomic(date_value: str, symbol: str, market: str) -> pl.DataFrame:
    initial = _state()
    first = _state(bid_qty=200)
    second = _state(bid_qty=200, ask_qty=200)
    return build_stock_day_atomic(
        [
            BookStateTransition(initial, first),
            BookStateTransition(first, second),
        ],
        [93_000_000, 112_959_000],
        date=date_value,
        symbol=symbol,
        market=market,
        event_ordering_version="exchange_time_sequence_v2",
    )


def test_atomic_features_are_time_weighted_and_ofi_normalized() -> None:
    atomic = _atomic("2026-01-05", "000001", "SZ").row(0, named=True)

    expected_spread = 10_000 * 200 / 10_100
    expected_depth = (
        7_199 * math.log1p(300)
        + math.log1p(400)
    ) / 7_200
    assert atomic["stock_spread_bps"] == pytest.approx(expected_spread)
    assert atomic["stock_depth_l5_log"] == pytest.approx(expected_depth)
    assert atomic["stock_ofi_l1"] == pytest.approx(0.0)
    assert atomic["n_ofi_events"] == 2
    assert atomic["observed_seconds"] == pytest.approx(7_200.0)
    assert atomic["book_valid_ratio"] == pytest.approx(
        7_200 / CONTINUOUS_SESSION_SECONDS
    )


def test_atomic_ofi_keeps_zero_duration_same_timestamp_events() -> None:
    initial = _state()
    first = _state(bid_qty=200)
    second = _state(bid_qty=300)

    atomic = build_stock_day_atomic(
        [
            BookStateTransition(initial, first),
            BookStateTransition(first, second),
        ],
        [93_000_000, 93_000_000],
        date="2026-01-05",
        symbol="000001",
        market="SZ",
        event_ordering_version="exchange_time_sequence_v2",
    ).row(0, named=True)

    assert atomic["n_ofi_events"] == 2
    assert atomic["stock_ofi_l1"] == pytest.approx(1.0)


def _calendar(n: int = 25) -> list[str]:
    start = date(2026, 1, 1)
    return [(start + timedelta(days=index)).isoformat() for index in range(n)]


def _eod_and_universe(calendar: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    identities = [
        ("000001", "SZ", 0.000),
        ("300001", "SZ", 0.002),
        ("600000", "SH", 0.000),
        ("688001", "SH", -0.002),
    ]
    eod_rows: list[dict[str, object]] = []
    universe_rows: list[dict[str, object]] = []
    for day_index, date_value in enumerate(calendar):
        market_move = 0.001 * ((day_index % 5) - 2)
        for symbol, market, board_move in identities:
            stock_return = market_move + board_move
            eod_rows.append(
                {
                    "date": date_value,
                    "symbol": symbol,
                    "market": market,
                    "pre_close": 10.0,
                    "close": 10.0 * (1.0 + stock_return),
                    "total_notional": float(1_000 + 20 * day_index),
                }
            )
            universe_rows.append(
                {
                    "date": date_value,
                    "symbol": symbol,
                    "asof_date": date_value,
                    "universe_policy": "fixture_all_names_v1",
                }
            )
    return pl.DataFrame(eod_rows), pl.DataFrame(universe_rows)


def _signal_atomic(date_value: str) -> pl.DataFrame:
    return pl.concat(
        [
            _atomic(date_value, "000001", "SZ"),
            _atomic(date_value, "300001", "SZ"),
            _atomic(date_value, "600000", "SH"),
            _atomic(date_value, "688001", "SH"),
        ],
        how="vertical_relaxed",
    )


def test_l2_finalize_uses_continuous_trailing_calendar_and_ignores_future() -> None:
    calendar = _calendar()
    signal_date = calendar[-2]
    atomic = _signal_atomic(signal_date)
    eod, universe = _eod_and_universe(calendar)

    baseline = build_l2_regime_features(
        atomic,
        eod,
        universe,
        calendar,
        min_eod_coverage=1.0,
    )
    changed_future = eod.with_columns(
        pl.when(pl.col("date") == calendar[-1])
        .then(pl.col("close") * 100)
        .otherwise(pl.col("close"))
        .alias("close"),
        pl.when(pl.col("date") == calendar[-1])
        .then(pl.col("total_notional") * 1_000)
        .otherwise(pl.col("total_notional"))
        .alias("total_notional"),
    )
    actual = build_l2_regime_features(
        atomic,
        changed_future,
        universe,
        calendar,
        min_eod_coverage=1.0,
    )

    assert baseline.equals(actual)
    assert baseline.columns == [
        "date",
        "symbol",
        *L2_FEATURE_COLUMNS,
        "asof_date",
    ]
    assert baseline.height == 4
    assert baseline["asof_date"].unique().to_list() == [signal_date]
    assert baseline["market_amount_ratio_20d"][0] > 1.0
    assert baseline.filter(pl.col("symbol") == "300001")[
        "board_relative_strength_20d"
    ].item() > 0


def test_l2_finalize_rejects_atomic_market_identity_drift() -> None:
    calendar = _calendar()
    signal_date = calendar[-1]
    atomic = _signal_atomic(signal_date).with_columns(
        pl.when(pl.col("symbol") == "000001")
        .then(pl.lit("SH"))
        .otherwise(pl.col("market"))
        .alias("market")
    )
    eod, universe = _eod_and_universe(calendar)

    with pytest.raises(ValueError, match="market/board disagrees"):
        build_l2_regime_features(
            atomic,
            eod,
            universe,
            calendar,
            min_eod_coverage=1.0,
        )


def test_archive_and_final_manifest_bind_coverage_and_source_hashes(
    tmp_path: Path,
) -> None:
    calendar = _calendar()
    signal_date = calendar[-1]
    clean_day = tmp_path / "clean" / signal_date
    for frame in _signal_atomic(signal_date).partition_by("market"):
        row = frame.row(0, named=True)
        for item in frame.partition_by("symbol"):
            symbol = str(item["symbol"].item())
            symbol_dir = clean_day / str(row["market"]) / symbol
            symbol_dir.mkdir(parents=True)
            (symbol_dir / "events.parquet").touch()
            item.write_parquet(symbol_dir / "regime_atomic.parquet")
    coverage = write_coverage_receipt(
        workdir=tmp_path,
        clean_dir=clean_day,
        date=signal_date,
        symbols_sz=("000001", "300001"),
        symbols_sh=("600000", "688001"),
    )
    archive = tmp_path / "data" / "regime" / "stock_day_atomic" / f"{signal_date}.parquet"
    archive_atomic_day(
        clean_day,
        archive,
        date=signal_date,
        coverage_receipt=coverage,
    )
    assert archive.is_file()
    assert atomic_manifest_path(archive).is_file()

    eod, universe = _eod_and_universe(calendar)
    eod_path = tmp_path / "eod.parquet"
    universe_path = tmp_path / "universe.parquet"
    calendar_path = tmp_path / "calendar.txt"
    eod.write_parquet(eod_path)
    universe.write_parquet(universe_path)
    calendar_path.write_text("\n".join(calendar) + "\n", encoding="utf-8")
    output = tmp_path / "data" / "regime" / "features" / "regime_features_l2_v1.parquet"

    final_path, manifest = finalize_l2_regime_features(
        atomic_dir=archive.parent,
        eod_path=eod_path,
        universe_path=universe_path,
        calendar_path=calendar_path,
        output_path=output,
        min_eod_coverage=1.0,
        signal_dates=[signal_date],
    )

    assert final_path == output
    assert manifest == regime_manifest_path(output)
    assert pl.read_parquet(output).height == 4
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["artifact_version"] == "regime_features_l2_v1"
    assert payload["features"] == list(L2_FEATURE_COLUMNS)
    assert len(payload["signal_dates_sha256"]) == 64
    assert len(payload["parquet_sha256"]) == 64
