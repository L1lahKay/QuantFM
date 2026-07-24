from __future__ import annotations

import polars as pl
import pytest

from quant_fm.downstream.build_panel_from_minio import build_execution_panel
from quant_fm.downstream.return_spec import get_return_spec


def _daily() -> pl.DataFrame:
    rows = []
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    for symbol, offset in (("000001", 0.0), ("000002", 10.0)):
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "market": "SZ",
                    "close": 10.0 + offset + index,
                    "vwap": 10.0 + offset + index,
                    "is_st": False,
                    "is_new": False,
                    "is_halt": False,
                    "limit_locked": symbol == "000002" and index == 1,
                }
            )
    return pl.DataFrame(rows)


def test_execution_panel_uses_explicit_entry_and_exit_dates() -> None:
    calendar = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    panel = build_execution_panel(
        _daily(),
        signal_dates=calendar[:2],
        trading_calendar=calendar,
        spec=get_return_spec("vwap_t1_vwap_t2"),
    )
    first = panel.filter(
        (pl.col("date") == "2026-01-05") & (pl.col("symbol") == "000001")
    ).row(0, named=True)
    assert first["entry_date"] == "2026-01-06"
    assert first["exit_date"] == "2026-01-07"
    assert first["fwd_ret"] == pytest.approx(12 / 11 - 1)

    locked = panel.filter(
        (pl.col("date") == "2026-01-05") & (pl.col("symbol") == "000002")
    ).row(0, named=True)
    assert not locked["entry_fillable"]
    # 无法成交不等于没有市场收益；执行器负责拒单，IC 仍可观察价格变化。
    assert locked["fwd_ret"] is not None


def test_execution_panel_rejects_incomplete_future_calendar() -> None:
    calendar = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    with pytest.raises(ValueError, match="does not cover"):
        build_execution_panel(
            _daily(),
            signal_dates=["2026-01-07"],
            trading_calendar=calendar,
            spec=get_return_spec("vwap_t1_vwap_t2"),
        )
