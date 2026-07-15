"""Build model-ready event streams from replayable market data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def build_event_stream(order_data: pd.DataFrame, *, market: str) -> pd.DataFrame:
    """Convert merged PyLOB market rows to a standard event stream."""
    df = order_data.copy()
    df["market"] = market.upper()
    df["event_idx"] = np.arange(len(df), dtype=np.int64)
    df["delta_t"] = df["int_time"].diff().fillna(0).clip(lower=0).astype("int64")
    df["session_phase"] = df["int_time"].map(_session_phase)
    df["event_type"] = df.apply(_event_type, axis=1)
    df["side"] = df["bsflag"].map(_side).fillna("UNKNOWN")
    df["price"] = np.where(df["type"] == "T", df["trade_price"], df["order_price"])
    df["volume"] = np.where(
        df["type"] == "T",
        df["trade_volume"],
        df["order_volume"],
    )
    df["log_volume"] = np.log1p(np.maximum(df["volume"], 0) / 100.0)
    return df[
        [
            "symbol",
            "market",
            "event_idx",
            "int_time",
            "local_time",
            "serial",
            "delta_t",
            "session_phase",
            "event_type",
            "side",
            "price",
            "volume",
            "log_volume",
            "orderorino",
            "buy_id",
            "sell_id",
        ]
    ]


def _session_phase(int_time: int) -> str:
    if int_time < 93000000:
        return "OPEN_AUCTION"
    if int_time < 113000000:
        return "CONTINUOUS_AM"
    if int_time < 130000000:
        return "MIDDAY_BREAK"
    if int_time < 145700000:
        return "CONTINUOUS_PM"
    return "CLOSE_AUCTION"


def _event_type(row: pd.Series) -> str:
    if row["type"] == "T":
        return "CANCEL" if row["trade_type"] == "C" else "TRADE"
    if row["order_type"] == "D":
        return "CANCEL"
    return "ADD"


def _side(value: object) -> str:
    if value in {"B", "b'B'"}:
        return "BUY"
    if value in {"S", "b'S'"}:
        return "SELL"
    return "UNKNOWN"
