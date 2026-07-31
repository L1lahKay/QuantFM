"""Build model-ready event streams from replayable market data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pylob.event_ordering import (
    LEGACY_LOCAL_TIME_V1,
    validate_event_ordering_version,
)

if TYPE_CHECKING:
    import pandas as pd

EVENT_STREAM_CONTRACT_VERSION = "2.0"


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


def event_stream_contract_path(events_path: Path) -> Path:
    """Return the non-invasive sidecar path for one clean event parquet."""
    path = Path(events_path)
    return path.with_suffix(f"{path.suffix}.contract.json")


def write_event_stream_contract(
    events_path: Path,
    *,
    event_ordering_version: str,
) -> Path:
    """Atomically write the ordering provenance for a clean event parquet."""
    version = validate_event_ordering_version(event_ordering_version)
    destination = event_stream_contract_path(events_path)
    payload = {
        "artifact_version": EVENT_STREAM_CONTRACT_VERSION,
        "event_ordering_version": version,
        "exchange_order_key": ["int_time", "serial"],
        "identical_key_tie_break": "stable_input_order",
    }
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def read_event_stream_contract(events_path: Path) -> dict[str, object]:
    """
    Read ordering provenance, inferring legacy semantics for old sidecar-less data.

    Existing ``events.parquet`` files predate this contract and were produced by
    ``local_time`` sorting. They remain readable and are identified explicitly as
    inferred legacy artifacts instead of being mislabeled as causal.
    """
    path = event_stream_contract_path(events_path)
    if not path.is_file():
        return {
            "artifact_version": "1.0",
            "event_ordering_version": LEGACY_LOCAL_TIME_V1,
            "inferred_legacy": True,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"invalid event stream contract object: {path}"
        raise TypeError(msg)
    version = validate_event_ordering_version(
        str(payload.get("event_ordering_version", ""))
    )
    return {**payload, "event_ordering_version": version, "inferred_legacy": False}


def event_stream_contract_matches(events_path: Path, *, version: str) -> bool:
    """Return whether an existing clean event artifact matches the requested order."""
    expected = validate_event_ordering_version(version)
    return read_event_stream_contract(events_path)["event_ordering_version"] == expected


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
