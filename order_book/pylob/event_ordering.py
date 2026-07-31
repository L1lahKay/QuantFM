"""Versioned ordering contract for replayable exchange event streams."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

LEGACY_LOCAL_TIME_V1 = "local_time_v1"
CAUSAL_EXCHANGE_TIME_V2 = "exchange_time_sequence_v2"
DEFAULT_EVENT_ORDERING_VERSION = CAUSAL_EXCHANGE_TIME_V2
SUPPORTED_EVENT_ORDERING_VERSIONS = frozenset(
    {LEGACY_LOCAL_TIME_V1, CAUSAL_EXCHANGE_TIME_V2}
)


def validate_event_ordering_version(version: str) -> str:
    """Return a supported event-ordering version or raise explicitly."""
    normalized = str(version)
    if normalized not in SUPPORTED_EVENT_ORDERING_VERSIONS:
        msg = (
            f"unsupported event_ordering_version={normalized!r}; expected one of "
            f"{sorted(SUPPORTED_EVENT_ORDERING_VERSIONS)}"
        )
        raise ValueError(msg)
    return normalized


def order_market_events(
    events: pl.DataFrame,
    *,
    version: str = DEFAULT_EVENT_ORDERING_VERSION,
    time_column: str = "int_time",
    sequence_column: str = "serial",
) -> pl.DataFrame:
    """
    Stably order merged order/trade rows according to a frozen contract.

    The causal contract uses exchange ``int_time`` followed by the exchange
    sequence number. Rows with identical keys retain their input order. The
    legacy contract remains available solely for reproducing existing artifacts.
    """
    version = validate_event_ordering_version(version)
    if version == LEGACY_LOCAL_TIME_V1:
        if "local_time" not in events.columns:
            msg = "legacy local_time ordering requires column 'local_time'"
            raise ValueError(msg)
        return events.sort("local_time", maintain_order=True)

    missing = [
        column
        for column in (time_column, sequence_column)
        if column not in events.columns
    ]
    if missing:
        msg = f"missing exchange ordering columns: {missing}"
        raise ValueError(msg)
    return events.sort([time_column, sequence_column], maintain_order=True)


def exchange_ordering_columns(
    events: pl.DataFrame,
    *,
    time_column: str = "int_time",
    sequence_columns: Sequence[str] = (
        "exchange_seqnum",
        "source_seqnum",
        "serial",
        "event_idx",
    ),
) -> tuple[str, ...]:
    """Resolve the strongest available exchange-order key for an event frame."""
    if time_column not in events.columns:
        msg = f"missing exchange ordering column: {time_column!r}"
        raise ValueError(msg)
    sequence = next(
        (column for column in sequence_columns if column in events.columns), None
    )
    if sequence is None:
        msg = f"no exchange sequence column found among {tuple(sequence_columns)!r}"
        raise ValueError(msg)
    columns = [time_column, sequence]
    if sequence != "event_idx" and "event_idx" in events.columns:
        columns.append("event_idx")
    return tuple(columns)


def assert_exchange_ordered(
    events: pl.DataFrame,
    *,
    time_column: str = "int_time",
    sequence_columns: Sequence[str] = (
        "exchange_seqnum",
        "source_seqnum",
        "serial",
        "event_idx",
    ),
) -> tuple[str, ...]:
    """Reject a frame that is not monotonic under its exchange ordering key."""
    columns = exchange_ordering_columns(
        events,
        time_column=time_column,
        sequence_columns=sequence_columns,
    )
    if events.height < 2:
        return columns
    keys = events.select(
        pl.col(column).cast(pl.Int64, strict=True).alias(column) for column in columns
    )
    if any(keys[column].null_count() for column in columns):
        msg = f"exchange ordering columns contain nulls: {columns}"
        raise ValueError(msg)
    arrays = [keys[column].to_numpy() for column in columns]
    equal_prefix = np.ones(events.height - 1, dtype=bool)
    violation = np.zeros(events.height - 1, dtype=bool)
    for values in arrays:
        previous = values[:-1]
        current = values[1:]
        violation |= equal_prefix & (current < previous)
        equal_prefix &= current == previous
    if violation.any():
        row = int(np.flatnonzero(violation)[0] + 1)
        before = tuple(int(values[row - 1]) for values in arrays)
        after = tuple(int(values[row]) for values in arrays)
        msg = (
            f"event stream is not exchange ordered by {columns}: "
            f"row {row - 1}={before} -> row {row}={after}"
        )
        raise ValueError(msg)
    return columns


def validate_order_if_present(
    events: pl.DataFrame,
    *,
    version: str,
) -> tuple[str, ...] | None:
    """Validate causal ordering when index columns are present in a generic frame."""
    version = validate_event_ordering_version(version)
    if version == LEGACY_LOCAL_TIME_V1 or "int_time" not in events.columns:
        return None
    return assert_exchange_ordered(events)


__all__ = [
    "CAUSAL_EXCHANGE_TIME_V2",
    "DEFAULT_EVENT_ORDERING_VERSION",
    "LEGACY_LOCAL_TIME_V1",
    "SUPPORTED_EVENT_ORDERING_VERSIONS",
    "assert_exchange_ordered",
    "exchange_ordering_columns",
    "order_market_events",
    "validate_event_ordering_version",
    "validate_order_if_present",
]
