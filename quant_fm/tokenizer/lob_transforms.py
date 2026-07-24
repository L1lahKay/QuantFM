"""Causal feature transforms for compact limit-order-book snapshots."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pylob.book_state import BookState, BookStateTransition

POST_BOOK_FEATURES: tuple[str, ...] = (
    "book_valid_post",
    "spread_ticks_post",
    "microprice_delta_ticks_post",
    "imbalance_l1_post",
    "imbalance_l5_post",
    "imbalance_l10_post",
    "log_bid_depth_l5_post",
    "log_ask_depth_l5_post",
)

PRE_EVENT_FEATURES: tuple[str, ...] = ("event_price_distance_ticks_pre",)

_FEATURE_DTYPES: dict[str, pl.DataType] = {
    "book_valid_post": pl.Boolean,
    "spread_ticks_post": pl.Int64,
    "microprice_delta_ticks_post": pl.Float64,
    "imbalance_l1_post": pl.Float64,
    "imbalance_l5_post": pl.Float64,
    "imbalance_l10_post": pl.Float64,
    "log_bid_depth_l5_post": pl.Float64,
    "log_ask_depth_l5_post": pl.Float64,
    "event_price_distance_ticks_pre": pl.Float64,
}


def book_state_to_post_features(
    state: BookState,
) -> dict[str, bool | float | int | None]:
    """Map one post-event snapshot to explicitly suffixed model fields."""
    return {
        "book_valid_post": state.valid,
        "spread_ticks_post": state.spread_ticks,
        "microprice_delta_ticks_post": state.microprice_delta_ticks,
        "imbalance_l1_post": state.imbalance_1,
        "imbalance_l5_post": state.imbalance_5,
        "imbalance_l10_post": state.imbalance_10,
        "log_bid_depth_l5_post": math.log1p(state.bid_depth_5),
        "log_ask_depth_l5_post": math.log1p(state.ask_depth_5),
    }


def event_price_distance_ticks_pre(
    event_price: int | float | None,
    state_pre: BookState,
    *,
    tick_size: int = 100,
) -> float | None:
    """
    Return signed event-price distance from the pre-event midpoint.

    Both quotes must exist. A positive result is above the previous midpoint and a
    negative result is below it. ``event_price`` must use the same integer price
    scale as :class:`~pylob.book_state.BookState`.
    """
    if tick_size <= 0:
        msg = "tick_size must be positive"
        raise ValueError(msg)
    if event_price is None or not state_pre.valid:
        return None
    price = float(event_price)
    if not math.isfinite(price) or price <= 0:
        return None
    if state_pre.bid1 is None or state_pre.ask1 is None:
        return None
    midpoint = (state_pre.bid1 + state_pre.ask1) / 2.0
    return (price - midpoint) / tick_size


def transitions_to_feature_frame(
    transitions: Sequence[BookStateTransition],
    *,
    event_prices: Sequence[int | float | None] | None = None,
    tick_size: int = 100,
) -> pl.DataFrame:
    """
    Convert aligned transitions to a schema-ready feature frame.

    ``post_event_state`` supplies every ``*_post`` field. If prices are supplied,
    the distance feature is computed exclusively from ``pre_event_state``.
    """
    n_rows = len(transitions)
    if event_prices is not None and len(event_prices) != n_rows:
        msg = "event_prices and transitions must have identical lengths"
        raise ValueError(msg)

    rows: list[dict[str, bool | float | int | None]] = []
    for index, transition in enumerate(transitions):
        row = book_state_to_post_features(transition.post_event_state)
        if event_prices is not None:
            row["event_price_distance_ticks_pre"] = event_price_distance_ticks_pre(
                event_prices[index],
                transition.pre_event_state,
                tick_size=tick_size,
            )
        rows.append(row)

    columns = list(POST_BOOK_FEATURES)
    if event_prices is not None:
        columns.extend(PRE_EVENT_FEATURES)
    if not rows:
        return _empty_feature_frame(include_pre=event_prices is not None)
    return pl.DataFrame(rows).select(
        pl.col(column).cast(_FEATURE_DTYPES[column], strict=True).alias(column)
        for column in columns
    )


def sort_by_exchange_sequence(
    events: pl.DataFrame,
    *,
    time_column: str = "int_time",
    sequence_column: str | None = None,
) -> pl.DataFrame:
    """
    Stably order events by exchange time and exchange sequence number.

    ``local_time`` is intentionally not considered because it can contain feed
    reception latency. Duplicate ``(time, sequence)`` keys retain input order.
    """
    if sequence_column is None:
        sequence_column = _find_sequence_column(events)
    missing = [
        column
        for column in (time_column, sequence_column)
        if column not in events.columns
    ]
    if missing:
        msg = f"missing exchange ordering columns: {missing}"
        raise ValueError(msg)
    return events.sort([time_column, sequence_column], maintain_order=True)


def _find_sequence_column(events: pl.DataFrame) -> str:
    for column in ("exchange_seqnum", "source_seqnum", "serial"):
        if column in events.columns:
            return column
    msg = "no exchange sequence column found"
    raise ValueError(msg)


def _empty_feature_frame(*, include_pre: bool) -> pl.DataFrame:
    schema = {column: _FEATURE_DTYPES[column] for column in POST_BOOK_FEATURES}
    if include_pre:
        schema["event_price_distance_ticks_pre"] = pl.Float64
    return pl.DataFrame(schema=schema)


__all__ = [
    "POST_BOOK_FEATURES",
    "PRE_EVENT_FEATURES",
    "book_state_to_post_features",
    "event_price_distance_ticks_pre",
    "sort_by_exchange_sequence",
    "transitions_to_feature_frame",
]
