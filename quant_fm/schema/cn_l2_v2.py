"""Versioned A-share Level-2 schema with compact causal book state."""

from __future__ import annotations

import polars as pl
import pyarrow as pa

from quant_fm.schema.cn_l2_v1 import (
    CANONICAL_COLUMNS as V1_CANONICAL_COLUMNS,
)
from quant_fm.schema.cn_l2_v1 import canonical_arrow_schema as v1_arrow_schema
from quant_fm.schema.cn_l2_v1 import events_to_canonical as events_to_canonical_v1
from quant_fm.tokenizer.lob_transforms import (
    POST_BOOK_FEATURES,
    PRE_EVENT_FEATURES,
)

SCHEMA_VERSION = "cn_l2_v2"
BOOK_STATE_TIMING = "post_event"

V2_FEATURE_COLUMNS: tuple[str, ...] = (
    "exchange_seqnum",
    "time_of_day_ms",
    *POST_BOOK_FEATURES,
    *PRE_EVENT_FEATURES,
)

CANONICAL_COLUMNS: tuple[str, ...] = (*V1_CANONICAL_COLUMNS, *V2_FEATURE_COLUMNS)

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


def events_to_canonical(
    events: pl.DataFrame,
    *,
    date: str,
    market: str,
    book_features: pl.DataFrame,
) -> pl.DataFrame:
    """
    Convert exchange events plus aligned causal book features to ``cn_l2_v2``.

    Unlike v1, book features are required. This prevents a dataset labelled v2 from
    silently containing placeholder book state. The eight ``*_post`` columns must
    have been captured immediately after their corresponding event; the optional
    price-distance input must have been computed from that event's pre-state.

    Parameters
    ----------
    events
        PyLOB standard event stream, in exchange event order.
    date
        Trading date in ``YYYY-MM-DD`` form.
    market
        ``"SH"`` or ``"SZ"``.
    book_features
        Row-aligned output from
        :func:`quant_fm.tokenizer.lob_transforms.transitions_to_feature_frame`.
    """
    if len(book_features) != len(events):
        msg = "book_features and events must have identical row counts"
        raise ValueError(msg)
    missing = [column for column in POST_BOOK_FEATURES if column not in book_features]
    if missing:
        msg = f"missing required post-event book features: {missing}"
        raise ValueError(msg)
    if book_features["book_valid_post"].null_count() > 0:
        msg = "book_valid_post cannot contain null values"
        raise ValueError(msg)

    canonical = events_to_canonical_v1(events, date=date, market=market).with_columns(
        pl.lit(SCHEMA_VERSION).alias("schema_version"),
        _time_of_day_ms_expr("int_time").alias("time_of_day_ms"),
    )

    if "exchange_seqnum" in events.columns:
        exchange_seqnum = events["exchange_seqnum"].cast(pl.Int64, strict=True)
    else:
        exchange_seqnum = canonical["source_seqnum"].cast(pl.Int64, strict=True)
    canonical = canonical.with_columns(exchange_seqnum.alias("exchange_seqnum"))

    features = _normalize_feature_frame(book_features)
    return canonical.hstack(features).select(CANONICAL_COLUMNS)


def canonical_arrow_schema() -> pa.Schema:
    """Return the stable Arrow schema for :data:`CANONICAL_COLUMNS`."""
    fields = list(v1_arrow_schema())
    fields.extend(
        [
            pa.field("exchange_seqnum", pa.int64(), nullable=False),
            pa.field("time_of_day_ms", pa.int64(), nullable=False),
            pa.field("book_valid_post", pa.bool_(), nullable=False),
            pa.field("spread_ticks_post", pa.int64()),
            pa.field("microprice_delta_ticks_post", pa.float64()),
            pa.field("imbalance_l1_post", pa.float64()),
            pa.field("imbalance_l5_post", pa.float64()),
            pa.field("imbalance_l10_post", pa.float64()),
            pa.field("log_bid_depth_l5_post", pa.float64()),
            pa.field("log_ask_depth_l5_post", pa.float64()),
            pa.field("event_price_distance_ticks_pre", pa.float64()),
        ]
    )
    return pa.schema(fields)


def _normalize_feature_frame(book_features: pl.DataFrame) -> pl.DataFrame:
    normalized = book_features
    if "event_price_distance_ticks_pre" not in normalized.columns:
        normalized = normalized.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("event_price_distance_ticks_pre")
        )
    return normalized.select(
        pl.col(column).cast(dtype, strict=True).alias(column)
        for column, dtype in _FEATURE_DTYPES.items()
    )


def _time_of_day_ms_expr(column: str) -> pl.Expr:
    packed = pl.col(column).cast(pl.Int64)
    milliseconds = packed % 1_000
    seconds = (packed // 1_000) % 100
    minutes = (packed // 100_000) % 100
    hours = (packed // 10_000_000) % 100
    return (hours * 3_600_000 + minutes * 60_000 + seconds * 1_000 + milliseconds).cast(
        pl.Int64
    )


__all__ = [
    "BOOK_STATE_TIMING",
    "CANONICAL_COLUMNS",
    "SCHEMA_VERSION",
    "V2_FEATURE_COLUMNS",
    "canonical_arrow_schema",
    "events_to_canonical",
]
