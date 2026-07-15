"""Normalize raw Level-2 order and trade frames to PyLOB columns."""

from __future__ import annotations

import polars as pl

REQUIRED_COLUMNS = {
    "symbol",
    "int_time",
    "local_time",
    "exchange_time",
    "serial",
    "trade_type",
    "order_type",
    "orderorino",
    "sell_id",
    "buy_id",
    "bsflag",
    "trade_price",
    "order_price",
    "trade_volume",
    "order_volume",
}


def standardize_order_frame(
    df: pl.DataFrame,
    *,
    field_mapping: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Return an order dataframe with the columns expected by PyLOB."""
    return _standardize_common(df, field_mapping=field_mapping, is_trade=False)


def standardize_trade_frame(
    df: pl.DataFrame,
    *,
    field_mapping: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Return a trade dataframe with the columns expected by PyLOB."""
    return _standardize_common(df, field_mapping=field_mapping, is_trade=True)


def _standardize_common(
    df: pl.DataFrame,
    *,
    field_mapping: dict[str, str] | None,
    is_trade: bool,
) -> pl.DataFrame:
    if field_mapping:
        rename_map = {
            raw: std for raw, std in field_mapping.items() if raw in df.columns
        }
        df = df.rename(rename_map)

    df = _ensure_columns(df, is_trade=is_trade)

    return df.with_columns(
        pl.col("symbol").cast(pl.String).str.zfill(6).str.slice(0, 6),
        pl.col("int_time").cast(pl.Int64),
        pl.col("local_time").cast(pl.Int64),
        pl.col("exchange_time").cast(pl.Int64),
        pl.col("serial").cast(pl.Int64),
        pl.col("trade_type").cast(pl.String).fill_null("0"),
        pl.col("order_type").cast(pl.String).fill_null("0"),
        pl.col("orderorino").cast(pl.Int64).fill_null(0),
        pl.col("sell_id").cast(pl.Int64).fill_null(0),
        pl.col("buy_id").cast(pl.Int64).fill_null(0),
        pl.col("bsflag").cast(pl.String).str.to_uppercase().fill_null(""),
        pl.col("trade_price").cast(pl.Int64).fill_null(0),
        pl.col("order_price").cast(pl.Int64).fill_null(0),
        pl.col("trade_volume").cast(pl.Int64).fill_null(0),
        pl.col("order_volume").cast(pl.Int64).fill_null(0),
    )


def _ensure_columns(df: pl.DataFrame, *, is_trade: bool) -> pl.DataFrame:
    additions = []
    for col in REQUIRED_COLUMNS - set(df.columns):
        default = _default_value(col, is_trade=is_trade)
        additions.append(pl.lit(default).alias(col))
    return df.with_columns(additions) if additions else df


def _default_value(col: str, *, is_trade: bool) -> int | str:
    if col in {"symbol", "bsflag"}:
        return ""
    if col == "trade_type":
        return "0"
    if col == "order_type":
        return "0" if is_trade else "O"
    return 0
