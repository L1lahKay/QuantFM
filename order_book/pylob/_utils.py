"""Shared utilities for pylob internals."""

import polars as pl


def normalize_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    """统一列类型：UInt64->Int64，Binary->Utf8，避免 concat 时 SchemaError."""
    cast_map = {}
    for col in df.columns:
        if df[col].dtype == pl.UInt64:
            cast_map[col] = pl.Int64
        elif df[col].dtype == pl.Binary:
            cast_map[col] = pl.Utf8
    return df.cast(cast_map) if cast_map else df
