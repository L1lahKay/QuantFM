"""``date, symbol, score`` 对外交付契约。"""

from __future__ import annotations

import math

import polars as pl

SCORE_COLUMNS = ["date", "symbol", "score"]


def validate_scores(scores: pl.DataFrame) -> pl.DataFrame:
    """校验、规范化并稳定排序 score 信号表。"""
    if scores.columns != SCORE_COLUMNS:
        msg = f"score columns must be exactly {SCORE_COLUMNS}, got {scores.columns}"
        raise ValueError(msg)
    frame = scores.with_columns(
        pl.col("date").cast(pl.Utf8),
        pl.col("symbol").cast(pl.Utf8).str.zfill(6),
        pl.col("score").cast(pl.Float64),
    )
    if frame.is_empty():
        msg = "score table must not be empty"
        raise ValueError(msg)
    if any(frame[name].null_count() for name in SCORE_COLUMNS):
        msg = "score table contains null values"
        raise ValueError(msg)
    dates_ok = frame.select(
        pl.col("date").str.contains(r"^\d{4}-\d{2}-\d{2}$").all()
        & pl.col("date").str.to_date("%Y-%m-%d", strict=False).is_not_null().all()
    ).item()
    if not dates_ok:
        msg = "date must be a valid YYYY-MM-DD string"
        raise ValueError(msg)
    symbols_ok = frame.select(pl.col("symbol").str.contains(r"^\d{6}$").all()).item()
    if not symbols_ok:
        msg = "symbol must be a six-digit string without exchange suffix"
        raise ValueError(msg)
    if not all(math.isfinite(value) for value in frame["score"].to_list()):
        msg = "score must contain only finite values"
        raise ValueError(msg)
    if frame.select(["date", "symbol"]).is_duplicated().any():
        msg = "duplicate (date, symbol) keys in score table"
        raise ValueError(msg)
    return frame.sort(["date", "symbol"])
