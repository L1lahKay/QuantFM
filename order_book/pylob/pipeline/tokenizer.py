"""Field-level tokenizer for normalized order-flow events."""

from __future__ import annotations

import pandas as pd


def build_field_tokens(
    events: pd.DataFrame,
    *,
    n_price_bins: int = 32,
    n_volume_bins: int = 32,
    n_time_bins: int = 32,
) -> pd.DataFrame:
    """Discretize event fields into model-ready token columns."""
    tokens = events.copy()
    tokens["event_type_token"] = _category_codes(tokens["event_type"])
    tokens["side_token"] = _category_codes(tokens["side"])
    tokens["session_phase_token"] = _category_codes(tokens["session_phase"])
    tokens["price_bin"] = _safe_qcut(tokens["price"], n_price_bins)
    tokens["volume_bin"] = _safe_qcut(tokens["log_volume"], n_volume_bins)
    tokens["delta_t_bin"] = _safe_qcut(tokens["delta_t"], n_time_bins)
    return tokens


def _category_codes(series: pd.Series) -> pd.Series:
    return series.astype("category").cat.codes.astype("int16")


def _safe_qcut(series: pd.Series, bins: int) -> pd.Series:
    clean = series.replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    if clean.nunique(dropna=True) <= 1:
        return pd.Series(0, index=series.index, dtype="int16")
    clipped = clean.clip(clean.quantile(0.01), clean.quantile(0.99))
    return (
        pd.qcut(
            clipped, q=min(bins, clipped.nunique()), labels=False, duplicates="drop"
        )
        .fillna(0)
        .astype("int16")
    )
