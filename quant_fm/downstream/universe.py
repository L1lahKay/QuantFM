"""
Point-in-time stock-universe validation for Top-K ranking.

The ranker cannot infer whether a plain ``(date, symbol)`` file was actually
known on the signal date.  Strict production paths therefore require explicit
row-level ``asof_date`` provenance and a stable ``universe_policy`` identifier.
This module validates those claims; it intentionally does not manufacture PIT
membership from today's constituents.
"""

from __future__ import annotations

from datetime import date as calendar_date
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable

PIT_UNIVERSE_CONTRACT_VERSION = "pit_universe_v1"
PIT_REQUIRED_COLUMNS = frozenset({"date", "symbol", "asof_date", "universe_policy"})


def _iso_date(value: object, *, context: str) -> str:
    text = str(value)
    try:
        calendar_date.fromisoformat(text)
    except ValueError as exc:
        msg = f"{context} contains a non-ISO date: {text!r}"
        raise ValueError(msg) from exc
    return text


def cross_section_stats(frame: pl.DataFrame) -> dict[str, int | float | str | None]:
    """Return stable per-date width diagnostics for a keyed cross-section."""
    if frame.is_empty():
        return {
            "rows": 0,
            "days": 0,
            "date_min": None,
            "date_max": None,
            "names_min": 0,
            "names_median": 0.0,
            "names_max": 0,
        }
    counts = frame.group_by("date").len()["len"]
    return {
        "rows": frame.height,
        "days": frame["date"].n_unique(),
        "date_min": str(frame["date"].min()),
        "date_max": str(frame["date"].max()),
        "names_min": int(counts.min()),
        "names_median": float(counts.median()),
        "names_max": int(counts.max()),
    }


def validate_pit_universe(
    universe: pl.DataFrame,
    *,
    required_dates: Iterable[str],
    min_names_per_day: int,
    context: str,
    require_pit_metadata: bool = True,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """
    Validate, normalise and restrict a daily universe to required dates.

    In strict mode every membership row must carry ``asof_date <= date`` and a
    single non-empty ``universe_policy`` value.  The latter is compared between
    training and scoring so that two different selection recipes cannot be
    silently used with fixed Top-K cutoffs.
    """
    if min_names_per_day < 1:
        msg = "min_names_per_day must be >= 1"
        raise ValueError(msg)
    base_required = {"date", "symbol"}
    required_columns = (
        set(PIT_REQUIRED_COLUMNS) if require_pit_metadata else base_required
    )
    missing_columns = sorted(required_columns - set(universe.columns))
    if missing_columns:
        suffix = (
            "; a plain (date,symbol) file cannot prove point-in-time membership"
            if require_pit_metadata
            else ""
        )
        msg = f"{context} is missing columns {missing_columns}{suffix}"
        raise ValueError(msg)

    frame = universe.with_columns(
        pl.col("date").cast(pl.Utf8, strict=False),
        pl.col("symbol").cast(pl.Utf8, strict=False).str.zfill(6),
    )
    null_key_rows = frame.filter(
        pl.col("date").is_null() | pl.col("symbol").is_null()
    ).height
    if null_key_rows:
        msg = f"{context} contains {null_key_rows} null (date, symbol) keys"
        raise ValueError(msg)
    if frame.select(pl.struct(["date", "symbol"]).is_duplicated().any()).item():
        msg = f"{context} contains duplicate (date, symbol) keys"
        raise ValueError(msg)

    frame_dates = frame["date"].unique().to_list()
    for value in frame_dates:
        _iso_date(value, context=context)
    wanted = sorted(
        {_iso_date(value, context="required signal dates") for value in required_dates}
    )
    if not wanted:
        msg = f"{context} received no required signal dates"
        raise ValueError(msg)
    missing_dates = sorted(set(wanted) - set(frame_dates))
    if missing_dates:
        msg = f"{context} is missing signal dates: {missing_dates[:5]}"
        raise ValueError(msg)

    policy: str | None = None
    if require_pit_metadata:
        frame = frame.with_columns(
            pl.col("asof_date").cast(pl.Utf8, strict=False),
            pl.col("universe_policy").cast(pl.Utf8, strict=False),
        )
        null_metadata = frame.filter(
            pl.col("asof_date").is_null()
            | pl.col("universe_policy").is_null()
            | (pl.col("universe_policy").str.strip_chars() == "")
        ).height
        if null_metadata:
            msg = f"{context} contains {null_metadata} rows without PIT provenance"
            raise ValueError(msg)
        for value in frame["asof_date"].unique().to_list():
            _iso_date(value, context=f"{context} asof_date")
        future_membership = frame.filter(pl.col("asof_date") > pl.col("date")).height
        if future_membership:
            msg = (
                f"{context} contains {future_membership} memberships whose "
                "asof_date is after the signal date"
            )
            raise ValueError(msg)
        policies = sorted(frame["universe_policy"].unique().to_list())
        if len(policies) != 1:
            msg = f"{context} must contain exactly one universe_policy: {policies[:5]}"
            raise ValueError(msg)
        policy = str(policies[0])

    selected = frame.filter(pl.col("date").is_in(wanted)).sort(["date", "symbol"])
    stats = cross_section_stats(selected)
    observed_min = int(stats["names_min"] or 0)
    if observed_min < min_names_per_day:
        msg = (
            f"{context} is too narrow for Top-K: minimum={observed_min}, "
            f"required={min_names_per_day}"
        )
        raise ValueError(msg)
    contract: dict[str, Any] = {
        "format_version": (
            PIT_UNIVERSE_CONTRACT_VERSION
            if require_pit_metadata
            else "legacy_unverified"
        ),
        "verified": require_pit_metadata,
        "policy": policy,
        "asof_rule": "asof_date_lte_signal_date" if require_pit_metadata else None,
        "required_dates": len(wanted),
        "stats": stats,
    }
    return selected, contract


def validate_universe_alignment(
    training: dict[str, Any],
    scoring: dict[str, Any],
    *,
    median_ratio_tolerance: float = 0.25,
) -> dict[str, float | str]:
    """Require one policy and comparable widths across train and scoring."""
    if not 0.0 <= median_ratio_tolerance < 1.0:
        msg = "median_ratio_tolerance must be in [0, 1)"
        raise ValueError(msg)
    train_policy = training.get("policy")
    score_policy = scoring.get("policy")
    if train_policy is not None and train_policy != score_policy:
        msg = (
            "training and scoring universe policies differ: "
            f"training={train_policy!r}, scoring={score_policy!r}"
        )
        raise ValueError(msg)
    train_median = float(training.get("stats", {}).get("names_median") or 0.0)
    score_median = float(scoring.get("stats", {}).get("names_median") or 0.0)
    if train_median <= 0.0 or score_median <= 0.0:
        msg = "training/scoring universe widths must both be positive"
        raise ValueError(msg)
    ratio = train_median / score_median
    lower = 1.0 - median_ratio_tolerance
    upper = 1.0 + median_ratio_tolerance
    if not lower <= ratio <= upper:
        msg = (
            "training and scoring universe widths are misaligned for fixed Top-K "
            f"cutoffs: train_median={train_median}, score_median={score_median}, "
            f"ratio={ratio:.3f}, allowed=[{lower:.3f},{upper:.3f}]"
        )
        raise ValueError(msg)
    return {
        "policy": str(train_policy)
        if train_policy is not None
        else "legacy_unverified",
        "train_to_score_median_ratio": ratio,
        "median_ratio_tolerance": median_ratio_tolerance,
    }
