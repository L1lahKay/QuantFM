"""
Fail-fast preflight for strict Top-K ranker retraining inputs.

This command validates external calendars and PIT universes without creating or
backfilling either dataset.  It is intended to run before expensive quote-panel
construction and model training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from quant_fm.downstream.representation import validate_strict_topk_representation
from quant_fm.downstream.return_spec import get_return_spec, read_trading_calendar
from quant_fm.downstream.universe import (
    cross_section_stats,
    validate_pit_universe,
    validate_universe_alignment,
)
from quant_fm.embedding.contract import (
    assert_embedding_contract_compatible,
    load_embedding_contract,
    validate_embedding_columns,
)

if TYPE_CHECKING:
    from quant_fm.embedding.contract import EmbeddingContract


def _embedding_keys(path: Path, *, context: str) -> pl.DataFrame:
    schema = set(pl.read_parquet_schema(path).names())
    missing = sorted({"date", "symbol"} - schema)
    if missing:
        msg = f"{context} embeddings are missing columns: {missing}"
        raise ValueError(msg)
    keys = pl.read_parquet(path, columns=["date", "symbol"]).with_columns(
        pl.col("date").cast(pl.Utf8, strict=False),
        pl.col("symbol").cast(pl.Utf8, strict=False).str.zfill(6),
    )
    if keys.is_empty():
        msg = f"{context} embeddings are empty"
        raise ValueError(msg)
    if keys.filter(pl.col("date").is_null() | pl.col("symbol").is_null()).height:
        msg = f"{context} embeddings contain null keys"
        raise ValueError(msg)
    if keys.select(pl.struct(["date", "symbol"]).is_duplicated().any()).item():
        msg = f"{context} embeddings contain duplicate (date, symbol) keys"
        raise ValueError(msg)
    return keys


def _validate_horizon(
    signal_dates: list[str],
    calendar: list[str],
    *,
    return_spec: str,
    context: str,
) -> dict[str, Any]:
    spec = get_return_spec(return_spec)
    if spec.entry_day_lag < 1:
        msg = f"{context} return spec {return_spec!r} enters before score(T) is usable"
        raise ValueError(msg)
    positions = {value: index for index, value in enumerate(calendar)}
    missing = sorted(set(signal_dates) - set(positions))
    if missing:
        msg = f"{context} signal dates are absent from its trading calendar: {missing[:5]}"
        raise ValueError(msg)
    incomplete = [
        value
        for value in signal_dates
        if positions[value] + spec.entry_day_lag >= len(calendar)
        or positions[value] + spec.exit_day_lag >= len(calendar)
    ]
    if incomplete:
        msg = (
            f"{context} calendar does not cover exact T+1/T+2 mappings for "
            f"signal dates: {incomplete[:5]}"
        )
        raise ValueError(msg)
    mappings = [
        {
            "date": value,
            "entry_date": calendar[positions[value] + spec.entry_day_lag],
            "exit_date": calendar[positions[value] + spec.exit_day_lag],
        }
        for value in signal_dates
    ]
    return {
        "return_spec": spec.name,
        "signal_dates": len(signal_dates),
        "signal_date_min": min(signal_dates),
        "signal_date_max": max(signal_dates),
        "first_mapping": mappings[0],
        "last_mapping": mappings[-1],
    }


def _validate_period(
    *,
    embeddings_path: Path,
    calendar_path: Path,
    universe_path: Path,
    return_spec: str,
    min_names_per_day: int,
    context: str,
) -> tuple[dict[str, Any], dict[str, Any], EmbeddingContract]:
    embedding_contract = load_embedding_contract(
        embeddings_path,
        required=True,
        require_vocab=True,
    )
    if embedding_contract is None:  # pragma: no cover - required=True invariant
        msg = f"{context} embedding contract is missing"
        raise RuntimeError(msg)
    validate_embedding_columns(
        list(pl.read_parquet_schema(embeddings_path).names()),
        embedding_contract,
        context=f"{context} embeddings",
    )
    representation_gate = validate_strict_topk_representation(
        embedding_contract,
        context=f"{context} embeddings",
    )
    keys = _embedding_keys(embeddings_path, context=context)
    dates = sorted(str(value) for value in keys["date"].unique())
    calendar = read_trading_calendar(calendar_path)
    horizon = _validate_horizon(
        dates,
        calendar,
        return_spec=return_spec,
        context=context,
    )
    universe, universe_contract = validate_pit_universe(
        pl.read_parquet(universe_path),
        required_dates=dates,
        min_names_per_day=min_names_per_day,
        context=f"{context} PIT universe",
    )
    retained = keys.join(
        universe.select(["date", "symbol"]),
        on=["date", "symbol"],
        how="inner",
    )
    retained_stats = cross_section_stats(retained)
    if int(retained_stats["days"] or 0) != len(dates):
        msg = f"{context} loses complete signal dates after PIT/embedding join"
        raise ValueError(msg)
    if int(retained_stats["names_min"] or 0) < min_names_per_day:
        msg = (
            f"{context} has too few embedded PIT members after join: "
            f"minimum={retained_stats['names_min']}, required={min_names_per_day}"
        )
        raise ValueError(msg)
    effective_contract = {**universe_contract, "stats": retained_stats}
    return (
        {
            "embeddings": str(embeddings_path),
            "calendar": str(calendar_path),
            "universe": str(universe_path),
            "horizon": horizon,
            "embedding_representation": {
                "fingerprint": embedding_contract.fingerprint(),
                "strict_topk_gate": representation_gate,
            },
            "universe_contract": universe_contract,
            "retained_embedding_universe": retained_stats,
        },
        effective_contract,
        embedding_contract,
    )


def preflight_topk_inputs(
    *,
    train_embeddings: Path,
    oos_embeddings: Path,
    train_calendar: Path,
    oos_calendar: Path,
    train_universe: Path,
    oos_universe: Path,
    return_spec: str = "vwap_t1_vwap_t2",
    min_names_per_day: int = 350,
) -> dict[str, Any]:
    """Validate all external inputs needed by the strict retraining entry."""
    training, train_contract, train_embedding_contract = _validate_period(
        embeddings_path=train_embeddings,
        calendar_path=train_calendar,
        universe_path=train_universe,
        return_spec=return_spec,
        min_names_per_day=min_names_per_day,
        context="training",
    )
    scoring, score_contract, score_embedding_contract = _validate_period(
        embeddings_path=oos_embeddings,
        calendar_path=oos_calendar,
        universe_path=oos_universe,
        return_spec=return_spec,
        min_names_per_day=min_names_per_day,
        context="OOS scoring",
    )
    train_date_max = str(training["horizon"]["signal_date_max"])
    score_date_min = str(scoring["horizon"]["signal_date_min"])
    if train_date_max >= score_date_min:
        msg = (
            "training and OOS signal periods overlap: "
            f"training_end={train_date_max}, oos_start={score_date_min}"
        )
        raise ValueError(msg)
    assert_embedding_contract_compatible(
        train_embedding_contract,
        score_embedding_contract,
        context="strict Top-K training vs OOS preflight",
    )
    alignment = validate_universe_alignment(train_contract, score_contract)
    return {
        "status": "ready",
        "return_spec": return_spec,
        "min_names_per_day": min_names_per_day,
        "training": training,
        "oos_scoring": scoring,
        "universe_alignment": alignment,
    }


def main() -> None:
    """Run strict input preflight and optionally persist its JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-embeddings", type=Path, required=True)
    parser.add_argument("--oos-embeddings", type=Path, required=True)
    parser.add_argument("--train-calendar", type=Path, required=True)
    parser.add_argument("--oos-calendar", type=Path, required=True)
    parser.add_argument("--train-universe", type=Path, required=True)
    parser.add_argument("--oos-universe", type=Path, required=True)
    parser.add_argument("--return-spec", default="vwap_t1_vwap_t2")
    parser.add_argument("--min-names-per-day", type=int, default=350)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = preflight_topk_inputs(
        train_embeddings=args.train_embeddings,
        oos_embeddings=args.oos_embeddings,
        train_calendar=args.train_calendar,
        oos_calendar=args.oos_calendar,
        train_universe=args.train_universe,
        oos_universe=args.oos_universe,
        return_spec=args.return_spec,
        min_names_per_day=args.min_names_per_day,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.out)
    print(rendered)


if __name__ == "__main__":
    main()
