"""
Build reproducible Backbone-MoE benchmark and regime-adaptation date plans.

The strict benchmark keeps a chronological 300/60/100 split and leaves every
remaining date locked.  The adaptation track ends at a requested cutoff,
reserves an internal purged validation block for model selection, then permits
a final refit on all 300 adaptation dates.  Dates after the adaptation cutoff
remain shadow OOS and are never silently promoted to a full OOS claim.

The inventory can be supplied as a text file or discovered read-only from the
known MinIO HDS object layout.  MinIO bucket listing is not required.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from quant_fm.scripts.minio_config import load_read_config, read_bucket

if TYPE_CHECKING:
    from minio import Minio


def _validate_dates(values: list[str]) -> list[str]:
    """Return strictly sorted, unique ISO dates and reject malformed input."""
    dates = [value.strip() for value in values if value.strip()]
    for value in dates:
        date.fromisoformat(value)
    if dates != sorted(set(dates)):
        msg = "date inventory must be strictly sorted and unique"
        raise ValueError(msg)
    return dates


def _block(values: list[str]) -> dict[str, int | str | None]:
    return {
        "count": len(values),
        "start": values[0] if values else None,
        "end": values[-1] if values else None,
    }


def build_training_plan(
    dates: list[str],
    *,
    benchmark_train_days: int = 300,
    benchmark_validation_days: int = 60,
    benchmark_test_days: int = 100,
    adaptation_cutoff: str = "2026-04-30",
    adaptation_train_days: int = 300,
    adaptation_validation_days: int = 40,
    adaptation_purge_days: int = 2,
    required_shadow_oos_days: int = 100,
) -> tuple[dict[str, object], dict[str, list[str]]]:
    """Construct both plans and the concrete date lists written beside them."""
    dates = _validate_dates(dates)
    positive = (
        benchmark_train_days,
        benchmark_validation_days,
        benchmark_test_days,
        adaptation_train_days,
        adaptation_validation_days,
        required_shadow_oos_days,
    )
    if min(positive) < 1 or adaptation_purge_days < 0:
        msg = "day counts must be positive and purge must be non-negative"
        raise ValueError(msg)

    benchmark_required = (
        benchmark_train_days + benchmark_validation_days + benchmark_test_days
    )
    if len(dates) < benchmark_required:
        msg = f"benchmark needs {benchmark_required} dates, found {len(dates)}"
        raise ValueError(msg)
    train_stop = benchmark_train_days
    validation_stop = train_stop + benchmark_validation_days
    test_stop = validation_stop + benchmark_test_days
    benchmark_train = dates[:train_stop]
    benchmark_validation = dates[train_stop:validation_stop]
    benchmark_test = dates[validation_stop:test_stop]
    locked_oos = dates[test_stop:]

    eligible = [value for value in dates if value <= adaptation_cutoff]
    if len(eligible) < adaptation_train_days:
        msg = (
            f"adaptation needs {adaptation_train_days} dates through "
            f"{adaptation_cutoff}, found {len(eligible)}"
        )
        raise ValueError(msg)
    adaptation_final_train = eligible[-adaptation_train_days:]
    core_days = (
        adaptation_train_days
        - adaptation_validation_days
        - adaptation_purge_days
    )
    if core_days < 252:
        msg = (
            "adaptation development train must retain at least 252 dates; "
            f"found {core_days}"
        )
        raise ValueError(msg)
    adaptation_core = adaptation_final_train[:core_days]
    purge_stop = core_days + adaptation_purge_days
    adaptation_purge = adaptation_final_train[core_days:purge_stop]
    adaptation_validation = adaptation_final_train[purge_stop:]
    shadow_oos = [value for value in dates if value > adaptation_cutoff]
    shadow_ready = len(shadow_oos) >= required_shadow_oos_days

    files = {
        "inventory_dates": dates,
        "benchmark_train_dates": benchmark_train,
        "benchmark_validation_dates": benchmark_validation,
        "benchmark_test_dates": benchmark_test,
        "benchmark_pipeline_dates": dates[:test_stop],
        "benchmark_locked_oos_dates": locked_oos,
        "adaptation_development_train_dates": adaptation_core,
        "adaptation_purge_dates": adaptation_purge,
        "adaptation_validation_dates": adaptation_validation,
        "adaptation_development_pipeline_dates": (
            adaptation_core + adaptation_validation + shadow_oos
        ),
        "adaptation_final_train_dates": adaptation_final_train,
        "adaptation_shadow_oos_dates": shadow_oos,
        "adaptation_refit_pipeline_dates": adaptation_final_train + shadow_oos,
    }
    plan: dict[str, object] = {
        "plan_version": "1.0",
        "inventory": _block(dates),
        "strict_benchmark": {
            "purpose": "model selection and leakage-free performance claim",
            "train": _block(benchmark_train),
            "validation": _block(benchmark_validation),
            "test": _block(benchmark_test),
            "locked_oos": _block(locked_oos),
            "vocab_fit_scope": "benchmark_train_dates only",
        },
        "adaptation_2026q1": {
            "purpose": "deployment adaptation after consuming 2026Q1 volatility",
            "development_train": _block(adaptation_core),
            "purge": _block(adaptation_purge),
            "validation": _block(adaptation_validation),
            "final_refit_train": _block(adaptation_final_train),
            "shadow_oos": _block(shadow_oos),
            "required_shadow_oos_days": required_shadow_oos_days,
            "eligible_for_full_oos_claim": shadow_ready,
            "vocab_fit_scope": "adaptation_final_train_dates only after selection",
            "warning": (
                None
                if shadow_ready
                else "post-cutoff shadow OOS is shorter than the required claim window"
            ),
        },
        "invariants": {
            "chronological": True,
            "ranker_label_horizon_days": 2,
            "benchmark_dates_must_never_be_reassigned": True,
            "adaptation_validation_can_enter_only_the_final_refit": True,
            "adaptation_period_cannot_be_reported_as_oos": True,
        },
    }
    return plan, files


def _hds_key(day: str, partition: int) -> str:
    dotted = day.replace("-", ".")
    month = dotted[:7]
    return (
        "HDS/SOURCE=zeus/DOMAIN=quote/DATASET=china_stock/"
        f"{month}/{dotted}/default/{partition}/all.parquet"
    )


def discover_complete_minio_dates(
    start: str,
    end: str,
    *,
    workers: int = 24,
) -> list[str]:
    """Read-only HEAD scan for days with snapshot, transaction and order data."""
    from minio import Minio

    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    if first > last:
        msg = "start date must not be after end date"
        raise ValueError(msg)
    candidates: list[str] = []
    cursor = first
    while cursor <= last:
        if cursor.weekday() < 5:
            candidates.append(cursor.isoformat())
        cursor += timedelta(days=1)

    config = load_read_config()
    client: Minio = Minio(
        config.endpoint,
        access_key=config.access_key,
        secret_key=config.secret_key,
        secure=config.secure,
    )
    bucket = read_bucket()

    def complete(day: str) -> str | None:
        try:
            for partition in (1, 2, 3):
                client.stat_object(bucket, _hds_key(day, partition))
        except Exception as exc:  # MinIO uses several SDK exception subclasses.
            code = getattr(exc, "code", None)
            if code in {"NoSuchKey", "NoSuchObject", "AccessDenied"}:
                if code == "AccessDenied":
                    raise
                return None
            raise
        return day

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        return [day for day in executor.map(complete, candidates) if day is not None]


def write_training_plan(
    out_dir: Path,
    plan: dict[str, object],
    files: dict[str, list[str]],
) -> Path:
    """Atomically write the JSON contract and all concrete date files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, values in files.items():
        destination = out_dir / f"{name}.txt"
        temporary = destination.with_suffix(".txt.tmp")
        temporary.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
        temporary.replace(destination)
    contract = out_dir / "training_plan.json"
    temporary = contract.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(contract)
    return contract


def main() -> None:
    """Parse CLI arguments and write the selected date-plan artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dates-file", type=Path)
    source.add_argument("--scan-minio", action="store_true")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--adaptation-cutoff", default="2026-04-30")
    args = parser.parse_args()

    if args.scan_minio:
        dates = discover_complete_minio_dates(
            args.start,
            args.end,
            workers=args.workers,
        )
    else:
        dates = _validate_dates(
            args.dates_file.read_text(encoding="utf-8").splitlines()
        )
    plan, files = build_training_plan(
        dates,
        adaptation_cutoff=args.adaptation_cutoff,
    )
    path = write_training_plan(args.out_dir, plan, files)
    print(path)


if __name__ == "__main__":
    main()
