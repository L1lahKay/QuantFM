"""Read-only ordering audit for V1/V2 token shards listed by a manifest."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow.parquet as pq

from quant_fm.manifest.build_manifest import Manifest
from quant_fm.manifest.validation import validate_manifest_shard_paths
from quant_fm.tokenizer.artifact_contract import read_token_contract

if TYPE_CHECKING:
    from typing import Any

_SEQUENCE_CANDIDATES = (
    "exchange_seqnum",
    "source_seqnum",
    "serial",
    "event_idx",
)


def _first_position(mask: np.ndarray, *, offset: int) -> int | None:
    positions = np.flatnonzero(mask)
    return int(offset + positions[0] + 1) if positions.size else None


def audit_token_shard_order(
    path: Path,
    *,
    batch_size: int = 262_144,
) -> dict[str, Any]:
    """
    Stream one parquet and count time, sequence, and stable-tie inversions.

    No parquet is modified. ``first_bad_row`` is the zero-based row containing
    the first event that moves backwards relative to its predecessor.
    """
    path = Path(path)
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    result: dict[str, Any] = {
        "path": str(path),
        "rows": int(parquet.metadata.num_rows),
        "int_time_inversions": 0,
        "same_time_sequence_inversions": 0,
        "stable_tie_event_idx_inversions": 0,
        "first_bad_row": None,
        "null_ordering_values": 0,
    }
    try:
        data_semantics = read_token_contract(path)
        result["data_semantics"] = data_semantics
        result["provenance_ok"] = not bool(data_semantics["inferred_legacy"])
        if data_semantics["inferred_legacy"]:
            result["provenance_error"] = "missing explicit token contract sidecar"
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        result["data_semantics_error"] = str(exc)
        result["provenance_ok"] = False
    if "int_time" not in names:
        result.update(
            {
                "ordering_columns": [],
                "error": "missing int_time",
                "ordered": False,
                "passed": False,
            }
        )
        return result
    sequence_column = next(
        (column for column in _SEQUENCE_CANDIDATES if column in names), None
    )
    if sequence_column is None:
        result.update(
            {
                "ordering_columns": ["int_time"],
                "error": "missing exchange sequence/event_idx",
                "ordered": False,
                "passed": False,
            }
        )
        return result
    event_idx_column = (
        "event_idx" if sequence_column != "event_idx" and "event_idx" in names else None
    )
    columns = ["int_time", sequence_column]
    if event_idx_column is not None:
        columns.append(event_idx_column)
    result["ordering_columns"] = columns
    if sequence_column == "event_idx":
        result["sequence_audit_limit"] = (
            "source exchange sequence is absent; V1 same-time sequence order cannot "
            "be recovered from post-sort event_idx"
        )

    previous: tuple[int, int, int | None] | None = None
    row_offset = 0
    first_bad: int | None = None
    for batch in parquet.iter_batches(columns=columns, batch_size=batch_size):
        nulls = sum(batch.column(index).null_count for index in range(len(columns)))
        result["null_ordering_values"] += int(nulls)
        if nulls:
            row_offset += batch.num_rows
            previous = None
            continue
        time_values = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
        sequence_values = (
            batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64)
        )
        event_values = (
            batch.column(2).to_numpy(zero_copy_only=False).astype(np.int64)
            if event_idx_column is not None
            else None
        )
        if not time_values.size:
            continue

        if previous is not None:
            boundary_time = bool(time_values[0] < previous[0])
            boundary_sequence = bool(
                time_values[0] == previous[0] and sequence_values[0] < previous[1]
            )
            boundary_tie = bool(
                event_values is not None
                and previous[2] is not None
                and time_values[0] == previous[0]
                and sequence_values[0] == previous[1]
                and event_values[0] < previous[2]
            )
            result["int_time_inversions"] += int(boundary_time)
            result["same_time_sequence_inversions"] += int(boundary_sequence)
            result["stable_tie_event_idx_inversions"] += int(boundary_tie)
            if first_bad is None and (
                boundary_time or boundary_sequence or boundary_tie
            ):
                first_bad = row_offset

        prior_time, current_time = time_values[:-1], time_values[1:]
        prior_sequence, current_sequence = sequence_values[:-1], sequence_values[1:]
        time_bad = current_time < prior_time
        sequence_bad = (current_time == prior_time) & (
            current_sequence < prior_sequence
        )
        result["int_time_inversions"] += int(time_bad.sum())
        result["same_time_sequence_inversions"] += int(sequence_bad.sum())
        tie_bad = np.zeros(time_bad.shape, dtype=bool)
        if event_values is not None:
            tie_bad = (
                (current_time == prior_time)
                & (current_sequence == prior_sequence)
                & (event_values[1:] < event_values[:-1])
            )
            result["stable_tie_event_idx_inversions"] += int(tie_bad.sum())
        if first_bad is None:
            bad = time_bad | sequence_bad | tie_bad
            first_bad = _first_position(bad, offset=row_offset)

        previous = (
            int(time_values[-1]),
            int(sequence_values[-1]),
            int(event_values[-1]) if event_values is not None else None,
        )
        row_offset += batch.num_rows

    result["first_bad_row"] = first_bad
    result["ordered"] = not (
        result["int_time_inversions"]
        or result["same_time_sequence_inversions"]
        or result["stable_tie_event_idx_inversions"]
        or result["null_ordering_values"]
    )
    result["passed"] = bool(result["ordered"] and result["provenance_ok"])
    return result


def _sample_manifest_shards(manifest: Manifest, limit: int) -> list[Any]:
    selected: list[Any] = []
    per_split = max(1, math.ceil(limit / 3))
    for split in ("train", "val", "test"):
        selected.extend(manifest.split(split)[:per_split])
    return selected[:limit]


def audit_manifest_token_ordering(
    manifest_path: Path,
    *,
    sample_shards: int = 500,
    full: bool = False,
    batch_size: int = 262_144,
) -> dict[str, Any]:
    """Audit a deterministic manifest sample, or every shard with ``full=True``."""
    if sample_shards < 1:
        msg = "sample_shards must be positive"
        raise ValueError(msg)
    manifest_path = Path(manifest_path)
    manifest = Manifest.load(manifest_path)
    if manifest.schema_version == "cn_l2_v2":
        validate_manifest_shard_paths(
            manifest,
            context="token ordering audit",
            expected_tokens_root=(
                manifest_path.parent.parent / "tokens"
                if manifest_path.parent.name == "data"
                else manifest_path.parent / "tokens"
            ),
        )
    selected = (
        manifest.shards if full else _sample_manifest_shards(manifest, sample_shards)
    )
    shards: list[dict[str, Any]] = []
    missing = 0
    for shard in selected:
        path = Path(shard.path)
        if not path.is_file():
            missing += 1
            shards.append({"path": str(path), "ordered": False, "error": "missing"})
            continue
        item = audit_token_shard_order(path, batch_size=batch_size)
        item.update({"split": shard.split, "date": shard.date, "symbol": shard.symbol})
        semantics = item.get("data_semantics")
        provenance_mismatches: dict[str, tuple[object, object]] = {}
        if isinstance(semantics, dict):
            for field, manifest_value in (
                ("schema_version", manifest.schema_version),
                ("vocab_sha256", manifest.vocab_sha256),
                ("event_ordering_version", manifest.event_ordering_version),
                (
                    "feature_transform_version",
                    manifest.feature_transform_version,
                ),
            ):
                actual = semantics.get(field)
                if actual != manifest_value:
                    provenance_mismatches[field] = (actual, manifest_value)
        if provenance_mismatches:
            item["manifest_provenance_mismatches"] = provenance_mismatches
            item["provenance_ok"] = False
        item["passed"] = bool(item.get("ordered") and item.get("provenance_ok"))
        shards.append(item)
    bad_ordering = [item for item in shards if not item.get("ordered", False)]
    bad_provenance = [item for item in shards if not item.get("provenance_ok", False)]
    failed = [item for item in shards if not item.get("passed", False)]
    return {
        "audit_version": "2.0",
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "schema_version": manifest.schema_version,
        "manifest_vocab_sha256": manifest.vocab_sha256,
        "manifest_event_ordering_version": manifest.event_ordering_version,
        "manifest_feature_transform_version": manifest.feature_transform_version,
        "full_scan": full,
        "sampled_shards": len(shards),
        "missing_shards": missing,
        "bad_shards": len(failed),
        "ordering_failure_shards": len(bad_ordering),
        "provenance_failure_shards": len(bad_provenance),
        "int_time_inversions": sum(
            int(item.get("int_time_inversions", 0)) for item in shards
        ),
        "same_time_sequence_inversions": sum(
            int(item.get("same_time_sequence_inversions", 0)) for item in shards
        ),
        "stable_tie_event_idx_inversions": sum(
            int(item.get("stable_tie_event_idx_inversions", 0)) for item in shards
        ),
        "ordered": not bad_ordering,
        "provenance_ok": not bad_provenance,
        "passed": not failed,
        "shards": shards,
    }


def main() -> None:
    """Run the read-only audit and optionally write its JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-shards", type=int, default=500)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--batch-size", type=int, default=262_144)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit_manifest_token_ordering(
        args.manifest,
        sample_shards=args.sample_shards,
        full=args.full,
        batch_size=args.batch_size,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(rendered, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_name(f".{args.out.name}.tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.out)
        print(args.out)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
