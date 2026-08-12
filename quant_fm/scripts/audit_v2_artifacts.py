"""以 Parquet 元数据为主的 V2 artifact 生产就绪审计。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pylob.event_ordering import CAUSAL_EXCHANGE_TIME_V2

from quant_fm.data_coverage import coverage_set_sha256, verify_dataset_coverage
from quant_fm.manifest.build_manifest import Manifest
from quant_fm.manifest.validation import sha256_file, validate_manifest_shard_paths
from quant_fm.scripts.audit_token_ordering import audit_token_shard_order
from quant_fm.tokenizer.artifact_contract import (
    assert_token_contract_matches,
    read_token_contract,
    token_contract_path,
)
from quant_fm.tokenizer.field_spec import BOOK_FIELD_SPECS_V2
from quant_fm.tokenizer.storage_encoding_v2 import (
    Q16_MAX,
    StorageEncodingMetadataV2,
    assert_storage_metadata_matches_vocab_v2,
)
from quant_fm.tokenizer.transforms import (
    FEATURE_TRANSFORM_CAUSAL_V2,
    reference_price_initialization,
)
from quant_fm.tokenizer.vocab_v2 import VocabV2

if TYPE_CHECKING:
    from typing import Any


def _issue(severity: str, code: str, message: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message}


def _audit_failure_records(
    data_dir: Path,
) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    """Load explicit replay gaps so a formal artifact cannot hide them."""
    gaps: dict[str, list[str]] = {}
    issues: list[dict[str, str]] = []
    for path in sorted((data_dir / ".failed").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(
                _issue(
                    "critical",
                    "failure_record_invalid",
                    f"无法读取失败标的记录 {path}: {exc}",
                )
            )
            continue
        if not isinstance(payload, list) or any(
            not isinstance(symbol, str) or not symbol for symbol in payload
        ):
            issues.append(
                _issue(
                    "critical",
                    "failure_record_invalid",
                    f"失败标的记录必须是非空字符串列表: {path}",
                )
            )
            continue
        symbols = sorted(set(payload))
        if symbols:
            gaps[path.stem] = symbols
    if gaps:
        summary = ", ".join(
            f"{date}({len(symbols)})" for date, symbols in sorted(gaps.items())
        )
        issues.append(
            _issue(
                "critical",
                "symbol_coverage_gaps",
                f"存在显式回放失败标的，正式 V2 不可就绪: {summary}",
            )
        )
    return gaps, issues


def _sample_shards(manifest: Manifest, limit: int) -> list[Any]:
    selected: list[Any] = []
    per_split = max(1, math.ceil(limit / 3))
    for split in ("train", "val", "test"):
        selected.extend(manifest.split(split)[:per_split])
    return selected[:limit]


def _physical_dtypes(
    parquet: pq.ParquetFile,
    columns: tuple[str, ...],
) -> dict[str, str]:
    """返回审计字段的 Arrow 物理 dtype，缺失列留给 schema 审计处理。"""
    schema = parquet.schema_arrow
    names = set(schema.names)
    return {
        column: str(schema.field(column).type) for column in columns if column in names
    }


def _encoded_column_ranges(
    parquet: pq.ParquetFile,
    columns: tuple[str, ...],
    *,
    batch_size: int = 262_144,
) -> tuple[dict[str, dict[str, int | None]], dict[str, int]]:
    """流式统计 encoded 列范围和 null 数，不把整个 shard 读进内存。"""
    ranges = {column: {"min": None, "max": None} for column in columns}
    null_counts = dict.fromkeys(columns, 0)
    if not columns:
        return ranges, null_counts
    for batch in parquet.iter_batches(columns=list(columns), batch_size=batch_size):
        for index, column in enumerate(columns):
            array = batch.column(index)
            null_counts[column] += int(array.null_count)
            extrema = pc.min_max(array).as_py()
            if extrema is None or extrema["min"] is None:
                continue
            minimum = int(extrema["min"])
            maximum = int(extrema["max"])
            current = ranges[column]
            current["min"] = (
                minimum if current["min"] is None else min(current["min"], minimum)
            )
            current["max"] = (
                maximum if current["max"] is None else max(current["max"], maximum)
            )
    return ranges, null_counts


def _audit_storage_encoding(
    path: Path,
    parquet: pq.ParquetFile,
    vocab: VocabV2,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """审计 legacy Float shard 或显式 Q16 shard 的物理存储契约。"""
    token_columns = tuple(vocab.token_field_sizes())
    value_columns = tuple(
        str(spec.value_column)
        for spec in vocab.field_specs
        if spec.value_column is not None
    )
    all_columns = (*token_columns, *value_columns)
    physical = _physical_dtypes(parquet, all_columns)
    summary: dict[str, Any] = {
        "mode": "unknown",
        "validated": False,
        "physical_dtypes": physical,
    }
    storage_issues: list[dict[str, str]] = []

    try:
        contract = read_token_contract(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        storage_issues.append(
            _issue(
                "critical",
                "token_storage_metadata_invalid",
                f"{path}: 无法读取 token sidecar：{exc}",
            )
        )
        summary["error"] = str(exc)
        return summary, storage_issues

    if "storage_encoding" not in contract:
        scalar_types = {
            column: parquet.schema_arrow.field(column).type
            for column in value_columns
            if column in parquet.schema_arrow.names
        }
        integer_scalars = sorted(
            column
            for column, dtype in scalar_types.items()
            if pa.types.is_integer(dtype)
        )
        invalid_scalars = sorted(
            column
            for column, dtype in scalar_types.items()
            if not pa.types.is_floating(dtype) and not pa.types.is_integer(dtype)
        )
        all_float32 = bool(scalar_types) and all(
            dtype == pa.float32() for dtype in scalar_types.values()
        )
        summary.update(
            {
                "mode": "legacy_float32" if all_float32 else "legacy_float",
                "format_version": "legacy_unquantized",
                "validated": not integer_scalars and not invalid_scalars,
            }
        )
        if integer_scalars:
            storage_issues.append(
                _issue(
                    "critical",
                    "token_storage_encoding_missing",
                    f"{path}: 整数 scalar 缺少 storage_encoding，无法安全反量化："
                    f"{integer_scalars}",
                )
            )
        if invalid_scalars:
            storage_issues.append(
                _issue(
                    "critical",
                    "token_storage_dtype_mismatch",
                    f"{path}: legacy scalar 必须是浮点列：{invalid_scalars}",
                )
            )
        return summary, storage_issues

    raw_metadata = contract.get("storage_encoding")
    if not isinstance(raw_metadata, dict):
        error = "storage_encoding must be a JSON object"
        storage_issues.append(
            _issue(
                "critical",
                "token_storage_metadata_invalid",
                f"{path}: {error}",
            )
        )
        summary.update({"mode": "q16", "error": error})
        return summary, storage_issues

    try:
        metadata = StorageEncodingMetadataV2.from_dict(raw_metadata)
        assert_storage_metadata_matches_vocab_v2(metadata, vocab)
    except (TypeError, ValueError) as exc:
        storage_issues.append(
            _issue(
                "critical",
                "token_storage_metadata_invalid",
                f"{path}: {exc}",
            )
        )
        summary.update({"mode": "q16", "error": str(exc)})
        return summary, storage_issues

    summary.update(
        {
            "mode": "q16",
            "format_version": metadata.format_version,
            "scheme": metadata.scheme,
            "metadata_sha256": metadata.metadata_sha256,
            "vocab_sha256": metadata.vocab_sha256,
            "token_storage_dtypes": {
                field.column: field.storage_dtype for field in metadata.token_fields
            },
            "scalar_storage": {
                field.column: {
                    "storage_dtype": field.storage_dtype,
                    "decoded_dtype": field.decoded_dtype,
                    "clip": field.clip,
                    "scale": field.scale,
                }
                for field in metadata.scalar_fields
            },
        }
    )

    schema = parquet.schema_arrow
    schema_names = set(schema.names)
    dtype_mismatches: dict[str, dict[str, str]] = {}
    valid_encoded_columns: list[str] = []
    for field in metadata.token_fields:
        expected = pa.uint8() if field.storage_dtype == "uint8" else pa.uint16()
        if field.column not in schema_names:
            continue
        actual = schema.field(field.column).type
        if actual != expected:
            dtype_mismatches[field.column] = {
                "expected": str(expected),
                "actual": str(actual),
            }
        else:
            valid_encoded_columns.append(field.column)
    for field in metadata.scalar_fields:
        if field.column not in schema_names:
            continue
        actual = schema.field(field.column).type
        if actual != pa.int16():
            dtype_mismatches[field.column] = {
                "expected": "int16",
                "actual": str(actual),
            }
        else:
            valid_encoded_columns.append(field.column)
    if dtype_mismatches:
        storage_issues.append(
            _issue(
                "critical",
                "token_storage_dtype_mismatch",
                f"{path}: encoded 物理 dtype 不符合契约：{dtype_mismatches}",
            )
        )

    ranges, null_counts = _encoded_column_ranges(
        parquet,
        tuple(valid_encoded_columns),
    )
    summary["observed_ranges"] = ranges
    nonzero_nulls = {
        column: count for column, count in null_counts.items() if count > 0
    }
    if nonzero_nulls:
        storage_issues.append(
            _issue(
                "critical",
                "token_storage_null_values",
                f"{path}: encoded 列包含 null：{nonzero_nulls}",
            )
        )

    out_of_range: dict[str, dict[str, int | None]] = {}
    token_by_column = {field.column: field for field in metadata.token_fields}
    scalar_columns = {field.column for field in metadata.scalar_fields}
    for column, observed in ranges.items():
        minimum = observed["min"]
        maximum = observed["max"]
        if minimum is None or maximum is None:
            continue
        if column in token_by_column:
            vocab_size = token_by_column[column].vocab_size
            if minimum < 0 or maximum >= vocab_size:
                out_of_range[column] = {
                    **observed,
                    "allowed_min": 0,
                    "allowed_max": vocab_size - 1,
                }
        elif column in scalar_columns and (minimum < -Q16_MAX or maximum > Q16_MAX):
            out_of_range[column] = {
                **observed,
                "allowed_min": -Q16_MAX,
                "allowed_max": Q16_MAX,
            }
    if out_of_range:
        storage_issues.append(
            _issue(
                "critical",
                "token_storage_value_out_of_range",
                f"{path}: encoded 值超出 token/Q16 契约：{out_of_range}",
            )
        )

    summary["validated"] = not storage_issues
    return summary, storage_issues


def audit_v2_artifacts(
    root: Path,
    *,
    sample_shards: int = 12,
    full_path_check: bool = False,
) -> dict[str, Any]:
    """检查 V2 vocab/manifest/token schema、切分泄漏和盘口字段契约。"""
    if sample_shards < 1:
        msg = "sample_shards must be positive"
        raise ValueError(msg)
    root = Path(root)
    data_dir = root / "data"
    manifest_path = data_dir / "manifest.json"
    vocab_path = data_dir / "vocab_v2.json"
    validation_plan = root / "validation_windows.json"
    issues: list[dict[str, str]] = []
    failed_symbol_gaps, failure_issues = _audit_failure_records(data_dir)
    issues.extend(failure_issues)
    for name, path in (("manifest", manifest_path), ("vocab_v2", vocab_path)):
        if not path.is_file():
            issues.append(_issue("critical", f"missing_{name}", f"缺少 {path}"))
    if issues:
        return {
            "audit_version": "1.0",
            "created_utc": datetime.now(tz=UTC).isoformat(),
            "root": str(root.resolve()),
            "contract_ready": False,
            "failed_symbol_gaps": failed_symbol_gaps,
            "issues": issues,
            "sampled_shards": [],
        }

    try:
        vocab = VocabV2.load(vocab_path)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        issues.append(_issue("critical", "invalid_vocab", str(exc)))
        vocab = None
    try:
        manifest = Manifest.load(manifest_path)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        issues.append(_issue("critical", "invalid_manifest", str(exc)))
        manifest = None
    if vocab is None or manifest is None:
        return {
            "audit_version": "1.0",
            "created_utc": datetime.now(tz=UTC).isoformat(),
            "root": str(root.resolve()),
            "contract_ready": False,
            "failed_symbol_gaps": failed_symbol_gaps,
            "issues": issues,
            "sampled_shards": [],
        }

    try:
        validate_manifest_shard_paths(
            manifest,
            context="V2 artifact audit",
            expected_tokens_root=root / "tokens",
        )
    except ValueError as exc:
        issues.append(
            _issue(
                "critical",
                "manifest_shard_path_invalid",
                str(exc),
            )
        )

    if (
        manifest.schema_version != vocab.schema_version
        or vocab.schema_version != "cn_l2_v2"
    ):
        issues.append(
            _issue(
                "critical",
                "schema_mismatch",
                f"manifest={manifest.schema_version}, vocab={vocab.schema_version}",
            )
        )
    if not vocab.data_semantics_explicit:
        issues.append(
            _issue(
                "critical",
                "implicit_data_semantics",
                "V2 vocab 缺少显式事件排序/EW-VWAP 初始化版本",
            )
        )
    if vocab.event_ordering_version != CAUSAL_EXCHANGE_TIME_V2:
        issues.append(
            _issue(
                "critical",
                "noncausal_event_ordering",
                f"event_ordering_version={vocab.event_ordering_version}",
            )
        )
    if vocab.feature_transform_version != FEATURE_TRANSFORM_CAUSAL_V2:
        issues.append(
            _issue(
                "critical",
                "noncausal_reference_price_initialization",
                "feature_transform_version="
                f"{vocab.feature_transform_version}, policy="
                f"{reference_price_initialization(vocab.feature_transform_version)}",
            )
        )
    for field_name, manifest_value, vocab_value in (
        (
            "event_ordering_version",
            manifest.event_ordering_version,
            vocab.event_ordering_version,
        ),
        (
            "feature_transform_version",
            manifest.feature_transform_version,
            vocab.feature_transform_version,
        ),
    ):
        if manifest_value != vocab_value:
            issues.append(
                _issue(
                    "critical",
                    "manifest_data_semantics_mismatch",
                    f"{field_name}: manifest={manifest_value!r}, vocab={vocab_value!r}",
                )
            )
    split_dates = {
        split: set(manifest.dates(split)) for split in ("train", "val", "test")
    }
    all_manifest_dates = sorted(set().union(*split_dates.values()))
    coverage_summary: dict[str, Any] | None = None
    try:
        coverage_summary = verify_dataset_coverage(
            root,
            expected_dates=all_manifest_dates,
            manifest=manifest,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        issues.append(
            _issue(
                "critical",
                "universe_coverage_invalid",
                str(exc),
            )
        )
    if (
        overlap := (split_dates["train"] & split_dates["val"])
        | (split_dates["train"] & split_dates["test"])
        | (split_dates["val"] & split_dates["test"])
    ):
        issues.append(
            _issue(
                "critical",
                "split_date_overlap",
                f"日期跨 split 重叠：{sorted(overlap)}",
            )
        )
    leaked = set(vocab.fit_dates) & (split_dates["val"] | split_dates["test"])
    if leaked:
        issues.append(
            _issue(
                "critical", "vocab_date_leakage", f"vocab 拟合泄漏：{sorted(leaked)}"
            )
        )
    outside_training_manifest = set(vocab.fit_dates) - split_dates["train"]
    if outside_training_manifest:
        issues.append(
            _issue(
                "critical",
                "vocab_fit_date_outside_train",
                "vocab 拟合日期不属于 manifest 训练切分："
                f"{sorted(outside_training_manifest)}",
            )
        )

    field_names = {spec.name for spec in vocab.field_specs}
    expected_book = {spec.name for spec in BOOK_FIELD_SPECS_V2}
    missing_book = sorted(expected_book - field_names)
    if missing_book:
        issues.append(
            _issue(
                "critical",
                "missing_book_fields",
                f"正式 V2 缺少盘口字段：{missing_book}",
            )
        )
    if not validation_plan.is_file():
        issues.append(
            _issue(
                "warning",
                "validation_plan_missing",
                "固定验证窗口尚未生成；训练首次启动前必须创建",
            )
        )

    shards_to_check = (
        manifest.shards if full_path_check else _sample_shards(manifest, sample_shards)
    )
    expected_columns = {
        column
        for spec in vocab.field_specs
        for column in (spec.token_column, spec.value_column)
        if column is not None and (spec.is_input or spec.is_target)
    }
    sampled: list[dict[str, Any]] = []
    for shard in shards_to_check:
        path = Path(shard.path)
        item: dict[str, Any] = {
            "path": str(path),
            "split": shard.split,
            "exists": path.is_file(),
        }
        if not path.is_file():
            issues.append(_issue("critical", "token_shard_missing", str(path)))
            sampled.append(item)
            continue
        parquet = pq.ParquetFile(path)
        rows = int(parquet.metadata.num_rows)
        parquet_sha256 = sha256_file(path)
        missing_columns = sorted(expected_columns - set(parquet.schema_arrow.names))
        item.update(
            {
                "rows": rows,
                "manifest_rows": shard.rows,
                "sha256": parquet_sha256,
                "manifest_sha256": shard.sha256,
                "missing_columns": missing_columns,
            }
        )
        if rows != shard.rows:
            issues.append(
                _issue(
                    "critical",
                    "row_count_mismatch",
                    f"{path}: manifest={shard.rows}, parquet={rows}",
                )
            )
        if not shard.sha256 or parquet_sha256 != shard.sha256:
            issues.append(
                _issue(
                    "critical",
                    "token_shard_hash_mismatch",
                    f"{path}: manifest={shard.sha256!r}, actual={parquet_sha256}",
                )
            )
        sidecar = token_contract_path(path)
        if not sidecar.is_file():
            issues.append(_issue("critical", "token_sidecar_missing", f"{sidecar}"))
        else:
            sidecar_sha256 = sha256_file(sidecar)
            item.update(
                {
                    "data_contract_sha256": sidecar_sha256,
                    "manifest_data_contract_sha256": shard.data_contract_sha256,
                }
            )
            if (
                not shard.data_contract_sha256
                or sidecar_sha256 != shard.data_contract_sha256
            ):
                issues.append(
                    _issue(
                        "critical",
                        "token_sidecar_hash_mismatch",
                        f"{sidecar}: manifest={shard.data_contract_sha256!r}, "
                        f"actual={sidecar_sha256}",
                    )
                )
        if missing_columns:
            issues.append(
                _issue(
                    "critical",
                    "token_schema_missing",
                    f"{path}: {missing_columns}",
                )
            )
        storage_encoding, storage_issues = _audit_storage_encoding(
            path,
            parquet,
            vocab,
        )
        item["storage_encoding"] = storage_encoding
        issues.extend(storage_issues)
        ordering = audit_token_shard_order(path)
        item["ordering"] = {
            key: ordering.get(key)
            for key in (
                "ordering_columns",
                "int_time_inversions",
                "same_time_sequence_inversions",
                "stable_tie_event_idx_inversions",
                "first_bad_row",
                "null_ordering_values",
                "ordered",
                "error",
            )
            if key in ordering
        }
        if not ordering.get("ordered", False):
            issues.append(
                _issue(
                    "critical",
                    "token_event_order_violation",
                    f"{path}: {item['ordering']}",
                )
            )
        try:
            assert_token_contract_matches(path, vocab)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            issues.append(
                _issue(
                    "critical",
                    "token_data_semantics_mismatch",
                    f"{path}: {exc}",
                )
            )
        sampled.append(item)

    manifest_sha256 = sha256_file(manifest_path)
    vocab_file_sha256 = sha256_file(vocab_path)
    coverage_sha256 = coverage_set_sha256(root)
    audit_input_sha256 = hashlib.sha256(
        json.dumps(
            {
                "coverage_sha256": coverage_sha256,
                "manifest_sha256": manifest_sha256,
                "vocab_file_sha256": vocab_file_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    critical = [item for item in issues if item["severity"] == "critical"]
    return {
        "audit_version": "2.0",
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "root": str(root.resolve()),
        "contract_ready": not critical,
        "audit_input_sha256": audit_input_sha256,
        "coverage_sha256": coverage_sha256,
        "coverage": coverage_summary,
        "manifest_sha256": manifest_sha256,
        "vocab_file_sha256": vocab_file_sha256,
        "content_coverage_required": True,
        "failed_symbol_gaps": failed_symbol_gaps,
        "schema_version": manifest.schema_version,
        "vocab_version": vocab.VOCAB_VERSION,
        "event_ordering_version": vocab.event_ordering_version,
        "feature_transform_version": vocab.feature_transform_version,
        "reference_price_initialization": reference_price_initialization(
            vocab.feature_transform_version
        ),
        "fit_dates": list(vocab.fit_dates),
        "split_counts": dict(Counter(shard.split for shard in manifest.shards)),
        "split_dates": {split: sorted(dates) for split, dates in split_dates.items()},
        "checked_all_paths": full_path_check,
        "sampled_shards": sampled,
        "issues": issues,
    }


def main() -> None:
    """运行审计并写 JSON；契约不就绪时返回退出码 2。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample-shards", type=int, default=12)
    parser.add_argument("--full-path-check", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit_v2_artifacts(
        args.root,
        sample_shards=args.sample_shards,
        full_path_check=args.full_path_check,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    print(args.out)
    if not result["contract_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
