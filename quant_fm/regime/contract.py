"""Regime 数据产物的稳定 schema、版本和哈希工具。"""

from __future__ import annotations

import hashlib
import json
from datetime import date as calendar_date
from pathlib import Path
from typing import Any

import polars as pl

ATOMIC_ARTIFACT_VERSION = "regime_stock_day_atomic_v1"
ATOMIC_FORMULA_VERSION = "l2_book_twa_ofi_v1"
REGIME_ARTIFACT_VERSION = "regime_features_l2_v1"
REGIME_FORMULA_VERSION = "l2_equal_weight_eod_v1"

ATOMIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "stock_spread_bps",
    "stock_depth_l5_log",
    "stock_ofi_l1",
)
ATOMIC_QUALITY_COLUMNS: tuple[str, ...] = (
    "book_valid_seconds",
    "observed_seconds",
    "continuous_session_seconds",
    "book_valid_ratio",
    "n_events",
    "n_ofi_events",
)
ATOMIC_COLUMNS: tuple[str, ...] = (
    "artifact_version",
    "formula_version",
    "date",
    "symbol",
    "market",
    "board",
    *ATOMIC_FEATURE_COLUMNS,
    *ATOMIC_QUALITY_COLUMNS,
    "feature_cutoff_ts",
    "event_ordering_version",
)

L2_FEATURE_COLUMNS: tuple[str, ...] = (
    "market_return_5d",
    "market_return_20d",
    "market_realized_vol_20d",
    "market_breadth",
    "market_amount_ratio_20d",
    "board_relative_strength_20d",
    *ATOMIC_FEATURE_COLUMNS,
)

_STRING_ATOMIC_COLUMNS = {
    "artifact_version",
    "formula_version",
    "date",
    "symbol",
    "market",
    "board",
    "feature_cutoff_ts",
    "event_ordering_version",
}
_INT_ATOMIC_COLUMNS = {"n_events", "n_ofi_events"}


def sha256_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    """流式计算一个产物的 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_paths(paths: list[Path]) -> str:
    """按稳定相对名称和文件内容聚合一组输入的哈希。"""
    digest = hashlib.sha256()
    for path in sorted(Path(value) for value in paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_manifest_path(path: Path) -> Path:
    """返回日级 atomic parquet 的 sidecar manifest 路径。"""
    return Path(path).with_suffix(".manifest.json")


def regime_manifest_path(path: Path) -> Path:
    """返回最终 Regime parquet 的 sidecar manifest 路径。"""
    return Path(path).with_suffix(".manifest.json")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """以稳定 JSON 格式原子写入 manifest。"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def validate_atomic_frame(
    frame: pl.DataFrame,
    *,
    expected_date: str | None = None,
) -> pl.DataFrame:
    """规范并严格校验逐股日 atomic 表；特征缺失保留给最终门禁处理。"""
    missing = sorted(set(ATOMIC_COLUMNS) - set(frame.columns))
    if missing:
        msg = f"Regime atomic frame is missing columns: {missing}"
        raise ValueError(msg)
    if frame.is_empty():
        msg = "Regime atomic frame must not be empty"
        raise ValueError(msg)

    expressions: list[pl.Expr] = []
    for column in ATOMIC_COLUMNS:
        if column in _STRING_ATOMIC_COLUMNS:
            expressions.append(pl.col(column).cast(pl.Utf8, strict=False))
        elif column in _INT_ATOMIC_COLUMNS:
            expressions.append(pl.col(column).cast(pl.Int64, strict=True))
        else:
            expressions.append(pl.col(column).cast(pl.Float64, strict=True))
    normalized = frame.select(expressions)

    if normalized.select(pl.struct(["date", "market", "symbol"]).is_duplicated().any()).item():
        msg = "Regime atomic frame contains duplicate (date, market, symbol) keys"
        raise ValueError(msg)
    if normalized.select(
        pl.any_horizontal(
            pl.col(name).is_null()
            for name in (
                "artifact_version",
                "formula_version",
                "date",
                "market",
                "symbol",
                "board",
                "feature_cutoff_ts",
                "event_ordering_version",
            )
        ).any()
    ).item():
        msg = "Regime atomic identity/provenance columns must not be null"
        raise ValueError(msg)
    if set(normalized["artifact_version"].unique()) != {ATOMIC_ARTIFACT_VERSION}:
        msg = "unsupported Regime atomic artifact version"
        raise ValueError(msg)
    if set(normalized["formula_version"].unique()) != {ATOMIC_FORMULA_VERSION}:
        msg = "unsupported Regime atomic formula version"
        raise ValueError(msg)
    dates = normalized["date"].unique().to_list()
    for value in dates:
        try:
            calendar_date.fromisoformat(str(value))
        except ValueError as exc:
            msg = f"Regime atomic frame contains a non-ISO date: {value!r}"
            raise ValueError(msg) from exc
    if expected_date is not None and dates != [expected_date]:
        msg = f"Regime atomic date mismatch: expected={expected_date}, actual={dates}"
        raise ValueError(msg)
    if not set(normalized["market"].unique()).issubset({"SH", "SZ"}):
        msg = "Regime atomic market must be SH or SZ"
        raise ValueError(msg)
    if normalized.filter(
        (pl.col("book_valid_seconds") < 0)
        | (pl.col("observed_seconds") < 0)
        | (pl.col("continuous_session_seconds") <= 0)
        | (pl.col("book_valid_ratio") < 0)
        | (pl.col("book_valid_ratio") > 1)
        | (pl.col("n_events") < 1)
        | (pl.col("n_ofi_events") < 0)
    ).height:
        msg = "Regime atomic quality metrics are outside their valid ranges"
        raise ValueError(msg)
    for name in (*ATOMIC_FEATURE_COLUMNS, *ATOMIC_QUALITY_COLUMNS[:-2]):
        finite = normalized[name].drop_nulls().is_finite()
        if not bool(finite.all()):
            msg = f"Regime atomic column {name!r} contains NaN/Inf"
            raise ValueError(msg)
    return normalized.sort(["date", "market", "symbol"])


__all__ = [
    "ATOMIC_ARTIFACT_VERSION",
    "ATOMIC_COLUMNS",
    "ATOMIC_FEATURE_COLUMNS",
    "ATOMIC_FORMULA_VERSION",
    "ATOMIC_QUALITY_COLUMNS",
    "L2_FEATURE_COLUMNS",
    "REGIME_ARTIFACT_VERSION",
    "REGIME_FORMULA_VERSION",
    "atomic_manifest_path",
    "regime_manifest_path",
    "sha256_file",
    "sha256_paths",
    "validate_atomic_frame",
    "write_json_atomic",
]
