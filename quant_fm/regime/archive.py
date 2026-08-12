"""把清洗期逐股小文件压成一个可恢复、可审计的日级 atomic 产物。"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from quant_fm.data_coverage import load_coverage_receipt, symbol_key
from quant_fm.regime.contract import (
    ATOMIC_ARTIFACT_VERSION,
    ATOMIC_FORMULA_VERSION,
    atomic_manifest_path,
    sha256_file,
    validate_atomic_frame,
    write_json_atomic,
)


def _validate_existing_archive(
    output_path: Path,
    *,
    date: str,
    coverage_path: Path,
) -> pl.DataFrame:
    """校验 resume 命中的日级 parquet 与 manifest。"""
    manifest_path = atomic_manifest_path(output_path)
    if not manifest_path.is_file():
        msg = f"Regime atomic archive lacks manifest: {manifest_path}"
        raise RuntimeError(msg)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("artifact_version") != ATOMIC_ARTIFACT_VERSION
        or payload.get("formula_version") != ATOMIC_FORMULA_VERSION
        or payload.get("date") != date
        or payload.get("parquet_sha256") != sha256_file(output_path)
        or payload.get("coverage_receipt_sha256") != sha256_file(coverage_path)
    ):
        msg = f"Regime atomic archive manifest mismatch: {manifest_path}"
        raise RuntimeError(msg)
    return validate_atomic_frame(pl.read_parquet(output_path), expected_date=date)


def archive_atomic_day(
    clean_dir: Path,
    output_path: Path,
    *,
    date: str,
    coverage_receipt: Path,
    skip_existing: bool = False,
) -> Path:
    """合并一天全部逐股 atomic 文件，并绑定 exact coverage receipt。"""
    destination = Path(output_path)
    coverage_path = Path(coverage_receipt)
    receipt = load_coverage_receipt(coverage_path)
    if receipt.get("date") != date:
        msg = f"coverage receipt date does not match Regime archive date: {date}"
        raise ValueError(msg)
    if destination.is_file() and skip_existing:
        _validate_existing_archive(
            destination,
            date=date,
            coverage_path=coverage_path,
        )
        return destination

    frames: list[pl.DataFrame] = []
    source_paths: list[Path] = []
    for market in ("SH", "SZ"):
        market_dir = Path(clean_dir) / market
        if not market_dir.is_dir():
            continue
        for path in sorted(market_dir.glob("*/regime_atomic.parquet")):
            frame = validate_atomic_frame(
                pl.read_parquet(path),
                expected_date=date,
            )
            frames.append(frame)
            source_paths.append(path)
    if not frames:
        msg = f"no Regime atomic files found under {clean_dir}"
        raise RuntimeError(msg)

    combined = validate_atomic_frame(
        pl.concat(frames, how="vertical_relaxed"),
        expected_date=date,
    )
    actual = {
        symbol_key(str(row["market"]), str(row["symbol"]))
        for row in combined.iter_rows(named=True)
    }
    expected = set(receipt["materialized"])
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        msg = (
            "Regime atomic coverage disagrees with clean receipt: "
            f"missing={missing[:8]} extra={extra[:8]}"
        )
        raise RuntimeError(msg)
    ordering_versions = combined["event_ordering_version"].unique().to_list()
    if len(ordering_versions) != 1:
        msg = f"Regime atomic day mixes event ordering versions: {ordering_versions}"
        raise RuntimeError(msg)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    combined.write_parquet(
        temporary,
        compression="zstd",
        compression_level=3,
        statistics=True,
    )
    temporary.replace(destination)
    write_json_atomic(
        atomic_manifest_path(destination),
        {
            "artifact_version": ATOMIC_ARTIFACT_VERSION,
            "formula_version": ATOMIC_FORMULA_VERSION,
            "date": date,
            "rows": combined.height,
            "source_files": len(source_paths),
            "event_ordering_version": str(ordering_versions[0]),
            "coverage_receipt": coverage_path.name,
            "coverage_receipt_sha256": sha256_file(coverage_path),
            "parquet_file": destination.name,
            "parquet_sha256": sha256_file(destination),
        },
    )
    return destination


__all__ = ["archive_atomic_day"]
