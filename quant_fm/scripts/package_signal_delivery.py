"""Validate and freeze a QuantFM signal directory into a backtest delivery package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_fm.signal.schema import SCORE_COLUMNS, validate_scores

EXPECTED_SCHEMA = {"date": "string", "symbol": "string", "score": "float64"}
EXPECTED_SEMANTICS = {
    "direction": "higher_is_more_bullish",
    "availability": "available_after_signal_date_close",
    "comparability": "cross_sectional_within_date",
}


def _sha256(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"invalid signal manifest {path}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(value, dict):
        msg = f"signal manifest must be a JSON object: {path}"
        raise TypeError(msg)
    return value


def validate_signal_directory(source_dir: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Validate the strict parquet contract and its production signal manifest."""
    source_dir = Path(source_dir)
    scores_path = source_dir / "scores.parquet"
    manifest_path = source_dir / "signal_manifest.json"
    missing = [str(path) for path in (scores_path, manifest_path) if not path.is_file()]
    if missing:
        msg = f"signal delivery is missing required file(s): {missing}"
        raise FileNotFoundError(msg)

    scores = pl.read_parquet(scores_path)
    if scores.columns != SCORE_COLUMNS:
        msg = f"scores columns must be exactly {SCORE_COLUMNS}, got {scores.columns}"
        raise ValueError(msg)
    actual_schema = scores.schema
    expected_dtypes = {"date": pl.String, "symbol": pl.String, "score": pl.Float64}
    mismatches = {
        name: f"expected {expected_dtypes[name]}, got {actual_schema[name]}"
        for name in SCORE_COLUMNS
        if actual_schema[name] != expected_dtypes[name]
    }
    if mismatches:
        msg = f"scores parquet dtype mismatch: {mismatches}"
        raise ValueError(msg)
    scores = validate_scores(scores)

    manifest = _read_manifest(manifest_path)
    data = manifest.get("data")
    if not isinstance(data, dict):
        msg = "signal manifest is missing object field 'data'"
        raise TypeError(msg)
    expected_data = {
        "file": "scores.parquet",
        "file_sha256": _sha256(scores_path),
        "schema": EXPECTED_SCHEMA,
        "primary_key": ["date", "symbol"],
        "rows": scores.height,
        "dates": scores["date"].n_unique(),
        "date_min": scores["date"].min(),
        "date_max": scores["date"].max(),
    }
    inconsistent = {
        key: {"expected": expected, "actual": data.get(key)}
        for key, expected in expected_data.items()
        if data.get(key) != expected
    }
    if inconsistent:
        msg = f"signal manifest does not match scores.parquet: {inconsistent}"
        raise ValueError(msg)
    if manifest.get("score_semantics") != EXPECTED_SEMANTICS:
        msg = "signal manifest score_semantics does not match the stable contract"
        raise ValueError(msg)
    return scores, manifest


def _readme(manifest: dict[str, Any], report_names: list[str]) -> str:
    data = manifest["data"]
    reports = (
        "\n".join(f"- `reports/{name}`" for name in report_names)
        if report_names
        else "- 无附加报告"
    )
    return f"""# QuantFM 回测信号交付包

## 主数据

- 文件：`scores.parquet`
- Schema：`date: string, symbol: string, score: float64`
- 主键：`(date, symbol)`
- 行数：{data["rows"]}
- 日期：{data["dates"]} 天，{data["date_min"]} 至 {data["date_max"]}

## 信号语义

- `score` 越大越看多，仅保证同一交易日截面可比。
- `score(T)` 在 T 日收盘后可用，最早于 T+1 执行。
- 缺失股票表示当日无信号，不得补零。
- 本文件不是预测收益率、概率或直接仓位。

## 完整性

`delivery_manifest.json` 记录包内文件大小与完整 SHA-256。接收端应在读取前复核。

## 附加验收报告

{reports}
"""


def package_signal_delivery(
    *,
    source_dir: Path,
    out_dir: Path,
    reports: tuple[Path, ...] = (),
    archive_path: Path | None = None,
) -> tuple[Path, Path | None]:
    """Validate, atomically freeze, hash and optionally archive a delivery."""
    source_dir = Path(source_dir).resolve()
    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        msg = f"refusing to overwrite existing delivery directory: {out_dir}"
        raise FileExistsError(msg)
    if archive_path is not None:
        archive_path = Path(archive_path).resolve()
        if archive_path.exists():
            msg = f"refusing to overwrite existing archive: {archive_path}"
            raise FileExistsError(msg)

    _, signal_manifest = validate_signal_directory(source_dir)
    resolved_reports = tuple(Path(path).resolve() for path in reports)
    missing_reports = [str(path) for path in resolved_reports if not path.is_file()]
    if missing_reports:
        msg = f"report file(s) do not exist: {missing_reports}"
        raise FileNotFoundError(msg)
    report_names = [path.name for path in resolved_reports]
    if len(report_names) != len(set(report_names)):
        msg = "report basenames must be unique"
        raise ValueError(msg)

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = out_dir.with_name(f".{out_dir.name}.tmp-{os.getpid()}")
    if stage.exists():
        msg = f"temporary delivery path already exists: {stage}"
        raise FileExistsError(msg)
    try:
        stage.mkdir()
        shutil.copy2(source_dir / "scores.parquet", stage / "scores.parquet")
        shutil.copy2(
            source_dir / "signal_manifest.json", stage / "signal_manifest.json"
        )
        if resolved_reports:
            reports_dir = stage / "reports"
            reports_dir.mkdir()
            for report in resolved_reports:
                shutil.copy2(report, reports_dir / report.name)
        (stage / "README.md").write_text(
            _readme(signal_manifest, report_names), encoding="utf-8"
        )

        file_entries: dict[str, dict[str, Any]] = {}
        for path in sorted(item for item in stage.rglob("*") if item.is_file()):
            relative = path.relative_to(stage).as_posix()
            file_entries[relative] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        package_manifest = {
            "format_version": "1.0",
            "package_type": "quantfm_backtest_signal",
            "created_utc": datetime.now(tz=UTC).isoformat(),
            "score_contract": {
                "schema": EXPECTED_SCHEMA,
                "primary_key": ["date", "symbol"],
                "semantics": EXPECTED_SEMANTICS,
            },
            "data": signal_manifest["data"],
            "files": file_entries,
        }
        (stage / "delivery_manifest.json").write_text(
            json.dumps(package_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        stage.replace(out_dir)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    if archive_path is not None:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_archive = archive_path.with_name(
            f".{archive_path.name}.tmp-{os.getpid()}"
        )
        try:
            with tarfile.open(temporary_archive, mode="w:gz") as archive:
                archive.add(out_dir, arcname=out_dir.name, recursive=True)
            temporary_archive.replace(archive_path)
        except Exception:
            temporary_archive.unlink(missing_ok=True)
            raise
    return out_dir, archive_path


def main() -> None:
    """Run the delivery packager."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--report",
        action="append",
        type=Path,
        default=[],
        help="Optional QA/audit report to include; may be repeated.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Optional .tar.gz output path; existing files are never overwritten.",
    )
    args = parser.parse_args()
    out_dir, archive = package_signal_delivery(
        source_dir=args.source_dir,
        out_dir=args.out_dir,
        reports=tuple(args.report),
        archive_path=args.archive,
    )
    print(f"delivery_dir={out_dir}")
    if archive is not None:
        print(f"archive={archive}")


if __name__ == "__main__":
    main()
