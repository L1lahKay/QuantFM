from __future__ import annotations

import hashlib
import json
import tarfile
from typing import TYPE_CHECKING

import polars as pl
import pytest

from quant_fm.scripts.package_signal_delivery import package_signal_delivery

if TYPE_CHECKING:
    from pathlib import Path


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    scores = pl.DataFrame(
        {
            "date": ["2026-01-05", "2026-01-05", "2026-01-06"],
            "symbol": ["000001", "000002", "000001"],
            "score": [0.5, -0.2, 0.8],
        },
        schema={"date": pl.String, "symbol": pl.String, "score": pl.Float64},
    )
    scores.write_parquet(source / "scores.parquet")
    manifest = {
        "format_version": "1.0",
        "score_semantics": {
            "direction": "higher_is_more_bullish",
            "availability": "available_after_signal_date_close",
            "comparability": "cross_sectional_within_date",
        },
        "data": {
            "file": "scores.parquet",
            "schema": {"date": "string", "symbol": "string", "score": "float64"},
            "primary_key": ["date", "symbol"],
            "rows": 3,
            "dates": 2,
            "date_min": "2026-01-05",
            "date_max": "2026-01-06",
        },
    }
    (source / "signal_manifest.json").write_text(json.dumps(manifest))
    return source


def test_package_signal_delivery_hashes_reports_and_archive(tmp_path: Path) -> None:
    source = _source(tmp_path)
    report = tmp_path / "signal_qa.md"
    report.write_text("# QA\n\nPASS\n", encoding="utf-8")
    out = tmp_path / "frozen_delivery"
    archive = tmp_path / "frozen_delivery.tar.gz"

    package_signal_delivery(
        source_dir=source,
        out_dir=out,
        reports=(report,),
        archive_path=archive,
    )

    assert (out / "scores.parquet").is_file()
    assert (out / "signal_manifest.json").is_file()
    assert (out / "README.md").is_file()
    assert (out / "reports/signal_qa.md").is_file()
    package_manifest = json.loads((out / "delivery_manifest.json").read_text())
    score_entry = package_manifest["files"]["scores.parquet"]
    assert (
        score_entry["sha256"]
        == hashlib.sha256((out / "scores.parquet").read_bytes()).hexdigest()
    )
    with tarfile.open(archive, mode="r:gz") as bundle:
        names = set(bundle.getnames())
    assert "frozen_delivery/scores.parquet" in names
    assert "frozen_delivery/delivery_manifest.json" in names


def test_package_rejects_manifest_mismatch_and_existing_destination(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    manifest_path = source / "signal_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["data"]["rows"] = 999
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match"):
        package_signal_delivery(source_dir=source, out_dir=tmp_path / "bad")

    source = _source(tmp_path / "second")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        package_signal_delivery(source_dir=source, out_dir=existing)
