from __future__ import annotations

import hashlib
import json
import tarfile
from typing import TYPE_CHECKING

import polars as pl
import pytest

from quant_fm.scripts.build_backtest_contract_fixture import (
    build_backtest_contract_fixture,
)
from quant_fm.scripts.package_signal_delivery import validate_signal_directory

if TYPE_CHECKING:
    from pathlib import Path


def test_fixture_matches_production_signal_contract(tmp_path: Path) -> None:
    root = build_backtest_contract_fixture(
        tmp_path / "fixture",
        created_utc="2026-07-28T00:00:00+00:00",
    )
    source_scores, source_manifest = validate_signal_directory(root / "source_signal")
    delivery_scores, delivery_manifest = validate_signal_directory(
        root / "backtest_delivery"
    )

    assert source_scores.equals(delivery_scores)
    assert source_manifest == delivery_manifest
    assert source_scores.columns == ["date", "symbol", "score"]
    assert source_scores.schema == {
        "date": pl.String,
        "symbol": pl.String,
        "score": pl.Float64,
    }
    assert source_scores.height == 8
    assert source_scores["date"].n_unique() == 2
    assert source_scores.select(["date", "symbol"]).is_duplicated().sum() == 0
    assert source_scores["symbol"].str.contains(r"^\d{6}$").all()
    assert source_scores["score"].is_finite().all()
    assert source_manifest["fixture"] == {
        "synthetic": True,
        "purpose": "backtest_interface_integration_only",
        "must_not_be_used_for_research_or_trading": True,
    }


def test_fixture_package_hashes_and_archive_are_complete(tmp_path: Path) -> None:
    root = build_backtest_contract_fixture(tmp_path / "fixture")
    delivery = root / "backtest_delivery"
    package_manifest = json.loads(
        (delivery / "delivery_manifest.json").read_text(encoding="utf-8")
    )
    assert package_manifest["package_type"] == "quantfm_backtest_signal"
    for relative, metadata in package_manifest["files"].items():
        payload = (delivery / relative).read_bytes()
        assert len(payload) == metadata["bytes"]
        assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]

    assert (delivery / "reports/SAMPLE_ONLY.md").is_file()
    with tarfile.open(root / "backtest_delivery.tar.gz", mode="r:gz") as archive:
        members = set(archive.getnames())
    assert "backtest_delivery/scores.parquet" in members
    assert "backtest_delivery/signal_manifest.json" in members
    assert "backtest_delivery/delivery_manifest.json" in members
    assert "backtest_delivery/reports/SAMPLE_ONLY.md" in members


def test_fixture_builder_never_overwrites_existing_path(tmp_path: Path) -> None:
    destination = tmp_path / "fixture"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_backtest_contract_fixture(destination)

    assert sentinel.read_text(encoding="utf-8") == "keep"
