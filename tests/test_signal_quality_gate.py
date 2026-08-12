from __future__ import annotations

import hashlib
import json
import sys
from typing import TYPE_CHECKING

import polars as pl
import pytest

from quant_fm.scripts.signal_quality_gate import evaluate_signal_quality, main

if TYPE_CHECKING:
    from pathlib import Path


def _scores(dates: list[str], *, names: int = 5) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "date": date,
                "symbol": f"{index + 1:06d}",
                "score": float(index + day / 100),
            }
            for day, date in enumerate(dates)
            for index in range(names)
        ],
        schema={"date": pl.Utf8, "symbol": pl.Utf8, "score": pl.Float64},
    )


def _manifest(path: Path, scores: pl.DataFrame) -> Path:
    dates = sorted(scores["date"].unique().to_list())
    payload = {
        "format_version": "1.0",
        "score_semantics": {
            "direction": "higher_is_more_bullish",
            "availability": "available_after_signal_date_close",
            "comparability": "cross_sectional_within_date",
        },
        "data": {
            "file": "scores.parquet",
            "file_sha256": hashlib.sha256(
                path.with_name("scores.parquet").read_bytes()
            ).hexdigest(),
            "schema": {"date": "string", "symbol": "string", "score": "float64"},
            "primary_key": ["date", "symbol"],
            "rows": scores.height,
            "dates": len(dates),
            "date_min": min(dates),
            "date_max": max(dates),
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _check(report: dict[str, object], check_id: str) -> dict[str, object]:
    checks = report["checks"]
    assert isinstance(checks, list)
    return next(item for item in checks if item["id"] == check_id)


def test_quality_gate_passes_and_reports_ic_baseline_and_turnover(
    tmp_path: Path,
) -> None:
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    scores = _scores(dates)
    scores_path = tmp_path / "scores.parquet"
    scores.write_parquet(scores_path)
    manifest_path = _manifest(tmp_path / "signal_manifest.json", scores)
    panel = scores.select("date", "symbol").with_columns(
        pl.col("symbol").cast(pl.Int64).cast(pl.Float64).alias("fwd_ret"),
        (-pl.col("symbol").cast(pl.Int64)).cast(pl.Float64).alias("factor_mom_1"),
        pl.lit(True).alias("eligible_at_signal"),
    )
    panel_path = tmp_path / "panel.parquet"
    panel.write_parquet(panel_path)
    report_dir = tmp_path / "quality"

    report = evaluate_signal_quality(
        scores_path=scores_path,
        manifest_path=manifest_path,
        panel_path=panel_path,
        expected_dates=dates,
        min_names_per_day=3,
        top_k=2,
        json_path=report_dir / "signal_quality.json",
        markdown_path=report_dir / "signal_quality.md",
    )

    assert report["status"] == "pass"
    assert report["evaluation"]["candidate_ic"]["statistics"][
        "mean_ic"
    ] == pytest.approx(1.0)
    assert report["evaluation"]["baseline"]["mean_ic"] == pytest.approx(-1.0)
    assert report["turnover"]["mean_top_k_turnover"] == 0.0
    assert _check(report, "manifest_contract")["status"] == "pass"
    raw_json = (report_dir / "signal_quality.json").read_text(encoding="utf-8")
    assert "NaN" not in raw_json
    assert json.loads(raw_json)["status"] == "pass"
    assert "## IC and baseline" in (report_dir / "signal_quality.md").read_text(
        encoding="utf-8"
    )


def test_quality_gate_fails_duplicates_nonfinite_constant_and_missing_date(
    tmp_path: Path,
) -> None:
    bad = pl.DataFrame(
        {
            "date": ["2026-01-05", "2026-01-05", "2026-01-05"],
            "symbol": ["000001", "000001", "000002"],
            "score": [1.0, float("nan"), 1.0],
        },
        schema={"date": pl.Utf8, "symbol": pl.Utf8, "score": pl.Float64},
    )
    scores_path = tmp_path / "scores.parquet"
    bad.write_parquet(scores_path)

    report = evaluate_signal_quality(
        scores_path=scores_path,
        expected_dates=["2026-01-05", "2026-01-06"],
        min_names_per_day=2,
        json_path=tmp_path / "quality.json",
        markdown_path=tmp_path / "quality.md",
    )

    assert report["status"] == "fail"
    assert _check(report, "finite_scores")["status"] == "fail"
    assert _check(report, "primary_key")["status"] == "fail"
    assert _check(report, "non_constant_cross_sections")["status"] == "fail"
    assert _check(report, "expected_dates")["status"] == "fail"
    assert (tmp_path / "quality.json").exists()
    assert (tmp_path / "quality.md").exists()


def test_quality_gate_fails_incomplete_panel_and_label_coverage(tmp_path: Path) -> None:
    scores = _scores(["2026-01-05", "2026-01-06"], names=4)
    scores_path = tmp_path / "scores.parquet"
    scores.write_parquet(scores_path)
    panel = scores.filter(
        ~((pl.col("date") == "2026-01-06") & (pl.col("symbol") == "000004"))
    ).with_columns(
        pl.when(pl.col("symbol") == "000003")
        .then(None)
        .otherwise(pl.col("score"))
        .alias("fwd_ret")
    )
    panel_path = tmp_path / "panel.parquet"
    panel.write_parquet(panel_path)

    report = evaluate_signal_quality(
        scores_path=scores_path,
        panel_path=panel_path,
        min_names_per_day=3,
        min_daily_coverage=0.9,
        min_label_coverage=0.95,
        json_path=tmp_path / "quality.json",
        markdown_path=tmp_path / "quality.md",
    )

    assert report["status"] == "fail"
    assert _check(report, "panel_key_coverage")["status"] == "fail"
    assert _check(report, "forward_return_coverage")["status"] == "fail"


def test_cli_returns_two_after_hard_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scores_path = tmp_path / "scores.parquet"
    pl.DataFrame(
        {"date": ["bad"], "symbol": ["1"], "score": [float("inf")]},
        schema={"date": pl.Utf8, "symbol": pl.Utf8, "score": pl.Float64},
    ).write_parquet(scores_path)
    out_dir = tmp_path / "report"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "signal_quality_gate",
            "--scores",
            str(scores_path),
            "--out-dir",
            str(out_dir),
            "--min-names-per-day",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    assert (out_dir / "signal_quality.json").exists()
    assert (out_dir / "signal_quality.md").exists()
