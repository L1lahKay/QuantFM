from __future__ import annotations

import json
from typing import TYPE_CHECKING

import polars as pl

from quant_fm.scripts.audit_ranker_inputs import (
    audit_ranker_inputs,
    write_audit_reports,
)

if TYPE_CHECKING:
    from pathlib import Path


def _embeddings(dates: list[str], *, columns: tuple[int, ...] = (0, 1)) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "date": date,
                "symbol": f"{symbol_index + 1:06d}",
                **{
                    f"emb_{column}": float(day_index + symbol_index + column)
                    for column in columns
                },
            }
            for day_index, date in enumerate(dates)
            for symbol_index in range(3)
        ]
    )


def _panel(dates: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "date": date,
                "symbol": f"{symbol_index + 1:06d}",
                "fwd_ret": float(symbol_index - 1) / 100.0,
            }
            for date in dates
            for symbol_index in range(3)
        ]
    )


def test_ranker_input_audit_passes_strict_oos_fixture(tmp_path: Path) -> None:
    train_embeddings = tmp_path / "train_embeddings.parquet"
    train_panel = tmp_path / "train_panel.parquet"
    oos_embeddings = tmp_path / "oos_embeddings.parquet"
    _embeddings(["2025-01-02", "2025-01-03"]).write_parquet(train_embeddings)
    _panel(["2025-01-02", "2025-01-03"]).write_parquet(train_panel)
    _embeddings(["2026-01-05"]).write_parquet(oos_embeddings)

    result = audit_ranker_inputs(train_embeddings, train_panel, oos_embeddings)

    assert result["status"] == "pass"
    assert result["ready_for_ranker_training"] is True
    assert result["ready_for_oos_scoring"] is True
    assert result["checks"]["train_panel_coverage"]["label_coverage"] == 1.0
    assert result["checks"]["feature_compatibility"]["compatible"] is True
    assert result["checks"]["temporal_split"]["oos_strictly_after_training"] is True
    assert result["issues"] == []


def test_ranker_input_audit_marks_missing_oos_as_preflight(tmp_path: Path) -> None:
    train_embeddings = tmp_path / "train_embeddings.parquet"
    train_panel = tmp_path / "train_panel.parquet"
    _embeddings(["2025-01-02"]).write_parquet(train_embeddings)
    _panel(["2025-01-02"]).write_parquet(train_panel)

    result = audit_ranker_inputs(
        train_embeddings,
        train_panel,
        tmp_path / "not_created_yet.parquet",
    )

    assert result["status"] == "preflight"
    assert result["ready_for_ranker_training"] is True
    assert result["ready_for_oos_scoring"] is False
    assert {item["code"] for item in result["issues"]} == {"oos_embeddings_pending"}


def test_ranker_input_audit_uses_label_horizon_as_temporal_boundary(
    tmp_path: Path,
) -> None:
    train_embeddings = tmp_path / "train_embeddings.parquet"
    train_panel = tmp_path / "train_panel.parquet"
    oos_embeddings = tmp_path / "oos_embeddings.parquet"
    dates = ["2025-12-30", "2025-12-31"]
    _embeddings(dates).write_parquet(train_embeddings)
    _panel(dates).with_columns(
        pl.when(pl.col("date") == "2025-12-31")
        .then(pl.lit("2026-01-05"))
        .otherwise(pl.lit("2025-12-31"))
        .alias("next_date")
    ).write_parquet(train_panel)
    _embeddings(["2026-01-05"]).write_parquet(oos_embeddings)

    result = audit_ranker_inputs(train_embeddings, train_panel, oos_embeddings)

    assert result["status"] == "fail"
    horizon = result["checks"]["label_horizon"]
    assert horizon["columns"] == ["next_date"]
    assert horizon["date_max"] == "2026-01-05"
    temporal = result["checks"]["temporal_split"]
    assert temporal["label_horizon_end_date"] == "2026-01-05"
    assert temporal["oos_strictly_after_training"] is False
    assert "oos_not_strictly_after_training" in {
        item["code"] for item in result["issues"]
    }


def test_ranker_input_audit_finds_coverage_dimension_and_time_failures(
    tmp_path: Path,
) -> None:
    train_embeddings = tmp_path / "train_embeddings.parquet"
    train_panel = tmp_path / "train_panel.parquet"
    oos_embeddings = tmp_path / "oos_embeddings.parquet"
    train = _embeddings(["2025-01-02", "2025-01-03"])
    pl.concat([train, train.head(1)]).write_parquet(train_embeddings)
    _panel(["2025-01-02"]).write_parquet(train_panel)
    _embeddings(["2025-01-03"], columns=(0, 2)).write_parquet(oos_embeddings)

    result = audit_ranker_inputs(
        train_embeddings,
        train_panel,
        oos_embeddings,
        min_train_coverage=0.9,
    )

    assert result["status"] == "fail"
    codes = {item["code"] for item in result["issues"]}
    assert {
        "embedding_duplicate_keys",
        "train_panel_key_coverage_below_threshold",
        "train_label_coverage_below_threshold",
        "embedding_columns_not_contiguous",
        "train_oos_feature_mismatch",
        "train_oos_key_overlap",
        "oos_not_strictly_after_training",
    } <= codes


def test_ranker_input_audit_reports_invalid_values_and_future_columns(
    tmp_path: Path,
) -> None:
    train_embeddings = tmp_path / "train_embeddings.parquet"
    train_panel = tmp_path / "train_panel.parquet"
    oos_embeddings = tmp_path / "oos_embeddings.parquet"
    train = _embeddings(["2025-01-02"])
    train.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(float("nan"))
        .otherwise(pl.col("emb_0"))
        .alias("emb_0")
    ).write_parquet(train_embeddings)
    panel = _panel(["2025-01-02"]).with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(float("inf"))
        .otherwise(pl.col("fwd_ret"))
        .alias("fwd_ret")
    )
    panel.write_parquet(train_panel)
    _embeddings(["2026-01-05"]).with_columns(
        pl.lit(0.0).alias("fwd_ret")
    ).write_parquet(oos_embeddings)

    result = audit_ranker_inputs(
        train_embeddings,
        train_panel,
        oos_embeddings,
        min_train_coverage=0.5,
    )

    codes = {item["code"] for item in result["issues"]}
    assert "embedding_values_invalid" in codes
    assert "panel_returns_nonfinite" in codes
    assert "oos_future_columns_present" in codes


def test_ranker_input_audit_writes_json_and_markdown(tmp_path: Path) -> None:
    train_embeddings = tmp_path / "train_embeddings.parquet"
    train_panel = tmp_path / "train_panel.parquet"
    _embeddings(["2025-01-02"]).write_parquet(train_embeddings)
    _panel(["2025-01-02"]).write_parquet(train_panel)
    result = audit_ranker_inputs(train_embeddings, train_panel)

    json_path, markdown_path = write_audit_reports(result, tmp_path / "audit")

    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "preflight"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "`PREFLIGHT`" in markdown
    assert "oos_embeddings_pending" in markdown
