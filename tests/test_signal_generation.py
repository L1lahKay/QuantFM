from __future__ import annotations

import json
from typing import TYPE_CHECKING

import polars as pl
import pytest

from quant_fm.downstream.make_features import (
    build_scoring_features,
    build_training_features,
)
from quant_fm.downstream.train_ranker import feature_columns, train_ranker
from quant_fm.signal.artifact import load_ranker_artifact, save_ranker_artifact
from quant_fm.signal.generate import generate_scores

if TYPE_CHECKING:
    from pathlib import Path


def _embeddings(dates: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "date": date,
                "symbol": f"{index + 1:06d}",
                "emb_0": float(index),
                "emb_1": float(day + index),
            }
            for day, date in enumerate(dates)
            for index in range(3)
        ]
    )


def _artifact(tmp_path: Path) -> tuple[Path, Path]:
    embeddings = _embeddings(["2025-01-02", "2025-01-03"])
    panel = embeddings.select(["date", "symbol"]).with_columns(
        pl.int_range(pl.len()).mod(3).cast(pl.Float64).alias("fwd_ret")
    )
    training = build_training_features(embeddings, panel, min_names_per_day=2)
    model, history = train_ranker(training, epochs=1, device="cpu", seed=7)
    checkpoint = tmp_path / "ranker.pt"
    metadata = tmp_path / "ranker_metadata.json"
    save_ranker_artifact(
        model,
        checkpoint,
        metadata,
        feature_columns=feature_columns(training),
        training_end_date="2025-01-03",
        seed=7,
        history=history,
    )
    return checkpoint, metadata


def test_scoring_features_have_no_label_dependency() -> None:
    embeddings = _embeddings(["2026-01-05"])
    result = build_scoring_features(embeddings)
    assert result.columns == ["date", "symbol", "emb_0", "emb_1"]
    assert "label" not in result.columns
    assert "fwd_ret" not in result.columns
    with pytest.raises(ValueError, match="forbidden future"):
        build_scoring_features(embeddings.with_columns(pl.lit(0.1).alias("fwd_ret")))


def test_ranker_artifact_round_trip(tmp_path: Path) -> None:
    checkpoint, metadata_path = _artifact(tmp_path)
    model, metadata = load_ranker_artifact(checkpoint, metadata_path, device="cpu")
    assert model.proj.in_features == 2
    assert metadata["feature_columns"] == ["emb_0", "emb_1"]


def test_generate_scores_without_test_panel(tmp_path: Path) -> None:
    checkpoint, metadata = _artifact(tmp_path)
    embeddings_path = tmp_path / "latest_embeddings.parquet"
    _embeddings(["2026-01-05"]).write_parquet(embeddings_path)
    output = tmp_path / "delivery"

    first = generate_scores(
        embeddings_path=embeddings_path,
        ranker_path=checkpoint,
        ranker_metadata_path=metadata,
        out_dir=output,
        device="cpu",
    )
    values = pl.read_parquet(first)
    second_values = pl.read_parquet(
        generate_scores(
            embeddings_path=embeddings_path,
            ranker_path=checkpoint,
            ranker_metadata_path=metadata,
            out_dir=output,
            device="cpu",
        )
    )
    assert values.columns == ["date", "symbol", "score"]
    assert values.equals(second_values)
    assert {path.name for path in output.iterdir()} == {
        "scores.parquet",
        "signal_manifest.json",
    }
    manifest = json.loads((output / "signal_manifest.json").read_text())
    assert manifest["data"]["rows"] == 3
    assert manifest["data"]["date_max"] == "2026-01-05"


def test_generate_scores_rejects_in_sample_date(tmp_path: Path) -> None:
    checkpoint, metadata = _artifact(tmp_path)
    embeddings_path = tmp_path / "embeddings.parquet"
    _embeddings(["2025-01-03"]).write_parquet(embeddings_path)
    with pytest.raises(ValueError, match="strictly after"):
        generate_scores(
            embeddings_path=embeddings_path,
            ranker_path=checkpoint,
            ranker_metadata_path=metadata,
            out_dir=tmp_path / "delivery",
        )
