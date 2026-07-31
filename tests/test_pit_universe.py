from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from quant_fm.downstream.universe import (
    validate_pit_universe,
    validate_universe_alignment,
)
from quant_fm.embedding.contract import (
    AFTER_CLOSE_AVAILABILITY,
    CAUSAL_OVERLAPPING_ENCODER,
    EMBEDDING_CONTRACT_VERSION,
    STOCK_DAY_GRANULARITY,
    STRICT_EVENT_ORDERING_VERSION,
    STRICT_FEATURE_TRANSFORM_VERSION,
    EmbeddingContract,
    write_embedding_contract,
)
from quant_fm.embedding.pooling_spec import DEFAULT_V2_MULTI_SCALE_OUTPUTS
from quant_fm.scripts.preflight_topk_ranker import preflight_topk_inputs


def _universe(
    dates: list[str],
    symbols: list[str],
    *,
    policy: str = "liquid_a_share_v1",
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "date": date,
                "symbol": symbol,
                "asof_date": date,
                "universe_policy": policy,
            }
            for date in dates
            for symbol in symbols
        ]
    )


def _embeddings(path: Path, dates: list[str], symbols: list[str]) -> None:
    frame = pl.DataFrame(
        [
            {
                "date": date,
                "symbol": symbol,
                **{f"emb_{dim}": float(index + dim) for dim in range(4)},
            }
            for date in dates
            for index, symbol in enumerate(symbols)
        ]
    )
    frame.write_parquet(path)
    write_embedding_contract(
        path,
        EmbeddingContract(
            format_version=EMBEDDING_CONTRACT_VERSION,
            fm_checkpoint_sha256="a" * 64,
            vocab_sha256="b" * 64,
            schema_version="cn_l2_v2",
            book_state_timing="post_event",
            pooling_version="hierarchical_selected_v2",
            granularity=STOCK_DAY_GRANULARITY,
            context=2048,
            chunk_stride=512,
            pooling="multi_scale",
            last_k=256,
            dtype="bf16",
            encoder_width=1,
            pooling_components=DEFAULT_V2_MULTI_SCALE_OUTPUTS,
            pooling_scalar_components=(),
            embedding_columns=tuple(f"emb_{index}" for index in range(4)),
            embedding_width=4,
            signal_availability=AFTER_CLOSE_AVAILABILITY,
            encoder_semantics=CAUSAL_OVERLAPPING_ENCODER,
            event_ordering_version=STRICT_EVENT_ORDERING_VERSION,
            feature_transform_version=STRICT_FEATURE_TRANSFORM_VERSION,
        ),
    )


def _write_calendar(path: Path, dates: list[str]) -> None:
    path.write_text("\n".join(dates) + "\n", encoding="utf-8")


def test_strict_pit_universe_rejects_unverifiable_plain_membership() -> None:
    plain = pl.DataFrame({"date": ["2025-01-02"], "symbol": ["1"]})

    with pytest.raises(ValueError, match="cannot prove point-in-time"):
        validate_pit_universe(
            plain,
            required_dates=["2025-01-02"],
            min_names_per_day=1,
            context="training PIT universe",
        )


def test_strict_pit_universe_rejects_future_asof_membership() -> None:
    universe = _universe(["2025-01-02"], ["1"]).with_columns(
        pl.lit("2025-01-03").alias("asof_date")
    )

    with pytest.raises(ValueError, match="asof_date is after"):
        validate_pit_universe(
            universe,
            required_dates=["2025-01-02"],
            min_names_per_day=1,
            context="training PIT universe",
        )


def test_training_and_scoring_universe_policy_must_match() -> None:
    train = {
        "policy": "liquid_a_share_v1",
        "stats": {"names_median": 1000.0},
    }
    scoring = {
        "policy": "csi300_constituents_v1",
        "stats": {"names_median": 1000.0},
    }

    with pytest.raises(ValueError, match="policies differ"):
        validate_universe_alignment(train, scoring)


def test_preflight_validates_both_pit_universes_and_exact_horizons(
    tmp_path: Path,
) -> None:
    symbols = ["000001", "000002"]
    train_dates = ["2025-01-02", "2025-01-03"]
    oos_dates = ["2026-01-05"]
    train_embeddings = tmp_path / "train.parquet"
    oos_embeddings = tmp_path / "oos.parquet"
    train_universe = tmp_path / "train_universe.parquet"
    oos_universe = tmp_path / "oos_universe.parquet"
    train_calendar = tmp_path / "train_calendar.txt"
    oos_calendar = tmp_path / "oos_calendar.txt"
    _embeddings(train_embeddings, train_dates, symbols)
    _embeddings(oos_embeddings, oos_dates, symbols)
    _universe(train_dates, symbols).write_parquet(train_universe)
    _universe(oos_dates, symbols).write_parquet(oos_universe)
    _write_calendar(
        train_calendar,
        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"],
    )
    _write_calendar(
        oos_calendar,
        ["2026-01-05", "2026-01-06", "2026-01-07"],
    )

    report = preflight_topk_inputs(
        train_embeddings=train_embeddings,
        oos_embeddings=oos_embeddings,
        train_calendar=train_calendar,
        oos_calendar=oos_calendar,
        train_universe=train_universe,
        oos_universe=oos_universe,
        min_names_per_day=2,
    )

    assert report["status"] == "ready"
    assert report["training"]["horizon"]["last_mapping"] == {
        "date": "2025-01-03",
        "entry_date": "2025-01-06",
        "exit_date": "2025-01-07",
    }
    assert report["universe_alignment"]["train_to_score_median_ratio"] == 1.0


def test_preflight_rejects_calendar_without_last_signal_t2(tmp_path: Path) -> None:
    symbols = ["000001", "000002"]
    train_embeddings = tmp_path / "train.parquet"
    oos_embeddings = tmp_path / "oos.parquet"
    train_universe = tmp_path / "train_universe.parquet"
    oos_universe = tmp_path / "oos_universe.parquet"
    train_calendar = tmp_path / "train_calendar.txt"
    oos_calendar = tmp_path / "oos_calendar.txt"
    _embeddings(train_embeddings, ["2025-01-03"], symbols)
    _embeddings(oos_embeddings, ["2026-01-05"], symbols)
    _universe(["2025-01-03"], symbols).write_parquet(train_universe)
    _universe(["2026-01-05"], symbols).write_parquet(oos_universe)
    _write_calendar(train_calendar, ["2025-01-03", "2025-01-06"])
    _write_calendar(
        oos_calendar,
        ["2026-01-05", "2026-01-06", "2026-01-07"],
    )

    with pytest.raises(ValueError, match=r"exact T\+1/T\+2"):
        preflight_topk_inputs(
            train_embeddings=train_embeddings,
            oos_embeddings=oos_embeddings,
            train_calendar=train_calendar,
            oos_calendar=oos_calendar,
            train_universe=train_universe,
            oos_universe=oos_universe,
            min_names_per_day=2,
        )
