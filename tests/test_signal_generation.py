from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from typing import TYPE_CHECKING

import polars as pl
import pytest
import torch

from quant_fm.downstream.make_features import (
    build_scoring_features,
    build_training_features,
)
from quant_fm.downstream.train_ranker import (
    RankerObjectiveConfig,
    feature_columns,
    train_ranker,
)
from quant_fm.embedding.contract import (
    AFTER_CLOSE_AVAILABILITY,
    CAUSAL_CHUNKED_ENCODER,
    CAUSAL_OVERLAPPING_ENCODER,
    EMBEDDING_CONTRACT_VERSION,
    STOCK_DAY_GRANULARITY,
    STRICT_EVENT_ORDERING_VERSION,
    STRICT_FEATURE_TRANSFORM_VERSION,
    EmbeddingContract,
    write_embedding_contract,
)
from quant_fm.embedding.pooling_spec import DEFAULT_V2_MULTI_SCALE_OUTPUTS
from quant_fm.signal.artifact import (
    SIGNAL_FEATURE_TARGET_SPEC_VERSION,
    load_ranker_artifact,
    save_ranker_artifact,
    validate_ranker_training_contract,
)
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


def _embedding_contract(
    *,
    pooling: str = "mean",
    checkpoint: str = "a" * 64,
    vocab: str = "b" * 64,
) -> EmbeddingContract:
    return EmbeddingContract(
        format_version=EMBEDDING_CONTRACT_VERSION,
        fm_checkpoint_sha256=checkpoint,
        vocab_sha256=vocab,
        schema_version="cn_l2_v1",
        book_state_timing="none",
        pooling_version="flat_v1",
        granularity=STOCK_DAY_GRANULARITY,
        context=2048,
        chunk_stride=2048,
        pooling=pooling,
        last_k=256,
        dtype="bf16",
        encoder_width=2,
        pooling_components=(pooling,),
        pooling_scalar_components=(),
        embedding_columns=("emb_0", "emb_1"),
        embedding_width=2,
        signal_availability=AFTER_CLOSE_AVAILABILITY,
        encoder_semantics=CAUSAL_CHUNKED_ENCODER,
        event_ordering_version="local_time_v1",
        feature_transform_version="ew_vwap_future_backfill_v1",
    )


def _strict_embedding_contract(*, checkpoint: str, vocab: str) -> EmbeddingContract:
    width = len(DEFAULT_V2_MULTI_SCALE_OUTPUTS)
    return EmbeddingContract(
        format_version=EMBEDDING_CONTRACT_VERSION,
        fm_checkpoint_sha256=checkpoint,
        vocab_sha256=vocab,
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
        embedding_columns=tuple(f"emb_{index}" for index in range(width)),
        embedding_width=width,
        signal_availability=AFTER_CLOSE_AVAILABILITY,
        encoder_semantics=CAUSAL_OVERLAPPING_ENCODER,
        event_ordering_version=STRICT_EVENT_ORDERING_VERSION,
        feature_transform_version=STRICT_FEATURE_TRANSFORM_VERSION,
    )


def _wide_embeddings(dates: list[str], *, names: int, width: int) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "date": date,
                "symbol": f"{index + 1:06d}",
                **{
                    f"emb_{column}": float(day + index + column)
                    for column in range(width)
                },
            }
            for day, date in enumerate(dates)
            for index in range(names)
        ]
    )


def _write_embeddings(
    path: Path,
    frame: pl.DataFrame,
    *,
    contract: EmbeddingContract | None = None,
) -> None:
    frame.write_parquet(path)
    if contract is not None:
        write_embedding_contract(path, contract)


def _strict_training_contract() -> dict[str, object]:
    objective = json.loads(json.dumps(asdict(RankerObjectiveConfig())))
    width_stats = {
        "rows": 11_200,
        "days": 32,
        "date_min": "2025-01-02",
        "date_max": "2025-02-14",
        "names_min": 350,
        "names_median": 350.0,
        "names_max": 350,
    }
    return {
        "execution_contract": {
            "verified": True,
            "mode": "strict_execution_panel",
            "execution_contract_version": "calendar_indexed_execution_v1",
            "return_spec": "vwap_t1_vwap_t2",
            "signal_availability": "available_after_signal_date_close",
            "entry_day_lag": 1,
            "exit_day_lag": 2,
            "entry_price_field": "vwap",
            "exit_price_field": "vwap",
            "trading_calendar_sha256": "c" * 64,
            "calendar_date_count": 64,
            "calendar_reverified": True,
        },
        "representation_gate": {
            "verified": True,
            "format_version": "stock_day_embedding_v2",
            "schema_version": "cn_l2_v2",
            "book_state_timing": "post_event",
            "event_ordering_version": "exchange_time_sequence_v2",
            "feature_transform_version": "ew_vwap_causal_nan_v2",
            "encoder_semantics": "causal_overlap_unique_emit_v2",
            "context": 2048,
            "chunk_stride": 512,
            "pooling_version": "hierarchical_selected_v2",
        },
        "time_split": {
            "validation_enabled": True,
            "available_days": 32,
            "train_days": 20,
            "purge_days": 2,
            "val_days": 10,
            "train_end": "2025-01-29",
            "val_start": "2025-02-03",
            "val_end": "2025-02-14",
        },
        "feature_target_spec_version": SIGNAL_FEATURE_TARGET_SPEC_VERSION,
        "universe": {
            "mode": "daily_pit_file",
            "sha256": "d" * 64,
            "daily_names_min": 350,
            "contract": {
                "format_version": "pit_universe_v1",
                "verified": True,
                "policy": "liquid_a_share_v1",
                "asof_rule": "asof_date_lte_signal_date",
                "required_dates": 32,
                "stats": width_stats,
            },
            "retained_training_features": width_stats,
        },
        "selection": {
            "best_epoch": 3,
            "best_val_ic": 0.1,
            "best_val_ndcg": 0.5,
            "best_selection_score": 0.53,
            "stopped_early": False,
        },
        "objective": objective,
    }


def _artifact(
    tmp_path: Path,
    *,
    embedding_contract: EmbeddingContract | None = None,
    training_contract: dict[str, object] | None = None,
    allow_legacy_training_contract: bool = True,
    objective: RankerObjectiveConfig | None = None,
    training_embeddings: pl.DataFrame | None = None,
) -> tuple[Path, Path]:
    embeddings = (
        training_embeddings
        if training_embeddings is not None
        else _embeddings(["2025-01-02", "2025-01-03"])
    )
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
        label_end_date="2025-01-03",
        seed=7,
        objective=objective or RankerObjectiveConfig(),
        embedding_contract=embedding_contract or _embedding_contract(),
        allow_legacy_training_contract=allow_legacy_training_contract,
        history=history,
        training_contract=training_contract,
    )
    return checkpoint, metadata


def _strict_signal_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    fm_path = tmp_path / "fm.pt"
    vocab_path = tmp_path / "vocab.json"
    fm_bytes = b"frozen-fm-checkpoint"
    vocab_bytes = b'{"format_version":"test-vocab"}'
    fm_path.write_bytes(fm_bytes)
    vocab_path.write_bytes(vocab_bytes)
    contract = _strict_embedding_contract(
        checkpoint=sha256(fm_bytes).hexdigest(),
        vocab=sha256(vocab_bytes).hexdigest(),
    )
    checkpoint, metadata = _artifact(
        tmp_path,
        embedding_contract=contract,
        training_contract=_strict_training_contract(),
        allow_legacy_training_contract=False,
        training_embeddings=_wide_embeddings(
            ["2025-01-02", "2025-01-03"], names=3, width=4
        ),
    )
    scoring_path = tmp_path / "strict_scoring.parquet"
    _write_embeddings(
        scoring_path,
        _wide_embeddings(["2026-01-05"], names=350, width=4),
        contract=contract,
    )
    return checkpoint, metadata, scoring_path, fm_path, vocab_path


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
    model, metadata = load_ranker_artifact(
        checkpoint,
        metadata_path,
        device="cpu",
        allow_legacy_training_contract=True,
    )
    assert model.proj.in_features == 2
    assert metadata["feature_columns"] == ["emb_0", "emb_1"]
    assert metadata["artifact_version"] == "2.0"
    assert metadata["objective"]["ndcg_ks"] == [50, 300, 350]
    assert metadata["legacy_training_contract"]


def test_save_rejects_empty_training_contract_by_default(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"training_contract\.objective"):
        _artifact(tmp_path, allow_legacy_training_contract=False)


def test_load_rejects_empty_training_contract_by_default(tmp_path: Path) -> None:
    checkpoint, metadata_path = _artifact(tmp_path)

    with pytest.raises(ValueError, match=r"training_contract\.objective"):
        load_ranker_artifact(checkpoint, metadata_path, device="cpu")


def test_strict_training_contract_round_trip(tmp_path: Path) -> None:
    checkpoint, metadata_path, _scoring, _fm, _vocab = _strict_signal_fixture(tmp_path)

    _model, metadata = load_ranker_artifact(
        checkpoint,
        metadata_path,
        device="cpu",
    )

    assert "legacy_training_contract" not in metadata
    assert metadata["training_contract"]["universe"]["mode"] == "daily_pit_file"


def test_strict_training_contract_must_match_frozen_embedding(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"representation_gate\.schema_version"):
        _artifact(
            tmp_path,
            training_contract=_strict_training_contract(),
            allow_legacy_training_contract=False,
        )


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        (("execution_contract", "verified"), False, "verified must equal True"),
        (
            ("execution_contract", "calendar_reverified"),
            False,
            "calendar_reverified must equal True",
        ),
        (
            ("execution_contract", "return_spec"),
            "close_t_close_t1",
            "return_spec must equal 'vwap_t1_vwap_t2'",
        ),
        (("representation_gate", "verified"), False, "verified must equal True"),
        (("universe", "mode"), "legacy_unverified", "mode must equal"),
        (("universe", "contract", "verified"), False, "verified must equal True"),
        (("time_split", "validation_enabled"), False, "validation_enabled"),
        (("selection", "best_val_ic"), None, "best_val_ic must be"),
    ],
)
def test_strict_training_contract_rejects_incomplete_or_unverified_fields(
    path: tuple[str, ...],
    value: object,
    expected: str,
) -> None:
    contract = deepcopy(_strict_training_contract())
    cursor = contract
    for key in path[:-1]:
        child = cursor[key]
        assert isinstance(child, dict)
        cursor = child
    cursor[path[-1]] = value

    with pytest.raises(ValueError, match=expected):
        validate_ranker_training_contract(
            contract,
            objective=json.loads(json.dumps(asdict(RankerObjectiveConfig()))),
        )


def test_strict_artifact_rejects_custom_objective(tmp_path: Path) -> None:
    custom = RankerObjectiveConfig(global_ic_weight=0.31)

    with pytest.raises(ValueError, match="must equal the frozen"):
        _artifact(
            tmp_path,
            objective=custom,
            training_contract=_strict_training_contract(),
            allow_legacy_training_contract=False,
        )


def test_v2_artifact_rejects_missing_aux_head(tmp_path: Path) -> None:
    checkpoint, metadata_path = _artifact(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    del payload["state_dict"]["aux_out.weight"]
    torch.save(payload, checkpoint)

    with pytest.raises(RuntimeError, match=r"aux_out\.weight"):
        load_ranker_artifact(
            checkpoint,
            metadata_path,
            device="cpu",
            allow_legacy_training_contract=True,
        )


def test_v2_artifact_cross_checks_objective(tmp_path: Path) -> None:
    checkpoint, metadata_path = _artifact(tmp_path)
    metadata = json.loads(metadata_path.read_text())
    metadata["objective"]["global_ic_weight"] = 0.31
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="objective does not match"):
        load_ranker_artifact(
            checkpoint,
            metadata_path,
            device="cpu",
            allow_legacy_training_contract=True,
        )


@pytest.mark.parametrize(
    ("training_end", "label_end", "expected"),
    [
        ("20250103", "2025-01-04", "canonical"),
        ("2025-01-04", "2025-01-03", "must not be after"),
    ],
)
def test_ranker_artifact_rejects_invalid_metadata_dates(
    tmp_path: Path,
    training_end: str,
    label_end: str,
    expected: str,
) -> None:
    checkpoint, metadata_path = _artifact(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["training_end_date"] = training_end
    metadata["label_end_date"] = label_end
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        load_ranker_artifact(
            checkpoint,
            metadata_path,
            allow_legacy_training_contract=True,
        )


def test_v1_artifact_requires_explicit_inference_migration(tmp_path: Path) -> None:
    checkpoint, metadata_path = _artifact(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["artifact_version"] = "1.0"
    payload["state_dict"] = {
        name: value
        for name, value in payload["state_dict"].items()
        if not name.startswith("aux_out.")
    }
    for field in ("objective", "training_contract"):
        payload.pop(field)
    torch.save(payload, checkpoint)
    metadata = json.loads(metadata_path.read_text())
    metadata["artifact_version"] = "1.0"
    for field in ("objective", "training_contract", "label_end_date"):
        metadata.pop(field)
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="explicit inference-only migration"):
        load_ranker_artifact(checkpoint, metadata_path, device="cpu")
    _model, migrated = load_ranker_artifact(
        checkpoint,
        metadata_path,
        device="cpu",
        allow_legacy_v1_inference=True,
        allow_legacy_training_contract=True,
    )
    assert migrated["legacy_inference_only"]


def test_generate_scores_without_test_panel(tmp_path: Path) -> None:
    checkpoint, metadata = _artifact(tmp_path)
    embeddings_path = tmp_path / "latest_embeddings.parquet"
    _write_embeddings(
        embeddings_path,
        _embeddings(["2026-01-05"]),
        contract=_embedding_contract(),
    )
    output = tmp_path / "delivery"

    first = generate_scores(
        embeddings_path=embeddings_path,
        ranker_path=checkpoint,
        ranker_metadata_path=metadata,
        out_dir=output,
        device="cpu",
        require_causal_representation=False,
        allow_legacy_training_contract=True,
    )
    values = pl.read_parquet(first)
    second_values = pl.read_parquet(
        generate_scores(
            embeddings_path=embeddings_path,
            ranker_path=checkpoint,
            ranker_metadata_path=metadata,
            out_dir=output,
            device="cpu",
            require_causal_representation=False,
            allow_legacy_training_contract=True,
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
    _write_embeddings(
        embeddings_path,
        _embeddings(["2025-01-03"]),
        contract=_embedding_contract(),
    )
    with pytest.raises(ValueError, match="strictly after"):
        generate_scores(
            embeddings_path=embeddings_path,
            ranker_path=checkpoint,
            ranker_metadata_path=metadata,
            out_dir=tmp_path / "delivery",
            require_causal_representation=False,
            allow_legacy_training_contract=True,
        )


def test_generate_scores_rejects_noncanonical_signal_date(tmp_path: Path) -> None:
    checkpoint, metadata = _artifact(tmp_path)
    embeddings_path = tmp_path / "embeddings.parquet"
    _write_embeddings(
        embeddings_path,
        _embeddings(["20260105"]),
        contract=_embedding_contract(),
    )

    with pytest.raises(ValueError, match="canonical YYYY-MM-DD"):
        generate_scores(
            embeddings_path=embeddings_path,
            ranker_path=checkpoint,
            ranker_metadata_path=metadata,
            out_dir=tmp_path / "delivery",
            require_causal_representation=False,
            allow_legacy_training_contract=True,
        )


@pytest.mark.parametrize(
    "contract",
    [
        _embedding_contract(pooling="last"),
        _embedding_contract(checkpoint="c" * 64),
    ],
    ids=["same-width-pooling", "same-width-checkpoint"],
)
def test_generate_scores_rejects_embedding_representation_mismatch(
    tmp_path: Path,
    contract: EmbeddingContract,
) -> None:
    checkpoint, metadata = _artifact(tmp_path)
    embeddings_path = tmp_path / "scoring.parquet"
    _write_embeddings(
        embeddings_path,
        _embeddings(["2026-01-05"]),
        contract=contract,
    )

    with pytest.raises(ValueError, match="embedding representation mismatch"):
        generate_scores(
            embeddings_path=embeddings_path,
            ranker_path=checkpoint,
            ranker_metadata_path=metadata,
            out_dir=tmp_path / "delivery",
            require_causal_representation=False,
            allow_legacy_training_contract=True,
        )


def test_generate_scores_requires_embedding_contract_by_default(tmp_path: Path) -> None:
    checkpoint, metadata = _artifact(tmp_path)
    embeddings_path = tmp_path / "legacy_embeddings.parquet"
    _embeddings(["2026-01-05"]).write_parquet(embeddings_path)

    with pytest.raises(ValueError, match="representation contract is missing"):
        generate_scores(
            embeddings_path=embeddings_path,
            ranker_path=checkpoint,
            ranker_metadata_path=metadata,
            out_dir=tmp_path / "delivery",
            require_causal_representation=False,
            allow_legacy_training_contract=True,
        )


def test_strict_ranker_requires_explicit_scoring_pit_universe(tmp_path: Path) -> None:
    checkpoint, metadata_path, embeddings_path, fm_path, vocab_path = (
        _strict_signal_fixture(tmp_path)
    )

    with pytest.raises(ValueError, match=r"scoring requires.*PIT --universe"):
        generate_scores(
            embeddings_path=embeddings_path,
            ranker_path=checkpoint,
            ranker_metadata_path=metadata_path,
            out_dir=tmp_path / "delivery",
            fm_checkpoint_path=fm_path,
            vocab_path=vocab_path,
        )


def test_strict_generate_requires_explicit_fm_and_vocab_identity(
    tmp_path: Path,
) -> None:
    checkpoint, metadata_path, embeddings_path, _fm_path, _vocab_path = (
        _strict_signal_fixture(tmp_path)
    )

    with pytest.raises(ValueError, match=r"requires explicit.*missing --fm-checkpoint"):
        generate_scores(
            embeddings_path=embeddings_path,
            ranker_path=checkpoint,
            ranker_metadata_path=metadata_path,
            out_dir=tmp_path / "delivery",
        )


@pytest.mark.parametrize(
    ("tampered", "expected"),
    [("fm", "--fm-checkpoint SHA-256"), ("vocab", "--vocab SHA-256")],
)
def test_strict_generate_rejects_identity_hash_mismatch(
    tmp_path: Path,
    tampered: str,
    expected: str,
) -> None:
    checkpoint, metadata_path, embeddings_path, fm_path, vocab_path = (
        _strict_signal_fixture(tmp_path)
    )
    target = fm_path if tampered == "fm" else vocab_path
    target.write_bytes(b"tampered")

    with pytest.raises(ValueError, match=expected):
        generate_scores(
            embeddings_path=embeddings_path,
            ranker_path=checkpoint,
            ranker_metadata_path=metadata_path,
            out_dir=tmp_path / "delivery",
            fm_checkpoint_path=fm_path,
            vocab_path=vocab_path,
        )


def test_strict_generate_accepts_verified_identity_and_pit_universe(
    tmp_path: Path,
) -> None:
    checkpoint, metadata_path, embeddings_path, fm_path, vocab_path = (
        _strict_signal_fixture(tmp_path)
    )
    universe_path = tmp_path / "universe.parquet"
    pl.DataFrame(
        {
            "date": ["2026-01-05"] * 350,
            "symbol": [f"{index + 1:06d}" for index in range(350)],
            "asof_date": ["2026-01-05"] * 350,
            "universe_policy": ["liquid_a_share_v1"] * 350,
        }
    ).write_parquet(universe_path)

    scores_path = generate_scores(
        embeddings_path=embeddings_path,
        ranker_path=checkpoint,
        ranker_metadata_path=metadata_path,
        out_dir=tmp_path / "delivery",
        fm_checkpoint_path=fm_path,
        vocab_path=vocab_path,
        universe_path=universe_path,
    )

    assert pl.read_parquet(scores_path).height == 350
