"""Verify one lineage across the accepted FM, its data, and its embeddings."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import date as calendar_date
from pathlib import Path
from typing import Any

import polars as pl
import torch
import yaml

from quant_fm.downstream.representation import (
    STRICT_BOOK_STATE_TIMING,
    STRICT_CHUNK_STRIDE,
    STRICT_CONTEXT,
    STRICT_POOLING_VERSION,
    STRICT_SCHEMA_VERSION,
    validate_strict_topk_representation,
)
from quant_fm.embedding.contract import (
    STRICT_EVENT_ORDERING_VERSION,
    STRICT_FEATURE_TRANSFORM_VERSION,
    assert_embedding_contract_compatible,
    load_embedding_contract,
    validate_embedding_columns,
)
from quant_fm.embedding.pooling_spec import DEFAULT_V2_MULTI_SCALE_OUTPUTS
from quant_fm.manifest.build_manifest import Manifest
from quant_fm.manifest.validation import sha256_file
from quant_fm.monitoring.acceptance import (
    compare_pretrain_evaluations,
    validate_pretrain_acceptance,
)
from quant_fm.pretrain.data_contract import (
    build_pretrain_data_contract,
    load_checkpoint_contract,
    validate_checkpoint_data_contract,
)
from quant_fm.pretrain.train import validate_pretrain_split_contract
from quant_fm.tokenizer.artifact_contract import stable_vocab_sha256
from quant_fm.tokenizer.vocab import Vocab
from quant_fm.tokenizer.vocab_v2 import VocabV2

LINEAGE_REPORT_VERSION = "strict_pretrain_lineage_v1"


def _load_json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{context} is not valid JSON: {path}"
        raise ValueError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"{context} must be a JSON object: {path}"
        raise TypeError(msg)
    return payload


def _existing_path(value: object, *, reference: Path, context: str) -> Path:
    if not isinstance(value, str) or not value:
        msg = f"{context} must be a non-empty path"
        raise ValueError(msg)
    raw = Path(value)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(reference.parent / raw)
    existing = {candidate.resolve() for candidate in candidates if candidate.is_file()}
    if not existing:
        msg = f"{context} does not exist: {value}"
        raise FileNotFoundError(msg)
    if len(existing) != 1:
        msg = f"{context} is ambiguous relative to {reference}: {value}"
        raise ValueError(msg)
    return existing.pop()


def _load_vocab(path: Path) -> Vocab | VocabV2:
    payload = _load_json_object(path, context="pretrain vocab")
    if payload.get("vocab_version") == VocabV2.VOCAB_VERSION:
        return VocabV2.load(path)
    return Vocab.load(path)


def _canonical_date(value: object, *, context: str) -> str:
    """Return one canonical ISO date or reject ambiguous date-like values."""
    if not isinstance(value, str) or not value:
        msg = f"{context} must be a non-empty YYYY-MM-DD string"
        raise ValueError(msg)
    try:
        parsed = calendar_date.fromisoformat(value)
    except ValueError as exc:
        msg = f"{context} must be a canonical YYYY-MM-DD date: {value!r}"
        raise ValueError(msg) from exc
    if parsed.isoformat() != value:
        msg = f"{context} must be a canonical YYYY-MM-DD date: {value!r}"
        raise ValueError(msg)
    return value


def _validate_manifest_date_contract(manifest: Manifest) -> None:
    """Require every manifest row to agree with canonical split boundaries."""
    if manifest.train_end is None or manifest.val_end is None:
        msg = "accepted FM manifest must declare train_end and val_end"
        raise ValueError(msg)
    train_end = _canonical_date(
        manifest.train_end,
        context="accepted FM manifest train_end",
    )
    val_end = _canonical_date(
        manifest.val_end,
        context="accepted FM manifest val_end",
    )
    if train_end >= val_end:
        msg = "accepted FM manifest train_end must be earlier than val_end"
        raise ValueError(msg)
    for field, value in (
        ("purge_days", manifest.purge_days),
        ("embargo_days", manifest.embargo_days),
    ):
        if type(value) is not int or value < 0:
            msg = f"accepted FM manifest {field} must be a non-negative integer"
            raise ValueError(msg)

    for index, shard in enumerate(manifest.shards):
        shard_date = _canonical_date(
            shard.date,
            context=f"accepted FM manifest shard[{index}].date",
        )
        if shard_date <= train_end:
            expected_split = "train"
        elif shard_date <= val_end:
            expected_split = "val"
        else:
            expected_split = "test"
        if shard.split != expected_split:
            msg = (
                "accepted FM manifest shard split disagrees with its frozen "
                f"boundaries: date={shard_date}, split={shard.split!r}, "
                f"expected={expected_split!r}"
            )
            raise ValueError(msg)


def _validate_vocab_dates(vocab: VocabV2) -> None:
    """Require auditable, canonical fit dates for strict accepted vocabularies."""
    if not vocab.fit_dates:
        msg = "accepted FM vocab must record at least one fit date"
        raise ValueError(msg)
    for index, value in enumerate(vocab.fit_dates):
        _canonical_date(value, context=f"accepted FM vocab fit_dates[{index}]")


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    """Load enough checkpoint metadata to cross-check the external sidecar."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        msg = f"accepted FM checkpoint cannot be loaded: {path}"
        raise ValueError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"accepted FM checkpoint payload must be an object: {path}"
        raise TypeError(msg)
    return payload


def _validate_strict_checkpoint_representation(
    checkpoint_payload: Mapping[str, Any],
    checkpoint_contract: Mapping[str, Any],
    vocab: VocabV2,
) -> None:
    """Prove the checkpoint itself froze the representation claimed downstream."""
    if checkpoint_payload.get("fm_artifact_version") != VocabV2.VOCAB_VERSION:
        msg = "strict Top-K checkpoint must declare fm_artifact_version='2.0'"
        raise ValueError(msg)
    config = checkpoint_payload.get("config")
    if not isinstance(config, Mapping):
        msg = "strict Top-K checkpoint config must be an object"
        raise TypeError(msg)
    expected_config: dict[str, object] = {
        "vocab_version": VocabV2.VOCAB_VERSION,
        "vocab_sha256": stable_vocab_sha256(vocab),
        "schema_version": STRICT_SCHEMA_VERSION,
        "event_ordering_version": STRICT_EVENT_ORDERING_VERSION,
        "feature_transform_version": STRICT_FEATURE_TRANSFORM_VERSION,
        "book_state_timing": STRICT_BOOK_STATE_TIMING,
        "context_horizon": STRICT_CONTEXT,
        "pooling_version": STRICT_POOLING_VERSION,
        "pooling_method": "multi_scale",
        "pooling_outputs": [*DEFAULT_V2_MULTI_SCALE_OUTPUTS],
        "pooling_stride": STRICT_CHUNK_STRIDE,
        "field_specs": [spec.to_dict() for spec in vocab.field_specs],
    }
    mismatches = {
        field: {"checkpoint": config.get(field), "required": expected}
        for field, expected in expected_config.items()
        if config.get(field) != expected
    }
    sidecar_model = checkpoint_contract.get("model_data_contract")
    if not isinstance(sidecar_model, Mapping):
        msg = "strict Top-K checkpoint model_data_contract must be an object"
        raise TypeError(msg)
    for field in (
        "vocab_version",
        "vocab_sha256",
        "schema_version",
        "event_ordering_version",
        "feature_transform_version",
    ):
        expected = expected_config[field]
        if sidecar_model.get(field) != expected:
            mismatches[f"model_data_contract.{field}"] = {
                "checkpoint": sidecar_model.get(field),
                "required": expected,
            }
    if mismatches:
        msg = f"accepted FM checkpoint is not the strict V2 Top-K representation: {mismatches}"
        raise ValueError(msg)


def _embedding_artifact(
    path: Path,
    *,
    contract: Any,
    context: str,
) -> dict[str, Any]:
    """Bind one validated embedding parquet's live bytes, keys, and date range."""
    path = Path(path).resolve()
    if not path.is_file():
        msg = f"{context} embedding parquet is missing: {path}"
        raise FileNotFoundError(msg)
    schema = list(pl.read_parquet_schema(path).names())
    missing = sorted({"date", "symbol"} - set(schema))
    if missing:
        msg = f"{context} embeddings are missing key columns: {missing}"
        raise ValueError(msg)
    validate_embedding_columns(schema, contract, context=context)
    keys = pl.read_parquet(path, columns=["date", "symbol"])
    if keys.is_empty():
        msg = f"{context} embeddings are empty"
        raise ValueError(msg)
    keys = keys.select(
        pl.col("date").cast(pl.Utf8, strict=False),
        pl.col("symbol").cast(pl.Utf8, strict=False),
    )
    if keys.filter(pl.any_horizontal(pl.all().is_null())).height:
        msg = f"{context} embeddings contain null date/symbol keys"
        raise ValueError(msg)
    if keys.filter(
        (pl.col("date").str.strip_chars() == "")
        | (pl.col("symbol").str.strip_chars() == "")
    ).height:
        msg = f"{context} embeddings contain blank date/symbol keys"
        raise ValueError(msg)
    if keys.select(pl.struct(["date", "symbol"]).is_duplicated().any()).item():
        msg = f"{context} embeddings contain duplicate (date, symbol) keys"
        raise ValueError(msg)
    raw_dates = keys["date"].unique().to_list()
    dates = sorted(
        {
            _canonical_date(str(value), context=f"{context} embedding date")
            for value in raw_dates
        }
    )
    if not dates:  # pragma: no cover - non-empty keys imply at least one date
        msg = f"{context} embeddings contain no dates"
        raise ValueError(msg)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": keys.height,
        "dates": len(dates),
        "date_start": dates[0],
        "date_end": dates[-1],
    }


def _reverify_acceptance_sources(
    acceptance_path: Path,
    acceptance: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    candidate_path = _existing_path(
        acceptance["candidate"],
        reference=acceptance_path,
        context="accepted candidate evaluation",
    )
    baseline_path = _existing_path(
        acceptance["baseline"],
        reference=acceptance_path,
        context="accepted baseline evaluation",
    )
    recomputed = compare_pretrain_evaluations(
        candidate_path,
        baseline_path,
        noninferiority_tolerance=float(acceptance["noninferiority_tolerance"]),
    )
    immutable_fields = (
        "candidate",
        "candidate_sha256",
        "baseline",
        "baseline_sha256",
        "validation_plan_source_fingerprint",
        "validation_windows",
        "primary_metric",
        "candidate_value",
        "baseline_value",
        "relative_change",
        "noninferiority_tolerance",
        "accepted",
        "decision",
        "per_field_ce",
    )
    mismatches = {
        field: {"acceptance": acceptance[field], "recomputed": recomputed[field]}
        for field in immutable_fields
        if acceptance[field] != recomputed[field]
    }
    if mismatches:
        msg = f"pretrain acceptance no longer matches its source reports: {mismatches}"
        raise ValueError(msg)
    return candidate_path, _load_json_object(
        candidate_path,
        context="accepted candidate evaluation",
    )


def validate_pretrain_lineage(
    *,
    acceptance_path: Path,
    train_embeddings: Path,
    oos_embeddings: Path,
    expected_training_end: str | None = None,
) -> dict[str, Any]:
    """Return a fail-closed lineage report for strict Top-K retraining."""
    if expected_training_end is not None:
        _canonical_date(
            expected_training_end,
            context="expected FM training end",
        )

    acceptance_path = Path(acceptance_path).resolve()
    acceptance = validate_pretrain_acceptance(acceptance_path)
    candidate_path, candidate = _reverify_acceptance_sources(
        acceptance_path,
        acceptance,
    )
    if candidate.get("split") != "val":
        msg = "accepted candidate evaluation must use split=val"
        raise ValueError(msg)
    checkpoint_path = _existing_path(
        candidate.get("checkpoint"),
        reference=candidate_path,
        context="accepted FM checkpoint",
    )
    config_path = _existing_path(
        candidate.get("config"),
        reference=candidate_path,
        context="accepted FM config",
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        msg = f"accepted FM config has no data mapping: {config_path}"
        raise TypeError(msg)
    manifest_path = _existing_path(
        config["data"].get("manifest"),
        reference=config_path,
        context="accepted FM manifest",
    )
    vocab_path = _existing_path(
        config["data"].get("vocab"),
        reference=config_path,
        context="accepted FM vocab",
    )
    manifest = Manifest.load(manifest_path)
    vocab = _load_vocab(vocab_path)
    if not isinstance(vocab, VocabV2):
        msg = (
            "strict Top-K lineage requires a genuine VocabV2 artifact; a V1 vocab "
            "cannot become V2 by changing schema_version"
        )
        raise TypeError(msg)
    _validate_manifest_date_contract(manifest)
    _validate_vocab_dates(vocab)
    split_contract = validate_pretrain_split_contract(
        manifest,
        vocab,
        require_validation=True,
        min_validation_dates=int(config["data"].get("min_validation_dates", 5)),
        min_test_dates=int(config["data"].get("min_test_dates", 5)),
    )
    expected_data_contract = build_pretrain_data_contract(
        manifest_path=manifest_path,
        manifest=manifest,
        vocab_path=vocab_path,
        vocab=vocab,
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint_contract = load_checkpoint_contract(
        checkpoint_path,
        expected_checkpoint_sha256=checkpoint_sha256,
        required=True,
    )
    if checkpoint_contract is None:  # pragma: no cover - required=True invariant
        msg = "accepted FM checkpoint contract is unexpectedly missing"
        raise RuntimeError(msg)
    checkpoint_payload = _load_checkpoint_payload(checkpoint_path)
    _validate_strict_checkpoint_representation(
        checkpoint_payload,
        checkpoint_contract,
        vocab,
    )
    validate_checkpoint_data_contract(
        checkpoint_contract,
        checkpoint_payload=checkpoint_payload,
        vocab=vocab,
        expected_pretrain_data_contract=expected_data_contract,
    )

    train_contract = load_embedding_contract(
        Path(train_embeddings), required=True, require_vocab=True
    )
    score_contract = load_embedding_contract(
        Path(oos_embeddings), required=True, require_vocab=True
    )
    if train_contract is None or score_contract is None:  # pragma: no cover
        msg = "strict embedding contracts are unexpectedly missing"
        raise RuntimeError(msg)
    validate_strict_topk_representation(
        train_contract,
        context="training embeddings in pretrain lineage",
    )
    validate_strict_topk_representation(
        score_contract,
        context="OOS embeddings in pretrain lineage",
    )
    assert_embedding_contract_compatible(
        train_contract,
        score_contract,
        context="pretrain lineage train vs OOS embeddings",
    )
    vocab_sha256 = stable_vocab_sha256(vocab)
    identity_mismatches: dict[str, dict[str, str | None]] = {}
    for name, actual, expected in (
        (
            "fm_checkpoint_sha256",
            train_contract.fm_checkpoint_sha256,
            checkpoint_sha256,
        ),
        ("vocab_sha256", train_contract.vocab_sha256, vocab_sha256),
    ):
        if actual != expected:
            identity_mismatches[name] = {"embedding": actual, "accepted": expected}
    if identity_mismatches:
        msg = f"embeddings do not come from the accepted FM lineage: {identity_mismatches}"
        raise ValueError(msg)

    derived_training_end = expected_data_contract.get("effective_training_end")
    if not isinstance(derived_training_end, str):
        msg = "accepted FM data contract has no effective_training_end"
        raise TypeError(msg)
    _canonical_date(
        derived_training_end,
        context="accepted FM effective_training_end",
    )
    if (
        expected_training_end is not None
        and expected_training_end != derived_training_end
    ):
        msg = (
            "declared FM training end does not match accepted checkpoint lineage: "
            f"declared={expected_training_end}, derived={derived_training_end}"
        )
        raise ValueError(msg)

    train_artifact = _embedding_artifact(
        Path(train_embeddings),
        contract=train_contract,
        context="training",
    )
    oos_artifact = _embedding_artifact(
        Path(oos_embeddings),
        contract=score_contract,
        context="OOS",
    )
    train_end = str(train_artifact["date_end"])
    oos_start = str(oos_artifact["date_start"])
    if train_end >= oos_start:
        msg = (
            "training and OOS embedding periods overlap: "
            f"training_end={train_end}, oos_start={oos_start}"
        )
        raise ValueError(msg)
    if derived_training_end >= oos_start:
        msg = (
            "accepted FM training horizon overlaps OOS embeddings: "
            f"training_end={derived_training_end}, oos_start={oos_start}"
        )
        raise ValueError(msg)

    return {
        "format_version": LINEAGE_REPORT_VERSION,
        "status": "verified",
        "effective_training_end": derived_training_end,
        "declared_training_end": expected_training_end,
        "training_embedding_start_date": train_artifact["date_start"],
        "training_embedding_end_date": train_end,
        "oos_start_date": oos_start,
        "oos_end_date": oos_artifact["date_end"],
        "training_embeddings": train_artifact,
        "oos_embeddings": oos_artifact,
        "acceptance": {
            "path": str(acceptance_path),
            "sha256": sha256_file(acceptance_path),
            "validation_plan_source_fingerprint": acceptance[
                "validation_plan_source_fingerprint"
            ],
        },
        "candidate_evaluation": {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
        },
        "fm_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": expected_data_contract["manifest_sha256"],
            "split_contract": split_contract,
        },
        "vocab": {
            "path": str(vocab_path),
            "sha256": vocab_sha256,
        },
        "pretrain_data_contract": expected_data_contract,
        "embedding_contract_fingerprint": train_contract.fingerprint(),
    }


def main() -> None:
    """Validate and atomically persist the strict FM lineage report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--train-embeddings", type=Path, required=True)
    parser.add_argument("--oos-embeddings", type=Path, required=True)
    parser.add_argument("--expected-training-end")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = validate_pretrain_lineage(
            acceptance_path=args.acceptance,
            train_embeddings=args.train_embeddings,
            oos_embeddings=args.oos_embeddings,
            expected_training_end=args.expected_training_end,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
