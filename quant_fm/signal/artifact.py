"""冻结 Ranker 的可复现保存与加载。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import UTC, datetime
from datetime import date as calendar_date
from typing import TYPE_CHECKING, Any

import torch

from quant_fm.downstream.representation import (
    STRICT_BOOK_STATE_TIMING,
    STRICT_CHUNK_STRIDE,
    STRICT_CONTEXT,
    STRICT_POOLING_VERSION,
    STRICT_SCHEMA_VERSION,
)
from quant_fm.downstream.return_spec import (
    AFTER_CLOSE_AVAILABILITY,
    EXECUTION_CONTRACT_VERSION,
)
from quant_fm.downstream.train_ranker import (
    CrossSectionalRanker,
    RankerConfig,
    RankerObjectiveConfig,
)
from quant_fm.downstream.universe import PIT_UNIVERSE_CONTRACT_VERSION
from quant_fm.embedding.contract import (
    CAUSAL_OVERLAPPING_ENCODER,
    EMBEDDING_CONTRACT_VERSION,
    STRICT_EVENT_ORDERING_VERSION,
    STRICT_FEATURE_TRANSFORM_VERSION,
    EmbeddingContract,
)

if TYPE_CHECKING:
    from pathlib import Path

ARTIFACT_VERSION = "2.0"
_LEGACY_ARTIFACT_VERSION = "1.0"
SIGNAL_FEATURE_TARGET_SPEC_VERSION = "strict_exec_percentile_mad_head_gain_v1"
PRODUCTION_RETURN_SPEC = "vwap_t1_vwap_t2"


def _json_mapping(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        msg = f"{context} must be a JSON object"
        raise ValueError(msg)  # noqa: TRY004 - contract violations use one API error
    try:
        payload = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        msg = f"{context} must contain only finite JSON values"
        raise ValueError(msg) from exc
    if not isinstance(payload, dict):  # pragma: no cover - guarded above
        msg = f"{context} must be a JSON object"
        raise ValueError(msg)  # noqa: TRY004 - contract violations use one API error
    return payload


def _required_mapping(
    payload: dict[str, Any], field: str, *, context: str
) -> dict[str, Any]:
    value = payload.get(field)
    if not isinstance(value, dict):
        msg = f"{context}.{field} must be a JSON object"
        raise ValueError(msg)  # noqa: TRY004 - contract violations use one API error
    return value


def _required_int(
    payload: dict[str, Any],
    field: str,
    *,
    context: str,
    minimum: int,
) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        msg = f"{context}.{field} must be an integer >= {minimum}"
        raise ValueError(msg)
    return value


def _required_finite_number(
    payload: dict[str, Any], field: str, *, context: str
) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{context}.{field} must be a finite number"
        raise ValueError(msg)  # noqa: TRY004 - contract violations use one API error
    number = float(value)
    if not math.isfinite(number):
        msg = f"{context}.{field} must be a finite number"
        raise ValueError(msg)
    return number


def _required_iso_date(payload: dict[str, Any], field: str, *, context: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        msg = f"{context}.{field} must be an ISO date"
        raise ValueError(msg)  # noqa: TRY004 - contract violations use one API error
    try:
        parsed = calendar_date.fromisoformat(value)
    except ValueError as exc:
        msg = f"{context}.{field} must be an ISO date"
        raise ValueError(msg) from exc
    if parsed.isoformat() != value:
        msg = f"{context}.{field} must use canonical YYYY-MM-DD ISO format"
        raise ValueError(msg)
    return value


def _validate_artifact_dates(training_end_date: object, label_end_date: object) -> None:
    values = {
        "training_end_date": training_end_date,
        "label_end_date": label_end_date,
    }
    parsed: dict[str, calendar_date] = {}
    for field in values:
        text = _required_iso_date(values, field, context="ranker metadata")
        parsed[field] = calendar_date.fromisoformat(text)
    if parsed["training_end_date"] > parsed["label_end_date"]:
        msg = "ranker metadata training_end_date must not be after label_end_date"
        raise ValueError(msg)


def _require_exact(
    payload: dict[str, Any], field: str, expected: object, *, context: str
) -> None:
    actual = payload.get(field)
    if actual != expected or type(actual) is not type(expected):
        msg = f"{context}.{field} must equal {expected!r}; got {actual!r}"
        raise ValueError(msg)


def _validate_width_stats(payload: dict[str, Any], *, context: str) -> None:
    rows = _required_int(payload, "rows", context=context, minimum=1)
    days = _required_int(payload, "days", context=context, minimum=1)
    date_min = _required_iso_date(payload, "date_min", context=context)
    date_max = _required_iso_date(payload, "date_max", context=context)
    names_min = _required_int(payload, "names_min", context=context, minimum=1)
    names_median = _required_finite_number(payload, "names_median", context=context)
    names_max = _required_int(payload, "names_max", context=context, minimum=1)
    if date_min > date_max:
        msg = f"{context}.date_min must not be after date_max"
        raise ValueError(msg)
    if not names_min <= names_median <= names_max:
        msg = f"{context} has inconsistent names_min/names_median/names_max"
        raise ValueError(msg)
    if not days * names_min <= rows <= days * names_max:
        msg = f"{context}.rows is inconsistent with the per-day width bounds"
        raise ValueError(msg)


def validate_ranker_training_contract(
    training_contract: dict[str, Any],
    *,
    objective: dict[str, Any],
    embedding_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and normalise the fail-closed production Ranker contract."""
    contract = _json_mapping(training_contract, context="ranker training_contract")
    objective_payload = _json_mapping(objective, context="ranker objective")
    production_objective = _json_mapping(
        asdict(RankerObjectiveConfig()), context="production ranker objective"
    )
    if objective_payload != production_objective:
        msg = (
            "strict production ranker objective must equal the frozen "
            "RankerObjectiveConfig defaults; use the explicit legacy/research "
            "training-contract override for custom objectives"
        )
        raise ValueError(msg)
    embedded_objective = _required_mapping(
        contract, "objective", context="ranker training_contract"
    )
    if embedded_objective != objective_payload:
        msg = "training_contract.objective does not match top-level objective"
        raise ValueError(msg)

    execution = _required_mapping(
        contract, "execution_contract", context="ranker training_contract"
    )
    execution_context = "ranker training_contract.execution_contract"
    execution_expected: dict[str, object] = {
        "verified": True,
        "mode": "strict_execution_panel",
        "execution_contract_version": EXECUTION_CONTRACT_VERSION,
        "return_spec": PRODUCTION_RETURN_SPEC,
        "signal_availability": AFTER_CLOSE_AVAILABILITY,
        "entry_day_lag": 1,
        "exit_day_lag": 2,
        "entry_price_field": "vwap",
        "exit_price_field": "vwap",
        "calendar_reverified": True,
    }
    for field, expected in execution_expected.items():
        _require_exact(execution, field, expected, context=execution_context)
    calendar_hash = execution.get("trading_calendar_sha256")
    if (
        not isinstance(calendar_hash, str)
        or len(calendar_hash) != 64
        or any(char not in "0123456789abcdef" for char in calendar_hash.lower())
    ):
        msg = f"{execution_context}.trading_calendar_sha256 must be a SHA-256"
        raise ValueError(msg)
    _required_int(
        execution, "calendar_date_count", context=execution_context, minimum=1
    )

    representation = _required_mapping(
        contract, "representation_gate", context="ranker training_contract"
    )
    representation_context = "ranker training_contract.representation_gate"
    representation_expected: dict[str, object] = {
        "verified": True,
        "format_version": EMBEDDING_CONTRACT_VERSION,
        "schema_version": STRICT_SCHEMA_VERSION,
        "book_state_timing": STRICT_BOOK_STATE_TIMING,
        "event_ordering_version": STRICT_EVENT_ORDERING_VERSION,
        "feature_transform_version": STRICT_FEATURE_TRANSFORM_VERSION,
        "encoder_semantics": CAUSAL_OVERLAPPING_ENCODER,
        "context": STRICT_CONTEXT,
        "chunk_stride": STRICT_CHUNK_STRIDE,
        "pooling_version": STRICT_POOLING_VERSION,
    }
    for field, expected in representation_expected.items():
        _require_exact(representation, field, expected, context=representation_context)
    if embedding_contract is not None:
        embedding_payload = _json_mapping(
            embedding_contract, context="ranker embedding_contract"
        )
        for field in representation_expected:
            if field == "verified":
                continue
            if embedding_payload.get(field) != representation[field]:
                msg = (
                    f"training representation_gate.{field} does not match "
                    "the frozen ranker embedding_contract"
                )
                raise ValueError(msg)

    _require_exact(
        contract,
        "feature_target_spec_version",
        SIGNAL_FEATURE_TARGET_SPEC_VERSION,
        context="ranker training_contract",
    )

    universe = _required_mapping(
        contract, "universe", context="ranker training_contract"
    )
    universe_context = "ranker training_contract.universe"
    _require_exact(universe, "mode", "daily_pit_file", context=universe_context)
    universe_hash = universe.get("sha256")
    if (
        not isinstance(universe_hash, str)
        or len(universe_hash) != 64
        or any(char not in "0123456789abcdef" for char in universe_hash.lower())
    ):
        msg = f"{universe_context}.sha256 must be a SHA-256"
        raise ValueError(msg)
    daily_names_min = _required_int(
        universe, "daily_names_min", context=universe_context, minimum=1
    )
    required_names = max(int(value) for value in objective_payload["ndcg_ks"])
    if daily_names_min < required_names:
        msg = (
            f"{universe_context}.daily_names_min={daily_names_min} is below "
            f"the production Top-K cutoff {required_names}"
        )
        raise ValueError(msg)
    pit_contract = _required_mapping(universe, "contract", context=universe_context)
    pit_context = f"{universe_context}.contract"
    for field, expected in {
        "format_version": PIT_UNIVERSE_CONTRACT_VERSION,
        "verified": True,
        "asof_rule": "asof_date_lte_signal_date",
    }.items():
        _require_exact(pit_contract, field, expected, context=pit_context)
    policy = pit_contract.get("policy")
    if not isinstance(policy, str) or not policy.strip():
        msg = f"{pit_context}.policy must be a non-empty stable policy identifier"
        raise ValueError(msg)
    required_dates = _required_int(
        pit_contract, "required_dates", context=pit_context, minimum=1
    )
    pit_stats = _required_mapping(pit_contract, "stats", context=pit_context)
    _validate_width_stats(pit_stats, context=f"{pit_context}.stats")
    if pit_stats["days"] != required_dates:
        msg = f"{pit_context}.required_dates must equal contract.stats.days"
        raise ValueError(msg)
    retained_stats = _required_mapping(
        universe, "retained_training_features", context=universe_context
    )
    _validate_width_stats(
        retained_stats, context=f"{universe_context}.retained_training_features"
    )
    if retained_stats["names_min"] != daily_names_min:
        msg = (
            f"{universe_context}.daily_names_min must equal "
            "retained_training_features.names_min"
        )
        raise ValueError(msg)

    split = _required_mapping(
        contract, "time_split", context="ranker training_contract"
    )
    split_context = "ranker training_contract.time_split"
    _require_exact(split, "validation_enabled", True, context=split_context)
    available_days = _required_int(
        split, "available_days", context=split_context, minimum=1
    )
    train_days = _required_int(split, "train_days", context=split_context, minimum=1)
    purge_days = _required_int(split, "purge_days", context=split_context, minimum=2)
    val_days = _required_int(split, "val_days", context=split_context, minimum=1)
    if available_days != train_days + purge_days + val_days:
        msg = (
            f"{split_context}.available_days must equal "
            "train_days + purge_days + val_days"
        )
        raise ValueError(msg)
    train_end = _required_iso_date(split, "train_end", context=split_context)
    val_start = _required_iso_date(split, "val_start", context=split_context)
    val_end = _required_iso_date(split, "val_end", context=split_context)
    if not train_end < val_start <= val_end:
        msg = f"{split_context} must be chronological and purged"
        raise ValueError(msg)

    selection = _required_mapping(
        contract, "selection", context="ranker training_contract"
    )
    selection_context = "ranker training_contract.selection"
    _required_int(selection, "best_epoch", context=selection_context, minimum=0)
    best_ic = _required_finite_number(
        selection, "best_val_ic", context=selection_context
    )
    best_ndcg = _required_finite_number(
        selection, "best_val_ndcg", context=selection_context
    )
    best_score = _required_finite_number(
        selection, "best_selection_score", context=selection_context
    )
    stopped_early = selection.get("stopped_early")
    if not isinstance(stopped_early, bool):
        msg = f"{selection_context}.stopped_early must be a boolean"
        raise ValueError(msg)  # noqa: TRY004 - contract violations use one API error
    expected_score = (
        float(objective_payload["head_weight"]) * best_ndcg
        + float(objective_payload["global_ic_weight"]) * best_ic
    )
    if not math.isclose(best_score, expected_score, rel_tol=1e-6, abs_tol=1e-8):
        msg = (
            f"{selection_context}.best_selection_score does not match the "
            "frozen selection objective"
        )
        raise ValueError(msg)
    return contract


def _model_config(model: CrossSectionalRanker) -> RankerConfig:
    row_layers = [layer for layer in model.layers if hasattr(layer, "net")]
    attention_layers = [layer for layer in model.layers if hasattr(layer, "attn")]
    return RankerConfig(
        in_dim=model.proj.in_features,
        hidden=model.proj.out_features,
        depth=len(row_layers),
        n_heads=attention_layers[0].attn.num_heads if attention_layers else 4,
        dropout=float(row_layers[0].net[2].p) if row_layers else 0.0,
        use_attention=bool(attention_layers),
    )


def save_ranker_artifact(
    model: CrossSectionalRanker,
    checkpoint_path: Path,
    metadata_path: Path,
    *,
    feature_columns: list[str],
    training_end_date: str,
    label_end_date: str,
    seed: int,
    objective: RankerObjectiveConfig,
    embedding_contract: EmbeddingContract | dict[str, Any] | None,
    allow_legacy_embedding_contract: bool = False,
    allow_legacy_training_contract: bool = False,
    history: list[dict[str, float | int | None]] | list[float] | None = None,
    training_contract: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> None:
    """原子保存冻结权重及不含机器绝对路径的元数据。"""
    if not feature_columns:
        msg = "feature_columns must not be empty"
        raise ValueError(msg)
    _validate_artifact_dates(training_end_date, label_end_date)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = _model_config(model)
    objective.validate()
    objective_payload = json.loads(json.dumps(asdict(objective)))
    if embedding_contract is None:
        if not allow_legacy_embedding_contract:
            msg = "strict ranker artifact requires an embedding representation contract"
            raise ValueError(msg)
        embedding_payload = None
    else:
        parsed_embedding_contract = (
            embedding_contract
            if isinstance(embedding_contract, EmbeddingContract)
            else EmbeddingContract.from_dict(embedding_contract, require_vocab=True)
        )
        parsed_embedding_contract.validate(require_vocab=True)
        embedding_payload = parsed_embedding_contract.to_dict()
    raw_training_contract = training_contract or {}
    contract_payload = (
        _json_mapping(raw_training_contract, context="ranker training_contract")
        if allow_legacy_training_contract
        else validate_ranker_training_contract(
            raw_training_contract,
            objective=objective_payload,
            embedding_contract=embedding_payload,
        )
    )
    if len(feature_columns) != cfg.in_dim:
        msg = (
            f"feature column count {len(feature_columns)} does not match "
            f"ranker in_dim {cfg.in_dim}"
        )
        raise ValueError(msg)
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "config": asdict(cfg),
        "objective": objective_payload,
        "embedding_contract": embedding_payload,
        "training_contract": contract_payload,
        "feature_columns": feature_columns,
        "state_dict": model.state_dict(),
    }
    checkpoint_tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, checkpoint_tmp)
    checkpoint_tmp.replace(checkpoint_path)

    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "training_end_date": training_end_date,
        "label_end_date": label_end_date,
        "seed": seed,
        "feature_columns": feature_columns,
        "config": asdict(cfg),
        "objective": objective_payload,
        "embedding_contract": embedding_payload,
        "training_contract": contract_payload,
        "training_history": history or [],
        "provenance": provenance or {},
    }
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_tmp.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metadata_tmp.replace(metadata_path)


def load_ranker_artifact(
    checkpoint_path: Path,
    metadata_path: Path,
    *,
    device: str = "cpu",
    allow_legacy_v1_inference: bool = False,
    allow_legacy_embedding_contract: bool = False,
    allow_legacy_training_contract: bool = False,
) -> tuple[CrossSectionalRanker, dict[str, Any]]:
    """加载 Ranker，并交叉校验权重与 sidecar 元数据。"""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    versions = {metadata.get("artifact_version"), payload.get("artifact_version")}
    if len(versions) != 1:
        msg = "ranker checkpoint and metadata artifact versions differ"
        raise ValueError(msg)
    version = versions.pop()
    if version == _LEGACY_ARTIFACT_VERSION and not allow_legacy_v1_inference:
        msg = "v1 ranker artifact requires explicit inference-only migration"
        raise ValueError(msg)
    if version not in {ARTIFACT_VERSION, _LEGACY_ARTIFACT_VERSION}:
        msg = f"unsupported ranker artifact version: {version}"
        raise ValueError(msg)
    required = {"config", "feature_columns"}
    if not required <= metadata.keys() or not required <= payload.keys():
        msg = "ranker artifact is missing config or feature_columns"
        raise ValueError(msg)
    if metadata["config"] != payload["config"]:
        msg = "ranker checkpoint config does not match metadata"
        raise ValueError(msg)
    if metadata["feature_columns"] != payload["feature_columns"]:
        msg = "ranker feature columns do not match metadata"
        raise ValueError(msg)
    if "training_end_date" not in metadata:
        msg = "ranker metadata is missing training_end_date"
        raise ValueError(msg)
    _required_iso_date(metadata, "training_end_date", context="ranker metadata")
    if len(payload["feature_columns"]) != payload["config"]["in_dim"]:
        msg = "ranker feature column count does not match configured in_dim"
        raise ValueError(msg)
    if version == ARTIFACT_VERSION:
        for field in ("objective", "training_contract"):
            if field not in metadata or field not in payload:
                msg = f"v2 ranker artifact is missing {field}"
                raise ValueError(msg)
            if metadata[field] != payload[field]:
                msg = f"ranker checkpoint {field} does not match metadata"
                raise ValueError(msg)
        if "label_end_date" not in metadata:
            msg = "v2 ranker metadata is missing label_end_date"
            raise ValueError(msg)
        _validate_artifact_dates(
            metadata["training_end_date"],
            metadata["label_end_date"],
        )
        RankerObjectiveConfig(**payload["objective"]).validate()
        if allow_legacy_training_contract:
            metadata = {**metadata, "legacy_training_contract": True}
        else:
            validate_ranker_training_contract(
                payload["training_contract"],
                objective=payload["objective"],
                embedding_contract=payload.get("embedding_contract"),
            )
    elif not allow_legacy_training_contract:
        msg = (
            "v1 ranker artifact has no production training contract; enable the "
            "explicit legacy training-contract diagnostic override"
        )
        raise ValueError(msg)
    else:
        metadata = {**metadata, "legacy_training_contract": True}
    embedding_values = {
        "metadata": metadata.get("embedding_contract"),
        "checkpoint": payload.get("embedding_contract"),
    }
    if embedding_values["metadata"] != embedding_values["checkpoint"]:
        msg = "ranker checkpoint embedding contract does not match metadata"
        raise ValueError(msg)
    if embedding_values["checkpoint"] is None:
        if not allow_legacy_embedding_contract:
            msg = "ranker artifact is missing an embedding representation contract"
            raise ValueError(msg)
        metadata = {**metadata, "legacy_embedding_contract": True}
    else:
        EmbeddingContract.from_dict(
            embedding_values["checkpoint"],
            require_vocab=True,
        )

    model = CrossSectionalRanker(RankerConfig(**payload["config"])).to(
        torch.device(device)
    )
    if version == ARTIFACT_VERSION:
        model.load_state_dict(payload["state_dict"], strict=True)
    else:
        incompatible = model.load_state_dict(payload["state_dict"], strict=False)
        if set(incompatible.missing_keys) != {"aux_out.weight", "aux_out.bias"}:
            msg = f"unexpected missing v1 ranker weights: {incompatible.missing_keys}"
            raise ValueError(msg)
        if incompatible.unexpected_keys:
            msg = f"unexpected v1 ranker weights: {incompatible.unexpected_keys}"
            raise ValueError(msg)
        metadata = {**metadata, "legacy_inference_only": True}
    model.eval()
    return model, metadata
