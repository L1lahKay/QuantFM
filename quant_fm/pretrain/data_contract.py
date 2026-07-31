"""Auditable data provenance carried by every new FM checkpoint."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date as calendar_date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from quant_fm.manifest.validation import (
    sha256_file,
    validate_manifest_vocab_contract,
)
from quant_fm.tokenizer.artifact_contract import stable_vocab_sha256

if TYPE_CHECKING:
    from quant_fm.manifest.build_manifest import Manifest

PRETRAIN_DATA_CONTRACT_VERSION = "pretrain_data_contract_v2"
PRETRAIN_CHECKPOINT_CONTRACT_VERSION = "pretrain_checkpoint_contract_v1"

_PRETRAIN_DATA_CONTRACT_FIELDS = frozenset(
    {
        "format_version",
        "manifest_sha256",
        "vocab_artifact_sha256",
        "vocab_sha256",
        "schema_version",
        "event_ordering_version",
        "feature_transform_version",
        "manifest_train_start",
        "manifest_train_end",
        "manifest_validation_start",
        "manifest_validation_end",
        "vocab_fit_start",
        "vocab_fit_end",
        "effective_training_end",
    }
)


class PretrainVocab(Protocol):
    """Vocabulary fields frozen into the pretraining lineage."""

    schema_version: str
    fit_dates: tuple[str, ...]
    event_ordering_version: str
    feature_transform_version: str
    data_semantics_explicit: bool

    def to_json(self) -> str:
        """Return the stable vocabulary serialization."""


def _validate_sha256(value: object, *, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        msg = f"FM pretrain data contract {field} must be a full SHA-256"
        raise ValueError(msg)
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        msg = f"FM pretrain data contract {field} must be hexadecimal"
        raise ValueError(msg) from exc


def _validate_contract_date(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"FM pretrain data contract {field} must be YYYY-MM-DD or null"
        raise TypeError(msg)
    try:
        parsed = calendar_date.fromisoformat(value)
    except ValueError as exc:
        msg = f"FM pretrain data contract {field} must be canonical YYYY-MM-DD"
        raise ValueError(msg) from exc
    if parsed.isoformat() != value:
        msg = f"FM pretrain data contract {field} must be canonical YYYY-MM-DD"
        raise ValueError(msg)
    return value


def validate_pretrain_data_contract(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact, versioned FM data horizon and artifact identities."""
    missing = sorted(_PRETRAIN_DATA_CONTRACT_FIELDS - set(payload))
    if missing:
        msg = f"FM pretrain data contract is missing fields: {missing}"
        raise ValueError(msg)
    unknown = sorted(set(payload) - _PRETRAIN_DATA_CONTRACT_FIELDS)
    if unknown:
        msg = f"FM pretrain data contract contains unknown fields: {unknown}"
        raise ValueError(msg)
    if payload.get("format_version") != PRETRAIN_DATA_CONTRACT_VERSION:
        msg = "unsupported nested FM pretrain_data_contract version"
        raise ValueError(msg)
    for field in ("manifest_sha256", "vocab_artifact_sha256", "vocab_sha256"):
        _validate_sha256(payload[field], field=field)
    for field in (
        "schema_version",
        "event_ordering_version",
        "feature_transform_version",
    ):
        if not isinstance(payload[field], str) or not payload[field]:
            msg = f"FM pretrain data contract {field} must be a non-empty string"
            raise ValueError(msg)
    dates = {
        field: _validate_contract_date(payload[field], field=field)
        for field in (
            "manifest_train_start",
            "manifest_train_end",
            "manifest_validation_start",
            "manifest_validation_end",
            "vocab_fit_start",
            "vocab_fit_end",
            "effective_training_end",
        )
    }
    for prefix in ("manifest_train", "manifest_validation", "vocab_fit"):
        start = dates[f"{prefix}_start"]
        end = dates[f"{prefix}_end"]
        if (start is None) != (end is None):
            msg = f"FM pretrain data contract has incomplete {prefix} date range"
            raise ValueError(msg)
        if start is not None and end is not None and start > end:
            msg = f"FM pretrain data contract has reversed {prefix} date range"
            raise ValueError(msg)
    end_candidates = [
        dates[field]
        for field in (
            "manifest_train_end",
            "manifest_validation_end",
            "vocab_fit_end",
        )
        if dates[field] is not None
    ]
    expected_effective_end = max(end_candidates) if end_candidates else None
    if dates["effective_training_end"] != expected_effective_end:
        msg = (
            "FM pretrain data contract effective_training_end does not include "
            "train/vocab-fit/validation selection horizons"
        )
        raise ValueError(msg)
    return dict(payload)


def build_pretrain_data_contract(
    *,
    manifest_path: Path,
    manifest: Manifest,
    vocab_path: Path,
    vocab: PretrainVocab,
) -> dict[str, object]:
    """Build the immutable training-date and token-semantics provenance."""
    validate_manifest_vocab_contract(
        manifest,
        vocab,
        context="pretrain data contract",
    )
    train_dates = manifest.dates("train")
    validation_dates = manifest.dates("val")
    fit_dates = sorted({str(value) for value in vocab.fit_dates})
    manifest_train_start = min(train_dates) if train_dates else None
    manifest_train_end = max(train_dates) if train_dates else None
    manifest_validation_start = min(validation_dates) if validation_dates else None
    manifest_validation_end = max(validation_dates) if validation_dates else None
    vocab_fit_start = min(fit_dates) if fit_dates else None
    vocab_fit_end = max(fit_dates) if fit_dates else None
    end_candidates = [
        value
        for value in (manifest_train_end, vocab_fit_end, manifest_validation_end)
        if value is not None
    ]
    payload = {
        "format_version": PRETRAIN_DATA_CONTRACT_VERSION,
        "manifest_sha256": sha256_file(Path(manifest_path)),
        "vocab_artifact_sha256": sha256_file(Path(vocab_path)),
        "vocab_sha256": stable_vocab_sha256(vocab),
        "schema_version": vocab.schema_version,
        "event_ordering_version": vocab.event_ordering_version,
        "feature_transform_version": vocab.feature_transform_version,
        "manifest_train_start": manifest_train_start,
        "manifest_train_end": manifest_train_end,
        "manifest_validation_start": manifest_validation_start,
        "manifest_validation_end": manifest_validation_end,
        "vocab_fit_start": vocab_fit_start,
        "vocab_fit_end": vocab_fit_end,
        "effective_training_end": max(end_candidates) if end_candidates else None,
    }
    return validate_pretrain_data_contract(payload)


def checkpoint_contract_path(checkpoint_path: Path) -> Path:
    """Return the immutable metadata sidecar path for an FM checkpoint."""
    path = Path(checkpoint_path)
    return path.with_name(f"{path.name}.contract.json")


def model_data_contract(config: Mapping[str, Any]) -> dict[str, object]:
    """Extract the model fields that must agree with token data provenance."""
    return {
        "vocab_version": str(config.get("vocab_version", "1.0")),
        "vocab_sha256": str(config.get("vocab_sha256", "")),
        "schema_version": str(config.get("schema_version", "cn_l2_v1")),
        "event_ordering_version": str(
            config.get("event_ordering_version", "local_time_v1")
        ),
        "feature_transform_version": str(
            config.get(
                "feature_transform_version",
                "ew_vwap_future_backfill_v1",
            )
        ),
    }


def write_checkpoint_contract(
    checkpoint_path: Path,
    *,
    config: Mapping[str, Any],
    pretrain_data_contract: Mapping[str, object],
) -> Path:
    """Atomically bind checkpoint bytes to their full data lineage."""
    validated_pretrain_contract = validate_pretrain_data_contract(
        pretrain_data_contract
    )
    destination = checkpoint_contract_path(checkpoint_path)
    payload = {
        "format_version": PRETRAIN_CHECKPOINT_CONTRACT_VERSION,
        "checkpoint_sha256": sha256_file(Path(checkpoint_path)),
        "model_data_contract": model_data_contract(config),
        "pretrain_data_contract": validated_pretrain_contract,
    }
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_checkpoint_contract(
    checkpoint_path: Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    required: bool = True,
) -> dict[str, Any] | None:
    """Load a sidecar and verify that it names the live checkpoint bytes."""
    path = checkpoint_contract_path(checkpoint_path)
    if not path.is_file():
        if required:
            msg = f"FM checkpoint data contract is missing: {path}"
            raise ValueError(msg)
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"FM checkpoint data contract must be a JSON object: {path}"
        raise TypeError(msg)
    if payload.get("format_version") != PRETRAIN_CHECKPOINT_CONTRACT_VERSION:
        msg = f"unsupported FM checkpoint data contract: {path}"
        raise ValueError(msg)
    required_fields = {
        "checkpoint_sha256",
        "model_data_contract",
        "pretrain_data_contract",
    }
    missing = sorted(required_fields - set(payload))
    if missing:
        msg = f"FM checkpoint data contract is missing fields: {missing}"
        raise ValueError(msg)
    allowed_fields = required_fields | {"format_version"}
    unknown = sorted(set(payload) - allowed_fields)
    if unknown:
        msg = f"FM checkpoint data contract contains unknown fields: {unknown}"
        raise ValueError(msg)
    live_hash = sha256_file(Path(checkpoint_path))
    if (
        expected_checkpoint_sha256 is not None
        and expected_checkpoint_sha256 != live_hash
    ):
        msg = f"provided FM checkpoint SHA-256 is stale or incorrect: {checkpoint_path}"
        raise ValueError(msg)
    if payload["checkpoint_sha256"] != live_hash:
        msg = (
            f"FM checkpoint SHA-256 disagrees with its data contract: {checkpoint_path}"
        )
        raise ValueError(msg)
    model_contract = payload["model_data_contract"]
    if not isinstance(model_contract, Mapping):
        msg = "FM checkpoint model_data_contract must be an object"
        raise TypeError(msg)
    expected_model_fields = set(model_data_contract({}))
    if set(model_contract) != expected_model_fields:
        msg = (
            "FM checkpoint model_data_contract fields are not exact: "
            f"expected={sorted(expected_model_fields)}, "
            f"actual={sorted(model_contract)}"
        )
        raise ValueError(msg)
    pretrain_contract = payload["pretrain_data_contract"]
    if not isinstance(pretrain_contract, Mapping):
        msg = "FM checkpoint pretrain_data_contract must be an object"
        raise TypeError(msg)
    validate_pretrain_data_contract(pretrain_contract)
    return payload


def validate_checkpoint_data_contract(
    checkpoint_contract: Mapping[str, Any] | None,
    *,
    checkpoint_payload: Mapping[str, Any] | None,
    vocab: PretrainVocab,
    expected_pretrain_data_contract: Mapping[str, object] | None = None,
) -> None:
    """Cross-check checkpoint sidecar, payload config, vocab, and current data."""
    if checkpoint_contract is None:
        if vocab.data_semantics_explicit:
            msg = (
                "explicit/causal FM checkpoint is missing pretrain_data_contract; "
                "old checkpoints are legacy/diagnostic only"
            )
            raise ValueError(msg)
        return
    sidecar_model = checkpoint_contract.get("model_data_contract")
    if not isinstance(sidecar_model, Mapping):
        msg = "FM checkpoint model_data_contract must be an object"
        raise TypeError(msg)
    expected_model = {
        "vocab_version": (
            "2.0" if getattr(vocab, "VOCAB_VERSION", None) == "2.0" else "1.0"
        ),
        "vocab_sha256": stable_vocab_sha256(vocab),
        "schema_version": vocab.schema_version,
        "event_ordering_version": vocab.event_ordering_version,
        "feature_transform_version": vocab.feature_transform_version,
    }
    mismatches = {
        field: (sidecar_model.get(field), expected)
        for field, expected in expected_model.items()
        if sidecar_model.get(field) != expected
    }
    if mismatches:
        msg = f"FM checkpoint/vocab data contract mismatch: {mismatches}"
        raise ValueError(msg)
    sidecar_pretrain = checkpoint_contract.get("pretrain_data_contract")
    if not isinstance(sidecar_pretrain, Mapping):
        msg = "FM checkpoint pretrain_data_contract must be an object"
        raise TypeError(msg)
    validate_pretrain_data_contract(sidecar_pretrain)
    pretrain_mismatches = {
        field: (sidecar_pretrain.get(field), expected)
        for field, expected in expected_model.items()
        if field != "vocab_version" and sidecar_pretrain.get(field) != expected
    }
    if pretrain_mismatches:
        msg = (
            "FM checkpoint pretrain data lineage disagrees with vocab: "
            f"{pretrain_mismatches}"
        )
        raise ValueError(msg)
    if checkpoint_payload is not None:
        checkpoint_config = checkpoint_payload.get("config")
        if not isinstance(checkpoint_config, Mapping):
            msg = "FM checkpoint config must be an object"
            raise TypeError(msg)
        payload_model = model_data_contract(checkpoint_config)
        if dict(sidecar_model) != payload_model:
            msg = "FM checkpoint payload config disagrees with its contract sidecar"
            raise ValueError(msg)
        payload_pretrain = checkpoint_payload.get("pretrain_data_contract")
        if payload_pretrain != sidecar_pretrain:
            msg = "FM checkpoint payload pretrain data contract disagrees with sidecar"
            raise ValueError(msg)
    if expected_pretrain_data_contract is not None and checkpoint_contract.get(
        "pretrain_data_contract"
    ) != dict(expected_pretrain_data_contract):
        msg = (
            "FM checkpoint pretrain_data_contract does not match current manifest/vocab"
        )
        raise ValueError(msg)


__all__ = [
    "PRETRAIN_CHECKPOINT_CONTRACT_VERSION",
    "PRETRAIN_DATA_CONTRACT_VERSION",
    "build_pretrain_data_contract",
    "checkpoint_contract_path",
    "load_checkpoint_contract",
    "model_data_contract",
    "validate_checkpoint_data_contract",
    "validate_pretrain_data_contract",
    "write_checkpoint_contract",
]
