"""固定验证窗口上的预训练非劣验收。"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from quant_fm.pretrain.validation_sampler import ValidationSamplePlan

if TYPE_CHECKING:
    from typing import Any


_ACCEPTANCE_REQUIRED_FIELDS = {
    "acceptance_version",
    "created_utc",
    "candidate",
    "candidate_sha256",
    "baseline",
    "baseline_sha256",
    "validation_plan_source_fingerprint",
    "validation_plan_sha256",
    "validation_windows",
    "evaluated_windows",
    "evaluation_prediction_counts",
    "train_unigram_plan_source_fingerprint",
    "train_unigram_plan_sha256",
    "train_unigram_windows",
    "train_unigram_evaluated_windows",
    "train_unigram_prediction_counts",
    "normalization_target_fields",
    "checkpoint_target_fields",
    "train_unigram_counts",
    "train_unigram_counts_sha256",
    "normalization_contract_sha256",
    "unigram_source",
    "train_unigram_entropy",
    "candidate_checkpoint",
    "candidate_checkpoint_sha256",
    "baseline_checkpoint",
    "baseline_checkpoint_sha256",
    "primary_metric",
    "candidate_value",
    "baseline_value",
    "relative_change",
    "noninferiority_tolerance",
    "accepted",
    "decision",
    "per_field_ce",
}

_ACCEPTANCE_VERSION = "8.0"
_NORMALIZATION_CONTRACT_VERSION = "train_unigram_normalization_v3"
_METRIC_REL_TOL = 1e-12
_METRIC_ABS_TOL = 1e-12
DEFAULT_NONINFERIORITY_TOLERANCE = 0.01


def _validated_tolerance(value: object, *, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        msg = f"{field_name} must be a finite non-negative number"
        raise ValueError(msg)
    return float(value)


def _sha256_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _canonical_unigram_counts(
    value: object,
    *,
    target_fields: list[str],
    context: str,
) -> dict[str, list[int]]:
    if not isinstance(value, dict) or set(value) != set(target_fields):
        msg = f"{context} train_unigram_counts fields are inconsistent"
        raise ValueError(msg)
    canonical: dict[str, list[int]] = {}
    for field in target_fields:
        counts = value[field]
        if (
            not isinstance(counts, list)
            or not counts
            or any(type(count) is not int or count < 0 for count in counts)
        ):
            msg = (
                f"{context} train_unigram_counts[{field!r}] must be a non-empty "
                "non-negative integer list"
            )
            raise ValueError(msg)
        canonical[field] = list(counts)
    return canonical


def _entropy_from_counts(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return float("nan")
    return -math.fsum(
        (count / total) * math.log(count / total) for count in counts if count > 0
    )


def _metric_map(
    value: object,
    *,
    field_name: str,
    target_fields: list[str],
    positive: bool = False,
    nonnegative: bool = False,
) -> dict[str, float]:
    """Return an exact-field, finite metric map with canonical float values."""
    if not isinstance(value, dict) or set(value) != set(target_fields):
        msg = f"pretraining evaluation {field_name} fields are inconsistent"
        raise ValueError(msg)
    normalized: dict[str, float] = {}
    for field in target_fields:
        metric = value[field]
        if (
            isinstance(metric, bool)
            or not isinstance(metric, int | float)
            or not math.isfinite(float(metric))
        ):
            msg = f"pretraining evaluation {field_name} values must be finite"
            raise ValueError(msg)
        number = float(metric)
        if positive and number <= 0:
            msg = f"pretraining evaluation {field_name} values must be positive"
            raise ValueError(msg)
        if nonnegative and number < 0:
            msg = f"pretraining evaluation {field_name} values must be non-negative"
            raise ValueError(msg)
        normalized[field] = number
    return normalized


def _finite_metric(
    report: dict[str, Any],
    *,
    field_name: str,
    nonnegative: bool = False,
) -> float:
    value = report.get(field_name)
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        msg = f"pretraining evaluation {field_name} must be finite numeric"
        raise ValueError(msg)
    number = float(value)
    if nonnegative and number < 0:
        msg = f"pretraining evaluation {field_name} must be non-negative"
        raise ValueError(msg)
    return number


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"pretraining evaluation must be a JSON object: {path}"
        raise TypeError(msg)
    if payload.get("split") != "val":
        msg = f"pretraining comparison requires split=val: {path}"
        raise ValueError(msg)
    return payload


def _canonical_live_path(value: object, *, field: str, report_path: Path) -> Path:
    if not isinstance(value, str) or not value:
        msg = f"pretraining evaluation {field} must be a non-empty path: {report_path}"
        raise ValueError(msg)
    source = Path(value)
    if not source.is_absolute() or str(source.resolve()) != value:
        msg = f"pretraining evaluation {field} must be a canonical absolute path"
        raise ValueError(msg)
    if not source.is_file():
        msg = f"pretraining evaluation {field} is missing: {source}"
        raise FileNotFoundError(msg)
    return source


def _live_checkpoint_target_fields(path: Path) -> list[str]:
    """Safely read the ordered prediction heads frozen inside a checkpoint."""
    safe_numpy_globals = [
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]
    try:
        with torch.serialization.safe_globals(safe_numpy_globals):
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
    except Exception as exc:
        msg = f"pretraining evaluation checkpoint cannot be safely loaded: {path}"
        raise ValueError(msg) from exc
    try:
        if not isinstance(payload, dict):
            msg = f"pretraining evaluation checkpoint payload must be an object: {path}"
            raise TypeError(msg)
        config = payload.get("config")
        if not isinstance(config, dict):
            msg = f"pretraining evaluation checkpoint config must be an object: {path}"
            raise TypeError(msg)
        if "target_fields" not in config:
            msg = "pretraining evaluation checkpoint has no explicit target_fields"
            raise ValueError(msg)
        value = config["target_fields"]
        if not isinstance(value, list | tuple):
            msg = "pretraining evaluation checkpoint target_fields must be a sequence"
            raise TypeError(msg)
        fields = list(value)
        if (
            not fields
            or not all(isinstance(field, str) and field for field in fields)
            or len(set(fields)) != len(fields)
        ):
            msg = (
                "pretraining evaluation checkpoint target_fields must be unique "
                "non-empty strings"
            )
            raise ValueError(msg)
        return fields
    finally:
        # Release mmap-backed tensor/storage references before verifying another report.
        del payload


def _verify_plan_identity(
    report: dict[str, Any],
    *,
    report_path: Path,
    prefix: str,
) -> dict[str, object]:
    path_field = f"{prefix}_plan"
    sha_field = f"{prefix}_plan_sha256"
    source_field = f"{prefix}_plan_source_fingerprint"
    windows_field = f"{prefix}_windows"
    plan_sha = report.get(sha_field)
    if not _is_sha256(plan_sha):
        msg = f"pretraining evaluation has no canonical {prefix} plan SHA-256: {report_path}"
        raise ValueError(msg)
    plan_path = _canonical_live_path(
        report.get(path_field),
        field=path_field,
        report_path=report_path,
    )
    plan = ValidationSamplePlan.load(plan_path)
    if plan.sha256 != plan_sha:
        msg = f"pretraining evaluation {prefix} plan identity has changed: {plan_path}"
        raise ValueError(msg)
    if plan.source_fingerprint != report.get(source_field):
        msg = f"pretraining evaluation {prefix} plan source is inconsistent"
        raise ValueError(msg)
    if len(plan.windows) != report.get(windows_field):
        msg = f"pretraining evaluation {prefix} window count is inconsistent"
        raise ValueError(msg)
    return {
        "path": str(plan_path),
        "sha256": plan_sha,
        "source_fingerprint": plan.source_fingerprint,
        "windows": len(plan.windows),
        "max_windows": plan.max_windows,
        "max_predictions_per_field": sum(
            max(window.length - 1, 0) for window in plan.windows
        ),
    }


def _verify_evaluation_identity(
    report: dict[str, Any],
    *,
    report_path: Path,
) -> dict[str, object]:
    """Verify exact validation, normalization, and checkpoint identities."""
    validation_plan = _verify_plan_identity(
        report,
        report_path=report_path,
        prefix="validation",
    )
    train_plan = _verify_plan_identity(
        report,
        report_path=report_path,
        prefix="train_unigram",
    )

    target_fields = report.get("normalization_target_fields")
    if (
        not isinstance(target_fields, list)
        or not target_fields
        or not all(isinstance(field, str) and field for field in target_fields)
        or len(set(target_fields)) != len(target_fields)
    ):
        msg = (
            "pretraining evaluation normalization_target_fields must be unique strings"
        )
        raise ValueError(msg)
    checkpoint_target_fields = report.get("checkpoint_target_fields")
    if checkpoint_target_fields != target_fields:
        msg = (
            "pretraining evaluation normalization_target_fields must exactly match "
            "checkpoint_target_fields"
        )
        raise ValueError(msg)
    if train_plan["max_windows"] != train_plan["windows"] or not isinstance(
        train_plan["max_windows"], int
    ):
        msg = "pretraining evaluation train-unigram plan is not an exact-N plan"
        raise ValueError(msg)
    evaluated_windows = report.get("evaluated_windows")
    if (
        type(evaluated_windows) is not int
        or evaluated_windows <= 0
        or evaluated_windows != validation_plan["windows"]
    ):
        msg = "pretraining evaluation must consume the complete frozen validation plan"
        raise ValueError(msg)
    prediction_counts = report.get("n_predictions")
    if not isinstance(prediction_counts, dict) or set(prediction_counts) != set(
        target_fields
    ):
        msg = "pretraining evaluation prediction-count fields are inconsistent"
        raise ValueError(msg)
    max_predictions = validation_plan["max_predictions_per_field"]
    assert isinstance(max_predictions, int)
    if any(
        type(value) is not int or value <= 0 or value > max_predictions
        for value in prediction_counts.values()
    ):
        msg = (
            "pretraining evaluation prediction counts must be positive and "
            "consistent with the validation plan"
        )
        raise ValueError(msg)
    canonical_prediction_counts = {
        field: prediction_counts[field] for field in target_fields
    }
    unigram_counts = _canonical_unigram_counts(
        report.get("train_unigram_counts"),
        target_fields=target_fields,
        context="pretraining evaluation",
    )
    counts_sha = report.get("train_unigram_counts_sha256")
    normalization_sha = report.get("normalization_contract_sha256")
    if not _is_sha256(counts_sha) or not _is_sha256(normalization_sha):
        msg = "pretraining evaluation normalization identities must be SHA-256 values"
        raise ValueError(msg)
    if _canonical_sha256(unigram_counts) != counts_sha:
        msg = "pretraining evaluation train_unigram_counts SHA-256 is invalid"
        raise ValueError(msg)
    train_evaluated_windows = report.get("train_unigram_evaluated_windows")
    if (
        type(train_evaluated_windows) is not int
        or train_evaluated_windows <= 0
        or train_evaluated_windows != train_plan["windows"]
    ):
        msg = "pretraining evaluation must consume the complete train-unigram plan"
        raise ValueError(msg)
    train_prediction_counts = report.get("train_unigram_prediction_counts")
    if not isinstance(train_prediction_counts, dict) or set(
        train_prediction_counts
    ) != set(target_fields):
        msg = "pretraining evaluation train-unigram prediction fields are inconsistent"
        raise ValueError(msg)
    max_train_predictions = train_plan["max_predictions_per_field"]
    assert isinstance(max_train_predictions, int)
    canonical_train_prediction_counts: dict[str, int] = {}
    for field in target_fields:
        prediction_count = train_prediction_counts[field]
        if (
            type(prediction_count) is not int
            or prediction_count <= 0
            or prediction_count > max_train_predictions
            or prediction_count != sum(unigram_counts[field])
        ):
            msg = (
                "pretraining evaluation train-unigram prediction count is "
                f"inconsistent with counts/plan: {field}"
            )
            raise ValueError(msg)
        canonical_train_prediction_counts[field] = prediction_count
    if report.get("unigram_source") != "train":
        msg = "pretraining evaluation normalized CE must use train unigram counts"
        raise ValueError(msg)
    train_entropy = _metric_map(
        report.get("train_unigram_entropy"),
        field_name="train_unigram_entropy",
        target_fields=target_fields,
        positive=True,
    )
    for field in target_fields:
        expected_entropy = _entropy_from_counts(unigram_counts[field])
        if not math.isclose(
            train_entropy[field],
            expected_entropy,
            rel_tol=_METRIC_REL_TOL,
            abs_tol=_METRIC_ABS_TOL,
        ):
            msg = (
                "pretraining evaluation train_unigram_entropy is inconsistent "
                f"with counts: {field}"
            )
            raise ValueError(msg)
    normalization_contract = {
        "format_version": _NORMALIZATION_CONTRACT_VERSION,
        "target_fields": target_fields,
        "train_unigram_plan_sha256": train_plan["sha256"],
        "train_unigram_plan_source_fingerprint": train_plan["source_fingerprint"],
        "train_unigram_windows": train_plan["windows"],
        "train_unigram_counts": unigram_counts,
        "train_unigram_counts_sha256": counts_sha,
        "train_unigram_entropy": train_entropy,
    }
    if _canonical_sha256(normalization_contract) != normalization_sha:
        msg = "pretraining evaluation normalization contract SHA-256 is invalid"
        raise ValueError(msg)
    per_field_ce = _metric_map(
        report.get("per_field_ce"),
        field_name="per_field_ce",
        target_fields=target_fields,
        nonnegative=True,
    )
    normalized_fields = _metric_map(
        report.get("ce_over_unigram_entropy"),
        field_name="normalized CE",
        target_fields=target_fields,
        nonnegative=True,
    )
    for field in target_fields:
        expected = per_field_ce[field] / train_entropy[field]
        if not math.isclose(
            normalized_fields[field],
            expected,
            rel_tol=_METRIC_REL_TOL,
            abs_tol=_METRIC_ABS_TOL,
        ):
            msg = (
                "pretraining evaluation normalized CE is inconsistent with "
                f"per-field CE and train entropy: {field}"
            )
            raise ValueError(msg)
    total_ce = _finite_metric(
        report,
        field_name="total_ce",
        nonnegative=True,
    )
    expected_total_ce = math.fsum(per_field_ce.values())
    if not math.isclose(
        total_ce,
        expected_total_ce,
        rel_tol=_METRIC_REL_TOL,
        abs_tol=_METRIC_ABS_TOL,
    ):
        msg = "pretraining evaluation total_ce is inconsistent with per-field CE"
        raise ValueError(msg)
    total_normalized_ce = _finite_metric(
        report,
        field_name="total_normalized_ce",
        nonnegative=True,
    )
    expected_total_normalized_ce = math.fsum(normalized_fields.values())
    if not math.isclose(
        total_normalized_ce,
        expected_total_normalized_ce,
        rel_tol=_METRIC_REL_TOL,
        abs_tol=_METRIC_ABS_TOL,
    ):
        msg = (
            "pretraining evaluation total_normalized_ce is inconsistent with "
            "per-field normalized CE"
        )
        raise ValueError(msg)

    checkpoint_sha = report.get("checkpoint_sha256")
    if not _is_sha256(checkpoint_sha):
        msg = f"pretraining evaluation has no checkpoint SHA-256: {report_path}"
        raise ValueError(msg)

    checkpoint_path = _canonical_live_path(
        report.get("checkpoint"),
        field="checkpoint",
        report_path=report_path,
    )
    if _sha256_file(checkpoint_path) != checkpoint_sha:
        msg = (
            f"pretraining evaluation checkpoint SHA-256 has changed: {checkpoint_path}"
        )
        raise ValueError(msg)
    live_checkpoint_target_fields = _live_checkpoint_target_fields(checkpoint_path)
    if _sha256_file(checkpoint_path) != checkpoint_sha:
        msg = (
            "pretraining evaluation checkpoint changed while its target_fields "
            f"were being verified: {checkpoint_path}"
        )
        raise ValueError(msg)
    if live_checkpoint_target_fields != checkpoint_target_fields:
        msg = (
            "pretraining evaluation checkpoint_target_fields do not exactly match "
            "the ordered target_fields frozen in the live checkpoint"
        )
        raise ValueError(msg)
    return {
        "validation_plan": validation_plan,
        "train_unigram_plan": train_plan,
        "normalization_contract_sha256": normalization_sha,
        "normalization_target_fields": target_fields,
        "checkpoint_target_fields": live_checkpoint_target_fields,
        "train_unigram_counts": unigram_counts,
        "train_unigram_counts_sha256": counts_sha,
        "train_unigram_entropy": train_entropy,
        "train_unigram_evaluated_windows": train_evaluated_windows,
        "train_unigram_prediction_counts": canonical_train_prediction_counts,
        "per_field_ce": per_field_ce,
        "ce_over_unigram_entropy": normalized_fields,
        "total_ce": total_ce,
        "total_normalized_ce": total_normalized_ce,
        "evaluated_windows": evaluated_windows,
        "prediction_counts": canonical_prediction_counts,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
    }


def _primary_metric(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> tuple[str, float, float]:
    """Return raw CE only; empirical normalization counts are diagnostic."""
    candidate_value = candidate.get("total_ce")
    baseline_value = baseline.get("total_ce")
    if (
        not isinstance(candidate_value, bool)
        and isinstance(candidate_value, int | float)
        and not isinstance(baseline_value, bool)
        and isinstance(baseline_value, int | float)
        and math.isfinite(float(candidate_value))
        and math.isfinite(float(baseline_value))
        and float(candidate_value) >= 0
        and float(baseline_value) > 0
    ):
        return "total_ce", float(candidate_value), float(baseline_value)
    msg = "evaluation reports do not share a finite positive raw total_ce metric"
    raise ValueError(msg)


def compare_pretrain_evaluations(
    candidate_path: Path,
    baseline_path: Path,
    *,
    noninferiority_tolerance: float = DEFAULT_NONINFERIORITY_TOLERANCE,
) -> dict[str, Any]:
    """比较同一 validation plan 的候选与基线，并执行相对 CE 非劣门槛。"""
    noninferiority_tolerance = _validated_tolerance(
        noninferiority_tolerance,
        field_name="noninferiority_tolerance",
    )
    candidate_path = Path(candidate_path).resolve()
    baseline_path = Path(baseline_path).resolve()
    if candidate_path == baseline_path:
        msg = "candidate and baseline evaluations must be different artifacts"
        raise ValueError(msg)
    candidate = _load_report(candidate_path)
    baseline = _load_report(baseline_path)
    candidate_identity = _verify_evaluation_identity(
        candidate,
        report_path=candidate_path,
    )
    baseline_identity = _verify_evaluation_identity(
        baseline,
        report_path=baseline_path,
    )
    candidate_validation = candidate_identity["validation_plan"]
    baseline_validation = baseline_identity["validation_plan"]
    candidate_train = candidate_identity["train_unigram_plan"]
    baseline_train = baseline_identity["train_unigram_plan"]
    assert isinstance(candidate_validation, dict)
    assert isinstance(baseline_validation, dict)
    assert isinstance(candidate_train, dict)
    assert isinstance(baseline_train, dict)
    if candidate_validation["sha256"] != baseline_validation["sha256"]:
        msg = "candidate and baseline must use the same canonical validation plan"
        raise ValueError(msg)
    if candidate_train["sha256"] != baseline_train["sha256"]:
        msg = "candidate and baseline must use the same canonical train-unigram plan"
        raise ValueError(msg)
    if (
        candidate_identity["normalization_contract_sha256"]
        != baseline_identity["normalization_contract_sha256"]
    ):
        msg = "candidate and baseline normalized CE denominators differ"
        raise ValueError(msg)
    if (
        candidate_identity["normalization_target_fields"]
        != baseline_identity["normalization_target_fields"]
        or candidate_identity["checkpoint_target_fields"]
        != baseline_identity["checkpoint_target_fields"]
        or candidate_identity["train_unigram_entropy"]
        != baseline_identity["train_unigram_entropy"]
    ):
        msg = (
            "candidate and baseline checkpoint/normalization field or entropy "
            "contracts differ"
        )
        raise ValueError(msg)
    if (
        candidate_identity["train_unigram_counts"]
        != baseline_identity["train_unigram_counts"]
        or candidate_identity["train_unigram_counts_sha256"]
        != baseline_identity["train_unigram_counts_sha256"]
        or candidate_identity["train_unigram_evaluated_windows"]
        != baseline_identity["train_unigram_evaluated_windows"]
        or candidate_identity["train_unigram_prediction_counts"]
        != baseline_identity["train_unigram_prediction_counts"]
    ):
        msg = "candidate and baseline train-unigram counts/consumption differ"
        raise ValueError(msg)
    if (
        candidate_identity["evaluated_windows"]
        != baseline_identity["evaluated_windows"]
        or candidate_identity["prediction_counts"]
        != baseline_identity["prediction_counts"]
    ):
        msg = "candidate and baseline evaluated window/prediction counts differ"
        raise ValueError(msg)
    candidate_fingerprint = candidate.get("validation_plan_source_fingerprint")
    baseline_fingerprint = baseline.get("validation_plan_source_fingerprint")
    if (
        not isinstance(candidate_fingerprint, str)
        or not candidate_fingerprint
        or candidate_fingerprint != baseline_fingerprint
    ):
        msg = "candidate and baseline must use the same validation plan fingerprint"
        raise ValueError(msg)
    validation_windows = candidate.get("validation_windows")
    if (
        type(validation_windows) is not int
        or validation_windows <= 0
        or validation_windows != baseline.get("validation_windows")
    ):
        msg = "candidate and baseline validation window counts differ"
        raise ValueError(msg)

    metric, candidate_value, baseline_value = _primary_metric(
        candidate_identity,
        baseline_identity,
    )
    relative_change = (candidate_value - baseline_value) / baseline_value
    accepted = relative_change <= noninferiority_tolerance
    candidate_fields = candidate_identity["per_field_ce"]
    baseline_fields = baseline_identity["per_field_ce"]
    assert isinstance(candidate_fields, dict)
    assert isinstance(baseline_fields, dict)
    expected_fields = set(candidate_identity["normalization_target_fields"])
    if (
        set(candidate_fields) != expected_fields
        or set(baseline_fields) != expected_fields
    ):
        msg = (
            "candidate and baseline per-field CE sets must match normalization targets"
        )
        raise ValueError(msg)
    shared_fields = sorted(expected_fields)
    fields: dict[str, dict[str, float]] = {}
    for field in shared_fields:
        candidate_ce = candidate_fields[field]
        baseline_ce = baseline_fields[field]
        if (
            isinstance(candidate_ce, bool)
            or not isinstance(candidate_ce, int | float)
            or not math.isfinite(float(candidate_ce))
            or isinstance(baseline_ce, bool)
            or not isinstance(baseline_ce, int | float)
            or not math.isfinite(float(baseline_ce))
        ):
            msg = f"per-field CE metric must be finite numeric: {field}"
            raise ValueError(msg)
        fields[field] = {
            "candidate_ce": float(candidate_ce),
            "baseline_ce": float(baseline_ce),
            "absolute_delta": float(candidate_ce) - float(baseline_ce),
        }
    return {
        "acceptance_version": _ACCEPTANCE_VERSION,
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "candidate": str(candidate_path),
        "candidate_sha256": _sha256_file(candidate_path),
        "baseline": str(baseline_path),
        "baseline_sha256": _sha256_file(baseline_path),
        "validation_plan_source_fingerprint": candidate_fingerprint,
        "validation_plan_sha256": candidate_validation["sha256"],
        "validation_windows": candidate.get("validation_windows"),
        "evaluated_windows": candidate_identity["evaluated_windows"],
        "evaluation_prediction_counts": candidate_identity["prediction_counts"],
        "train_unigram_plan_source_fingerprint": candidate_train["source_fingerprint"],
        "train_unigram_plan_sha256": candidate_train["sha256"],
        "train_unigram_windows": candidate_train["windows"],
        "train_unigram_evaluated_windows": candidate_identity[
            "train_unigram_evaluated_windows"
        ],
        "train_unigram_prediction_counts": candidate_identity[
            "train_unigram_prediction_counts"
        ],
        "normalization_target_fields": candidate_identity[
            "normalization_target_fields"
        ],
        "checkpoint_target_fields": candidate_identity["checkpoint_target_fields"],
        "train_unigram_counts": candidate_identity["train_unigram_counts"],
        "train_unigram_counts_sha256": candidate_identity[
            "train_unigram_counts_sha256"
        ],
        "normalization_contract_sha256": candidate_identity[
            "normalization_contract_sha256"
        ],
        "unigram_source": "train",
        "train_unigram_entropy": candidate_identity["train_unigram_entropy"],
        "candidate_checkpoint": candidate_identity["checkpoint"],
        "candidate_checkpoint_sha256": candidate_identity["checkpoint_sha256"],
        "baseline_checkpoint": baseline_identity["checkpoint"],
        "baseline_checkpoint_sha256": baseline_identity["checkpoint_sha256"],
        "primary_metric": metric,
        "candidate_value": candidate_value,
        "baseline_value": baseline_value,
        "relative_change": relative_change,
        "noninferiority_tolerance": noninferiority_tolerance,
        "accepted": accepted,
        "decision": "PASS" if accepted else "FAIL",
        "per_field_ce": fields,
    }


def validate_pretrain_acceptance(
    path: Path,
    *,
    expected_noninferiority_tolerance: float = DEFAULT_NONINFERIORITY_TOLERANCE,
) -> dict[str, Any]:
    """Load a v8 acceptance artifact and reverify its sources and explicit PASS."""
    expected_tolerance = _validated_tolerance(
        expected_noninferiority_tolerance,
        field_name="expected_noninferiority_tolerance",
    )
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"pretrain acceptance is not valid JSON: {path}"
        raise ValueError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"pretrain acceptance must be a JSON object: {path}"
        raise TypeError(msg)

    missing = sorted(_ACCEPTANCE_REQUIRED_FIELDS - set(payload))
    if missing:
        msg = f"pretrain acceptance is missing required fields: {missing}"
        raise ValueError(msg)
    if payload["acceptance_version"] != _ACCEPTANCE_VERSION:
        msg = (
            "unsupported pretrain acceptance version: "
            f"{payload['acceptance_version']!r}"
        )
        raise ValueError(msg)
    try:
        created = datetime.fromisoformat(payload["created_utc"])
    except (TypeError, ValueError) as exc:
        msg = "pretrain acceptance created_utc must be an ISO timestamp"
        raise ValueError(msg) from exc
    if created.tzinfo is None:
        msg = "pretrain acceptance created_utc must include a timezone"
        raise ValueError(msg)
    if payload["primary_metric"] != "total_ce":
        msg = f"unsupported pretrain acceptance metric: {payload['primary_metric']!r}"
        raise ValueError(msg)
    if (
        not isinstance(payload["validation_plan_source_fingerprint"], str)
        or not payload["validation_plan_source_fingerprint"]
    ):
        msg = "pretrain acceptance has no validation-plan fingerprint"
        raise ValueError(msg)
    if not _is_sha256(payload["validation_plan_sha256"]):
        msg = "pretrain acceptance has no canonical validation-plan SHA-256"
        raise ValueError(msg)
    validation_windows = payload["validation_windows"]
    if type(validation_windows) is not int or validation_windows <= 0:
        msg = "pretrain acceptance validation_windows must be a positive integer"
        raise ValueError(msg)
    evaluated_windows = payload["evaluated_windows"]
    if (
        type(evaluated_windows) is not int
        or evaluated_windows <= 0
        or evaluated_windows != validation_windows
    ):
        msg = "pretrain acceptance must cover the complete validation plan"
        raise ValueError(msg)
    if (
        not isinstance(payload["train_unigram_plan_source_fingerprint"], str)
        or not payload["train_unigram_plan_source_fingerprint"]
    ):
        msg = "pretrain acceptance has no train-unigram source fingerprint"
        raise ValueError(msg)
    if not _is_sha256(payload["train_unigram_plan_sha256"]):
        msg = "pretrain acceptance has no canonical train-unigram plan SHA-256"
        raise ValueError(msg)
    train_windows = payload["train_unigram_windows"]
    if type(train_windows) is not int or train_windows <= 0:
        msg = "pretrain acceptance train_unigram_windows must be positive"
        raise ValueError(msg)
    target_fields = payload["normalization_target_fields"]
    if (
        not isinstance(target_fields, list)
        or not target_fields
        or not all(isinstance(field, str) and field for field in target_fields)
        or len(set(target_fields)) != len(target_fields)
    ):
        msg = "pretrain acceptance normalization_target_fields are invalid"
        raise ValueError(msg)
    if payload["checkpoint_target_fields"] != target_fields:
        msg = (
            "pretrain acceptance normalization_target_fields must exactly match "
            "checkpoint_target_fields"
        )
        raise ValueError(msg)
    prediction_counts = payload["evaluation_prediction_counts"]
    if not isinstance(prediction_counts, dict) or set(prediction_counts) != set(
        target_fields
    ):
        msg = "pretrain acceptance prediction-count fields are inconsistent"
        raise ValueError(msg)
    if any(
        type(value) is not int or value <= 0 for value in prediction_counts.values()
    ):
        msg = "pretrain acceptance prediction counts must be positive integers"
        raise ValueError(msg)
    train_evaluated_windows = payload["train_unigram_evaluated_windows"]
    if (
        type(train_evaluated_windows) is not int
        or train_evaluated_windows <= 0
        or train_evaluated_windows != train_windows
    ):
        msg = "pretrain acceptance must cover the complete train-unigram plan"
        raise ValueError(msg)
    unigram_counts = _canonical_unigram_counts(
        payload["train_unigram_counts"],
        target_fields=target_fields,
        context="pretrain acceptance",
    )
    for field in ("train_unigram_counts_sha256", "normalization_contract_sha256"):
        if not _is_sha256(payload[field]):
            msg = f"pretrain acceptance {field} must be a SHA-256"
            raise ValueError(msg)
    if _canonical_sha256(unigram_counts) != payload["train_unigram_counts_sha256"]:
        msg = "pretrain acceptance train_unigram_counts SHA-256 is invalid"
        raise ValueError(msg)
    train_prediction_counts = payload["train_unigram_prediction_counts"]
    if not isinstance(train_prediction_counts, dict) or set(
        train_prediction_counts
    ) != set(target_fields):
        msg = "pretrain acceptance train-unigram prediction fields are inconsistent"
        raise ValueError(msg)
    for field in target_fields:
        prediction_count = train_prediction_counts[field]
        if (
            type(prediction_count) is not int
            or prediction_count <= 0
            or prediction_count != sum(unigram_counts[field])
        ):
            msg = (
                "pretrain acceptance train-unigram prediction count is inconsistent "
                f"with counts: {field}"
            )
            raise ValueError(msg)
    entropy = payload["train_unigram_entropy"]
    if payload["unigram_source"] != "train":
        msg = "pretrain acceptance unigram_source must be train"
        raise ValueError(msg)
    if not isinstance(entropy, dict) or set(entropy) != set(target_fields):
        msg = "pretrain acceptance train_unigram_entropy fields are inconsistent"
        raise ValueError(msg)
    canonical_entropy: dict[str, float] = {}
    for field in target_fields:
        value = entropy[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
        ):
            msg = "pretrain acceptance train_unigram_entropy values must be finite"
            raise ValueError(msg)
        number = float(value)
        if number <= 0:
            msg = "pretrain acceptance train_unigram_entropy values must be positive"
            raise ValueError(msg)
        expected_entropy = _entropy_from_counts(unigram_counts[field])
        if not math.isclose(
            number,
            expected_entropy,
            rel_tol=_METRIC_REL_TOL,
            abs_tol=_METRIC_ABS_TOL,
        ):
            msg = (
                "pretrain acceptance train_unigram_entropy is inconsistent with "
                f"counts: {field}"
            )
            raise ValueError(msg)
        canonical_entropy[field] = number
    normalization_contract = {
        "format_version": _NORMALIZATION_CONTRACT_VERSION,
        "target_fields": target_fields,
        "train_unigram_plan_sha256": payload["train_unigram_plan_sha256"],
        "train_unigram_plan_source_fingerprint": payload[
            "train_unigram_plan_source_fingerprint"
        ],
        "train_unigram_windows": train_windows,
        "train_unigram_counts": unigram_counts,
        "train_unigram_counts_sha256": payload["train_unigram_counts_sha256"],
        "train_unigram_entropy": canonical_entropy,
    }
    if (
        _canonical_sha256(normalization_contract)
        != payload["normalization_contract_sha256"]
    ):
        msg = "pretrain acceptance normalization contract SHA-256 is invalid"
        raise ValueError(msg)
    if not isinstance(payload["per_field_ce"], dict):
        msg = "pretrain acceptance per_field_ce must be an object"
        raise TypeError(msg)

    for name in ("candidate_checkpoint", "baseline_checkpoint"):
        checkpoint = _canonical_live_path(
            payload[name],
            field=name,
            report_path=path,
        )
        hash_field = f"{name}_sha256"
        expected_hash = payload[hash_field]
        if not _is_sha256(expected_hash):
            msg = f"pretrain acceptance {hash_field} must be a SHA-256"
            raise ValueError(msg)
        if _sha256_file(checkpoint) != expected_hash:
            msg = f"pretrain acceptance {name} SHA-256 has changed: {checkpoint}"
            raise ValueError(msg)

    numeric: dict[str, float] = {}
    for field in (
        "candidate_value",
        "baseline_value",
        "relative_change",
        "noninferiority_tolerance",
    ):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int | float):
            msg = f"pretrain acceptance {field} must be numeric"
            raise TypeError(msg)
        number = float(value)
        if not math.isfinite(number):
            msg = f"pretrain acceptance {field} must be finite"
            raise ValueError(msg)
        numeric[field] = number
    if numeric["baseline_value"] <= 0:
        msg = "pretrain acceptance baseline_value must be positive"
        raise ValueError(msg)
    if numeric["candidate_value"] < 0:
        msg = "pretrain acceptance candidate_value must be non-negative"
        raise ValueError(msg)
    if numeric["noninferiority_tolerance"] < 0:
        msg = "pretrain acceptance noninferiority_tolerance must be non-negative"
        raise ValueError(msg)
    if numeric["noninferiority_tolerance"] != expected_tolerance:
        msg = (
            "pretrain acceptance noninferiority_tolerance does not match the "
            "independently configured expected tolerance: "
            f"artifact={numeric['noninferiority_tolerance']}, "
            f"expected={expected_tolerance}"
        )
        raise ValueError(msg)

    recomputed_change = (
        numeric["candidate_value"] - numeric["baseline_value"]
    ) / numeric["baseline_value"]
    if not math.isclose(
        numeric["relative_change"], recomputed_change, rel_tol=1e-12, abs_tol=1e-12
    ):
        msg = "pretrain acceptance relative_change is inconsistent with CE values"
        raise ValueError(msg)
    recomputed_pass = recomputed_change <= expected_tolerance
    accepted = payload["accepted"]
    if type(accepted) is not bool:
        msg = "pretrain acceptance accepted must be a boolean"
        raise ValueError(msg)
    expected_decision = "PASS" if accepted else "FAIL"
    if payload["decision"] != expected_decision:
        msg = "pretrain acceptance accepted and decision fields are inconsistent"
        raise ValueError(msg)
    if accepted != recomputed_pass:
        msg = (
            "pretrain acceptance decision is inconsistent with the noninferiority gate"
        )
        raise ValueError(msg)
    if not accepted:
        msg = "pretrain noninferiority acceptance did not pass"
        raise ValueError(msg)

    source_paths: dict[str, Path] = {}
    for name in ("candidate", "baseline"):
        value = payload[name]
        if not isinstance(value, str) or not value:
            msg = f"pretrain acceptance {name} must be a non-empty path"
            raise ValueError(msg)
        source = Path(value)
        if not source.is_absolute() or str(source.resolve()) != value:
            msg = f"pretrain acceptance {name} must be a canonical absolute path"
            raise ValueError(msg)
        if not source.is_file():
            msg = f"pretrain acceptance {name} source is missing: {source}"
            raise FileNotFoundError(msg)
        hash_field = f"{name}_sha256"
        expected_hash = payload[hash_field]
        if not _is_sha256(expected_hash):
            msg = f"pretrain acceptance {hash_field} must be a SHA-256"
            raise ValueError(msg)
        actual_hash = _sha256_file(source)
        if actual_hash != expected_hash:
            msg = f"pretrain acceptance {name} source SHA-256 has changed: {source}"
            raise ValueError(msg)
        source_paths[name] = source

    recomputed = compare_pretrain_evaluations(
        source_paths["candidate"],
        source_paths["baseline"],
        noninferiority_tolerance=expected_tolerance,
    )
    immutable_fields = _ACCEPTANCE_REQUIRED_FIELDS - {"created_utc"}
    mismatches = {
        field: {"acceptance": payload[field], "recomputed": recomputed[field]}
        for field in sorted(immutable_fields)
        if payload[field] != recomputed[field]
    }
    if mismatches:
        msg = f"pretrain acceptance no longer matches its source reports: {mismatches}"
        raise ValueError(msg)
    return payload


def render_acceptance_report(result: dict[str, Any]) -> str:
    """渲染预训练非劣结果 Markdown。"""
    rows = [
        "# 预训练模型非劣验收",
        "",
        f"决策：**{result['decision']}**  ",
        f"主指标：`{result['primary_metric']}`  ",
        f"候选：{result['candidate_value']:.6f}  ",
        f"基线：{result['baseline_value']:.6f}  ",
        f"相对变化：{100.0 * result['relative_change']:.3f}%  ",
        f"允许上限：{100.0 * result['noninferiority_tolerance']:.3f}%",
        "",
        "## 分字段 CE",
        "",
        "| 字段 | 候选 | 基线 | 差值 |",
        "|---|---:|---:|---:|",
    ]
    rows.extend(
        f"| {name} | {values['candidate_ce']:.6f} | "
        f"{values['baseline_ce']:.6f} | {values['absolute_delta']:.6f} |"
        for name, values in result["per_field_ce"].items()
    )
    rows.extend(
        [
            "",
            "PASS 仅由未使用经验 unigram 分母加权的 raw total CE 决定；归一化 CE 只作诊断。",
            "",
            "该门槛只判断预训练 validation 非劣；最终晋级仍需 RankIC、成本后回测和稳定性验收。",
            "",
        ]
    )
    return "\n".join(rows)
