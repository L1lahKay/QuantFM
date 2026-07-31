"""固定验证窗口上的预训练非劣验收。"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

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
    "validation_windows",
    "primary_metric",
    "candidate_value",
    "baseline_value",
    "relative_change",
    "noninferiority_tolerance",
    "accepted",
    "decision",
    "per_field_ce",
}

_ACCEPTANCE_VERSION = "2.0"


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


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"pretraining evaluation must be a JSON object: {path}"
        raise TypeError(msg)
    if payload.get("split") != "val":
        msg = f"pretraining comparison requires split=val: {path}"
        raise ValueError(msg)
    return payload


def _primary_metric(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> tuple[str, float, float]:
    for name in ("total_normalized_ce", "total_ce"):
        candidate_value = candidate.get(name)
        baseline_value = baseline.get(name)
        if (
            not isinstance(candidate_value, bool)
            and isinstance(candidate_value, int | float)
            and not isinstance(baseline_value, bool)
            and isinstance(baseline_value, int | float)
            and math.isfinite(float(candidate_value))
            and math.isfinite(float(baseline_value))
            and float(baseline_value) > 0
        ):
            return name, float(candidate_value), float(baseline_value)
    msg = "evaluation reports do not share a finite positive CE metric"
    raise ValueError(msg)


def compare_pretrain_evaluations(
    candidate_path: Path,
    baseline_path: Path,
    *,
    noninferiority_tolerance: float = 0.01,
) -> dict[str, Any]:
    """比较同一 validation plan 的候选与基线，并执行相对 CE 非劣门槛。"""
    if (
        isinstance(noninferiority_tolerance, bool)
        or not isinstance(noninferiority_tolerance, int | float)
        or not math.isfinite(float(noninferiority_tolerance))
        or noninferiority_tolerance < 0
    ):
        msg = "noninferiority_tolerance must be a finite non-negative number"
        raise ValueError(msg)
    candidate_path = Path(candidate_path).resolve()
    baseline_path = Path(baseline_path).resolve()
    if candidate_path == baseline_path:
        msg = "candidate and baseline evaluations must be different artifacts"
        raise ValueError(msg)
    candidate = _load_report(candidate_path)
    baseline = _load_report(baseline_path)
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

    metric, candidate_value, baseline_value = _primary_metric(candidate, baseline)
    relative_change = (candidate_value - baseline_value) / baseline_value
    accepted = relative_change <= noninferiority_tolerance
    candidate_fields = candidate.get("per_field_ce", candidate.get("ce", {}))
    baseline_fields = baseline.get("per_field_ce", baseline.get("ce", {}))
    if not isinstance(candidate_fields, dict) or not isinstance(baseline_fields, dict):
        msg = "candidate and baseline per-field CE metrics must be objects"
        raise TypeError(msg)
    shared_fields = sorted(set(candidate_fields) & set(baseline_fields))
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
        "validation_windows": candidate.get("validation_windows"),
        "primary_metric": metric,
        "candidate_value": candidate_value,
        "baseline_value": baseline_value,
        "relative_change": relative_change,
        "noninferiority_tolerance": noninferiority_tolerance,
        "accepted": accepted,
        "decision": "PASS" if accepted else "FAIL",
        "per_field_ce": fields,
    }


def validate_pretrain_acceptance(path: Path) -> dict[str, Any]:
    """Load a v2 acceptance artifact and reverify its sources and explicit PASS."""
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
    if payload["primary_metric"] not in {"total_normalized_ce", "total_ce"}:
        msg = f"unsupported pretrain acceptance metric: {payload['primary_metric']!r}"
        raise ValueError(msg)
    if (
        not isinstance(payload["validation_plan_source_fingerprint"], str)
        or not payload["validation_plan_source_fingerprint"]
    ):
        msg = "pretrain acceptance has no validation-plan fingerprint"
        raise ValueError(msg)
    validation_windows = payload["validation_windows"]
    if type(validation_windows) is not int or validation_windows <= 0:
        msg = "pretrain acceptance validation_windows must be a positive integer"
        raise ValueError(msg)
    if not isinstance(payload["per_field_ce"], dict):
        msg = "pretrain acceptance per_field_ce must be an object"
        raise TypeError(msg)

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
    if numeric["noninferiority_tolerance"] < 0:
        msg = "pretrain acceptance noninferiority_tolerance must be non-negative"
        raise ValueError(msg)

    recomputed_change = (
        numeric["candidate_value"] - numeric["baseline_value"]
    ) / numeric["baseline_value"]
    if not math.isclose(
        numeric["relative_change"], recomputed_change, rel_tol=1e-12, abs_tol=1e-12
    ):
        msg = "pretrain acceptance relative_change is inconsistent with CE values"
        raise ValueError(msg)
    recomputed_pass = recomputed_change <= numeric["noninferiority_tolerance"]
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
        noninferiority_tolerance=numeric["noninferiority_tolerance"],
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
            "该门槛只判断预训练 validation 非劣；最终晋级仍需 RankIC、成本后回测和稳定性验收。",
            "",
        ]
    )
    return "\n".join(rows)
