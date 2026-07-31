import json
from pathlib import Path

import pytest

from quant_fm.monitoring.acceptance import (
    compare_pretrain_evaluations,
    render_acceptance_report,
    validate_pretrain_acceptance,
)


def _report(path: Path, *, loss: float, fingerprint: str = "same") -> None:
    path.write_text(
        json.dumps(
            {
                "split": "val",
                "validation_plan_source_fingerprint": fingerprint,
                "validation_windows": 800,
                "total_normalized_ce": loss,
                "total_ce": loss * 2,
                "per_field_ce": {"tok_evt_type": loss / 2},
            }
        ),
        encoding="utf-8",
    )


def test_noninferiority_gate_passes_within_one_percent(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=5.04)
    _report(baseline, loss=5.0)
    result = compare_pretrain_evaluations(candidate, baseline)
    assert result["acceptance_version"] == "2.0"
    assert len(result["candidate_sha256"]) == 64
    assert len(result["baseline_sha256"]) == 64
    assert result["accepted"] is True
    assert result["relative_change"] == pytest.approx(0.008)
    assert "决策：**PASS**" in render_acceptance_report(result)


def test_noninferiority_gate_fails_regression(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=5.2)
    _report(baseline, loss=5.0)
    result = compare_pretrain_evaluations(candidate, baseline)
    assert result["accepted"] is False
    assert result["decision"] == "FAIL"


def test_noninferiority_gate_rejects_different_windows(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=5.0, fingerprint="candidate")
    _report(baseline, loss=5.0, fingerprint="baseline")
    with pytest.raises(ValueError, match="same validation plan"):
        compare_pretrain_evaluations(candidate, baseline)


def _acceptance_artifact(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    artifact = tmp_path / "acceptance.json"
    _report(candidate, loss=5.04)
    _report(baseline, loss=5.0)
    payload = compare_pretrain_evaluations(candidate, baseline)
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    return artifact, payload


def test_strict_acceptance_validator_requires_consistent_explicit_pass(
    tmp_path: Path,
) -> None:
    artifact, payload = _acceptance_artifact(tmp_path)
    assert validate_pretrain_acceptance(artifact) == payload

    payload["decision"] = "FAIL"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="accepted and decision"):
        validate_pretrain_acceptance(artifact)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("acceptance_version", "1.0", "unsupported pretrain acceptance version"),
        ("accepted", False, "decision is inconsistent with the noninferiority gate"),
        ("relative_change", 0.5, "relative_change is inconsistent"),
    ],
)
def test_strict_acceptance_validator_fails_closed(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    artifact, payload = _acceptance_artifact(tmp_path)
    payload[field] = value
    if field == "accepted":
        payload["decision"] = "FAIL"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_pretrain_acceptance(artifact)


def test_strict_acceptance_validator_rejects_malformed_or_partial_json(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "acceptance.json"
    artifact.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        validate_pretrain_acceptance(artifact)

    artifact.write_text('{"accepted": true, "decision": "PASS"}', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        validate_pretrain_acceptance(artifact)


def test_strict_acceptance_validator_rejects_changed_source_report(
    tmp_path: Path,
) -> None:
    artifact, payload = _acceptance_artifact(tmp_path)
    candidate = Path(str(payload["candidate"]))
    _report(candidate, loss=4.9)

    with pytest.raises(ValueError, match="source SHA-256 has changed"):
        validate_pretrain_acceptance(artifact)


def test_strict_acceptance_validator_recomputes_source_decision(
    tmp_path: Path,
) -> None:
    artifact, payload = _acceptance_artifact(tmp_path)
    payload["candidate_value"] = 5.03
    payload["relative_change"] = (5.03 - 5.0) / 5.0
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="no longer matches its source reports"):
        validate_pretrain_acceptance(artifact)
