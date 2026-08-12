import json
import math
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import torch

from quant_fm.manifest.build_manifest import ShardEntry
from quant_fm.monitoring import acceptance as acceptance_module
from quant_fm.monitoring.acceptance import (
    compare_pretrain_evaluations,
    render_acceptance_report,
    validate_pretrain_acceptance,
)
from quant_fm.pretrain.validation_sampler import (
    ValidationSamplePlan,
    build_validation_sample_plan,
)


def _canonical_sha(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(raw).hexdigest()


def _refresh_normalization_sha(payload: dict[str, object]) -> None:
    contract = {
        "format_version": "train_unigram_normalization_v3",
        "target_fields": payload["normalization_target_fields"],
        "train_unigram_plan_sha256": payload["train_unigram_plan_sha256"],
        "train_unigram_plan_source_fingerprint": payload[
            "train_unigram_plan_source_fingerprint"
        ],
        "train_unigram_windows": payload["train_unigram_windows"],
        "train_unigram_counts": payload["train_unigram_counts"],
        "train_unigram_counts_sha256": payload["train_unigram_counts_sha256"],
        "train_unigram_entropy": payload["train_unigram_entropy"],
    }
    payload["normalization_contract_sha256"] = _canonical_sha(contract)


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    return -math.fsum(
        (count / total) * math.log(count / total) for count in counts if count > 0
    )


def _plan(
    path: Path,
    *,
    seed: int = 7,
    max_windows: int = 2,
    split: str = "val",
) -> Path:
    shards = [
        ShardEntry(
            market="SH",
            symbol="600000",
            date="2026-01-05",
            path="/tokens/600000.parquet",
            rows=32,
            sha256="a" * 64,
            split=split,
        ),
        ShardEntry(
            market="SZ",
            symbol="000001",
            date="2026-01-06",
            path="/tokens/000001.parquet",
            rows=32,
            sha256="b" * 64,
            split=split,
        ),
    ]
    build_validation_sample_plan(
        shards,
        context=8,
        stride=8,
        min_len=4,
        seed=seed,
        max_windows=max_windows,
    ).save(path)
    return path


def _report(
    path: Path,
    *,
    loss: float,
    plan_path: Path | None = None,
    train_plan_path: Path | None = None,
    target_fields: tuple[str, ...] = ("tok_evt_type",),
    checkpoint_target_fields: tuple[str, ...] | None = None,
    counts_salt: int = 0,
) -> None:
    if plan_path is None:
        plan_path = path.parent / "validation-plan.json"
        if not plan_path.exists():
            _plan(plan_path)
    plan = ValidationSamplePlan.load(plan_path)
    if train_plan_path is None:
        train_plan_path = path.parent / "train-unigram-plan.json"
        if not train_plan_path.exists():
            _plan(train_plan_path, seed=11, max_windows=3, split="train")
    train_plan = ValidationSamplePlan.load(train_plan_path)
    counts_payload = {
        field: [0, counts_salt + index + 1, 2]
        for index, field in enumerate(target_fields)
    }
    counts_sha = _canonical_sha(counts_payload)
    entropy = {field: _entropy(counts_payload[field]) for field in target_fields}
    normalization_contract = {
        "format_version": "train_unigram_normalization_v3",
        "target_fields": list(target_fields),
        "train_unigram_plan_sha256": train_plan.sha256,
        "train_unigram_plan_source_fingerprint": train_plan.source_fingerprint,
        "train_unigram_windows": len(train_plan.windows),
        "train_unigram_counts": counts_payload,
        "train_unigram_counts_sha256": counts_sha,
        "train_unigram_entropy": entropy,
    }
    per_field_normalized_loss = loss / len(target_fields)
    per_field_ce = {
        field: per_field_normalized_loss * entropy[field] for field in target_fields
    }
    prediction_count = sum(max(window.length - 1, 0) for window in plan.windows)
    checkpoint = path.with_suffix(".pt")
    torch.save(
        {
            "config": {
                "target_fields": list(checkpoint_target_fields or target_fields),
            },
            "model_state": {},
        },
        checkpoint,
    )
    path.write_text(
        json.dumps(
            {
                "split": "val",
                "validation_plan": str(plan_path.resolve()),
                "validation_plan_source_fingerprint": plan.source_fingerprint,
                "validation_plan_sha256": plan.sha256,
                "validation_windows": len(plan.windows),
                "evaluated_windows": len(plan.windows),
                "n_predictions": dict.fromkeys(target_fields, prediction_count),
                "train_unigram_plan": str(train_plan_path.resolve()),
                "train_unigram_plan_source_fingerprint": (
                    train_plan.source_fingerprint
                ),
                "train_unigram_plan_sha256": train_plan.sha256,
                "train_unigram_windows": len(train_plan.windows),
                "train_unigram_evaluated_windows": len(train_plan.windows),
                "train_unigram_prediction_counts": {
                    field: sum(counts_payload[field]) for field in target_fields
                },
                "normalization_target_fields": list(target_fields),
                "checkpoint_target_fields": list(target_fields),
                "train_unigram_counts": counts_payload,
                "train_unigram_counts_sha256": counts_sha,
                "normalization_contract_sha256": _canonical_sha(normalization_contract),
                "unigram_source": "train",
                "train_unigram_entropy": entropy,
                "ce_over_unigram_entropy": dict.fromkeys(
                    target_fields, per_field_normalized_loss
                ),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
                "total_normalized_ce": loss,
                "total_ce": math.fsum(per_field_ce.values()),
                "per_field_ce": per_field_ce,
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
    assert result["acceptance_version"] == "8.0"
    assert result["primary_metric"] == "total_ce"
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


def test_noninferiority_gate_rejects_negative_candidate_ce(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=-1.0)
    _report(baseline, loss=1.0)

    with pytest.raises(ValueError, match="per_field_ce values must be non-negative"):
        compare_pretrain_evaluations(candidate, baseline)


def test_noninferiority_gate_rejects_different_windows(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    candidate_plan = _plan(tmp_path / "candidate-plan.json", max_windows=1)
    baseline_plan = _plan(tmp_path / "baseline-plan.json", max_windows=2)
    _report(candidate, loss=5.0, plan_path=candidate_plan)
    _report(baseline, loss=5.0, plan_path=baseline_plan)
    with pytest.raises(ValueError, match="same canonical validation plan"):
        compare_pretrain_evaluations(candidate, baseline)


def test_noninferiority_rejects_partial_consumption_of_same_plan(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=5.0)
    _report(baseline, loss=5.0)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["max_batches"] = 1
    payload["evaluated_windows"] -= 1
    payload["n_predictions"]["tok_evt_type"] -= 1
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="complete frozen validation plan"):
        compare_pretrain_evaluations(candidate, baseline)


def test_noninferiority_rejects_different_prediction_counts(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=5.0)
    _report(baseline, loss=5.0)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["n_predictions"]["tok_evt_type"] -= 1
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="window/prediction counts differ"):
        compare_pretrain_evaluations(candidate, baseline)


def test_noninferiority_rejects_different_train_unigram_plans(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    candidate_train = _plan(
        tmp_path / "candidate-train.json", max_windows=1, split="train"
    )
    baseline_train = _plan(
        tmp_path / "baseline-train.json", max_windows=2, split="train"
    )
    _report(candidate, loss=5.0, train_plan_path=candidate_train)
    _report(baseline, loss=5.0, train_plan_path=baseline_train)

    with pytest.raises(ValueError, match="same canonical train-unigram plan"):
        compare_pretrain_evaluations(candidate, baseline)


def test_noninferiority_rejects_changed_unigram_counts_or_entropy(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=5.0, counts_salt=0)
    _report(baseline, loss=5.0, counts_salt=1)
    with pytest.raises(ValueError, match="normalized CE denominators differ"):
        compare_pretrain_evaluations(candidate, baseline)

    _report(baseline, loss=5.0, counts_salt=0)
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload["train_unigram_entropy"]["tok_evt_type"] = 1.1
    _refresh_normalization_sha(payload)
    baseline.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="entropy is inconsistent with counts"):
        compare_pretrain_evaluations(candidate, baseline)


def test_forged_shared_counts_cannot_turn_raw_ce_regression_into_pass(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    target_fields = ("tok_evt_type", "tok_side")
    _report(candidate, loss=1.0, target_fields=target_fields)
    _report(baseline, loss=1.0, target_fields=target_fields)

    # These internally self-consistent empirical counts are deliberately not
    # derived from the live plan.  They give the first head a larger denominator
    # and can make normalized CE look better while raw CE regresses by 45%.
    forged_counts = {
        "tok_evt_type": [0, 5, 5],
        "tok_side": [0, 9, 1],
    }
    entropy = {field: _entropy(counts) for field, counts in forged_counts.items()}
    raw_ce = {
        candidate: {"tok_evt_type": 2.9, "tok_side": 0.0},
        baseline: {"tok_evt_type": 0.0, "tok_side": 2.0},
    }
    normalized_totals: dict[Path, float] = {}
    for path in (candidate, baseline):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["train_unigram_counts"] = forged_counts
        payload["train_unigram_counts_sha256"] = _canonical_sha(forged_counts)
        payload["train_unigram_entropy"] = entropy
        payload["train_unigram_prediction_counts"] = dict.fromkeys(target_fields, 10)
        payload["per_field_ce"] = raw_ce[path]
        payload["total_ce"] = math.fsum(raw_ce[path].values())
        payload["ce_over_unigram_entropy"] = {
            field: raw_ce[path][field] / entropy[field] for field in target_fields
        }
        payload["total_normalized_ce"] = math.fsum(
            payload["ce_over_unigram_entropy"].values()
        )
        normalized_totals[path] = payload["total_normalized_ce"]
        _refresh_normalization_sha(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")

    normalized_change = (
        normalized_totals[candidate] - normalized_totals[baseline]
    ) / normalized_totals[baseline]
    assert normalized_change < -0.01

    result = compare_pretrain_evaluations(candidate, baseline)
    assert result["primary_metric"] == "total_ce"
    assert result["relative_change"] == pytest.approx(0.45)
    assert result["decision"] == "FAIL"


def test_noninferiority_rejects_tampered_unigram_counts_preimage(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=5.0)
    _report(baseline, loss=5.0)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["train_unigram_counts"]["tok_evt_type"][1] += 1
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="train_unigram_counts SHA-256 is invalid"):
        compare_pretrain_evaluations(candidate, baseline)


def test_noninferiority_rejects_rehashed_counts_with_wrong_prediction_sum(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=5.0)
    _report(baseline, loss=5.0)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["train_unigram_counts"]["tok_evt_type"][1] += 1
    payload["train_unigram_counts_sha256"] = _canonical_sha(
        payload["train_unigram_counts"]
    )
    _refresh_normalization_sha(payload)
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="inconsistent with counts/plan"):
        compare_pretrain_evaluations(candidate, baseline)


def test_noninferiority_rejects_self_consistent_forged_entropy(
    tmp_path: Path,
) -> None:
    """Changing the claimed denominator cannot manufacture a normalized-CE pass."""
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=5.2)
    _report(baseline, loss=5.0)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    forged_entropy = payload["train_unigram_entropy"]["tok_evt_type"] * 2
    payload["train_unigram_entropy"]["tok_evt_type"] = forged_entropy
    payload["ce_over_unigram_entropy"]["tok_evt_type"] = 4.9
    payload["total_normalized_ce"] = 4.9
    payload["per_field_ce"]["tok_evt_type"] = 4.9 * forged_entropy
    payload["total_ce"] = payload["per_field_ce"]["tok_evt_type"]
    _refresh_normalization_sha(payload)
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="entropy is inconsistent with counts"):
        compare_pretrain_evaluations(candidate, baseline)


def test_noninferiority_rejects_train_unigram_consumption_tamper(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=5.0)
    _report(baseline, loss=5.0)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["train_unigram_evaluated_windows"] -= 1
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="complete train-unigram plan"):
        compare_pretrain_evaluations(candidate, baseline)


def test_noninferiority_rejects_checkpoint_target_subset(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=5.0)
    _report(baseline, loss=5.0)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["checkpoint_target_fields"] = []
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly match checkpoint_target_fields"):
        compare_pretrain_evaluations(candidate, baseline)


@pytest.mark.parametrize(
    ("reported_fields", "checkpoint_fields"),
    [
        (("tok_evt_type",), ("tok_evt_type", "tok_side")),
        (("tok_side", "tok_evt_type"), ("tok_evt_type", "tok_side")),
    ],
)
def test_noninferiority_rejects_heads_omitted_or_reordered_from_live_checkpoint(
    tmp_path: Path,
    reported_fields: tuple[str, ...],
    checkpoint_fields: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(
        candidate,
        loss=5.0,
        target_fields=reported_fields,
        checkpoint_target_fields=checkpoint_fields,
    )
    _report(
        baseline,
        loss=5.0,
        target_fields=reported_fields,
        checkpoint_target_fields=checkpoint_fields,
    )

    with pytest.raises(ValueError, match="ordered target_fields frozen"):
        compare_pretrain_evaluations(candidate, baseline)

    live_reader = acceptance_module._live_checkpoint_target_fields
    monkeypatch.setattr(
        acceptance_module,
        "_live_checkpoint_target_fields",
        lambda _path: list(reported_fields),
    )
    forged = compare_pretrain_evaluations(candidate, baseline)
    artifact = tmp_path / "forged-acceptance.json"
    artifact.write_text(json.dumps(forged), encoding="utf-8")
    monkeypatch.setattr(
        acceptance_module,
        "_live_checkpoint_target_fields",
        live_reader,
    )
    with pytest.raises(ValueError, match="ordered target_fields frozen"):
        validate_pretrain_acceptance(artifact)


def test_checkpoint_replacement_during_target_field_load_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=5.0)
    _report(baseline, loss=5.0)
    candidate_checkpoint = candidate.with_suffix(".pt")
    real_load = acceptance_module.torch.load
    replaced = False

    def replacing_load(*args, **kwargs):
        nonlocal replaced
        payload = real_load(*args, **kwargs)
        if not replaced:
            replaced = True
            torch.save(
                {
                    "config": {"target_fields": ["tok_evt_type"]},
                    "model_state": {"changed": torch.ones(1)},
                },
                candidate_checkpoint,
            )
        return payload

    monkeypatch.setattr(acceptance_module.torch, "load", replacing_load)
    with pytest.raises(ValueError, match="changed while its target_fields"):
        compare_pretrain_evaluations(candidate, baseline)


def test_live_checkpoint_head_reader_supports_resumable_numpy_rng_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "resume.pt"
    torch.save(
        {
            "config": {"target_fields": ["tok_evt_type", "tok_side"]},
            "model_state": {"weight": torch.ones(2)},
            "runtime_state_by_rank": [
                {"rng": {"numpy": np.random.RandomState(7).get_state()}}
            ],
        },
        checkpoint,
    )

    assert acceptance_module._live_checkpoint_target_fields(checkpoint) == [
        "tok_evt_type",
        "tok_side",
    ]


def test_noninferiority_rejects_per_field_set_mismatch(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    target_fields = ("tok_evt_type", "tok_side")
    _report(candidate, loss=5.0, target_fields=target_fields)
    _report(baseline, loss=5.0, target_fields=target_fields)
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload["per_field_ce"].pop("tok_side")
    baseline.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="per_field_ce fields are inconsistent"):
        compare_pretrain_evaluations(candidate, baseline)


def test_noninferiority_rejects_non_train_unigram_source(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=5.0)
    _report(baseline, loss=5.0)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["unigram_source"] = "evaluation_fallback"
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must use train unigram counts"):
        compare_pretrain_evaluations(candidate, baseline)


def test_noninferiority_rejects_nonpositive_train_entropy(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=5.0)
    _report(baseline, loss=5.0)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["train_unigram_entropy"]["tok_evt_type"] = 0.0
    _refresh_normalization_sha(payload)
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="entropy values must be positive"):
        compare_pretrain_evaluations(candidate, baseline)


def test_noninferiority_rejects_entropy_tampered_outside_contract(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=5.0)
    _report(baseline, loss=5.0)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["train_unigram_entropy"]["tok_evt_type"] = 2.0
    payload["ce_over_unigram_entropy"]["tok_evt_type"] = 2.5
    payload["total_normalized_ce"] = 2.5
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="entropy is inconsistent with counts"):
        compare_pretrain_evaluations(candidate, baseline)


def test_noninferiority_rejects_tampered_normalized_field(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=5.0)
    _report(baseline, loss=5.0)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["ce_over_unigram_entropy"]["tok_evt_type"] += 0.25
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="normalized CE is inconsistent"):
        compare_pretrain_evaluations(candidate, baseline)


def test_noninferiority_rejects_tampered_per_field_ce(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=5.0)
    _report(baseline, loss=5.0)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["per_field_ce"]["tok_evt_type"] += 0.25
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="normalized CE is inconsistent"):
        compare_pretrain_evaluations(candidate, baseline)


def test_noninferiority_rejects_negative_normalized_field(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=1.0)
    _report(baseline, loss=1.0)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["ce_over_unigram_entropy"]["tok_evt_type"] = -1.0
    payload["total_normalized_ce"] = -1.0
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="normalized CE values must be non-negative"):
        compare_pretrain_evaluations(candidate, baseline)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("total_ce", 5.25, "total_ce is inconsistent"),
        ("total_normalized_ce", 5.25, "total_normalized_ce is inconsistent"),
        ("total_ce", -0.25, "total_ce must be non-negative"),
        (
            "total_normalized_ce",
            -0.25,
            "total_normalized_ce must be non-negative",
        ),
    ],
)
def test_noninferiority_rejects_tampered_ce_total(
    tmp_path: Path,
    field: str,
    value: float,
    message: str,
) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=5.0)
    _report(baseline, loss=5.0)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload[field] = value
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        compare_pretrain_evaluations(candidate, baseline)


def test_comparison_rejects_tampered_live_train_unigram_plan(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=5.04)
    _report(baseline, loss=5.0)
    train_plan = tmp_path / "train-unigram-plan.json"
    payload = json.loads(train_plan.read_text(encoding="utf-8"))
    payload["seed"] += 1
    train_plan.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical SHA-256 is invalid"):
        compare_pretrain_evaluations(candidate, baseline)


def _acceptance_artifact(
    tmp_path: Path,
    *,
    tolerance: float = 0.01,
) -> tuple[Path, dict[str, object]]:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    artifact = tmp_path / "acceptance.json"
    _report(candidate, loss=5.04)
    _report(baseline, loss=5.0)
    payload = compare_pretrain_evaluations(
        candidate,
        baseline,
        noninferiority_tolerance=tolerance,
    )
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


def test_acceptance_tolerance_is_an_independent_validator_policy(
    tmp_path: Path,
) -> None:
    custom_artifact, custom_payload = _acceptance_artifact(
        tmp_path / "custom",
        tolerance=0.02,
    )
    assert (
        validate_pretrain_acceptance(
            custom_artifact,
            expected_noninferiority_tolerance=0.02,
        )
        == custom_payload
    )
    with pytest.raises(ValueError, match="independently configured expected"):
        validate_pretrain_acceptance(custom_artifact)

    default_artifact, _ = _acceptance_artifact(tmp_path / "default")
    with pytest.raises(ValueError, match="independently configured expected"):
        validate_pretrain_acceptance(
            default_artifact,
            expected_noninferiority_tolerance=0.02,
        )


def test_artifact_cannot_self_authorize_a_wider_tolerance(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    artifact = tmp_path / "acceptance.json"
    _report(candidate, loss=5.2)
    _report(baseline, loss=5.0)
    payload = compare_pretrain_evaluations(candidate, baseline)
    assert payload["accepted"] is False
    payload["noninferiority_tolerance"] = 1.0
    payload["accepted"] = True
    payload["decision"] = "PASS"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="independently configured expected"):
        validate_pretrain_acceptance(artifact)


@pytest.mark.parametrize("expected", [True, -0.01, float("nan"), float("inf")])
def test_acceptance_rejects_invalid_expected_tolerance_before_artifact_io(
    tmp_path: Path,
    expected: object,
) -> None:
    with pytest.raises(ValueError, match="expected_noninferiority_tolerance"):
        validate_pretrain_acceptance(
            tmp_path / "missing.json",
            expected_noninferiority_tolerance=expected,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("acceptance_version", "7.0", "unsupported pretrain acceptance version"),
        ("accepted", False, "decision is inconsistent with the noninferiority gate"),
        ("relative_change", 0.5, "relative_change is inconsistent"),
        ("candidate_value", -1.0, "candidate_value must be non-negative"),
        ("baseline_value", -1.0, "baseline_value must be positive"),
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


def test_comparison_rejects_checkpoint_changed_after_evaluation(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    _report(candidate, loss=5.04)
    _report(baseline, loss=5.0)
    candidate.with_suffix(".pt").write_bytes(b"replacement checkpoint")

    with pytest.raises(ValueError, match="checkpoint SHA-256 has changed"):
        compare_pretrain_evaluations(candidate, baseline)


def test_strict_acceptance_validator_recomputes_source_decision(
    tmp_path: Path,
) -> None:
    artifact, payload = _acceptance_artifact(tmp_path)
    payload["candidate_value"] -= 0.001
    payload["relative_change"] = (
        payload["candidate_value"] - payload["baseline_value"]
    ) / payload["baseline_value"]
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="no longer matches its source reports"):
        validate_pretrain_acceptance(artifact)


def test_strict_acceptance_validator_rejects_counts_preimage_tamper(
    tmp_path: Path,
) -> None:
    artifact, payload = _acceptance_artifact(tmp_path)
    payload["train_unigram_counts"]["tok_evt_type"][1] += 1
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="train_unigram_counts SHA-256 is invalid"):
        validate_pretrain_acceptance(artifact)


def test_strict_acceptance_validator_rejects_rehashed_counts_sum_tamper(
    tmp_path: Path,
) -> None:
    artifact, payload = _acceptance_artifact(tmp_path)
    payload["train_unigram_counts"]["tok_evt_type"][1] += 1
    payload["train_unigram_counts_sha256"] = _canonical_sha(
        payload["train_unigram_counts"]
    )
    _refresh_normalization_sha(payload)
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="prediction count is inconsistent"):
        validate_pretrain_acceptance(artifact)
