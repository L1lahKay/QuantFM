import math

import numpy as np
import pytest
import torch
from torch import nn

from quant_fm.manifest.build_manifest import ShardEntry
from quant_fm.pretrain.eval import (
    _collect_unigram_counts_with_stats,
    _normalization_contract,
    collect_unigram_counts,
    evaluation_batch_size,
    field_diagnostics,
    per_field_gradient_norms,
    require_exact_plan_windows,
    resolve_checkpoint_target_fields,
    resolve_unigram_windows,
    unigram_entropy,
)
from quant_fm.pretrain.heads import MultiHead
from quant_fm.pretrain.validation_sampler import build_validation_sample_plan


class _FixedLogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size = batch["tok_evt_type"].size(0)
        return {"tok_evt_type": self.fixed_logits.expand(batch_size, -1, -1)}


class _TinyCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(4, 3)
        self.head = MultiHead(3, {"tok_evt_type": 4})

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.embedding(batch["tok_evt_type"])

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.head(self.encode(batch))


def _batch() -> dict[str, torch.Tensor]:
    return {
        "tok_evt_type": torch.tensor([[1, 1, 2, 2, 3]], dtype=torch.long),
        "attention_mask": torch.ones((1, 5), dtype=torch.bool),
        "length": torch.tensor([5]),
    }


def test_field_diagnostics_reports_token_weighted_metrics() -> None:
    # Predictions at t=0..3 are [1, 1, 2, 3], while targets are [1, 2, 2, 3].
    logits = torch.full((1, 5, 4), -2.0)
    for position, predicted in enumerate((1, 1, 2, 3)):
        logits[0, position, predicted] = 2.0
    model = _FixedLogitModel(logits)
    model.train()

    report = field_diagnostics(
        model,
        [_batch()],  # type: ignore[arg-type]
        torch.device("cpu"),
        ("tok_evt_type",),
        train_unigram_counts={"tok_evt_type": np.array([0, 1, 2, 1])},
        max_batches=1,
        copy_epsilon=0.1,
    )

    field = report.fields["tok_evt_type"]
    targets = torch.tensor([1, 2, 2, 3])
    expected_ce = torch.nn.functional.cross_entropy(logits[0, :4], targets).item()
    expected_copy_ce = (-2 * math.log(0.9) - 2 * math.log(0.05)) / 4
    assert field.ce == pytest.approx(expected_ce)
    assert field.perplexity == pytest.approx(math.exp(expected_ce))
    assert field.top1_accuracy == pytest.approx(0.75)
    assert field.balanced_accuracy == pytest.approx((1.0 + 0.5 + 1.0) / 3)
    assert field.unigram_entropy == pytest.approx(unigram_entropy([0, 1, 2, 1]))
    assert field.normalized_ce == pytest.approx(field.ce / field.unigram_entropy)
    assert field.copy_baseline_ce == pytest.approx(expected_copy_ce)
    assert report.n_predictions == {"tok_evt_type": 4}
    assert report.evaluated_windows == 1
    assert report.unigram_source == "train"
    assert model.training  # Evaluation restores the caller's model mode.

    payload = report.to_dict()
    assert payload["ce"] == payload["per_field_ce"]
    assert payload["copy_previous_event_ce"]["tok_evt_type"] == pytest.approx(
        expected_copy_ce
    )


def test_field_diagnostics_fully_consumes_loader_without_batch_cap() -> None:
    logits = torch.zeros((1, 5, 4))
    report = field_diagnostics(
        _FixedLogitModel(logits),
        [_batch(), _batch()],  # type: ignore[arg-type]
        torch.device("cpu"),
        ("tok_evt_type",),
        train_unigram_counts={"tok_evt_type": np.array([0, 1, 2, 1])},
        max_batches=None,
    )

    assert report.evaluated_windows == 2
    assert report.n_predictions == {"tok_evt_type": 8}


def test_unigram_window_limit_is_an_exact_device_independent_count() -> None:
    assert (
        resolve_unigram_windows(
            unigram_windows=137,
            legacy_unigram_max_batches=None,
        )
        == 137
    )
    assert (
        resolve_unigram_windows(
            unigram_windows=None,
            legacy_unigram_max_batches=137,
        )
        == 137
    )
    assert (
        resolve_unigram_windows(
            unigram_windows=None,
            legacy_unigram_max_batches=None,
        )
        == 200
    )
    with pytest.raises(ValueError, match="must agree"):
        resolve_unigram_windows(
            unigram_windows=137,
            legacy_unigram_max_batches=200,
        )


def test_unigram_plan_identity_is_independent_of_device_batch_size() -> None:
    cfg = {"optim": {"micro_batch_size": 8}}
    assert evaluation_batch_size(cfg, torch.device("cpu")) == 1
    assert evaluation_batch_size(cfg, torch.device("cuda")) == 8
    window_limit = resolve_unigram_windows(
        unigram_windows=3,
        legacy_unigram_max_batches=None,
    )
    shards = [
        ShardEntry(
            market="SZ",
            symbol="000001",
            date="2026-01-05",
            path="/tokens/000001.parquet",
            rows=32,
            sha256="a" * 64,
            split="train",
        )
    ]
    plans = [
        build_validation_sample_plan(
            shards,
            context=8,
            stride=8,
            min_len=4,
            seed=42,
            max_windows=window_limit,
        )
        for _device in (torch.device("cpu"), torch.device("cuda"))
    ]
    assert plans[0].sha256 == plans[1].sha256
    assert len(plans[0].windows) == 3


def test_exact_plan_window_requirement_rejects_candidate_shortage() -> None:
    shards = [
        ShardEntry(
            market="SZ",
            symbol="000001",
            date="2026-01-05",
            path="/tokens/000001.parquet",
            rows=8,
            sha256="a" * 64,
            split="train",
        )
    ]
    exact = build_validation_sample_plan(
        shards,
        context=8,
        stride=8,
        min_len=4,
        seed=42,
        max_windows=1,
    )
    require_exact_plan_windows(exact, requested_windows=1, context="test plan")

    short = build_validation_sample_plan(
        shards,
        context=8,
        stride=8,
        min_len=4,
        seed=42,
        max_windows=2,
    )
    with pytest.raises(ValueError, match="requires exactly 2 windows"):
        require_exact_plan_windows(short, requested_windows=2, context="test plan")

    stratum_short = build_validation_sample_plan(
        [
            ShardEntry(
                market="SZ",
                symbol="000001",
                date="2026-01-05",
                path="/tokens/many.parquet",
                rows=32,
                sha256="b" * 64,
                split="train",
            )
        ],
        context=8,
        stride=8,
        min_len=4,
        seed=42,
        max_windows=2,
        windows_per_stratum=1,
    )
    assert stratum_short.total_candidate_windows == 4
    with pytest.raises(ValueError, match="requires exactly 2 windows"):
        require_exact_plan_windows(
            stratum_short,
            requested_windows=2,
            context="stratified test plan",
        )


@pytest.mark.parametrize(
    "configured",
    [
        ("tok_side",),
        ["tok_evt_type"],
        ["tok_side", "tok_evt_type"],
        ["tok_side", "tok_evt_type", "extra"],
    ],
)
def test_checkpoint_target_fields_reject_yaml_substitution(
    configured: object,
) -> None:
    with pytest.raises(ValueError, match="exactly match checkpoint"):
        resolve_checkpoint_target_fields(
            ("tok_evt_type", "tok_side"),
            configured,
        )


def test_checkpoint_target_fields_are_the_only_source_of_truth() -> None:
    expected = ("tok_evt_type", "tok_side")
    assert resolve_checkpoint_target_fields(expected, None) == expected
    assert resolve_checkpoint_target_fields(expected, list(expected)) == expected
    with pytest.raises(ValueError, match="unique non-empty"):
        resolve_checkpoint_target_fields((), None)
    with pytest.raises(ValueError, match="unique non-empty"):
        resolve_checkpoint_target_fields(("tok_evt_type", "tok_evt_type"), None)


def test_normalization_v3_hash_binds_counts_preimage() -> None:
    shards = [
        ShardEntry(
            market="SZ",
            symbol="000001",
            date="2026-01-05",
            path="/tokens/000001.parquet",
            rows=8,
            sha256="a" * 64,
            split="train",
        )
    ]
    plan = build_validation_sample_plan(
        shards,
        context=8,
        stride=8,
        min_len=4,
        seed=42,
        max_windows=1,
    )
    first = _normalization_contract(
        target_fields=("tok_evt_type",),
        train_plan=plan,
        unigram_counts={"tok_evt_type": np.array([0, 2, 1])},
    )
    second = _normalization_contract(
        target_fields=("tok_evt_type",),
        train_plan=plan,
        unigram_counts={"tok_evt_type": np.array([0, 1, 2])},
    )

    assert first["format_version"] == "train_unigram_normalization_v3"
    assert first["train_unigram_counts"] == {"tok_evt_type": [0, 2, 1]}
    assert first["sha256"] != second["sha256"]


def test_collect_unigram_counts_uses_only_valid_next_targets() -> None:
    batch = {
        "tok_evt_type": torch.tensor([[1, 2, 3, 0], [2, 2, 0, 0]]),
        "attention_mask": torch.tensor(
            [[True, True, True, False], [True, True, False, False]]
        ),
        "length": torch.tensor([3, 2]),
    }
    counts = collect_unigram_counts(
        [batch],  # type: ignore[arg-type]
        ("tok_evt_type",),
        field_sizes={"tok_evt_type": 5},
        max_batches=1,
    )
    # Sequence one contributes targets 2, 3; sequence two contributes target 2.
    assert counts["tok_evt_type"].tolist() == [0, 0, 2, 1, 0]


def test_unigram_count_stats_bind_masked_per_field_consumption() -> None:
    batch = {
        "tok_evt_type": torch.tensor([[1, 2, 3, 1], [2, 2, 1, 3]]),
        "tok_side": torch.tensor([[1, 2, 0, 1], [2, 0, 0, 1]]),
        "mask_tok_evt_type": torch.ones((2, 4), dtype=torch.bool),
        "mask_tok_side": torch.tensor(
            [[True, True, False, True], [True, False, False, True]]
        ),
        "attention_mask": torch.ones((2, 4), dtype=torch.bool),
        "length": torch.tensor([4, 4]),
    }
    counts, evaluated_windows, prediction_counts = _collect_unigram_counts_with_stats(
        [batch],  # type: ignore[arg-type]
        ("tok_evt_type", "tok_side"),
        field_sizes={"tok_evt_type": 5, "tok_side": 4},
        max_batches=None,
    )

    assert evaluated_windows == 2
    assert prediction_counts == {"tok_evt_type": 6, "tok_side": 3}
    assert {
        field: int(values.sum()) for field, values in counts.items()
    } == prediction_counts
    assert prediction_counts["tok_side"] < 2 * (4 - 1)


def test_unigram_entropy_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        unigram_entropy([1, -1])


def test_field_diagnostics_rejects_empty_targets() -> None:
    logits = torch.zeros((1, 1, 4))
    model = _FixedLogitModel(logits)
    batch = {
        "tok_evt_type": torch.tensor([[1]]),
        "attention_mask": torch.tensor([[True]]),
        "length": torch.tensor([1]),
    }
    with pytest.raises(ValueError, match="no valid next-event"):
        field_diagnostics(
            model,
            [batch],  # type: ignore[arg-type]
            torch.device("cpu"),
            ("tok_evt_type",),
            train_unigram_counts={"tok_evt_type": np.array([0, 1, 1, 1])},
            max_batches=1,
        )


def test_per_field_gradient_norm_reports_shared_hidden_pressure() -> None:
    model = _TinyCausalModel()
    model.train()
    norms = per_field_gradient_norms(
        model,  # type: ignore[arg-type]
        [_batch()],  # type: ignore[arg-type]
        torch.device("cpu"),
        ("tok_evt_type",),
    )

    assert math.isfinite(norms["tok_evt_type"])
    assert norms["tok_evt_type"] > 0
    assert model.training
