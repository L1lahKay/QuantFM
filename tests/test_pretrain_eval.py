import math

import numpy as np
import pytest
import torch
from torch import nn

from quant_fm.pretrain.eval import (
    collect_unigram_counts,
    field_diagnostics,
    per_field_gradient_norms,
    unigram_entropy,
)
from quant_fm.pretrain.heads import MultiHead


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
    assert report.unigram_source == "train"
    assert model.training  # Evaluation restores the caller's model mode.

    payload = report.to_dict()
    assert payload["ce"] == payload["per_field_ce"]
    assert payload["copy_previous_event_ce"]["tok_evt_type"] == pytest.approx(
        expected_copy_ce
    )


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
