from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from quant_fm.pretrain.heads import (  # noqa: E402
    TargetSpec,
    next_event_loss_v2,
    target_specs_from_config,
)
from quant_fm.pretrain.train import evaluate  # noqa: E402


def _batch(target: list[int], *, event_type: list[int] | None = None):
    length = len(target)
    return {
        "tok_target": torch.tensor([target]),
        "tok_evt_type": torch.tensor([event_type or [1] * length]),
        "attention_mask": torch.ones((1, length), dtype=torch.bool),
    }


def test_entropy_normalization_and_weight_are_explicit() -> None:
    logits = {"tok_target": torch.zeros(1, 2, 3, requires_grad=True)}
    spec = TargetSpec(name="tok_target", entropy=math.log(3), weight=2.0)
    output = next_event_loss_v2(logits, _batch([1, 2]), (spec,))

    assert output.per_field["tok_target"].item() == pytest.approx(math.log(3))
    assert output.normalized_per_field["tok_target"].item() == pytest.approx(1.0)
    assert output.total.item() == pytest.approx(2.0)


def test_applicability_mask_blocks_invalid_position_gradients() -> None:
    prediction = torch.zeros(1, 4, 7, requires_grad=True)
    logits = {"tok_target": prediction}
    spec = TargetSpec(name="tok_target", applicable_event_ids=(5,))
    batch = _batch([1, 2, 3, 4], event_type=[1, 1, 5, 1])

    output = next_event_loss_v2(logits, batch, (spec,))
    output.total.backward()

    assert output.valid_counts["tok_target"] == 1
    assert torch.count_nonzero(prediction.grad[0, 0]).item() == 0
    assert torch.count_nonzero(prediction.grad[0, 1]).item() > 0
    assert torch.count_nonzero(prediction.grad[0, 2:]).item() == 0


def test_v2_omitted_session_target_is_disabled() -> None:
    specs = target_specs_from_config(
        ("tok_evt_type", "tok_session"),
        {"targets": {"tok_evt_type": {"type": "ce", "weight": 1.0}}},
    )

    assert specs is not None
    assert {spec.name: spec.weight for spec in specs} == {
        "tok_evt_type": 1.0,
        "tok_session": 0.0,
    }


def test_ordinal_penalty_prefers_adjacent_bin() -> None:
    near = torch.full((1, 2, 7), -5.0)
    far = near.clone()
    near[0, 0, 4] = 5.0
    far[0, 0, 6] = 5.0
    spec = TargetSpec(
        name="tok_target",
        loss_type="ordinal_ce",
        ordinal_weight=1.0,
    )
    batch = _batch([1, 3])

    near_loss = next_event_loss_v2({"tok_target": near}, batch, (spec,))
    far_loss = next_event_loss_v2({"tok_target": far}, batch, (spec,))

    assert near_loss.per_field["tok_target"].item() == pytest.approx(
        far_loss.per_field["tok_target"].item()
    )
    assert near_loss.total.item() < far_loss.total.item()


def test_all_na_task_is_finite_and_backward_safe() -> None:
    prediction = torch.randn(1, 3, 5, requires_grad=True)
    spec = TargetSpec(name="tok_target", ignore_ids=(0, 2))
    output = next_event_loss_v2(
        {"tok_target": prediction},
        _batch([2, 2, 2]),
        (spec,),
    )

    assert output.valid_counts["tok_target"] == 0
    assert torch.isfinite(output.total)
    output.total.backward()
    assert torch.count_nonzero(prediction.grad).item() == 0


class _EvaluationModel(torch.nn.Module):
    def forward(self, batch):
        target = batch["tok_target"]
        logits = torch.zeros((*target.shape, 4), device=target.device)
        if int(batch["case"].item()) == 1:
            logits[:, :-1].scatter_(2, target[:, 1:].unsqueeze(-1), 5.0)
        return {"tok_target": logits}


def test_validation_loss_is_token_weighted_across_batches() -> None:
    first = {
        **_batch([1, 2]),
        "case": torch.tensor(0),
    }
    second = {
        **_batch([1, 2, 3, 1]),
        "case": torch.tensor(1),
    }
    spec = TargetSpec(name="tok_target")
    first_loss = next_event_loss_v2(
        _EvaluationModel()(first), first, (spec,)
    ).total.item()
    second_loss = next_event_loss_v2(
        _EvaluationModel()(second), second, (spec,)
    ).total.item()

    actual = evaluate(
        _EvaluationModel(),  # type: ignore[arg-type]
        [first, second],  # type: ignore[arg-type]
        torch.device("cpu"),
        ("tok_target",),
        target_specs=(spec,),
        max_batches=2,
    )

    assert actual == pytest.approx((first_loss + 3 * second_loss) / 4)
