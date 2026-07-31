from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch

from quant_fm.downstream.train_ranker import (
    CrossSectionalRanker,
    RankerConfig,
    RankerObjectiveConfig,
    fit_ranker,
    ranker_objective_loss,
    sampled_lambda_ndcg_loss,
)


def _gain(label: torch.Tensor) -> torch.Tensor:
    return ((label - 0.5).clamp_min(0.0) / 0.5).square()


def test_lambda_ndcg_gradient_corrects_reversed_head_order() -> None:
    label = torch.linspace(0.0, 1.0, 12)
    gain = _gain(label)
    pred = torch.linspace(1.0, -1.0, 12, requires_grad=True)
    objective = RankerObjectiveConfig(
        ndcg_ks=(3, 6, 10),
        ndcg_k_weights=(0.2, 0.6, 0.2),
        global_ic_weight=0.0,
        aux_huber_weight=0.0,
        pair_samples_per_day=512,
    )

    loss = sampled_lambda_ndcg_loss(
        pred,
        label,
        gain,
        objective=objective,
        seed=17,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert pred.grad is not None
    assert pred.grad[-1] < 0  # gradient descent raises the best name's score
    updated = (pred - 0.05 * pred.grad).detach()
    assert updated[-1] - updated[0] > pred.detach()[-1] - pred.detach()[0]
    updated_loss = sampled_lambda_ndcg_loss(
        updated,
        label,
        gain,
        objective=objective,
        seed=17,
    )
    assert updated_loss < loss.detach()


def test_lambda_ndcg_is_finite_deterministic_and_bounded_for_small_n() -> None:
    pred = torch.tensor([0.2, -0.1, 0.4], requires_grad=True)
    label = torch.tensor([0.5, 0.0, 1.0])
    gain = _gain(label)
    objective = RankerObjectiveConfig(pair_samples_per_day=8192)

    first = sampled_lambda_ndcg_loss(
        pred,
        label,
        gain,
        objective=objective,
        seed=91,
    )
    second = sampled_lambda_ndcg_loss(
        pred,
        label,
        gain,
        objective=objective,
        seed=91,
    )
    first.backward()

    assert torch.isfinite(first)
    assert torch.equal(first.detach(), second.detach())
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_auxiliary_head_uses_shared_hidden_but_not_score_head() -> None:
    torch.manual_seed(3)
    model = CrossSectionalRanker(
        RankerConfig(
            in_dim=3,
            hidden=8,
            depth=1,
            n_heads=2,
            dropout=0.0,
            use_attention=False,
        )
    )
    x = torch.randn(7, 3)
    label = torch.linspace(0.0, 1.0, 7)
    aux_target = torch.linspace(-1.0, 1.0, 7)
    score, aux_pred = model.forward_with_aux(x)
    objective = RankerObjectiveConfig(
        head_weight=0.0,
        global_ic_weight=0.0,
        aux_huber_weight=0.05,
        pair_samples_per_day=32,
    )

    loss = ranker_objective_loss(
        score,
        label,
        _gain(label),
        aux_pred=aux_pred,
        aux_target=aux_target,
        objective=objective,
    )
    loss.backward()

    assert score.shape == aux_pred.shape == (7,)
    assert torch.equal(model(x), score.detach())
    assert model.aux_out.weight.grad is not None
    assert model.aux_out.weight.grad.abs().sum() > 0
    assert model.proj.weight.grad is not None
    assert model.proj.weight.grad.abs().sum() > 0
    assert model.out.weight.grad is not None
    assert torch.count_nonzero(model.out.weight.grad) == 0


def test_model_and_sampled_loss_are_permutation_equivariant() -> None:
    torch.manual_seed(5)
    model = CrossSectionalRanker(
        RankerConfig(
            in_dim=4,
            hidden=8,
            depth=1,
            n_heads=2,
            dropout=0.0,
            use_attention=True,
        )
    ).eval()
    x = torch.randn(16, 4)
    label = torch.linspace(0.0, 1.0, 16)
    gain = _gain(label)
    permutation = torch.tensor([7, 2, 15, 0, 9, 4, 12, 1, 14, 6, 3, 11, 5, 8, 13, 10])
    score, aux = model.forward_with_aux(x)
    permuted_score, permuted_aux = model.forward_with_aux(x[permutation])

    assert torch.allclose(permuted_score, score[permutation], atol=1e-6)
    assert torch.allclose(permuted_aux, aux[permutation], atol=1e-6)
    objective = RankerObjectiveConfig(
        ndcg_ks=(5, 10, 15),
        ndcg_k_weights=(0.2, 0.6, 0.2),
        aux_huber_weight=0.0,
        pair_samples_per_day=64,
    )
    original_loss = ranker_objective_loss(
        score,
        label,
        gain,
        objective=objective,
        seed=123,
    )
    permuted_loss = ranker_objective_loss(
        permuted_score,
        label[permutation],
        gain[permutation],
        objective=objective,
        seed=123,
    )
    assert torch.allclose(original_loss, permuted_loss, atol=1e-7)


def test_fit_ranker_reports_exact_multi_k_metrics_and_requires_aux_target() -> None:
    rows = [
        {
            "date": date,
            "symbol": f"{index:06d}",
            "label": index / 5,
            "head_gain": max((index / 5 - 0.5) / 0.5, 0.0) ** 2,
            "aux_target": (index - 2.5) / 2.5,
            "emb_0": float(index),
            "emb_1": float(index % 2),
        }
        for date in ("2025-01-01", "2025-01-02")
        for index in range(6)
    ]
    features = pl.DataFrame(rows)
    result = fit_ranker(
        features,
        epochs=1,
        hidden=8,
        depth=1,
        dropout=0.0,
        use_attention=False,
        seed=11,
    )

    assert result.objective.ndcg_ks == (50, 300, 350)
    assert np.isfinite(result.history[0]["train_ndcg"])
    assert all(f"train_ndcg_{cutoff}" in result.history[0] for cutoff in (50, 300, 350))
    with pytest.raises(ValueError, match="must contain aux_target"):
        fit_ranker(
            features.drop("aux_target"),
            epochs=1,
            hidden=8,
            depth=1,
            dropout=0.0,
            use_attention=False,
        )
