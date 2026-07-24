import torch

from quant_fm.moe.config import RegimeMoEConfig
from quant_fm.moe.temporal_moe import TemporalRegimeMoE


def test_moe_has_no_cross_sample_or_order_leakage() -> None:
    torch.manual_seed(9)
    config = RegimeMoEConfig(
        enabled=True,
        n_experts=3,
        top_k=2,
        expert_hidden=10,
        router_hidden=7,
        capacity_factor=10.0,
        dropout=0.0,
    )
    model = TemporalRegimeMoE(5, 4, config).eval()
    hidden = torch.randn(6, 5)
    features = torch.randn(6, 4)
    expected = model(hidden, features).hidden

    changed_hidden = hidden.clone()
    changed_features = features.clone()
    changed_hidden[5] *= 100
    changed_features[5] *= -100
    changed = model(changed_hidden, changed_features).hidden
    assert torch.equal(expected[:5], changed[:5])

    permutation = torch.tensor([3, 0, 5, 2, 1, 4])
    permuted = model(hidden[permutation], features[permutation]).hidden
    assert torch.allclose(permuted, expected[permutation], atol=1e-6)


def test_moe_eval_is_independent_of_batch_capacity() -> None:
    torch.manual_seed(29)
    config = RegimeMoEConfig(
        enabled=True,
        n_experts=2,
        top_k=1,
        expert_hidden=10,
        router_hidden=7,
        capacity_factor=0.1,
        dropout=0.0,
    )
    model = TemporalRegimeMoE(5, 4, config).eval()
    hidden = torch.randn(32, 5)
    features = torch.randn(32, 4)

    expected = model(hidden[:1], features[:1])
    actual = model(hidden, features)

    assert torch.allclose(expected.hidden, actual.hidden[:1], atol=1e-6)
    assert expected.overflow_rate == 0
    assert actual.overflow_rate == 0
