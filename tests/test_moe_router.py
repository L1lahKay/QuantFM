import torch

from quant_fm.moe.artifact import (
    load_regime_moe_metadata,
    save_regime_moe_artifact,
)
from quant_fm.moe.config import RegimeMoEConfig
from quant_fm.moe.regime_features import (
    RegimeFeatureNormalizer,
    RegimeFeatureSpec,
)
from quant_fm.moe.router import TopKRouter
from quant_fm.moe.temporal_moe import TemporalRegimeMoE


def test_router_contract_and_gradients() -> None:
    torch.manual_seed(3)
    router = TopKRouter(5, 4, top_k=2, hidden_dim=8)
    features = torch.randn(12, 5, requires_grad=True)
    output = router(features)

    assert output.probabilities.shape == (12, 4)
    assert output.topk_indices.shape == (12, 2)
    assert torch.allclose(output.topk_weights.sum(-1), torch.ones(12))
    assert 0 <= output.entropy <= 1
    (output.load_balance_loss + output.router_z_loss).backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_temporal_moe_shared_residual_and_all_experts_receive_load() -> None:
    config = RegimeMoEConfig(
        enabled=True,
        n_experts=4,
        top_k=2,
        expert_hidden=12,
        router_hidden=8,
        capacity_factor=4.0,
    )
    model = TemporalRegimeMoE(6, 3, config)
    hidden = torch.randn(32, 6, requires_grad=True)
    features = torch.randn(32, 3)
    output = model(hidden, features)

    assert output.hidden.shape == hidden.shape
    assert output.overflow_rate == 0
    load = torch.bincount(output.router.topk_indices.flatten(), minlength=4)
    assert int((load > 0).sum()) >= 2
    (output.hidden.square().mean() + output.auxiliary_loss).backward()
    assert hidden.grad is not None


def test_regime_normalizer_and_inference_artifact_round_trip(tmp_path) -> None:
    specs = (
        RegimeFeatureSpec("market_vol"),
        RegimeFeatureSpec("industry_strength", availability_lag=1),
    )
    values = torch.tensor([[1.0, 4.0], [3.0, 8.0]])
    normalizer = RegimeFeatureNormalizer.fit(values, specs, fit_end="2025-12-31")
    transformed = normalizer.transform(values)
    assert torch.allclose(transformed.mean(0), torch.zeros(2))

    config = RegimeMoEConfig(
        enabled=True, n_experts=2, top_k=1, expert_hidden=4, router_hidden=4
    )
    model = TemporalRegimeMoE(3, 2, config)
    path = tmp_path / "regime.pt"
    save_regime_moe_artifact(
        path,
        model,
        config,
        normalizer,
        data_cutoff="2025-12-31",
        base_model_sha256="abc123",
    )
    payload = load_regime_moe_metadata(path)
    assert payload["artifact_version"] == "regime_moe_v1"
    assert "optimizer_state" not in payload
    restored = RegimeFeatureNormalizer.from_dict(payload["feature_normalizer"])
    assert restored.fit_end == "2025-12-31"
