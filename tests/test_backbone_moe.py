import torch

from quant_fm.moe.config import BackboneMoEConfig
from quant_fm.pretrain.dataset import FIELD_ORDER
from quant_fm.pretrain.model import OrderFlowFM
from quant_fm.tokenizer.vocab import default_vocab


def test_top_layer_sparse_moe_forward_and_router_gradient() -> None:
    config = BackboneMoEConfig(
        enabled=True,
        layer_indices=(1,),
        n_routed_experts=3,
        top_k=1,
        shared_expert_hidden=12,
        routed_expert_hidden=16,
        capacity_factor=2.0,
    )
    model = OrderFlowFM.from_vocab(
        default_vocab(n_bins=4),
        d_model=16,
        n_layers=2,
        n_heads=4,
        ffn_hidden=24,
        dropout=0.0,
        max_seq_len=12,
        backbone_moe=config,
    )
    batch = {field: torch.randint(1, 4, (2, 8)) for field in FIELD_ORDER}
    batch["attention_mask"] = torch.ones(2, 8, dtype=torch.bool)
    logits = model(batch)
    auxiliary = model.moe_auxiliary_loss()
    assert auxiliary is not None
    assert torch.isfinite(auxiliary)
    loss = sum(value.float().square().mean() for value in logits.values()) + auxiliary
    loss.backward()
    router = model.blocks[1].ffn.router
    assert any(parameter.grad is not None for parameter in router.parameters())


def test_sparse_moe_ignores_right_padding_during_capacity_routing() -> None:
    torch.manual_seed(17)
    config = BackboneMoEConfig(
        enabled=True,
        layer_indices=(0,),
        n_routed_experts=1,
        top_k=1,
        shared_expert_hidden=0,
        routed_expert_hidden=12,
        capacity_factor=0.5,
    )
    model = OrderFlowFM.from_vocab(
        default_vocab(n_bins=4),
        d_model=8,
        n_layers=1,
        n_heads=2,
        ffn_hidden=12,
        dropout=0.0,
        max_seq_len=8,
        backbone_moe=config,
    ).train()
    prefix = {field: torch.randint(1, 4, (1, 2)) for field in FIELD_ORDER}
    prefix["attention_mask"] = torch.ones(1, 2, dtype=torch.bool)
    padded = {
        field: torch.nn.functional.pad(values, (0, 6))
        for field, values in prefix.items()
        if field != "attention_mask"
    }
    padded["attention_mask"] = torch.tensor(
        [[True, True, False, False, False, False, False, False]]
    )

    expected = model(prefix)
    actual = model(padded)

    for field in expected:
        assert torch.allclose(expected[field], actual[field][:, :2], atol=1e-6)


def test_sparse_moe_eval_is_independent_of_batch_capacity() -> None:
    torch.manual_seed(23)
    config = BackboneMoEConfig(
        enabled=True,
        layer_indices=(0,),
        n_routed_experts=2,
        top_k=1,
        shared_expert_hidden=0,
        routed_expert_hidden=10,
        capacity_factor=0.1,
    )
    model = OrderFlowFM.from_vocab(
        default_vocab(n_bins=4),
        d_model=8,
        n_layers=1,
        n_heads=2,
        ffn_hidden=12,
        dropout=0.0,
        max_seq_len=4,
        backbone_moe=config,
    ).eval()
    batch = {field: torch.randint(1, 4, (8, 4)) for field in FIELD_ORDER}
    batch["attention_mask"] = torch.ones(8, 4, dtype=torch.bool)
    single = {field: values[:1] for field, values in batch.items()}

    expected = model(single)
    actual = model(batch)

    for field in expected:
        assert torch.allclose(expected[field], actual[field][:1], atol=1e-6)
