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
