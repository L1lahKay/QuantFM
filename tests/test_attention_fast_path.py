import torch

from quant_fm.pretrain.model import CausalAttention, OrderFlowFMConfig, _rope_cache


def test_fused_causal_path_matches_explicit_mask() -> None:
    torch.manual_seed(1)
    config = OrderFlowFMConfig(
        field_sizes={"x": 4},
        input_fields=("x",),
        target_fields=("x",),
        d_model=16,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        max_seq_len=8,
    )
    attention = CausalAttention(config).eval()
    values = torch.randn(2, 8, 16)
    mask = torch.ones(2, 8, dtype=torch.bool)
    cos, sin = _rope_cache(8, 4, 10_000.0, values.device)
    fused = attention(values, cos, sin, mask, full_mask=True)
    explicit = attention(values, cos, sin, mask, full_mask=False)
    assert torch.allclose(fused, explicit, atol=1e-6)
