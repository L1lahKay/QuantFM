from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from quant_fm.pretrain.field_fusion import (  # noqa: E402
    EventFieldFusion,
    FieldFusionConfig,
)
from quant_fm.pretrain.model import OrderFlowFM  # noqa: E402
from quant_fm.pretrain.train import load_checkpoint  # noqa: E402
from quant_fm.tokenizer.vocab import default_vocab  # noqa: E402


def _parts(n_fields: int, dim: int = 8):
    return [torch.ones(2, 4, dim) * (index + 1) for index in range(n_fields)]


def test_legacy_sum_matches_original_behavior() -> None:
    fusion = EventFieldFusion(
        n_fields=3,
        input_dim=8,
        d_model=8,
        config=FieldFusionConfig(method="legacy_sum", input_norm=False),
    )
    parts = _parts(3)
    assert torch.equal(fusion(parts), sum(parts))


@pytest.mark.parametrize("method", ["scaled_sum", "gated_sum"])
def test_scaled_fusions_have_stable_shape_and_gradients(method: str) -> None:
    fusion = EventFieldFusion(
        n_fields=4,
        input_dim=8,
        d_model=8,
        config=FieldFusionConfig(method=method),
    )
    parts = [value.requires_grad_() for value in _parts(4)]
    output = fusion(parts)
    assert output.shape == (2, 4, 8)
    output.sum().backward()
    assert all(value.grad is not None for value in parts)
    if method == "gated_sum":
        assert fusion.gate_logits is not None
        assert fusion.gate_logits.grad is not None


def test_concat_projection_uses_small_field_vectors() -> None:
    fusion = EventFieldFusion(
        n_fields=3,
        input_dim=4,
        d_model=16,
        config=FieldFusionConfig(method="concat_mlp", field_dim=4),
    )
    output = fusion(_parts(3, dim=4))
    assert output.shape == (2, 4, 16)


def test_field_dropout_only_changes_training_output() -> None:
    torch.manual_seed(0)
    fusion = EventFieldFusion(
        n_fields=4,
        input_dim=8,
        d_model=8,
        config=FieldFusionConfig(
            method="scaled_sum", field_dropout=0.75, input_norm=False
        ),
    )
    parts = _parts(4)
    fusion.eval()
    expected = fusion(parts)
    assert torch.equal(expected, fusion(parts))
    fusion.train()
    actual = fusion(parts)
    assert not torch.equal(expected, actual)


def test_legacy_checkpoint_without_fusion_metadata_still_loads(tmp_path) -> None:
    model = OrderFlowFM.from_vocab(
        default_vocab(n_bins=4),
        d_model=16,
        n_layers=1,
        n_heads=4,
        ffn_mult=2.0,
        dropout=0.0,
        max_seq_len=16,
    )
    cfg = model.cfg
    legacy_config = {
        "field_sizes": cfg.field_sizes,
        "input_fields": list(cfg.input_fields),
        "target_fields": list(cfg.target_fields),
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "ffn_mult": cfg.ffn_mult,
        "dropout": cfg.dropout,
        "max_seq_len": cfg.max_seq_len,
        "rope_theta": cfg.rope_theta,
    }
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model_state": model.state_dict(), "config": legacy_config}, checkpoint)

    restored = load_checkpoint(checkpoint, torch.device("cpu"))
    assert restored.cfg.field_fusion == "legacy_sum"
    assert restored.cfg.vocab_version == "1.0"
