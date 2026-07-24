import torch

from quant_fm.pretrain.model import OrderFlowFM
from quant_fm.tokenizer.vocab import default_vocab


def test_rope_cache_is_reused_and_not_in_state_dict() -> None:
    model = OrderFlowFM.from_vocab(
        default_vocab(n_bins=4),
        d_model=16,
        n_layers=1,
        n_heads=4,
        max_seq_len=16,
    )
    first = model._get_rope(8, torch.device("cpu"), torch.float32)
    second = model._get_rope(12, torch.device("cpu"), torch.float32)
    assert (
        first[0].untyped_storage().data_ptr() == second[0].untyped_storage().data_ptr()
    )
    assert not any("rope" in key for key in model.state_dict())
