import torch

from quant_fm.pretrain.model import OrderFlowFM
from quant_fm.pretrain.train import (
    TrainState,
    _resolve_resume_path,
    _save_checkpoint,
    load_checkpoint,
)
from quant_fm.tokenizer.vocab import default_vocab


def test_inference_checkpoint_excludes_optimizer_and_loads(tmp_path) -> None:
    model = OrderFlowFM.from_vocab(
        default_vocab(n_bins=4),
        d_model=16,
        n_layers=1,
        n_heads=4,
        max_seq_len=8,
    )
    path = tmp_path / "inference.pt"
    _save_checkpoint(model, model.cfg, path, train_state=TrainState(update_step=3))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "optimizer_state" not in payload
    assert payload["train_state"]["update_step"] == 3
    restored = load_checkpoint(path, torch.device("cpu"))
    assert restored.cfg.ffn_hidden is None


def test_auto_resume_ignores_inference_only_checkpoints(tmp_path) -> None:
    for name in ("best.pt", "final.pt"):
        (tmp_path / name).touch()

    assert _resolve_resume_path(tmp_path, "auto") is None
    resumable = tmp_path / "final_resume.pt"
    resumable.touch()
    assert _resolve_resume_path(tmp_path, "auto") == resumable
