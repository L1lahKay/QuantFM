from __future__ import annotations

import hashlib

import polars as pl
import torch
import yaml

from quant_fm.manifest.build_manifest import Manifest, ShardEntry
from quant_fm.pretrain.dataset import FIELD_ORDER
from quant_fm.pretrain.train import train
from quant_fm.tokenizer.vocab import default_vocab


def test_train_runs_one_cpu_optimizer_update(tmp_path) -> None:
    vocab = default_vocab(n_bins=4)
    vocab_path = tmp_path / "vocab.json"
    vocab.save(vocab_path)

    token_path = tmp_path / "tokens.parquet"
    pl.DataFrame({field: [1, 1, 1, 1] for field in FIELD_ORDER}).write_parquet(
        token_path
    )
    shard = ShardEntry(
        market="SH",
        symbol="600000",
        date="2025-01-02",
        path=str(token_path),
        rows=4,
        sha256=hashlib.sha256(token_path.read_bytes()).hexdigest(),
        split="train",
    )
    manifest_path = tmp_path / "manifest.json"
    Manifest(shards=[shard], vocab_path=str(vocab_path)).save(manifest_path)

    out_dir = tmp_path / "run"
    config = {
        "seed": 7,
        "data": {
            "manifest": str(manifest_path),
            "vocab": str(vocab_path),
            "context": 4,
            "stride": 4,
            "min_len": 2,
            "cache_size": 1,
            "num_workers": 0,
        },
        "model": {
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 4,
            "ffn_mult": 2.0,
            "dropout": 0.0,
            "max_seq_len": 4,
            "rope_theta": 10_000.0,
        },
        "optim": {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "betas": [0.9, 0.95],
            "grad_clip": 1.0,
            "warmup_steps": 0,
            "max_update_steps": 1,
            "batch_size": 1,
            "grad_accum": 1,
            "precision": "fp32",
        },
        "runtime": {
            "out_dir": str(out_dir),
            "log_every": 1,
            "eval_every": 10,
            "ckpt_every": 10,
            "save_best": False,
            "fsdp": False,
            "device": "cpu",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    train(config_path)

    resumable = torch.load(
        out_dir / "final_resume.pt", map_location="cpu", weights_only=False
    )
    inference = torch.load(out_dir / "final.pt", map_location="cpu", weights_only=False)
    assert resumable["train_state"]["micro_step"] == 1
    assert resumable["train_state"]["update_step"] == 1
    assert resumable["train_state"]["non_pad_tokens_seen"] == 4
    assert "optimizer_state" in resumable
    assert "optimizer_state" not in inference
