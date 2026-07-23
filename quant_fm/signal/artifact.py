"""冻结 Ranker 的可复现保存与加载。"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import torch

from quant_fm.downstream.train_ranker import CrossSectionalRanker, RankerConfig

if TYPE_CHECKING:
    from pathlib import Path

ARTIFACT_VERSION = "1.0"


def _model_config(model: CrossSectionalRanker) -> RankerConfig:
    row_layers = [layer for layer in model.layers if hasattr(layer, "net")]
    attention_layers = [layer for layer in model.layers if hasattr(layer, "attn")]
    return RankerConfig(
        in_dim=model.proj.in_features,
        hidden=model.proj.out_features,
        depth=len(row_layers),
        n_heads=attention_layers[0].attn.num_heads if attention_layers else 4,
        dropout=float(row_layers[0].net[2].p) if row_layers else 0.0,
        use_attention=bool(attention_layers),
    )


def save_ranker_artifact(
    model: CrossSectionalRanker,
    checkpoint_path: Path,
    metadata_path: Path,
    *,
    feature_columns: list[str],
    training_end_date: str,
    seed: int,
    history: list[float] | None = None,
    provenance: dict[str, Any] | None = None,
) -> None:
    """原子保存冻结权重及不含机器绝对路径的元数据。"""
    if not feature_columns:
        msg = "feature_columns must not be empty"
        raise ValueError(msg)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = _model_config(model)
    if len(feature_columns) != cfg.in_dim:
        msg = (
            f"feature column count {len(feature_columns)} does not match "
            f"ranker in_dim {cfg.in_dim}"
        )
        raise ValueError(msg)
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "config": asdict(cfg),
        "feature_columns": feature_columns,
        "state_dict": model.state_dict(),
    }
    checkpoint_tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(payload, checkpoint_tmp)
    checkpoint_tmp.replace(checkpoint_path)

    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "training_end_date": training_end_date,
        "seed": seed,
        "feature_columns": feature_columns,
        "config": asdict(cfg),
        "train_history_ic": history or [],
        "provenance": provenance or {},
    }
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_tmp.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metadata_tmp.replace(metadata_path)


def load_ranker_artifact(
    checkpoint_path: Path,
    metadata_path: Path,
    *,
    device: str = "cpu",
) -> tuple[CrossSectionalRanker, dict[str, Any]]:
    """加载 Ranker，并交叉校验权重与 sidecar 元数据。"""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    for source in (metadata, payload):
        if source.get("artifact_version") != ARTIFACT_VERSION:
            msg = (
                f"unsupported ranker artifact version: {source.get('artifact_version')}"
            )
            raise ValueError(msg)
    required = {"config", "feature_columns"}
    if not required <= metadata.keys() or not required <= payload.keys():
        msg = "ranker artifact is missing config or feature_columns"
        raise ValueError(msg)
    if metadata["config"] != payload["config"]:
        msg = "ranker checkpoint config does not match metadata"
        raise ValueError(msg)
    if metadata["feature_columns"] != payload["feature_columns"]:
        msg = "ranker feature columns do not match metadata"
        raise ValueError(msg)
    if "training_end_date" not in metadata:
        msg = "ranker metadata is missing training_end_date"
        raise ValueError(msg)
    if len(payload["feature_columns"]) != payload["config"]["in_dim"]:
        msg = "ranker feature column count does not match configured in_dim"
        raise ValueError(msg)
    model = CrossSectionalRanker(RankerConfig(**payload["config"])).to(
        torch.device(device)
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, metadata
