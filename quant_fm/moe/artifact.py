"""Regime-MoE 的可审计推理 artifact。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn

    from quant_fm.moe.config import RegimeMoEConfig
    from quant_fm.moe.regime_features import RegimeFeatureNormalizer

ARTIFACT_VERSION = "regime_moe_v1"


def save_regime_moe_artifact(
    path: Path,
    model: nn.Module,
    config: RegimeMoEConfig,
    normalizer: RegimeFeatureNormalizer,
    *,
    data_cutoff: str,
    base_model_sha256: str,
) -> None:
    """只保存推理必需内容，排除 optimizer 等大状态。"""
    if not data_cutoff or not base_model_sha256:
        msg = "data_cutoff and base_model_sha256 are required"
        raise ValueError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "artifact_version": ARTIFACT_VERSION,
            "model_state": model.state_dict(),
            "moe_config": config.to_dict(),
            "feature_normalizer": normalizer.to_dict(),
            "data_cutoff": data_cutoff,
            "base_model_sha256": base_model_sha256,
        },
        path,
    )


def load_regime_moe_metadata(path: Path) -> dict[str, object]:
    """在 CPU 上加载并验证 artifact 契约。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("artifact_version") != ARTIFACT_VERSION:
        msg = "unsupported Regime-MoE artifact version"
        raise ValueError(msg)
    return payload
