"""Regime-MoE 的可审计推理 artifact。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    hidden_dim = getattr(model, "hidden_dim", None)
    regime_feature_dim = getattr(model, "regime_feature_dim", None)
    if not isinstance(hidden_dim, int) or not isinstance(regime_feature_dim, int):
        msg = "Regime-MoE artifact model must declare hidden/feature dimensions"
        raise TypeError(msg)
    if data_cutoff != normalizer.fit_end:
        msg = "data_cutoff must match the train-only normalizer fit_end"
        raise ValueError(msg)
    torch.save(
        {
            "artifact_version": ARTIFACT_VERSION,
            "model_state": model.state_dict(),
            "moe_config": config.to_dict(),
            "feature_normalizer": normalizer.to_dict(),
            "hidden_dim": hidden_dim,
            "regime_feature_dim": regime_feature_dim,
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
    required = {
        "model_state",
        "moe_config",
        "feature_normalizer",
        "data_cutoff",
        "base_model_sha256",
    }
    missing = required - set(payload)
    if missing:
        msg = f"Regime-MoE artifact is missing fields: {sorted(missing)}"
        raise ValueError(msg)
    return payload


def load_regime_moe_artifact(
    path: Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, RegimeFeatureNormalizer, dict[str, Any]]:
    """严格重建可推理的 Temporal Regime-MoE 与冻结 normalizer。"""
    from quant_fm.moe.config import RegimeMoEConfig
    from quant_fm.moe.regime_features import RegimeFeatureNormalizer
    from quant_fm.moe.temporal_moe import TemporalRegimeMoE

    payload = load_regime_moe_metadata(path)
    dimension_fields = {"hidden_dim", "regime_feature_dim"}
    if not dimension_fields <= set(payload):
        msg = (
            "legacy Regime-MoE artifact lacks reconstruction dimensions; "
            "metadata remains readable but inference reconstruction is unavailable"
        )
        raise ValueError(msg)
    config = RegimeMoEConfig.from_dict(payload["moe_config"])  # type: ignore[arg-type]
    normalizer = RegimeFeatureNormalizer.from_dict(
        payload["feature_normalizer"]  # type: ignore[arg-type]
    )
    hidden_dim = payload["hidden_dim"]
    regime_feature_dim = payload["regime_feature_dim"]
    if (
        isinstance(hidden_dim, bool)
        or not isinstance(hidden_dim, int)
        or hidden_dim < 1
        or isinstance(regime_feature_dim, bool)
        or not isinstance(regime_feature_dim, int)
        or regime_feature_dim != len(normalizer.specs)
    ):
        msg = "invalid Regime-MoE artifact dimensions"
        raise ValueError(msg)
    if payload["data_cutoff"] != normalizer.fit_end:
        msg = "Regime-MoE artifact cutoff does not match normalizer fit_end"
        raise ValueError(msg)
    model = TemporalRegimeMoE(hidden_dim, regime_feature_dim, config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)  # type: ignore[arg-type]
    model.eval()
    return model, normalizer, payload
