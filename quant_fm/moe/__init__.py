"""QuantFM 的因果 Regime-MoE 与可选稀疏 FFN。"""

from quant_fm.moe.artifact import (
    load_regime_moe_artifact,
    load_regime_moe_metadata,
    save_regime_moe_artifact,
)
from quant_fm.moe.backbone import SparseMoEFeedForward
from quant_fm.moe.config import (
    BackboneMoEConfig,
    RegimeMoEConfig,
    TemporalRegimeTrainingConfig,
)
from quant_fm.moe.regime_features import (
    RegimeFeatureNormalizer,
    RegimeFeatureSpec,
    attach_regime_features,
    validate_regime_feature_frame,
)
from quant_fm.moe.router import RouterOutput, TopKRouter
from quant_fm.moe.temporal_moe import (
    MoEOutput,
    RegimeIntradayModel,
    TemporalRegimeMoE,
)

__all__ = [
    "BackboneMoEConfig",
    "MoEOutput",
    "RegimeFeatureNormalizer",
    "RegimeFeatureSpec",
    "RegimeIntradayModel",
    "RegimeMoEConfig",
    "RouterOutput",
    "SparseMoEFeedForward",
    "TemporalRegimeMoE",
    "TemporalRegimeTrainingConfig",
    "TopKRouter",
    "attach_regime_features",
    "load_regime_moe_artifact",
    "load_regime_moe_metadata",
    "save_regime_moe_artifact",
    "validate_regime_feature_frame",
]
