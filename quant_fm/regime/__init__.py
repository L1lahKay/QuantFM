"""Temporal Regime-MoE 的可审计 Level-2 特征生产流水线。"""

from quant_fm.regime.archive import archive_atomic_day
from quant_fm.regime.atomic import build_stock_day_atomic
from quant_fm.regime.contract import (
    ATOMIC_ARTIFACT_VERSION,
    ATOMIC_FORMULA_VERSION,
    L2_FEATURE_COLUMNS,
    REGIME_ARTIFACT_VERSION,
    REGIME_FORMULA_VERSION,
)
from quant_fm.regime.finalize import (
    build_l2_regime_features,
    finalize_l2_regime_features,
)

__all__ = [
    "ATOMIC_ARTIFACT_VERSION",
    "ATOMIC_FORMULA_VERSION",
    "L2_FEATURE_COLUMNS",
    "REGIME_ARTIFACT_VERSION",
    "REGIME_FORMULA_VERSION",
    "archive_atomic_day",
    "build_l2_regime_features",
    "build_stock_day_atomic",
    "finalize_l2_regime_features",
]
