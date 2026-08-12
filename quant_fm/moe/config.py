"""MoE 配置与静态校验。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import yaml

from quant_fm.moe.regime_features import RegimeFeatureSpec

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class RegimeMoEConfig:
    """股日/时间聚合层的轻量 Regime-MoE 配置。"""

    enabled: bool = False
    n_experts: int = 4
    top_k: int = 2
    expert_hidden: int = 256
    router_hidden: int = 128
    dropout: float = 0.0
    temperature: float = 1.0
    capacity_factor: float = 1.25
    load_balance_weight: float = 0.01
    router_z_loss_weight: float = 0.001

    def __post_init__(self) -> None:
        if self.n_experts < 1:
            msg = "n_experts must be positive"
            raise ValueError(msg)
        if not 1 <= self.top_k <= self.n_experts:
            msg = "top_k must be in [1, n_experts]"
            raise ValueError(msg)
        if self.expert_hidden < 1 or self.router_hidden < 1:
            msg = "expert_hidden and router_hidden must be positive"
            raise ValueError(msg)
        if not 0.0 <= self.dropout < 1.0:
            msg = "dropout must be in [0, 1)"
            raise ValueError(msg)
        if self.temperature <= 0.0 or self.capacity_factor <= 0.0:
            msg = "temperature and capacity_factor must be positive"
            raise ValueError(msg)
        if self.load_balance_weight < 0.0 or self.router_z_loss_weight < 0.0:
            msg = "router loss weights must be non-negative"
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> RegimeMoEConfig:
        """从 YAML/metadata 映射加载，未知字段显式失败。"""
        return cls() if not value else cls(**value)

    def to_dict(self) -> dict[str, Any]:
        """返回可写入 artifact 的普通字典。"""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TemporalRegimeTrainingConfig:
    """Temporal Regime-MoE 训练入口的完整、可审计配置。"""

    moe_router_version: str
    placement: str
    moe: RegimeMoEConfig
    feature_specs: tuple[RegimeFeatureSpec, ...]

    def __post_init__(self) -> None:
        if self.moe_router_version != "1.0":
            msg = "unsupported moe_router_version"
            raise ValueError(msg)
        if self.placement != "temporal_aggregator":
            msg = "Temporal Regime-MoE placement must be temporal_aggregator"
            raise ValueError(msg)
        if not self.moe.enabled:
            msg = "Temporal Regime-MoE training config must be enabled"
            raise ValueError(msg)
        if not self.feature_specs:
            msg = "Temporal Regime-MoE requires at least one regime feature"
            raise ValueError(msg)
        names = [spec.name for spec in self.feature_specs]
        if len(set(names)) != len(names):
            msg = "regime feature names must be unique"
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TemporalRegimeTrainingConfig:
        """从 YAML 映射加载，并拒绝未知或缺失字段。"""
        if not isinstance(value, dict):
            msg = "Temporal Regime-MoE config must be a mapping"
            raise TypeError(msg)
        required = {
            "moe_router_version",
            "placement",
            "enabled",
            "regime_features",
        }
        missing = required - set(value)
        if missing:
            msg = f"Temporal Regime-MoE config is missing fields: {sorted(missing)}"
            raise ValueError(msg)
        moe_fields = set(RegimeMoEConfig.__dataclass_fields__)
        allowed = {"moe_router_version", "placement", "regime_features", *moe_fields}
        unknown = set(value) - allowed
        if unknown:
            msg = f"unknown Temporal Regime-MoE config fields: {sorted(unknown)}"
            raise ValueError(msg)
        raw_features = value["regime_features"]
        if not isinstance(raw_features, list):
            msg = "regime_features must be a list"
            raise TypeError(msg)
        try:
            feature_specs = tuple(RegimeFeatureSpec(**item) for item in raw_features)
        except (TypeError, ValueError) as exc:
            msg = "invalid regime_features declaration"
            raise ValueError(msg) from exc
        moe_payload = {field: value[field] for field in moe_fields if field in value}
        return cls(
            moe_router_version=str(value["moe_router_version"]),
            placement=str(value["placement"]),
            moe=RegimeMoEConfig.from_dict(moe_payload),
            feature_specs=feature_specs,
        )

    @classmethod
    def from_yaml(cls, path: Path) -> TemporalRegimeTrainingConfig:
        """从磁盘加载标准 Temporal Regime-MoE YAML。"""
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        """返回可稳定写入训练报告的普通字典。"""
        return {
            "moe_router_version": self.moe_router_version,
            "placement": self.placement,
            **self.moe.to_dict(),
            "regime_features": [asdict(spec) for spec in self.feature_specs],
        }


@dataclass(frozen=True, slots=True)
class BackboneMoEConfig:
    """顶部 Transformer FFN 的稀疏 MoE 配置。"""

    enabled: bool = False
    layer_indices: tuple[int, ...] = ()
    n_routed_experts: int = 4
    top_k: int = 1
    shared_expert_hidden: int = 1024
    routed_expert_hidden: int = 1792
    capacity_factor: float = 1.25
    load_balance_weight: float = 0.01
    router_z_loss_weight: float = 0.001

    def __post_init__(self) -> None:
        if self.n_routed_experts < 1:
            msg = "n_routed_experts must be positive"
            raise ValueError(msg)
        if not 1 <= self.top_k <= self.n_routed_experts:
            msg = "top_k must be in [1, n_routed_experts]"
            raise ValueError(msg)
        if self.shared_expert_hidden < 0 or self.routed_expert_hidden < 1:
            msg = "invalid shared/routed expert hidden width"
            raise ValueError(msg)
        if self.capacity_factor <= 0.0:
            msg = "capacity_factor must be positive"
            raise ValueError(msg)
        if self.load_balance_weight < 0.0 or self.router_z_loss_weight < 0.0:
            msg = "router loss weights must be non-negative"
            raise ValueError(msg)
        if len(set(self.layer_indices)) != len(self.layer_indices):
            msg = "layer_indices must not contain duplicates"
            raise ValueError(msg)
        if any(index < 0 for index in self.layer_indices):
            msg = "layer_indices must be non-negative"
            raise ValueError(msg)
        if self.enabled and not self.layer_indices:
            msg = "enabled backbone MoE requires at least one layer index"
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> BackboneMoEConfig:
        """从 YAML/metadata 加载，并规范化 layer list。"""
        if not value:
            return cls()
        payload = dict(value)
        payload["layer_indices"] = tuple(payload.get("layer_indices", ()))
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """返回 artifact 友好的字典。"""
        payload = asdict(self)
        payload["layer_indices"] = list(self.layer_indices)
        return payload
