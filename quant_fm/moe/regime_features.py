"""Regime 路由特征的时点约束与仅训练集标准化。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True, slots=True)
class RegimeFeatureSpec:
    """一个路由特征及其最小可用滞后（以交易日计）。"""

    name: str
    availability_lag: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            msg = "feature name must not be empty"
            raise ValueError(msg)
        if self.availability_lag < 0:
            msg = "availability_lag must be non-negative"
            raise ValueError(msg)


class RegimeFeatureNormalizer:
    """冻结的逐列标准化器；调用方必须只用训练期数据拟合。"""

    def __init__(
        self,
        specs: tuple[RegimeFeatureSpec, ...],
        mean: torch.Tensor,
        scale: torch.Tensor,
        *,
        fit_end: str,
    ) -> None:
        if not specs or len(specs) != mean.numel() or mean.shape != scale.shape:
            msg = "specs, mean and scale must have the same non-zero width"
            raise ValueError(msg)
        if not fit_end:
            msg = "fit_end is required to audit the train-only fit window"
            raise ValueError(msg)
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            msg = "normalizer statistics must be finite"
            raise ValueError(msg)
        if not torch.all(scale > 0):
            msg = "normalizer scale must be positive"
            raise ValueError(msg)
        self.specs = specs
        self.mean = mean.detach().float().cpu()
        self.scale = scale.detach().float().cpu()
        self.fit_end = fit_end

    @classmethod
    def fit(
        cls,
        values: torch.Tensor,
        specs: tuple[RegimeFeatureSpec, ...],
        *,
        fit_end: str,
        eps: float = 1e-6,
    ) -> RegimeFeatureNormalizer:
        """从显式训练窗口拟合；不接受缺失/无穷值。"""
        if values.ndim != 2 or values.size(1) != len(specs):
            msg = "values must have shape [samples, len(specs)]"
            raise ValueError(msg)
        if values.size(0) < 1 or not torch.isfinite(values).all():
            msg = "training features must be non-empty and finite"
            raise ValueError(msg)
        values = values.float()
        mean = values.mean(dim=0)
        scale = values.std(dim=0, unbiased=False).clamp_min(eps)
        return cls(specs, mean, scale, fit_end=fit_end)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """保持设备与浮点 dtype 的标准化变换。"""
        if values.size(-1) != len(self.specs):
            msg = "feature width does not match frozen normalizer"
            raise ValueError(msg)
        if not torch.isfinite(values).all():
            msg = "regime features must be finite"
            raise ValueError(msg)
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        scale = self.scale.to(device=values.device, dtype=values.dtype)
        return (values - mean) / scale

    def to_dict(self) -> dict[str, object]:
        """序列化为 artifact 元数据。"""
        return {
            "specs": [asdict(spec) for spec in self.specs],
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "fit_end": self.fit_end,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> RegimeFeatureNormalizer:
        """从 artifact 元数据恢复。"""
        specs = tuple(RegimeFeatureSpec(**item) for item in payload["specs"])  # type: ignore[arg-type]
        return cls(
            specs,
            torch.tensor(payload["mean"], dtype=torch.float32),
            torch.tensor(payload["scale"], dtype=torch.float32),
            fit_end=str(payload["fit_end"]),
        )
