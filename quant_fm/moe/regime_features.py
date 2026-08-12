"""Regime 路由特征的时点约束与仅训练集标准化。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import polars as pl
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


_FORBIDDEN_REGIME_COLUMNS = {
    "label",
    "fwd_ret",
    "xs_ret",
    "target_return",
    "aux_target",
    "head_gain",
}


def validate_regime_feature_frame(
    frame: pl.DataFrame,
    specs: tuple[RegimeFeatureSpec, ...],
) -> pl.DataFrame:
    """
    校验 PIT Regime 特征表并返回唯一的 ``date/symbol/features``。

    每个特征必须提供 ``<name>__asof_date``，或共享的 ``asof_date``。lag=0
    允许 as-of 等于信号日；lag>0 至少要求严格早于信号日。交易日级精确 lag
    仍由生成该表的上游日历流程负责。
    """
    if not specs:
        msg = "at least one regime feature spec is required"
        raise ValueError(msg)
    required = {"date", "symbol", *(spec.name for spec in specs)}
    missing = required - set(frame.columns)
    if missing:
        msg = f"regime feature frame is missing columns: {sorted(missing)}"
        raise ValueError(msg)
    forbidden = sorted(_FORBIDDEN_REGIME_COLUMNS & set(frame.columns))
    if forbidden:
        msg = f"regime feature frame contains future target columns: {forbidden}"
        raise ValueError(msg)

    frame = frame.with_columns(
        pl.col("date").cast(pl.Utf8),
        pl.col("symbol").cast(pl.Utf8).str.zfill(6),
    )
    if frame.filter(pl.col("date").is_null() | pl.col("symbol").is_null()).height:
        msg = "regime feature frame contains null keys"
        raise ValueError(msg)
    duplicates = frame.filter(pl.struct(["date", "symbol"]).is_duplicated())
    if not duplicates.is_empty():
        msg = "regime feature frame contains duplicate (date, symbol) keys"
        raise ValueError(msg)

    signal_date_expr = pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    invalid_signal_dates = frame.select(
        (
            signal_date_expr.is_null()
            | (signal_date_expr.dt.strftime("%Y-%m-%d") != pl.col("date"))
        )
        .any()
        .alias("invalid")
    ).item()
    if invalid_signal_dates:
        msg = "regime date must contain canonical YYYY-MM-DD dates"
        raise ValueError(msg)
    for spec in specs:
        try:
            values = frame.select(
                pl.col(spec.name).cast(pl.Float32, strict=False).alias(spec.name)
            )[spec.name]
        except (TypeError, ValueError, pl.exceptions.PolarsError) as exc:
            msg = f"regime feature {spec.name!r} must be numeric"
            raise ValueError(msg) from exc
        if values.null_count() or not bool(values.is_finite().all()):
            msg = f"regime feature {spec.name!r} contains null/NaN/Inf"
            raise ValueError(msg)
        frame = frame.with_columns(values)

        feature_asof = f"{spec.name}__asof_date"
        asof_column = feature_asof if feature_asof in frame.columns else "asof_date"
        if asof_column not in frame.columns:
            msg = (
                f"regime feature {spec.name!r} requires {feature_asof!r} "
                "or shared 'asof_date'"
            )
            raise ValueError(msg)
        asof_expr = (
            pl.col(asof_column)
            .cast(pl.Utf8)
            .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        )
        temporal = frame.select(
            signal_date_expr.alias("_signal_date"),
            asof_expr.alias("_asof_date"),
            pl.col(asof_column).cast(pl.Utf8).alias("_asof_text"),
        )
        invalid_asof = temporal.select(
            (
                pl.col("_asof_date").is_null()
                | (pl.col("_asof_date").dt.strftime("%Y-%m-%d") != pl.col("_asof_text"))
            )
            .any()
            .alias("invalid")
        ).item()
        if invalid_asof:
            msg = f"{asof_column} must contain canonical YYYY-MM-DD dates"
            raise ValueError(msg)
        violates_lag = temporal.select(
            (
                (pl.col("_asof_date") > pl.col("_signal_date"))
                if spec.availability_lag == 0
                else (pl.col("_asof_date") >= pl.col("_signal_date"))
            )
            .any()
            .alias("invalid")
        ).item()
        if violates_lag:
            relation = "<=" if spec.availability_lag == 0 else "<"
            msg = (
                f"regime feature {spec.name!r} violates availability_lag="
                f"{spec.availability_lag}: asof_date must be {relation} signal date"
            )
            raise ValueError(msg)
    return frame.select("date", "symbol", *(spec.name for spec in specs)).sort(
        ["date", "symbol"]
    )


def attach_regime_features(
    features: pl.DataFrame,
    regime_features: pl.DataFrame,
    specs: tuple[RegimeFeatureSpec, ...],
) -> pl.DataFrame:
    """按唯一主键附加已校验 Regime 特征，缺失匹配时失败。"""
    names = [spec.name for spec in specs]
    overlap = sorted(set(names) & set(features.columns))
    if overlap:
        msg = f"ranker features already contain regime columns: {overlap}"
        raise ValueError(msg)
    keyed = features.with_columns(
        pl.col("date").cast(pl.Utf8),
        pl.col("symbol").cast(pl.Utf8).str.zfill(6),
    )
    validated = validate_regime_feature_frame(regime_features, specs)
    joined = keyed.join(validated, on=["date", "symbol"], how="left")
    missing = joined.filter(pl.any_horizontal(pl.col(name).is_null() for name in names))
    if not missing.is_empty():
        examples = missing.select(["date", "symbol"]).head(5).rows()
        msg = f"regime features are missing ranker keys: {examples}"
        raise ValueError(msg)
    return joined
