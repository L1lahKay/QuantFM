"""
跨股票 interval embedding 的 PIT 对齐与张量化。

本模块只接收已经由单股模型汇总好的 interval embedding，不接收原始逐笔事件。
行业信息通过严格向后、且不允许同一时刻命中的 as-of join 注入，从数据边界阻止
未来行业分类泄漏。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

_ROW_ID = "__cross_asset_row_id"
_PREDICTION_KEY = "__cross_asset_prediction_key"
_EFFECTIVE_KEY = "__cross_asset_effective_key"


@dataclass(slots=True)
class AlignedCrossAssetPanel:
    """
    一个交易日（或一个连续片段）的同步跨股票面板。

    ``embeddings`` 的形状为 ``[T, N, D]``。缺失的股票/interval 位置填零，
    且必须通过 ``active_mask=False`` 识别；下游不得把零向量解释为真实观测。
    """

    embeddings: torch.Tensor
    active_mask: torch.Tensor
    industry_id: torch.Tensor
    prediction_times: tuple[Any, ...]
    symbols: tuple[Any, ...]
    max_industry_effective_time: Any | None

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 3:
            msg = "embeddings must have shape [time, stock, feature]"
            raise ValueError(msg)
        expected = self.embeddings.shape[:2]
        if self.active_mask.shape != expected:
            msg = "active_mask must match embeddings[:2]"
            raise ValueError(msg)
        if self.industry_id.shape != expected:
            msg = "industry_id must match embeddings[:2]"
            raise ValueError(msg)
        if len(self.prediction_times) != expected[0]:
            msg = "prediction_times must match the time axis"
            raise ValueError(msg)
        if len(self.symbols) != expected[1]:
            msg = "symbols must match the stock axis"
            raise ValueError(msg)

    def as_model_inputs(self) -> dict[str, torch.Tensor]:
        """返回模型可直接消费的张量，不包含任何未来标签。"""
        return {
            "own": self.embeddings,
            "active_mask": self.active_mask,
            "industry_id": self.industry_id,
        }


class CrossAssetPanelDataset(Dataset):
    """
    把若干独立面板暴露为 PyTorch Dataset。

    通常每个元素对应一个交易日。股票轴可以跨日变化，因此默认不提供会静默
    对齐股票代码的 collator；训练方应按日消费，或显式实现自己的 universe 对齐。
    """

    def __init__(self, panels: Sequence[AlignedCrossAssetPanel]) -> None:
        if not panels:
            msg = "panels must not be empty"
            raise ValueError(msg)
        self.panels = tuple(panels)

    def __len__(self) -> int:
        return len(self.panels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.panels[index].as_model_inputs()


def _require_columns(frame: pl.DataFrame, columns: set[str], *, name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        msg = f"{name} is missing required columns: {sorted(missing)}"
        raise ValueError(msg)


def _normalise_asof_keys(
    intervals: pl.DataFrame,
    industry_history: pl.DataFrame,
    *,
    prediction_time_col: str,
    effective_time_col: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    prediction_dtype = intervals.schema[prediction_time_col]
    effective_dtype = industry_history.schema[effective_time_col]
    if prediction_dtype.is_temporal() and effective_dtype.is_temporal():
        key_dtype: pl.DataType = pl.Datetime("ns")
    elif prediction_dtype.is_integer() and effective_dtype.is_integer():
        key_dtype = pl.Int64
    else:
        msg = (
            "prediction and industry effective times must both be temporal "
            "or both be integer timestamps"
        )
        raise TypeError(msg)
    left = intervals.with_columns(
        pl.col(prediction_time_col).cast(key_dtype).alias(_PREDICTION_KEY)
    )
    right = industry_history.with_columns(
        pl.col(effective_time_col).cast(key_dtype).alias(_EFFECTIVE_KEY)
    )
    return left, right


def validate_pit_industry_assignments(
    frame: pl.DataFrame,
    *,
    prediction_time_col: str = "prediction_time",
    effective_time_col: str = "industry_effective_time",
) -> Any | None:
    """
    验证每条已匹配行业记录都严格早于对应预测时刻。

    返回所有实际使用记录中的最大 effective time，便于写入运行日志或 artifact。
    未匹配行业允许为空，此时该股票的行业上下文会被禁用。
    """
    _require_columns(
        frame,
        {prediction_time_col, effective_time_col},
        name="joined interval frame",
    )
    matched = frame.filter(pl.col(effective_time_col).is_not_null())
    if matched.is_empty():
        return None
    prediction_dtype = matched.schema[prediction_time_col]
    effective_dtype = matched.schema[effective_time_col]
    if prediction_dtype.is_temporal() and effective_dtype.is_temporal():
        comparison_dtype: pl.DataType = pl.Datetime("ns")
    elif prediction_dtype.is_integer() and effective_dtype.is_integer():
        comparison_dtype = pl.Int64
    else:
        msg = (
            "prediction and assigned effective times must both be temporal "
            "or both be integer timestamps"
        )
        raise TypeError(msg)
    invalid = matched.filter(
        pl.col(effective_time_col).cast(comparison_dtype)
        >= pl.col(prediction_time_col).cast(comparison_dtype)
    )
    if not invalid.is_empty():
        example = invalid.select(
            prediction_time_col,
            effective_time_col,
        ).row(0, named=True)
        msg = (
            "PIT industry leakage: effective time must be strictly earlier than "
            f"prediction time; offending row={example}"
        )
        raise ValueError(msg)
    return matched[effective_time_col].max()


def join_pit_industry(
    intervals: pl.DataFrame,
    industry_history: pl.DataFrame,
    *,
    symbol_col: str = "symbol",
    prediction_time_col: str = "prediction_time",
    industry_col: str = "industry_id",
    effective_time_col: str = "effective_time",
    output_effective_time_col: str = "industry_effective_time",
) -> pl.DataFrame:
    """
    按股票做严格向后的 PIT 行业 as-of join。

    精确等于预测时刻的行业变更不会生效，只有
    ``effective_time < prediction_time`` 的记录可以进入模型。历史表同一股票同一
    effective time 不允许重复，避免依赖输入行顺序产生不确定结果。
    """
    _require_columns(
        intervals,
        {symbol_col, prediction_time_col},
        name="intervals",
    )
    _require_columns(
        industry_history,
        {symbol_col, effective_time_col, industry_col},
        name="industry_history",
    )
    reserved = {
        _ROW_ID,
        _PREDICTION_KEY,
        _EFFECTIVE_KEY,
        output_effective_time_col,
    }
    collision = reserved.intersection(intervals.columns)
    if collision:
        msg = f"intervals contains reserved/output columns: {sorted(collision)}"
        raise ValueError(msg)
    if industry_col in intervals.columns:
        msg = (
            f"intervals already contains {industry_col!r}; remove it and use the "
            "PIT history join explicitly"
        )
        raise ValueError(msg)
    if (
        intervals[symbol_col].null_count()
        or intervals[prediction_time_col].null_count()
    ):
        msg = "interval symbol and prediction time must not be null"
        raise ValueError(msg)
    if (
        industry_history[symbol_col].null_count()
        or industry_history[effective_time_col].null_count()
        or industry_history[industry_col].null_count()
    ):
        msg = "PIT industry history keys and industry ids must not be null"
        raise ValueError(msg)
    if not industry_history.schema[industry_col].is_integer():
        msg = "industry_id must use an integer dtype; reserve negative ids for unknown"
        raise TypeError(msg)

    duplicates = (
        industry_history.group_by(symbol_col, effective_time_col)
        .len()
        .filter(pl.col("len") > 1)
    )
    if not duplicates.is_empty():
        msg = "PIT industry history has duplicate (symbol, effective_time) rows"
        raise ValueError(msg)

    left, right = _normalise_asof_keys(
        intervals,
        industry_history,
        prediction_time_col=prediction_time_col,
        effective_time_col=effective_time_col,
    )
    left = left.with_row_index(_ROW_ID).sort(symbol_col, _PREDICTION_KEY)
    right = right.select(
        symbol_col,
        _EFFECTIVE_KEY,
        pl.col(effective_time_col).alias(output_effective_time_col),
        industry_col,
    ).sort(symbol_col, _EFFECTIVE_KEY)
    joined = left.join_asof(
        right,
        left_on=_PREDICTION_KEY,
        right_on=_EFFECTIVE_KEY,
        by=symbol_col,
        strategy="backward",
        allow_exact_matches=False,
        check_sortedness=False,
    )
    joined = (
        joined.sort(_ROW_ID)
        .drop(_ROW_ID, _PREDICTION_KEY, _EFFECTIVE_KEY)
        .with_columns(pl.col(industry_col).fill_null(-1).cast(pl.Int64))
    )
    validate_pit_industry_assignments(
        joined,
        prediction_time_col=prediction_time_col,
        effective_time_col=output_effective_time_col,
    )
    return joined


def align_interval_embeddings(
    frame: pl.DataFrame,
    *,
    symbol_col: str = "symbol",
    prediction_time_col: str = "prediction_time",
    embedding_col: str = "embedding",
    industry_col: str = "industry_id",
    effective_time_col: str = "industry_effective_time",
    dtype: torch.dtype = torch.float32,
) -> AlignedCrossAssetPanel:
    """
    把长表 interval embedding 对齐为 ``[T, N, D]``。

    缺失组合使用零向量和 ``active_mask=False``。输入必须包含 PIT join 留下的
    effective time 审计列；这能防止调用方绕过行业时点校验。
    """
    _require_columns(
        frame,
        {
            symbol_col,
            prediction_time_col,
            embedding_col,
            industry_col,
            effective_time_col,
        },
        name="interval embedding frame",
    )
    if frame.is_empty():
        msg = "interval embedding frame must not be empty"
        raise ValueError(msg)
    if frame[symbol_col].null_count() or frame[prediction_time_col].null_count():
        msg = "symbol and prediction time must not be null"
        raise ValueError(msg)
    duplicate = (
        frame.group_by(prediction_time_col, symbol_col).len().filter(pl.col("len") > 1)
    )
    if not duplicate.is_empty():
        msg = "duplicate interval embedding for the same (prediction_time, symbol)"
        raise ValueError(msg)

    max_effective = validate_pit_industry_assignments(
        frame,
        prediction_time_col=prediction_time_col,
        effective_time_col=effective_time_col,
    )
    prediction_times = tuple(frame[prediction_time_col].unique().sort().to_list())
    symbols = tuple(frame[symbol_col].unique().sort().to_list())
    time_index = {value: index for index, value in enumerate(prediction_times)}
    symbol_index = {value: index for index, value in enumerate(symbols)}

    first_embedding = torch.as_tensor(
        frame.row(0, named=True)[embedding_col],
        dtype=dtype,
    )
    if first_embedding.ndim != 1 or first_embedding.numel() == 0:
        msg = "each embedding must be a non-empty one-dimensional vector"
        raise ValueError(msg)
    feature_dim = first_embedding.numel()
    embeddings = torch.zeros(
        (len(prediction_times), len(symbols), feature_dim),
        dtype=dtype,
    )
    active_mask = torch.zeros(
        (len(prediction_times), len(symbols)),
        dtype=torch.bool,
    )
    industry_id = torch.full(
        (len(prediction_times), len(symbols)),
        -1,
        dtype=torch.long,
    )

    for row in frame.iter_rows(named=True):
        embedding = torch.as_tensor(row[embedding_col], dtype=dtype)
        if embedding.shape != (feature_dim,):
            msg = (
                "all embeddings must share one feature dimension; "
                f"expected {feature_dim}, got {tuple(embedding.shape)}"
            )
            raise ValueError(msg)
        if not torch.isfinite(embedding).all():
            msg = "active interval embeddings must contain only finite values"
            raise ValueError(msg)
        time_pos = time_index[row[prediction_time_col]]
        stock_pos = symbol_index[row[symbol_col]]
        embeddings[time_pos, stock_pos] = embedding
        active_mask[time_pos, stock_pos] = True
        industry_id[time_pos, stock_pos] = int(row[industry_col])

    return AlignedCrossAssetPanel(
        embeddings=embeddings,
        active_mask=active_mask,
        industry_id=industry_id,
        prediction_times=prediction_times,
        symbols=symbols,
        max_industry_effective_time=max_effective,
    )


def build_cross_asset_panel(
    intervals: pl.DataFrame,
    industry_history: pl.DataFrame,
    *,
    symbol_col: str = "symbol",
    prediction_time_col: str = "prediction_time",
    embedding_col: str = "embedding",
    industry_col: str = "industry_id",
    effective_time_col: str = "effective_time",
    output_effective_time_col: str = "industry_effective_time",
    dtype: torch.dtype = torch.float32,
) -> AlignedCrossAssetPanel:
    """先做严格 PIT 行业 join，再生成同步跨股票张量。"""
    joined = join_pit_industry(
        intervals,
        industry_history,
        symbol_col=symbol_col,
        prediction_time_col=prediction_time_col,
        industry_col=industry_col,
        effective_time_col=effective_time_col,
        output_effective_time_col=output_effective_time_col,
    )
    return align_interval_embeddings(
        joined,
        symbol_col=symbol_col,
        prediction_time_col=prediction_time_col,
        embedding_col=embedding_col,
        industry_col=industry_col,
        effective_time_col=output_effective_time_col,
        dtype=dtype,
    )
