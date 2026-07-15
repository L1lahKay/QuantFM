"""
组装横截面排序器的特征矩阵与标签。

将冻结 FM 嵌入与手工因子及时点正确的日频面板拼接，标签为 T+1 横截面超额收益排序
（可交易 VWAP 约定，横截面去均值）。剔除不可交易行（ST / 停牌 / 新股 / 涨跌停锁死），
避免模型学习无法实际交易的样本。

日频面板应为独立的、时点正确的 parquet，列::

    date, symbol, fwd_ret, is_st, is_halt, is_new, limit_locked

``fwd_ret`` 为可交易 VWAP 约定下的 T+1 前瞻收益。本模块不伪造收益；
请提供自有 PIT 行情流程产出的真实面板。
"""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)

_FILTER_FLAGS = ("is_st", "is_halt", "is_new", "limit_locked")


def _embedding_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("emb_")]


def build_features(
    embeddings: pl.DataFrame,
    panel: pl.DataFrame,
    *,
    factors: pl.DataFrame | None = None,
    min_names_per_day: int = 20,
) -> pl.DataFrame:
    """
    将嵌入、因子与标签拼接为整洁训练表。

    参数
    ----------
    embeddings
        :func:`quant_fm.embedding.extract_stock_day_embeddings` 的输出。
    panel
        时点正确的日频面板（见模块文档字符串）。
    factors
        可选 ``(date, symbol, <factor cols>)`` 手工因子表。
    min_names_per_day
        可交易标的少于此数的交易日丢弃（横截面过小）。

    返回
    -------
    polars.DataFrame
        列：``date, symbol, label, <emb_*>, <factor_*>``。
    """
    df = embeddings.join(panel, on=["date", "symbol"], how="inner")

    # 可交易性过滤：仅保留 T+1 可操作的标的。
    for flag in _FILTER_FLAGS:
        if flag in df.columns:
            df = df.filter(~pl.col(flag).cast(pl.Boolean).fill_null(False))

    if factors is not None:
        df = df.join(factors, on=["date", "symbol"], how="left")

    # 标签为横截面超额收益排序（按日）。
    df = df.with_columns(
        (pl.col("fwd_ret") - pl.col("fwd_ret").mean().over("date")).alias("xs_ret")
    )
    df = df.with_columns(
        (pl.col("xs_ret").rank("average").over("date") / pl.len().over("date")).alias(
            "label"
        )
    )

    # 丢弃过薄的横截面。
    counts = df.group_by("date").len().rename({"len": "_n"})
    df = df.join(counts, on="date").filter(pl.col("_n") >= min_names_per_day)

    keep = (
        ["date", "symbol", "label"]
        + _embedding_columns(df)
        + [c for c in df.columns if c.startswith("factor_")]
    )
    logger.info(
        "features: %d rows, %d days, %d embedding dims",
        df.height,
        df["date"].n_unique(),
        len(_embedding_columns(df)),
    )
    return df.select(keep).sort(["date", "symbol"])
