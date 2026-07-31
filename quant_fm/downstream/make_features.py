"""
组装横截面排序器的特征矩阵与标签。

将冻结 FM 嵌入与手工因子及时点正确的日频面板拼接，标签为执行期前瞻收益的
横截面百分位。严格执行面板只使用信号时点已知的 ``eligible_at_signal`` 过滤股票；
``entry_fillable`` / ``exit_fillable`` 属于未来执行结果，不参与训练股票池筛选。

日频面板应为独立的、时点正确的 parquet，列::

    date, symbol, fwd_ret, eligible_at_signal

旧面板的 ``is_st, is_halt, is_new, limit_locked`` 仍向后兼容。``fwd_ret`` 应由
所选 return spec 生成；本模块不伪造收益，请提供自有 PIT 行情流程产出的真实面板。
"""

from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)

_FILTER_FLAGS = ("is_st", "is_halt", "is_new", "limit_locked")
_ROBUST_SCALE_EPS = 1e-12
_FORBIDDEN_SCORING_COLUMNS = {
    "label",
    "fwd_ret",
    "xs_ret",
    "target_return",
    "aux_target",
    "head_gain",
}


def _embedding_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("emb_")]


def _normalise_keys(frame: pl.DataFrame) -> pl.DataFrame:
    """统一信号主键类型，避免字符串股票代码在 join 时丢失前导零。"""
    required = {"date", "symbol"}
    missing = required - set(frame.columns)
    if missing:
        msg = f"missing key columns: {sorted(missing)}"
        raise ValueError(msg)
    return frame.with_columns(
        pl.col("date").cast(pl.Utf8),
        pl.col("symbol").cast(pl.Utf8).str.zfill(6),
    )


def _prepare_keyed_frame(frame: pl.DataFrame, *, name: str) -> pl.DataFrame:
    """统一并校验 ``(date, symbol)`` 主键，防止 join 静默放大样本。"""
    frame = _normalise_keys(frame)
    null_keys = frame.filter(pl.col("date").is_null() | pl.col("symbol").is_null())
    if not null_keys.is_empty():
        msg = f"{name} contains null (date, symbol) keys"
        raise ValueError(msg)
    duplicate_keys = (
        frame.filter(pl.struct(["date", "symbol"]).is_duplicated())
        .select(["date", "symbol"])
        .unique()
        .sort(["date", "symbol"])
    )
    if not duplicate_keys.is_empty():
        examples = duplicate_keys.head(5).rows()
        msg = f"{name} contains duplicate (date, symbol) keys: {examples}"
        raise ValueError(msg)
    return frame


def _feature_columns(frame: pl.DataFrame) -> list[str]:
    return _embedding_columns(frame) + [
        column for column in frame.columns if column.startswith("factor_")
    ]


def _validate_finite_features(frame: pl.DataFrame, *, context: str) -> list[str]:
    """要求实际送入模型的全部特征均为有限数值。"""
    feature_cols = _feature_columns(frame)
    invalid: list[str] = []
    for name in feature_cols:
        try:
            values = frame.select(
                pl.col(name).cast(pl.Float64, strict=False).alias(name)
            )[name]
        except (TypeError, ValueError, pl.exceptions.PolarsError):
            invalid.append(name)
            continue
        if values.null_count() or not bool(values.is_finite().all()):
            invalid.append(name)
    if invalid:
        msg = (
            f"{context} features contain null/NaN/Inf or non-numeric values: {invalid}"
        )
        raise ValueError(msg)
    return feature_cols


def _forbidden_scoring_columns(frame: pl.DataFrame) -> list[str]:
    """找出可能把未来信息带入生产评分的标签/目标列。"""
    return sorted(
        column
        for column in frame.columns
        if column in _FORBIDDEN_SCORING_COLUMNS or column.startswith("target_")
    )


def build_training_features(
    embeddings: pl.DataFrame,
    panel: pl.DataFrame,
    *,
    factors: pl.DataFrame | None = None,
    universe: pl.DataFrame | None = None,
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
    universe
        可选的逐日 PIT ``(date, symbol)`` 股票池。提供时以 inner join 限定每个
        信号日的训练横截面；不得使用未来固定股票池回填历史。
    min_names_per_day
        可交易标的少于此数的交易日丢弃（横截面过小）。

    返回
    -------
    polars.DataFrame
        列：``date, symbol, label, target_return, aux_target, head_gain,
        <emb_*>, <factor_*>``。其中 ``label`` 是按日收益百分位，
        ``target_return`` 是按日去均值收益，``aux_target`` 是按日稳健标准化并
        截断到 ``[-3, 3]`` 的辅助回归目标，``head_gain`` 是头部排序增益。
    """
    if "fwd_ret" not in panel.columns:
        msg = "training panel must contain fwd_ret"
        raise ValueError(msg)
    embeddings = _prepare_keyed_frame(embeddings, name="embeddings")
    panel = _prepare_keyed_frame(panel, name="panel")
    if factors is not None:
        factors = _prepare_keyed_frame(factors, name="factors")
    if universe is not None:
        universe = _prepare_keyed_frame(universe, name="universe")

    df = embeddings.join(panel, on=["date", "symbol"], how="inner")
    if universe is not None:
        missing_universe_dates = sorted(
            set(df["date"].unique().to_list())
            - set(universe["date"].unique().to_list())
        )
        if missing_universe_dates:
            msg = (
                "universe is missing training signal dates: "
                f"{missing_universe_dates[:5]}"
            )
            raise ValueError(msg)
        df = df.join(
            universe.select(["date", "symbol"]),
            on=["date", "symbol"],
            how="inner",
        )

    # 严格执行面板以信号时点股票池为准。entry/exit_fillable 是未来成交结果，
    # 只能交给执行模拟器处理，不能用于事后删除训练样本。
    if "eligible_at_signal" in df.columns:
        df = df.filter(
            pl.col("eligible_at_signal").cast(pl.Boolean, strict=False).fill_null(False)
        )
    else:
        # 兼容尚未迁移到 execution panel 的旧日频面板。
        for flag in _FILTER_FLAGS:
            if flag in df.columns:
                df = df.filter(
                    ~pl.col(flag).cast(pl.Boolean, strict=False).fill_null(False)
                )

    # 无效收益不得进入均值、排序或稳健尺度统计。宽松 cast 允许数据接入层把
    # 无法解析的值统一视作缺失，而不是产生带 NaN/Inf 的训练标签。
    df = df.with_columns(pl.col("fwd_ret").cast(pl.Float64, strict=False))
    invalid_returns = df.select(
        (pl.col("fwd_ret").is_null() | ~pl.col("fwd_ret").is_finite()).sum().alias("n")
    ).item()
    if invalid_returns:
        logger.warning("dropping %d rows with null/NaN/Inf fwd_ret", invalid_returns)
        df = df.filter(pl.col("fwd_ret").is_not_null() & pl.col("fwd_ret").is_finite())
    if df.is_empty():
        msg = "no rows with finite fwd_ret after eligibility filtering"
        raise ValueError(msg)

    if factors is not None:
        df = df.join(factors, on=["date", "symbol"], how="left")
    _validate_finite_features(df, context="training")

    # 所有目标均逐日计算；下游 loss 再对日期等权聚合，避免较大的横截面获得
    # 更高训练权重。
    df = df.with_columns(
        (pl.col("fwd_ret") - pl.col("fwd_ret").mean().over("date")).alias("xs_ret")
    )
    df = df.with_columns(
        pl.col("xs_ret").median().over("date").alias("_daily_median"),
        pl.col("xs_ret").std().over("date").alias("_daily_std"),
        (
            (pl.col("xs_ret").rank("average").over("date") - 1.0)
            / pl.when(pl.len().over("date") > 1)
            .then(pl.len().over("date") - 1)
            .otherwise(1)
        ).alias("label"),
    )
    df = df.with_columns(
        (pl.col("xs_ret") - pl.col("_daily_median"))
        .abs()
        .median()
        .over("date")
        .alias("_daily_mad")
    )
    df = df.with_columns(
        (1.4826 * pl.col("_daily_mad")).alias("_mad_scale")
    ).with_columns(
        pl.when(
            pl.col("_mad_scale").is_finite()
            & (pl.col("_mad_scale") > _ROBUST_SCALE_EPS)
        )
        .then(pl.col("_mad_scale"))
        .when(
            pl.col("_daily_std").is_finite()
            & (pl.col("_daily_std") > _ROBUST_SCALE_EPS)
        )
        .then(pl.col("_daily_std"))
        .otherwise(1.0)
        .alias("_aux_scale")
    )
    df = df.with_columns(
        ((pl.col("xs_ret") - pl.col("_daily_median")) / pl.col("_aux_scale"))
        .clip(-3.0, 3.0)
        .alias("aux_target"),
        (((pl.col("label") - 0.5).clip(0.0, None) / 0.5) ** 2).alias("head_gain"),
    )

    # 丢弃过薄的横截面。
    counts = df.group_by("date").len().rename({"len": "_n"})
    df = df.join(counts, on="date").filter(pl.col("_n") >= min_names_per_day)

    keep = (
        [
            "date",
            "symbol",
            "label",
            pl.col("xs_ret").alias("target_return"),
            "aux_target",
            "head_gain",
        ]
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


def build_scoring_features(
    embeddings: pl.DataFrame,
    *,
    factors: pl.DataFrame | None = None,
    universe: pl.DataFrame | None = None,
    min_names_per_day: int = 1,
) -> pl.DataFrame:
    """
    构建不含未来标签的生产打分特征。

    该接口只依赖信号日已经产生的 embedding，以及可选的 PIT 因子/股票池。
    它有意不接收 panel，防止在线推理意外读取 ``fwd_ret``。
    """
    if min_names_per_day < 1:
        msg = "min_names_per_day must be >= 1"
        raise ValueError(msg)
    df = _prepare_keyed_frame(embeddings, name="embeddings")
    emb_cols = _embedding_columns(df)
    if not emb_cols:
        msg = "embeddings must contain at least one emb_* column"
        raise ValueError(msg)

    forbidden = _forbidden_scoring_columns(df)
    if forbidden:
        msg = f"scoring embeddings contain forbidden future columns: {forbidden}"
        raise ValueError(msg)

    if universe is not None:
        scoring_dates = set(df["date"].unique().to_list())
        keys = _prepare_keyed_frame(universe, name="universe").select(
            ["date", "symbol"]
        )
        missing_universe_dates = sorted(
            scoring_dates - set(keys["date"].unique().to_list())
        )
        if missing_universe_dates:
            msg = (
                "universe is missing scoring signal dates: "
                f"{missing_universe_dates[:5]}"
            )
            raise ValueError(msg)
        df = df.join(keys, on=["date", "symbol"], how="inner")
    if factors is not None:
        factors = _prepare_keyed_frame(factors, name="factors")
        forbidden = _forbidden_scoring_columns(factors)
        if forbidden:
            msg = f"scoring factors contain forbidden future columns: {forbidden}"
            raise ValueError(msg)
        df = df.join(factors, on=["date", "symbol"], how="left")

    feature_cols = _validate_finite_features(df, context="scoring")

    counts = df.group_by("date").len().rename({"len": "_n"})
    df = df.join(counts, on="date").filter(pl.col("_n") >= min_names_per_day)
    if df.is_empty():
        msg = "no rows available for scoring after universe/day filters"
        raise ValueError(msg)
    return df.select(["date", "symbol", *feature_cols]).sort(["date", "symbol"])


def build_features(
    embeddings: pl.DataFrame,
    panel: pl.DataFrame,
    *,
    factors: pl.DataFrame | None = None,
    universe: pl.DataFrame | None = None,
    min_names_per_day: int = 20,
) -> pl.DataFrame:
    """兼容旧调用；新代码应显式调用 :func:`build_training_features`。"""
    return build_training_features(
        embeddings,
        panel,
        factors=factors,
        universe=universe,
        min_names_per_day=min_names_per_day,
    )
