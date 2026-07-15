"""
cn_l2_v1：A 股 Level-2 数据的统一规范事件模式。

【新手】本文件定义「一条事件在磁盘上长什么样」：
  - 一行 = 一个市场事件（挂单 ADD / 撤单 CANCEL / 成交 EXEC）
  - symbol、date 只做索引，不进模型词表
  - price 单位是「元」（PyLOB 原始整数 ÷10000）

目标是将异构的上海（XSHG）与深圳（XSHE）委托 / 成交 / 快照记录
投影到单一抽象事件空间，使下游分词器与模型无需感知交易所特有差异。
"""

from __future__ import annotations

import polars as pl
import pyarrow as pa

SCHEMA_VERSION = "cn_l2_v1"

PRICE_SCALE = 10_000.0  # [导读] PyLOB 存价格常用整数，除以 1e4 得到元

# [导读] 写入 parquet 时列顺序固定；训练/分词都依赖这套列名
CANONICAL_COLUMNS: tuple[str, ...] = (
    "schema_version",
    "date",
    "exchange",
    "market",
    "symbol",
    "security_id",
    "board",
    "session",
    "event_source",
    "evt_type",
    "side",
    "price",
    "qty",
    "amount",
    "order_type",
    "level",
    "delta_t",
    "int_time",
    "local_time",
    "source_seqnum",
    "event_idx",
    "quality_flag",
)

# 固定类别取值空间（词表的确定性部分）。
EVT_TYPES: tuple[str, ...] = ("ADD", "CANCEL", "EXEC", "SNAP", "STATUS", "UNKNOWN")
SIDES: tuple[str, ...] = ("B", "S", "N")
SESSIONS: tuple[str, ...] = (
    "PRE_OPEN",
    "OPEN_CALL",
    "COOLING",
    "CONT_AM",
    "LUNCH",
    "CONT_PM",
    "CLOSE_CALL",
    "AFTER_CLOSE",
    "HALT",
    "UNKNOWN",
)
EVENT_SOURCES: tuple[str, ...] = ("ORDER", "TRADE", "SNAPSHOT", "QUEUE", "DERIVED")
ORDER_TYPES: tuple[str, ...] = ("LIMIT", "MARKET", "BEST", "CANCEL", "UNKNOWN")
BOARDS: tuple[str, ...] = (
    "MAIN",
    "STAR",
    "CHINEXT",
    "SME",
    "BSE",
    "UNKNOWN",
)

# 将 PyLOB 事件流词表（events.py）映射到 cn_l2_v1 取值。
_EVT_TYPE_MAP = {"ADD": "ADD", "CANCEL": "CANCEL", "TRADE": "EXEC"}
_SIDE_MAP = {"BUY": "B", "SELL": "S", "UNKNOWN": "N"}
_SESSION_MAP = {
    "OPEN_AUCTION": "OPEN_CALL",
    "CONTINUOUS_AM": "CONT_AM",
    "MIDDAY_BREAK": "LUNCH",
    "CONTINUOUS_PM": "CONT_PM",
    "CLOSE_AUCTION": "CLOSE_CALL",
}


def exchange_of(market: str) -> str:
    """返回市场代码对应的 MIC（``XSHG``/``XSHE``）。"""
    return "XSHG" if market.upper() == "SH" else "XSHE"


def board_of(security_id: str, market: str) -> str:
    """将 6 位证券代码归类到交易板块。"""
    sid = str(security_id).zfill(6)
    prefix2 = sid[:2]
    prefix3 = sid[:3]
    if market.upper() == "SH":
        if prefix3 == "688":
            return "STAR"
        if prefix2 in {"60"}:
            return "MAIN"
        return "UNKNOWN"
    # 深圳与北京在此共用 6 位代码空间。
    if prefix3 in {"300", "301"}:
        return "CHINEXT"
    if prefix3 in {"002", "003"}:
        return "SME"
    if prefix3 in {"000", "001"}:
        return "MAIN"
    if prefix2 in {"43", "83", "87", "88", "92"} or prefix3 == "920":
        return "BSE"
    return "UNKNOWN"


def _order_type_expr() -> pl.Expr:
    """从 PyLOB 事件流尽力推断规范 order_type。"""
    return (
        pl.when(pl.col("event_type") == "CANCEL")
        .then(pl.lit("CANCEL"))
        .when(pl.col("event_type") == "TRADE")
        .then(pl.lit("UNKNOWN"))
        .otherwise(pl.lit("LIMIT"))
    )


def events_to_canonical(
    events: pl.DataFrame,
    *,
    date: str,
    market: str,
) -> pl.DataFrame:
    """
    将 PyLOB ``events.parquet`` 数据帧转换为 cn_l2_v1 规范形式。

    参数
    ----------
    events
        :func:`pylob.pipeline.events.build_event_stream` 的输出（polars 格式）。
    date
        交易日，``YYYY-MM-DD`` 形式（作为索引列携带）。
    market
        ``"SH"`` 或 ``"SZ"``。

    返回
    -------
    polars.DataFrame
        列恰好为 :data:`CANONICAL_COLUMNS` 的数据帧。
    """
    market = market.upper()
    mic = exchange_of(market)
    suffix = "SH" if market == "SH" else "SZ"

    df = events.with_columns(
        pl.col("symbol").cast(pl.String).str.zfill(6).alias("security_id"),
        pl.col("event_type").cast(pl.String),
        pl.col("session_phase").cast(pl.String),
    )

    boards = [board_of(sid, market) for sid in df["security_id"].to_list()]

    out = df.with_columns(
        pl.lit(SCHEMA_VERSION).alias("schema_version"),
        pl.lit(date).alias("date"),
        pl.lit(mic).alias("exchange"),
        pl.lit("A_SHARE").alias("market"),
        (pl.col("security_id") + f".{suffix}").alias("symbol"),
        pl.Series("board", boards, dtype=pl.String),
        pl.col("session_phase")
        .replace_strict(_SESSION_MAP, default="UNKNOWN")
        .alias("session"),
        pl.when(pl.col("event_type") == "TRADE")
        .then(pl.lit("TRADE"))
        .otherwise(pl.lit("ORDER"))
        .alias("event_source"),
        pl.col("event_type")
        .replace_strict(_EVT_TYPE_MAP, default="UNKNOWN")
        .alias("evt_type"),
        pl.col("side").replace_strict(_SIDE_MAP, default="N").alias("side"),
        (pl.col("price").cast(pl.Float64) / PRICE_SCALE).alias("price"),
        pl.col("volume").cast(pl.Float64).alias("qty"),
        _order_type_expr().alias("order_type"),
        pl.lit(0, dtype=pl.Int32).alias("level"),
        pl.col("serial").cast(pl.Int64).alias("source_seqnum"),
        pl.lit(0, dtype=pl.Int32).alias("quality_flag"),
    ).with_columns(
        (pl.col("price") * pl.col("qty")).alias("amount"),
        pl.col("delta_t").cast(pl.Int64),
        pl.col("int_time").cast(pl.Int64),
        pl.col("local_time").cast(pl.Int64),
        pl.col("event_idx").cast(pl.Int64),
    )

    return out.select(CANONICAL_COLUMNS)


def canonical_arrow_schema() -> pa.Schema:
    """返回描述 :data:`CANONICAL_COLUMNS` 的 pyarrow 模式。"""
    return pa.schema(
        [
            ("schema_version", pa.string()),
            ("date", pa.string()),
            ("exchange", pa.string()),
            ("market", pa.string()),
            ("symbol", pa.string()),
            ("security_id", pa.string()),
            ("board", pa.string()),
            ("session", pa.string()),
            ("event_source", pa.string()),
            ("evt_type", pa.string()),
            ("side", pa.string()),
            ("price", pa.float64()),
            ("qty", pa.float64()),
            ("amount", pa.float64()),
            ("order_type", pa.string()),
            ("level", pa.int32()),
            ("delta_t", pa.int64()),
            ("int_time", pa.int64()),
            ("local_time", pa.int64()),
            ("source_seqnum", pa.int64()),
            ("event_idx", pa.int64()),
            ("quality_flag", pa.int32()),
        ]
    )
