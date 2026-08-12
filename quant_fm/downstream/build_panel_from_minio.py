"""
从 MinIO ``zeus-cn-quote`` 的日内快照（``default/1``）构建日频收益面板。

约定（可交易近似，适合先接下游 RankIC；非券商级 PIT 对账级别）：

* 价格单位：快照整数价 / 10000 → 元
* 日终价 ``close``：当日最后一条 snapshot 的 ``last_px``
* 日 VWAP ``vwap``：末笔 ``total_notional / total_vol``（累计成交金额/量）
* ``fwd_ret``：下一交易日 VWAP / 当日 VWAP - 1（无下一交易日则为 null）
* ``total_vol`` / ``total_notional``：保留日终累计量额，供 Regime 市场活跃度使用
* ``limit_locked``：成交后观测价格全天锁在涨停或跌停附近
* ``is_halt``：全日 ``total_vol==0``
* ``is_st`` / ``is_new``：L2 快照无法可靠判断 → 默认 ``False``
  （正式生产请换官方 ST/新股日历覆盖）

输出列与 :mod:`quant_fm.downstream.make_features` 对齐::

    date, symbol, fwd_ret, is_st, is_halt, is_new, limit_locked
    （额外保留 close, vwap, pre_close, market 便于排查）
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from pylob.pipeline.paths import zeus_default_object_key

from quant_fm.downstream.return_spec import (
    AFTER_CLOSE_AVAILABILITY,
    EXECUTION_CONTRACT_VERSION,
    RETURN_SPECS,
    get_return_spec,
    normalise_trading_calendar,
    read_trading_calendar,
    trading_calendar_sha256,
)
from quant_fm.schema.cn_l2_v1 import PRICE_SCALE
from quant_fm.scripts.minio_config import read_bucket, storage_options_for_read

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from quant_fm.downstream.return_spec import ReturnSpec


def _snapshot_uri(date: str) -> str:
    key = zeus_default_object_key(date, partition="1")
    return f"s3://{read_bucket()}/{key}"


def eod_from_snapshots(
    date: str,
    *,
    symbols: set[str] | None = None,
    storage_options: dict[str, str] | None = None,
) -> pl.DataFrame:
    """读一日 ``default/1``，按 ticker 取最后一条快照，得到日终字段。"""
    opts = storage_options or storage_options_for_read()
    uri = _snapshot_uri(date)
    logger.info("scan snapshots %s", uri)
    lf = pl.scan_parquet(uri, storage_options=opts)
    cols = [
        "ticker",
        "wind_code",
        "trading_day",
        "exch_time",
        "status",
        "pre_close_px",
        "last_px",
        "total_vol",
        "total_notional",
        "upper_limit_px",
        "lower_limit_px",
    ]
    lf = lf.select(cols)
    if symbols is not None:
        lf = lf.filter(pl.col("ticker").cast(pl.Utf8).is_in(sorted(symbols)))

    # 日终累计字段取末值；开高低只在已有成交量的快照上聚合，避免盘前昨收污染。
    eod = (
        lf.sort(["ticker", "exch_time"])
        .group_by("ticker", maintain_order=True)
        .agg(
            pl.col("wind_code").last(),
            pl.col("trading_day").last(),
            pl.col("exch_time").last(),
            pl.col("status").last(),
            pl.col("pre_close_px").last(),
            pl.col("last_px").last(),
            pl.col("last_px").filter(pl.col("total_vol") > 0).first().alias("open_px"),
            pl.col("last_px").filter(pl.col("total_vol") > 0).max().alias("high_px"),
            pl.col("last_px").filter(pl.col("total_vol") > 0).min().alias("low_px"),
            pl.col("total_vol").last(),
            pl.col("total_notional").last(),
            pl.col("upper_limit_px").last(),
            pl.col("lower_limit_px").last(),
        )
        .collect()
    )
    if eod.is_empty():
        logger.warning("no EOD rows for %s", date)
        return eod

    market = (
        pl.when(pl.col("wind_code").cast(pl.Utf8).str.ends_with(".SH"))
        .then(pl.lit("SH"))
        .when(pl.col("wind_code").cast(pl.Utf8).str.ends_with(".SZ"))
        .then(pl.lit("SZ"))
        .otherwise(pl.lit("UNK"))
    )
    out = eod.with_columns(
        pl.lit(date).alias("date"),
        pl.col("ticker").cast(pl.Utf8).str.zfill(6).alias("symbol"),
        market.alias("market"),
        (pl.col("last_px").cast(pl.Float64) / PRICE_SCALE).alias("close"),
        (pl.col("open_px").cast(pl.Float64) / PRICE_SCALE).alias("open"),
        (pl.col("high_px").cast(pl.Float64) / PRICE_SCALE).alias("high"),
        (pl.col("low_px").cast(pl.Float64) / PRICE_SCALE).alias("low"),
        (pl.col("pre_close_px").cast(pl.Float64) / PRICE_SCALE).alias("pre_close"),
        (
            pl.when(pl.col("total_vol") > 0)
            .then(
                # 快照里 cumulative notional/vol 通常已是「元」均价，勿再 / PRICE_SCALE
                pl.col("total_notional").cast(pl.Float64)
                / pl.col("total_vol").cast(pl.Float64)
            )
            .otherwise(None)
        ).alias("vwap"),
        (pl.col("upper_limit_px").cast(pl.Float64) / PRICE_SCALE).alias("upper"),
        (pl.col("lower_limit_px").cast(pl.Float64) / PRICE_SCALE).alias("lower"),
        pl.col("total_vol").cast(pl.Float64).alias("total_vol"),
        pl.col("total_notional").cast(pl.Float64).alias("total_notional"),
        (pl.col("total_vol").cast(pl.Int64) <= 0).alias("is_halt"),
    ).with_columns(
        (
            (
                (pl.col("low") >= pl.col("upper") * 0.9995)
                & (pl.col("high") <= pl.col("upper") * 1.0005)
            )
            | (
                (pl.col("high") <= pl.col("lower") * 1.0005)
                & (pl.col("low") >= pl.col("lower") * 0.9995)
            )
        ).alias("limit_locked"),
        pl.lit(False).alias("is_st"),
        pl.lit(False).alias("is_new"),
    )
    return out.select(
        [
            "date",
            "symbol",
            "market",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "vwap",
            "upper",
            "lower",
            "total_vol",
            "total_notional",
            "is_st",
            "is_halt",
            "is_new",
            "limit_locked",
        ]
    )


def attach_fwd_ret(
    daily: pl.DataFrame,
    *,
    price_col: str = "vwap",
    trading_days: list[str] | None = None,
) -> pl.DataFrame:
    """
    用**全局交易日历的下一交易日**对齐价格，计算 ``fwd_ret``。

    与按 symbol ``shift(-1)`` 的旧实现不同：这里 ``next_date`` 由**所有标的共享的
    交易日序列**决定，因此

    * 每个信号日 ``T`` 的前瞻区间恒为「T → 下一交易日」，horizon 一致；
    * 若某标的在下一交易日停牌 / 缺数据，则该行 ``fwd_ret = null``（不会像旧实现
      那样跨过缺失日、把多日收益误当作 1 日）。

    参数
    ----------
    price_col
        计算收益用的价格列（``vwap`` 或 ``close``）。
    trading_days
        交易日历（``YYYY-MM-DD`` 列表）。缺省时用 ``daily`` 中出现过的日期排序。
        **仅当传入的是连续交易日时，``fwd_ret`` 才具备「次日」语义。**
    """
    if daily.is_empty():
        return daily.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("next_date"),
            pl.lit(None, dtype=pl.Float64).alias("fwd_ret"),
        )

    present = sorted(daily["date"].unique().to_list())
    days = sorted(trading_days) if trading_days else present
    # 全局「date -> 下一交易日」映射（最后一天无下一日 → null）。
    next_map = pl.DataFrame(
        {"date": days[:-1], "next_date": days[1:]},
        schema={"date": pl.Utf8, "next_date": pl.Utf8},
    )
    daily = daily.join(next_map, on="date", how="left")

    # 用「同标的、下一交易日」的价格作为前瞻价。
    nxt = daily.select(
        pl.col("symbol"),
        pl.col("date").alias("next_date"),
        pl.col(price_col).alias("_next_px"),
    )
    out = daily.join(nxt, on=["symbol", "next_date"], how="left")
    return out.with_columns(
        (
            pl.when(pl.col("_next_px").is_not_null() & (pl.col(price_col) > 0))
            .then(pl.col("_next_px") / pl.col(price_col) - 1.0)
            .otherwise(None)
        ).alias("fwd_ret")
    ).drop(["_next_px"])


def build_execution_panel(
    daily: pl.DataFrame,
    *,
    signal_dates: list[str],
    trading_calendar: list[str],
    spec: ReturnSpec,
    require_complete_horizon: bool = True,
) -> pl.DataFrame:
    """
    构造以信号日为键、显式带建仓/退出日期的研究面板。

    目标组合只能使用 ``eligible_at_signal``；``entry_fillable`` 是下一交易日
    的实际成交结果，只能由回测执行器用于拒单，不能用于事后补选股票。
    """
    spec.validate()
    required = {
        "date",
        "symbol",
        spec.entry_price,
        spec.exit_price,
        "is_st",
        "is_new",
        "is_halt",
        "limit_locked",
    }
    missing = required - set(daily.columns)
    if missing:
        msg = f"daily quotes missing required columns: {sorted(missing)}"
        raise ValueError(msg)
    calendar = normalise_trading_calendar(trading_calendar)
    normalized_signal_dates = [str(value) for value in signal_dates]
    if len(normalized_signal_dates) != len(set(normalized_signal_dates)):
        msg = "signal_dates contains duplicate dates"
        raise ValueError(msg)
    if normalized_signal_dates != sorted(normalized_signal_dates):
        msg = "signal_dates must be strictly increasing"
        raise ValueError(msg)
    positions = {date: i for i, date in enumerate(calendar)}
    unknown = sorted(set(normalized_signal_dates) - set(positions))
    if unknown:
        msg = f"signal dates absent from trading calendar: {unknown[:5]}"
        raise ValueError(msg)

    mapping_rows: list[dict[str, str | int | None]] = []
    incomplete: list[str] = []
    for date in normalized_signal_dates:
        index = positions[date]
        entry_index = index + spec.entry_day_lag
        exit_index = index + spec.exit_day_lag
        entry_date = calendar[entry_index] if entry_index < len(calendar) else None
        exit_date = calendar[exit_index] if exit_index < len(calendar) else None
        if entry_date is None or exit_date is None:
            incomplete.append(date)
        mapping_rows.append(
            {
                "date": date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "signal_calendar_index": index,
                "entry_calendar_index": entry_index if entry_date is not None else None,
                "exit_calendar_index": exit_index if exit_date is not None else None,
            }
        )
    if incomplete and require_complete_horizon:
        msg = (
            "trading calendar does not cover the requested return horizon for "
            f"signal dates: {incomplete[:5]}"
        )
        raise ValueError(msg)

    mapping = pl.DataFrame(
        mapping_rows,
        schema={
            "date": pl.Utf8,
            "entry_date": pl.Utf8,
            "exit_date": pl.Utf8,
            "signal_calendar_index": pl.Int64,
            "entry_calendar_index": pl.Int64,
            "exit_calendar_index": pl.Int64,
        },
    )
    required_quote_dates = {
        str(value)
        for column in ("date", "entry_date", "exit_date")
        for value in mapping[column].drop_nulls().to_list()
    }
    available_quote_dates = {str(value) for value in daily["date"].unique().to_list()}
    missing_quote_dates = sorted(required_quote_dates - available_quote_dates)
    if missing_quote_dates and require_complete_horizon:
        msg = (
            "daily quotes do not cover signal/entry/exit trading dates: "
            f"{missing_quote_dates[:5]}"
        )
        raise ValueError(msg)
    base_columns = [
        "date",
        "symbol",
        *[name for name in ("market",) if name in daily.columns],
        "is_st",
        "is_new",
        "is_halt",
        "limit_locked",
    ]
    signal = daily.select(base_columns).filter(
        pl.col("date").is_in(normalized_signal_dates)
    )
    signal = signal.rename(
        {
            "is_st": "is_st_at_signal",
            "is_new": "is_new_at_signal",
            "is_halt": "is_halt_at_signal",
            "limit_locked": "limit_locked_at_signal",
        }
    ).with_columns(
        (
            ~pl.col("is_st_at_signal").fill_null(True)
            & ~pl.col("is_new_at_signal").fill_null(True)
            & ~pl.col("is_halt_at_signal").fill_null(True)
        ).alias("eligible_at_signal")
    )
    entry = daily.select(
        pl.col("date").alias("entry_date"),
        "symbol",
        pl.col(spec.entry_price).cast(pl.Float64).alias("entry_px"),
        pl.col("is_halt").alias("is_halt_entry"),
        pl.col("limit_locked").alias("limit_locked_entry"),
    )
    exit_quotes = daily.select(
        pl.col("date").alias("exit_date"),
        "symbol",
        pl.col(spec.exit_price).cast(pl.Float64).alias("exit_px"),
        pl.col("is_halt").alias("is_halt_exit"),
        pl.col("limit_locked").alias("limit_locked_exit"),
    )
    out = (
        signal.join(mapping, on="date", how="left")
        .join(entry, on=["symbol", "entry_date"], how="left")
        .join(exit_quotes, on=["symbol", "exit_date"], how="left")
        .with_columns(
            pl.lit(EXECUTION_CONTRACT_VERSION).alias("execution_contract_version"),
            pl.lit(spec.name).alias("return_spec"),
            pl.lit(AFTER_CLOSE_AVAILABILITY).alias("signal_availability"),
            pl.lit(trading_calendar_sha256(calendar)).alias("trading_calendar_sha256"),
            pl.lit(len(calendar)).cast(pl.Int32).alias("calendar_date_count"),
            pl.lit(spec.entry_day_lag).cast(pl.Int32).alias("entry_day_lag"),
            pl.lit(spec.exit_day_lag).cast(pl.Int32).alias("exit_day_lag"),
            pl.lit(spec.entry_price).alias("entry_price_field"),
            pl.lit(spec.exit_price).alias("exit_price_field"),
            (
                pl.col("entry_px").is_not_null()
                & (pl.col("entry_px") > 0)
                & ~pl.col("is_halt_entry").fill_null(True)
                & ~pl.col("limit_locked_entry").fill_null(True)
            ).alias("entry_fillable"),
            (
                pl.col("exit_px").is_not_null()
                & (pl.col("exit_px") > 0)
                & ~pl.col("is_halt_exit").fill_null(True)
                & ~pl.col("limit_locked_exit").fill_null(True)
            ).alias("exit_fillable"),
        )
        .with_columns(
            pl.when(
                pl.col("entry_px").is_not_null()
                & pl.col("exit_px").is_not_null()
                & (pl.col("entry_px") > 0)
            )
            .then(pl.col("exit_px") / pl.col("entry_px") - 1.0)
            .otherwise(None)
            .alias("fwd_ret")
        )
        .sort(["date", "symbol"])
    )
    if out.select(pl.struct(["date", "symbol"]).is_duplicated().any()).item():
        msg = "execution panel contains duplicate (date, symbol) keys"
        raise ValueError(msg)
    return out


def build_panel(
    dates: list[str],
    *,
    symbols: set[str] | None = None,
    price_col: str = "vwap",
    signal_dates: list[str] | None = None,
    return_spec: ReturnSpec | None = None,
    require_complete_horizon: bool = True,
) -> pl.DataFrame:
    """
    多日构建面板；``fwd_ret`` 由**全局下一交易日**对齐（见 :func:`attach_fwd_ret`）。

    ``dates`` 应为**连续交易日**并在末尾多带一天（用于最后一个信号日的 fwd_ret）；
    否则 ``fwd_ret`` 只是「下一采样日」的收益，不具备次日语义。
    """
    opts = storage_options_for_read()
    frames: list[pl.DataFrame] = []
    strict_execution = return_spec is not None and require_complete_horizon
    for d in dates:
        try:
            frames.append(eod_from_snapshots(d, symbols=symbols, storage_options=opts))
        except Exception:
            if strict_execution:
                logger.exception("failed to load required execution date %s", d)
                raise
            logger.exception("skip %s", d)
    if not frames:
        return pl.DataFrame()
    daily = pl.concat([f for f in frames if not f.is_empty()], how="vertical_relaxed")
    if return_spec is not None:
        return build_execution_panel(
            daily,
            signal_dates=signal_dates or list(dates),
            trading_calendar=list(dates),
            spec=return_spec,
            require_complete_horizon=require_complete_horizon,
        )
    return attach_fwd_ret(daily, price_col=price_col, trading_days=list(dates))


def main() -> None:
    """Build and persist a daily downstream label panel."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dates",
        help="逗号分隔交易日 YYYY-MM-DD；与 --dates-file / --from-embeddings 三选一",
    )
    parser.add_argument("--dates-file", type=Path)
    parser.add_argument(
        "--signal-dates-file",
        type=Path,
        help="score 信号日期；与完整 --calendar-file 分开，避免少估值日",
    )
    parser.add_argument(
        "--calendar-file",
        type=Path,
        help="覆盖 entry/exit horizon 的完整连续交易日历",
    )
    parser.add_argument(
        "--from-embeddings",
        type=Path,
        help="从 embedding parquet 推断 date/symbol 集合",
    )
    parser.add_argument(
        "--symbols",
        help="逗号分隔 6 位代码；默认全市场（很慢）或随 embeddings",
    )
    parser.add_argument(
        "--price",
        choices=("vwap", "close"),
        default="vwap",
        help="fwd_ret 用的价格（默认 vwap）",
    )
    parser.add_argument(
        "--return-spec",
        choices=sorted(RETURN_SPECS),
        help="显式可执行收益口径；给定后替代 legacy --price",
    )
    parser.add_argument(
        "--allow-incomplete-horizon",
        action="store_true",
        help="研究排查用；默认未来行情不完整即失败",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("quant_fm/runs/medium_try/panel/daily_panel.parquet"),
    )
    args = parser.parse_args()

    symbols: set[str] | None = None
    dates: list[str] = []
    signal_dates: list[str] = []

    def _read_dates(path: Path) -> list[str]:
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    if args.from_embeddings is not None:
        emb = pl.read_parquet(args.from_embeddings)
        signal_dates = sorted(emb["date"].unique().to_list())
        dates = signal_dates
        symbols = set(emb["symbol"].cast(pl.Utf8).str.zfill(6).to_list())
        logger.info(
            "from embeddings: %d dates, %d symbols", len(signal_dates), len(symbols)
        )
    elif args.signal_dates_file is not None:
        signal_dates = _read_dates(args.signal_dates_file)
        dates = signal_dates
    elif args.dates_file is not None:
        dates = _read_dates(args.dates_file)
        signal_dates = list(dates)
    elif args.dates:
        dates = [d.strip() for d in args.dates.split(",") if d.strip()]
        signal_dates = list(dates)
    else:
        parser.error("need --dates or --dates-file or --from-embeddings")

    if args.calendar_file is not None:
        dates = read_trading_calendar(args.calendar_file)
    if args.return_spec and args.calendar_file is None:
        parser.error("--return-spec requires an explicit --calendar-file")

    if args.symbols:
        symbols = {s.strip().zfill(6) for s in args.symbols.split(",") if s.strip()}

    # fwd_ret 需要多一天；若末日无下一交易日则该日为空，调用方可多传一天。
    panel = build_panel(
        dates,
        symbols=symbols,
        price_col=args.price,
        signal_dates=signal_dates,
        return_spec=get_return_spec(args.return_spec) if args.return_spec else None,
        require_complete_horizon=not args.allow_incomplete_horizon,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(args.out)
    n_ret = panel.filter(pl.col("fwd_ret").is_not_null()).height
    logger.info(
        "wrote %s rows=%d with_fwd_ret=%d → %s",
        args.out,
        panel.height,
        n_ret,
        panel.columns,
    )
    if n_ret == 0:
        logger.warning(
            "fwd_ret 全空：日历末尾没有下一交易日。请把 dates 多加一天（如下一开市日）再跑。"
        )


if __name__ == "__main__":
    main()
