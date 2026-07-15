"""
从 MinIO ``zeus-cn-quote`` 的日内快照（``default/1``）构建日频收益面板。

约定（可交易近似，适合先接下游 RankIC；非券商级 PIT 对账级别）：

* 价格单位：快照整数价 / 10000 → 元
* 日终价 ``close``：当日最后一条 snapshot 的 ``last_px``
* 日 VWAP ``vwap``：末笔 ``total_notional / total_vol``（累计成交金额/量）
* ``fwd_ret``：下一交易日 VWAP / 当日 VWAP - 1（无下一交易日则为 null）
* ``limit_locked``：收盘价触及涨跌停（相对 upper/lower 容差内）
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

import polars as pl
from pylob.pipeline.paths import zeus_default_object_key

from quant_fm.schema.cn_l2_v1 import PRICE_SCALE
from quant_fm.scripts.minio_config import read_bucket, storage_options_for_read

logger = logging.getLogger(__name__)


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

    # 每个 ticker 取 exch_time 最大的一行（日终）
    eod = (
        lf.sort(["ticker", "exch_time"])
        .group_by("ticker", maintain_order=True)
        .agg(pl.all().last())
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
        (pl.col("total_vol").cast(pl.Int64) <= 0).alias("is_halt"),
    ).with_columns(
        (
            (pl.col("close") >= pl.col("upper") * 0.9995)
            | (pl.col("close") <= pl.col("lower") * 1.0005)
        ).alias("limit_locked"),
        pl.lit(False).alias("is_st"),
        pl.lit(False).alias("is_new"),
    )
    return out.select(
        [
            "date",
            "symbol",
            "market",
            "close",
            "pre_close",
            "vwap",
            "upper",
            "lower",
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
) -> pl.DataFrame:
    """按 symbol 对齐下一交易日价格，计算 ``fwd_ret``。"""
    if daily.is_empty():
        return daily.with_columns(pl.lit(None, dtype=pl.Float64).alias("fwd_ret"))

    daily = daily.sort(["symbol", "date"])
    # shift within symbol: join date_i with date_{i+1} via window
    with_next = daily.with_columns(
        pl.col(price_col).shift(-1).over("symbol").alias("_next_px"),
        pl.col("date").shift(-1).over("symbol").alias("next_date"),
    )
    return with_next.with_columns(
        (
            pl.when(pl.col("_next_px").is_not_null() & (pl.col(price_col) > 0))
            .then(pl.col("_next_px") / pl.col(price_col) - 1.0)
            .otherwise(None)
        ).alias("fwd_ret")
    ).drop(["_next_px"])


def build_panel(
    dates: list[str],
    *,
    symbols: set[str] | None = None,
    price_col: str = "vwap",
) -> pl.DataFrame:
    """多日构建面板；``fwd_ret`` 需要日历上后一天存在才有值。"""
    opts = storage_options_for_read()
    frames: list[pl.DataFrame] = []
    for d in dates:
        try:
            frames.append(eod_from_snapshots(d, symbols=symbols, storage_options=opts))
        except Exception:
            logger.exception("skip %s", d)
    if not frames:
        return pl.DataFrame()
    daily = pl.concat([f for f in frames if not f.is_empty()], how="vertical_relaxed")
    return attach_fwd_ret(daily, price_col=price_col)


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
        "--out",
        type=Path,
        default=Path("quant_fm/runs/medium_try/panel/daily_panel.parquet"),
    )
    args = parser.parse_args()

    symbols: set[str] | None = None
    dates: list[str] = []

    if args.from_embeddings is not None:
        emb = pl.read_parquet(args.from_embeddings)
        dates = sorted(emb["date"].unique().to_list())
        symbols = set(emb["symbol"].cast(pl.Utf8).str.zfill(6).to_list())
        logger.info("from embeddings: %d dates, %d symbols", len(dates), len(symbols))
    elif args.dates_file is not None:
        dates = [
            ln.strip()
            for ln in args.dates_file.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    elif args.dates:
        dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    else:
        parser.error("need --dates or --dates-file or --from-embeddings")

    if args.symbols:
        symbols = {s.strip().zfill(6) for s in args.symbols.split(",") if s.strip()}

    # fwd_ret 需要多一天；若末日无下一交易日则该日为空，调用方可多传一天。
    panel = build_panel(dates, symbols=symbols, price_col=args.price)
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
