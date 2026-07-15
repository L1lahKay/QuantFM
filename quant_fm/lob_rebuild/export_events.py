"""
将 PyLOB 清洗产物转换为 cn_l2_v1 规范事件 parquet。

繁重工作（沪深订单簿重建）已由 :func:`pylob.pipeline.workflow.build_clean_dataset`
完成，并写出 ``{market}/{symbol}/events.parquet``。本模块读取这些事件并
重新投影到 cn_l2_v1 规范空间，*不*进行分箱（分箱在后续仅于训练窗口进行，以防泄漏）。

输出布局::

    {out_dir} / {market} / {symbol} / {date}.parquet

以便每个标的可累积多个交易日，供分片训练使用。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from quant_fm.schema.cn_l2_v1 import events_to_canonical

logger = logging.getLogger(__name__)


def events_frame_to_canonical(
    events: pl.DataFrame,
    *,
    date: str,
    market: str,
) -> pl.DataFrame:
    """单标的场景下对 :func:`events_to_canonical` 的薄封装。"""
    return events_to_canonical(events, date=date, market=market)


def canonicalize_clean_dir(
    clean_dir: Path,
    out_dir: Path,
    *,
    date: str,
    markets: tuple[str, ...] = ("SH", "SZ"),
    symbols: tuple[str, ...] | None = None,
) -> list[Path]:
    """
    规范化 ``clean_dir`` 下所有 ``events.parquet``。

    参数
    ----------
    clean_dir
        ``build_clean_dataset`` 写入的根目录（含 ``{market}/{symbol}``）。
    out_dir
        规范 ``{market}/{symbol}/{date}.parquet`` 的目标根目录。
    date
        交易日（``YYYY-MM-DD``），写入每一行。
    markets
        要扫描的市场。
    symbols
        可选的 6 位代码白名单；``None`` 表示扫描到的全部。

    返回
    -------
    list[pathlib.Path]
        已写入的规范 parquet 路径列表。
    """
    clean_dir = Path(clean_dir)
    out_dir = Path(out_dir)
    written: list[Path] = []

    for market in markets:
        market_dir = clean_dir / market
        if not market_dir.is_dir():
            continue
        for symbol_dir in sorted(market_dir.iterdir()):
            if not symbol_dir.is_dir():
                continue
            symbol = symbol_dir.name
            if symbols is not None and symbol not in symbols:
                continue
            events_path = symbol_dir / "events.parquet"
            if not events_path.exists():
                logger.warning("missing %s, skipping", events_path)
                continue

            events = pl.read_parquet(events_path)
            if events.is_empty():
                logger.warning("empty %s, skipping", events_path)
                continue

            canonical = events_frame_to_canonical(events, date=date, market=market)

            dst_dir = out_dir / market / symbol
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{date}.parquet"
            canonical.write_parquet(dst)
            written.append(dst)
            logger.info("wrote %s (%d events)", dst, canonical.height)

    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--markets", default="SH,SZ")
    parser.add_argument("--symbols", default=None)
    return parser.parse_args()


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    symbols = tuple(args.symbols.split(",")) if args.symbols else None
    markets = tuple(m.strip().upper() for m in args.markets.split(","))
    paths = canonicalize_clean_dir(
        args.clean_dir,
        args.out_dir,
        date=args.date,
        markets=markets,
        symbols=symbols,
    )
    logger.info("canonicalized %d symbol-days", len(paths))


if __name__ == "__main__":
    main()
