"""Regime 日级归档与全局 finalize 命令行入口。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from quant_fm.downstream.return_spec import read_trading_calendar
from quant_fm.regime.archive import archive_atomic_day
from quant_fm.regime.finalize import finalize_l2_regime_features


def main() -> None:
    """解析 Regime 数据生产子命令。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    archive = subparsers.add_parser("archive", help="合并一天的逐股 atomic 文件")
    archive.add_argument("--clean-dir", type=Path, required=True)
    archive.add_argument("--date", required=True)
    archive.add_argument("--coverage-receipt", type=Path, required=True)
    archive.add_argument("--out", type=Path, required=True)
    archive.add_argument("--resume", action="store_true")

    finalize = subparsers.add_parser("finalize", help="生成最终 Level-2 Regime 表")
    finalize.add_argument("--atomic-dir", type=Path, required=True)
    finalize.add_argument("--eod", type=Path, required=True)
    finalize.add_argument("--universe", type=Path, required=True)
    finalize.add_argument("--calendar", type=Path, required=True)
    finalize.add_argument(
        "--signal-dates-file",
        type=Path,
        help="可选；只发布该有序交易日文件指定的 atomic 日期",
    )
    finalize.add_argument("--out", type=Path, required=True)
    finalize.add_argument("--amount-column", default="total_notional")
    finalize.add_argument("--min-eod-coverage", type=float, default=0.98)
    finalize.add_argument("--min-board-names", type=int, default=1)
    finalize.add_argument("--min-book-valid-ratio", type=float, default=0.0)
    args = parser.parse_args()

    if args.command == "archive":
        archive_atomic_day(
            args.clean_dir,
            args.out,
            date=args.date,
            coverage_receipt=args.coverage_receipt,
            skip_existing=args.resume,
        )
        return
    finalize_l2_regime_features(
        atomic_dir=args.atomic_dir,
        eod_path=args.eod,
        universe_path=args.universe,
        calendar_path=args.calendar,
        output_path=args.out,
        amount_column=args.amount_column,
        min_eod_coverage=args.min_eod_coverage,
        min_board_names=args.min_board_names,
        min_book_valid_ratio=args.min_book_valid_ratio,
        signal_dates=(
            read_trading_calendar(args.signal_dates_file)
            if args.signal_dates_file is not None
            else None
        ),
    )


if __name__ == "__main__":
    main()
