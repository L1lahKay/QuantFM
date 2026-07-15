"""Shared helper functions for legacy quote validation examples."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pylob.orderbook_builder_sh import OrderBookSH
from pylob.orderbook_builder_sz import OrderBookSZ

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

DEFAULT_CUT_TIME = 151000000


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a compact console logger for example notebooks and scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def build_legacy_loader(
    date: str, broker: str, colo: str, base_archive_dir: str
) -> Any:
    """Create the legacy quote loader lazily to keep imports optional."""
    from zeus_quant.data.dataset.loader.quote import CSAQuoteLegacyLoader

    return CSAQuoteLegacyLoader(
        date=date,
        broker=broker,
        colo=colo,
        base_archive_dir=base_archive_dir,
    )


def detect_market(symbol: str) -> str:
    """Infer market from a 6-digit stock code."""
    if symbol.startswith("6"):
        return "SH"
    if symbol.startswith(("0", "3")):
        return "SZ"
    msg = f"无法从代码推断市场: {symbol}"
    raise ValueError(msg)


def create_orderbook(market: str) -> OrderBookSH | OrderBookSZ:
    """Create the matching order book for a market."""
    return OrderBookSH() if market == "SH" else OrderBookSZ()


def load_symbol_frames(
    loader: Any, quote_dir: str, symbol: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load transaction and order frames for a single symbol."""
    trade_df = loader.load(
        instruments=symbol,
        quote_type="transaction",
        quote_dir=quote_dir,
        evaluate_eagerly=True,
    )
    order_df = loader.load(
        instruments=symbol,
        quote_type="order",
        quote_dir=quote_dir,
        evaluate_eagerly=True,
    )
    return trade_df, order_df


def run_single_validation(
    trade_df: pl.DataFrame,
    order_df: pl.DataFrame,
    symbol: str,
    market: str,
    cut_time: int = DEFAULT_CUT_TIME,
) -> tuple[bool, pd.DataFrame, OrderBookSH | OrderBookSZ]:
    """Run the core replay workflow for one symbol."""
    orderbook = create_orderbook(market)
    order_data = orderbook.prepare_market_data(
        trade_df,
        order_df,
        symbol=symbol,
        cut_time=cut_time,
    )
    orderbook.process_workflow(order_data)
    matched = orderbook.compare_df(order_data)
    return matched, order_data, orderbook
