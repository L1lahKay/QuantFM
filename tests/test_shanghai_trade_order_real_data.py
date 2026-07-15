import os
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from pylob.orderbook_builder_sh import OrderBookSH

# Optional real-data fixtures (override via env; skipped if missing).
TRADE_PATH = os.getenv(
    "PYLOB_SH_TRADE_PATH",
    "tests/fixtures/trade_600000_SH_20260202.parquet",
)
ORDER_PATH = os.getenv(
    "PYLOB_SH_ORDER_PATH",
    "tests/fixtures/order_600000_SH_20260202.parquet",
)


CUT_TIME = int(os.getenv("SH_CUT_TIME", "150100000"))
CUT_SERIAL = None
MAX_ROWS = None


def skip_if_unavailable(path: str, label: str) -> None:
    """Skip real-data tests when the fixture path is missing or inaccessible."""
    try:
        if not Path(path).exists():
            pytest.skip(f"{label} parquet not found: {path}")
    except PermissionError:
        pytest.skip(f"No permission to access {label} parquet: {path}")
    except OSError as exc:
        pytest.skip(f"Unable to access {label} parquet: {path} ({exc})")


@pytest.fixture(scope="session")
def trade_df() -> pl.DataFrame:
    """Load session-scoped trade Parquet or skip if missing."""
    skip_if_unavailable(TRADE_PATH, "Trade")
    try:
        return pl.read_parquet(TRADE_PATH)
    except PermissionError:
        pytest.skip(f"No permission to read Trade parquet: {TRADE_PATH}")
    except OSError as exc:
        pytest.skip(f"Unable to read Trade parquet: {TRADE_PATH} ({exc})")


@pytest.fixture(scope="session")
def order_df() -> pl.DataFrame:
    """Load session-scoped order Parquet or skip if missing."""
    skip_if_unavailable(ORDER_PATH, "Order")
    try:
        return pl.read_parquet(ORDER_PATH)
    except PermissionError:
        pytest.skip(f"No permission to read Order parquet: {ORDER_PATH}")
    except OSError as exc:
        pytest.skip(f"Unable to read Order parquet: {ORDER_PATH} ({exc})")


@pytest.fixture
def orderbook() -> OrderBookSH:
    """Fresh ``OrderBookSH`` instance per test."""
    return OrderBookSH()


# 撤单对比
def compare_cancel_orders(order_df: pd.DataFrame, ob: OrderBookSH) -> bool:
    """Assert merged cancel events match ``get_cancels_table`` output."""
    need_cols = {"type", "order_type", "orderorino"}
    miss = need_cols - set(order_df.columns)
    if miss:
        msg = f"merged_df 缺少列 {miss}，无法做撤单对比"
        raise AssertionError(msg)

    real = order_df[(order_df["type"] == "O") & (order_df["order_type"] == "D")].copy()
    real = real[["orderorino"]].rename(columns={"orderorino": "订单ID"})
    real["订单ID"] = real["订单ID"].astype("int64")

    my_cancel_list = ob.get_cancels_table() if hasattr(ob, "get_cancels_table") else []
    book = pd.DataFrame(my_cancel_list or [])

    if book.empty:
        book = pd.DataFrame(columns=["订单ID"])
    elif "订单ID" not in book.columns:
        msg = f"get_cancels_table 输出缺少 '订单ID'，实际列={list(book.columns)}"
        raise AssertionError(msg)

    book = book[["订单ID"]].copy()
    book["订单ID"] = book["订单ID"].astype("int64")

    # 看是否存在同一单多次撤单

    real["idx"] = real.groupby("订单ID").cumcount()
    book["idx"] = book.groupby("订单ID").cumcount()

    merged = real.merge(
        book,
        on=["订单ID", "idx"],
        how="outer",
        indicator=True,
        suffixes=("_real", "_book"),
    )

    # _merge: left_only => 真实有但回测没有（漏撤）
    # _merge: right_only => 回测有但真实没有（多撤）
    bad = merged[merged["_merge"] != "both"]

    if not bad.empty:
        leak = bad[bad["_merge"] == "left_only"][["订单ID", "idx"]].head(50)
        extra = bad[bad["_merge"] == "right_only"][["订单ID", "idx"]].head(50)

        msg = (
            "撤单对比失败（outer merge 出现 NaN/不匹配）\n"
            f"- 真实有但回测没有（left_only）前50条：\n{leak}\n\n"
            f"- 回测有但真实没有（right_only）前50条：\n{extra}\n"
        )
        raise AssertionError(msg)

    return True


# 集合测试
class TestShanghaiIntegration600000:
    """End-to-end run on 600000.SH Parquet fixtures."""

    def test_full_run_compare_trade_and_cancel(
        self,
        orderbook: OrderBookSH,
        trade_df: pl.DataFrame,
        order_df: pl.DataFrame,
    ):
        """Run full workflow and assert trades and cancels match source."""
        merged_df = orderbook.prepare_market_data(
            trade_df_SH=trade_df,
            order_df_SH=order_df,
            symbol="600000",
            cut_time=CUT_TIME,
            cut_serial=CUT_SERIAL,
        )

        if merged_df is None or len(merged_df) == 0:
            pytest.skip("merged_df empty")

        if MAX_ROWS is not None and len(merged_df) > MAX_ROWS:
            merged_df = merged_df.head(MAX_ROWS).copy()

        orderbook.process_workflow(merged_df)

        ok_trade = orderbook.compare_df(merged_df)
        assert ok_trade is True, "compare_df 成交对比失败"

        # 4) 撤单对比（自定义）
        ok_cancel = compare_cancel_orders(merged_df, orderbook)
        assert ok_cancel is True, "撤单对比失败"
