"""深圳交易所订单处理逻辑测试（Mock Data）."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path.cwd()))

from pylob.data_types import Side, TradingPhase
from pylob.orderbook_builder_sz import OrderBookSZ

# ========== Fixtures ==========


@pytest.fixture
def sz_orderbook():
    """Return ``OrderBookSZ`` in continuous phase with test ``column_indices``."""
    ob = OrderBookSZ()
    ob.set_trading_phase(TradingPhase.CONTINUOUS)
    ob.first_continuous = True

    ob.column_indices = {
        "int_time": 0,
        "exchange_time": 1,
        "serial": 2,
        "type": 3,
        "trade_type": 4,
        "order_type": 5,
        "orderorino": 6,
        "sell_id": 7,
        "buy_id": 8,
        "bsflag": 9,
        "trade_price": 10,
        "order_price": 11,
        "trade_volume": 12,
        "order_volume": 13,
    }
    ob.process_num = 0
    ob.cont_count = 0
    return ob


# ========== Helpers ==========

DATA_LEN = 20


def create_order_record(
    ob: OrderBookSZ,
    order_id: int,
    price: int,
    volume: int,
    side: str = "B",
    order_type: str = "0",
    time: int = 93001000,
):
    """构造一条 O 记录（np.ndarray）."""
    r = np.zeros(DATA_LEN, dtype=object)
    idx = ob.column_indices

    r[idx["int_time"]] = int(time)
    r[idx["exchange_time"]] = int(time)
    r[idx["serial"]] = 1
    r[idx["type"]] = "O"

    r[idx["trade_type"]] = "0"
    r[idx["order_type"]] = order_type
    r[idx["orderorino"]] = int(order_id)
    r[idx["sell_id"]] = 0
    r[idx["buy_id"]] = 0
    r[idx["bsflag"]] = side
    r[idx["trade_price"]] = 0
    r[idx["order_price"]] = int(price)
    r[idx["trade_volume"]] = 0
    r[idx["order_volume"]] = int(volume)

    return r


def create_trade_record(
    ob: OrderBookSZ,
    buy_id: int,
    sell_id: int,
    price: int = 0,
    volume: int = 0,
    side: str = "B",
    trade_type: str = "0",
    time: int = 93002000,
):
    """构造一条 T 记录（np.ndarray）."""
    r = np.zeros(DATA_LEN, dtype=object)
    idx = ob.column_indices

    r[idx["int_time"]] = int(time)
    r[idx["exchange_time"]] = int(time)
    r[idx["serial"]] = 2
    r[idx["type"]] = "T"

    r[idx["trade_type"]] = trade_type
    r[idx["order_type"]] = "0"
    r[idx["orderorino"]] = 0
    r[idx["sell_id"]] = int(sell_id)
    r[idx["buy_id"]] = int(buy_id)
    r[idx["bsflag"]] = side
    r[idx["trade_price"]] = int(price)
    r[idx["order_price"]] = 0
    r[idx["trade_volume"]] = int(volume)
    r[idx["order_volume"]] = 0

    return r


def process_one(ob: OrderBookSZ, record: np.ndarray):
    """Feed a single market row through ``ob`` and bump process counter."""
    ob.row_data = record
    ob.process_single_market_record(record)
    ob.process_num += 1


# ========== Tests ==========


class TestShenzhenOrderRecordProcessing:
    """Shenzhen mock tests for order rows and phase transitions."""

    def test_tc_sz_001_order_record_add_buy_order(self, sz_orderbook):
        """Add buy limit order lands on bid side."""
        order = create_order_record(
            sz_orderbook,
            order_id=101,
            price=105000,
            volume=100,
            side="B",
            order_type="0",
        )
        process_one(sz_orderbook, order)

        assert 101 in sz_orderbook.orders
        assert sz_orderbook.orders[101].side == Side.BUY
        assert sz_orderbook.orders[101].quantity == 100
        assert 105000 in sz_orderbook.bids

    def test_tc_sz_002_trade_cancel_record_cancels_order(self, sz_orderbook):
        """Cancel trade row removes resting sell order."""
        order = create_order_record(
            sz_orderbook,
            order_id=202,
            price=106000,
            volume=80,
            side="S",
            order_type="0",
        )
        process_one(sz_orderbook, order)
        assert 202 in sz_orderbook.orders

        cancel_trade = create_trade_record(
            sz_orderbook,
            buy_id=0,
            sell_id=202,  # max(sell_id, buy_id) -> 202
            trade_type="C",
            side="S",
            time=93003000,
        )
        process_one(sz_orderbook, cancel_trade)

        assert 202 not in sz_orderbook.orders
        cancels = sz_orderbook.get_cancels_table()
        assert len(cancels) == 1
        assert cancels[0]["订单ID"] == 202

    def test_tc_sz_003_phase_switch_call_to_continuous_and_back(self, sz_orderbook):
        """Time-based rows switch call auction to continuous and back."""
        sz_orderbook.set_trading_phase(TradingPhase.CALL_AUCTION)

        # 09:30 后进入连续竞价
        r1 = create_order_record(
            sz_orderbook,
            order_id=301,
            price=100000,
            volume=10,
            side="B",
            time=93000001,
        )
        process_one(sz_orderbook, r1)
        assert sz_orderbook.trading_phase == TradingPhase.CONTINUOUS

        # 14:57:00 及以后进入收盘集合竞价
        r2 = create_order_record(
            sz_orderbook,
            order_id=302,
            price=100100,
            volume=10,
            side="S",
            time=145700000,
        )
        process_one(sz_orderbook, r2)
        assert sz_orderbook.trading_phase == TradingPhase.CALL_AUCTION


class TestShenzhenQueryMarketOrderType:
    """Tests for ``_query_market_order_type`` with synthetic ``trade_df_with_c``."""

    def test_tc_sz_query_001_trade_df_none(self, sz_orderbook):
        """Missing trade frame returns (-1, None)."""
        sz_orderbook.trade_df_with_c = None

        market_type, left_quantity = sz_orderbook._query_market_order_type(
            9999, Side.BUY
        )

        assert market_type == -1
        assert left_quantity is None

    def test_tc_sz_query_002_keep_cancel_when_buy_normal_side_mismatch(
        self, sz_orderbook
    ):
        """
        验证 BUY 下正常成交 bsflag 全不匹配时仍保留撤单 C.

        存在撤单 C 记录时，撤单必须保留.
        """
        order_id = 1001
        sz_orderbook.trade_df_with_c = pd.DataFrame(
            [
                {
                    "buy_id": order_id,
                    "sell_id": 9001,
                    "trade_type": "0",
                    "bsflag": "S",  # 不匹配 BUY 预期 B
                    "trade_price": 10000,
                    "trade_volume": 50,
                },
                {
                    "buy_id": order_id,
                    "sell_id": 9002,
                    "trade_type": "C",
                    "bsflag": "S",  # 撤单无条件保留
                    "trade_price": 0,
                    "trade_volume": 30,
                },
            ]
        )

        market_type, left_quantity = sz_orderbook._query_market_order_type(
            order_id, Side.BUY
        )

        assert market_type == -2
        assert left_quantity == 30

    def test_tc_sz_query_003_return_minus1_when_no_valid_normal_and_no_cancel(
        self, sz_orderbook
    ):
        """验证正常成交全部方向不匹配且没有撤单，返回 -1."""
        order_id = 1002
        sz_orderbook.trade_df_with_c = pd.DataFrame(
            [
                {
                    "buy_id": order_id,
                    "sell_id": 9101,
                    "trade_type": "0",
                    "bsflag": "S",  # 不匹配 BUY 预期 B
                    "trade_price": 10100,
                    "trade_volume": 40,
                }
            ]
        )

        market_type, left_quantity = sz_orderbook._query_market_order_type(
            order_id, Side.BUY
        )

        assert market_type == -1
        assert left_quantity is None

    def test_tc_sz_query_004_filter_out_mismatched_normal_trades(self, sz_orderbook):
        """验证仅匹配方向的正常成交应参与 unique price 判定."""
        order_id = 1003
        sz_orderbook.trade_df_with_c = pd.DataFrame(
            [
                {
                    "buy_id": order_id,
                    "sell_id": 9201,
                    "trade_type": "0",
                    "bsflag": "B",  # 匹配 BUY
                    "trade_price": 10200,
                    "trade_volume": 60,
                },
                {
                    "buy_id": order_id,
                    "sell_id": 9202,
                    "trade_type": "0",
                    "bsflag": "S",  # 不匹配，必须被过滤
                    "trade_price": 10300,
                    "trade_volume": 70,
                },
            ]
        )

        market_type, left_quantity = sz_orderbook._query_market_order_type(
            order_id, Side.BUY
        )

        assert market_type == 0
        assert left_quantity is None

    def test_tc_sz_query_005_keep_cancel_when_sell_normal_side_mismatch(
        self, sz_orderbook
    ):
        """
        验证 SELL 分支下 expected_flag='S' 时撤单保留逻辑同样适用.

        正常成交不匹配 + 存在撤单 -> 撤单保留.
        """
        order_id = 2001
        sz_orderbook.trade_df_with_c = pd.DataFrame(
            [
                {
                    "buy_id": 9301,
                    "sell_id": order_id,
                    "trade_type": "0",
                    "bsflag": "B",  # 不匹配 SELL 预期 S
                    "trade_price": 9800,
                    "trade_volume": 10,
                },
                {
                    "buy_id": 9302,
                    "sell_id": order_id,
                    "trade_type": "C",
                    "bsflag": "B",  # 撤单无条件保留
                    "trade_price": 0,
                    "trade_volume": 88,
                },
            ]
        )

        market_type, left_quantity = sz_orderbook._query_market_order_type(
            order_id, Side.SELL
        )

        assert market_type == -2
        assert left_quantity == 88
