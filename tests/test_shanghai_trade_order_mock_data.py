"""上海交易所订单处理逻辑测试."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path.cwd()))
from pylob.data_types import TradingPhase
from pylob.orderbook_builder_sh import OrderBookSH

# ========== Fixtures ==========


@pytest.fixture
def sh_orderbook():
    """Return ``OrderBookSH`` in continuous phase with test ``column_indices``."""
    ob = OrderBookSH()

    # 默认从 CALL_AUCTION 开始
    ob.set_trading_phase(TradingPhase.CONTINUOUS)
    ob.first_continuous = True
    # key 与 OrderBookSH.prepare_market_data 输出一致
    ob.column_indices = {
        "int_time": 0,
        "trade_type": 1,
        "order_type": 2,
        "orderorino": 3,
        "sell_id": 4,
        "buy_id": 5,
        "bsflag": 6,
        "trade_price": 7,
        "order_price": 8,
        "trade_volume": 9,
        "order_volume": 10,
        "type": 11,
    }
    ob.reset_market_state()

    return ob


# ========== Helpers ==========

DATA_LEN = 20


def create_trade_record(
    ob: OrderBookSH,
    trade_id: int,
    price: int,
    volume: int,
    side: str = "B",
    time: int = 93001000,
    trade_type: str = "0",
):
    """
    造一条 Trade record（np.ndarray），字段按新 column_indices.

    上海逻辑里 trade_id = max(sell_id, buy_id).
    """
    r = np.zeros(DATA_LEN, dtype=object)
    idx = ob.column_indices
    r[idx["type"]] = "T"
    r[idx["int_time"]] = int(time)
    r[idx["trade_type"]] = trade_type
    r[idx["bsflag"]] = side
    r[idx["trade_price"]] = int(price)
    r[idx["trade_volume"]] = int(volume)
    r[idx["buy_id"]] = trade_id if side == "B" else 0
    r[idx["sell_id"]] = trade_id if side == "S" else 0
    r[idx["orderorino"]] = 0
    r[idx["order_type"]] = "O"
    r[idx["order_price"]] = 0
    r[idx["order_volume"]] = 0
    return r


def create_order_record(
    ob: OrderBookSH,
    order_id: int,
    price: int,
    volume: int,
    side: str = "B",
    order_type: str = "O",
    time: int = 93002000,
):
    """造一条 Order record（np.ndarray），字段按新 column_indices."""
    r = np.zeros(DATA_LEN, dtype=object)
    idx = ob.column_indices

    r[idx["type"]] = "O"
    r[idx["int_time"]] = int(time)

    r[idx["orderorino"]] = int(order_id)
    r[idx["order_type"]] = order_type
    r[idx["bsflag"]] = side
    r[idx["order_price"]] = int(price)
    r[idx["order_volume"]] = int(volume)

    # trade 字段默认值
    r[idx["trade_type"]] = "0"
    r[idx["trade_price"]] = 0
    r[idx["trade_volume"]] = 0
    r[idx["buy_id"]] = 0
    r[idx["sell_id"]] = 0

    return r


def process_one(ob: OrderBookSH, record: np.ndarray):
    """Feed a single market row through ``ob``."""
    ob.row_data = record
    ob.process_single_market_record(record)


# ========== Tests ==========


class TestShanghaiTradeOrderProcessingNew:
    """Shanghai mock tests for trade accumulation and order interaction."""

    def test_tc_sh_001_single_trade_accumulation(self, sh_orderbook):
        """Single trade row should populate ``market_trades`` and status."""
        trade = create_trade_record(
            sh_orderbook, trade_id=101, price=105000, volume=100, side="B"
        )
        process_one(sh_orderbook, trade)

        assert 101 in sh_orderbook.market_trades
        assert sh_orderbook.current_trade_id == 101
        assert sh_orderbook.data_status == "T"

        saved = sh_orderbook.market_trades[101]
        idx = sh_orderbook.column_indices
        assert saved[idx["trade_price"]] == 105000
        assert saved[idx["trade_volume"]] == 100

    def test_tc_sh_002_multiple_trades_same_id(self, sh_orderbook):
        """Multiple buys same ID: volume sums; buy side keeps max price."""
        t1 = create_trade_record(sh_orderbook, 101, 105000, 50, "B")
        t2 = create_trade_record(sh_orderbook, 101, 105100, 30, "B")
        t3 = create_trade_record(sh_orderbook, 101, 104900, 20, "B")

        process_one(sh_orderbook, t1)
        process_one(sh_orderbook, t2)
        process_one(sh_orderbook, t3)

        idx = sh_orderbook.column_indices
        saved = sh_orderbook.market_trades[101]
        assert saved[idx["trade_volume"]] == 100
        assert saved[idx["trade_price"]] == 105100  # 买单取最高价

    def test_tc_sh_003_multiple_trades_sell_side_price(self, sh_orderbook):
        """Multiple sells same ID: volume sums; sell side keeps min price."""
        t1 = create_trade_record(sh_orderbook, 201, 105000, 50, "S")
        t2 = create_trade_record(sh_orderbook, 201, 105200, 30, "S")
        t3 = create_trade_record(sh_orderbook, 201, 104800, 20, "S")

        process_one(sh_orderbook, t1)
        process_one(sh_orderbook, t2)
        process_one(sh_orderbook, t3)

        idx = sh_orderbook.column_indices
        saved = sh_orderbook.market_trades[201]
        assert saved[idx["trade_volume"]] == 100
        assert saved[idx["trade_price"]] == 104800  # 卖单取最低价

    def test_tc_sh_004_trade_then_order_full_fill(self, sh_orderbook):
        """Trade then different order ID finalizes trade into resting order."""
        # trade -> order(不同ID) => finalize trade 为 order + add 新订单
        trade = create_trade_record(sh_orderbook, 101, 105000, 100, "B")
        process_one(sh_orderbook, trade)

        order = create_order_record(sh_orderbook, 102, 105000, 50, "B")
        process_one(sh_orderbook, order)

        assert 101 in sh_orderbook.orders
        assert sh_orderbook.orders[101].quantity == 100
        assert 102 in sh_orderbook.orders

    def test_tc_sh_005_trade_then_same_order_partial_fill(self, sh_orderbook):
        """Same ID partial: trade volume adds order remainder."""
        trade = create_trade_record(sh_orderbook, 101, 105000, 100, "B")
        process_one(sh_orderbook, trade)

        # order_volume 表示剩余量：部分成交 => trade_volume += order_volume
        order = create_order_record(sh_orderbook, 101, 105000, 100, "B")
        process_one(sh_orderbook, order)

        idx = sh_orderbook.column_indices
        saved_trade = sh_orderbook.market_trades[101]
        assert saved_trade[idx["trade_volume"]] == 200  # 100(trade) + 100(剩余)

    def test_tc_sh_006_sequential_different_trades(self, sh_orderbook):
        """New trade ID finalizes previous pending trade into book."""
        t1 = create_trade_record(sh_orderbook, 101, 105000, 100, "B")
        process_one(sh_orderbook, t1)

        # 不同 trade_id 到来，会 finalize 上一笔 pending trade
        t2 = create_trade_record(sh_orderbook, 102, 105100, 150, "B")
        process_one(sh_orderbook, t2)

        assert 101 in sh_orderbook.orders
        assert 102 in sh_orderbook.market_trades
        assert sh_orderbook.current_trade_id == 102

    def test_tc_sh_007_order_cancellation(self, sh_orderbook):
        """Order then cancel removes resting order and records cancel."""
        order1 = create_order_record(
            sh_orderbook, 101, 105000, 100, "B", order_type="O"
        )
        process_one(sh_orderbook, order1)
        assert 101 in sh_orderbook.orders

        cancel = create_order_record(
            sh_orderbook, 101, 105000, 100, "B", order_type="D"
        )
        process_one(sh_orderbook, cancel)

        assert 101 not in sh_orderbook.orders
        cancels = sh_orderbook.get_cancels_table()
        assert len(cancels) == 1
        assert cancels[0]["订单ID"] == 101

    def test_tc_sh_008_state_machine_transitions(self, sh_orderbook):
        """``data_status`` toggles O/T across trade and order rows."""
        assert sh_orderbook.data_status == "O"

        trade = create_trade_record(sh_orderbook, 101, 105000, 100, "B")
        process_one(sh_orderbook, trade)
        assert sh_orderbook.data_status == "T"

        order = create_order_record(sh_orderbook, 102, 105000, 50, "B")
        process_one(sh_orderbook, order)
        assert sh_orderbook.data_status == "O"

    def test_tc_sh_009_first_continuous_flag_new_behavior(self, sh_orderbook):
        """
        新框架：连续竞价第一条数据处理完，会自动 first_continuous=False.

        所以第二条 trade（不同ID）会触发 finalize 第一条 pending trade.
        """
        sh_orderbook.first_continuous = True

        t1 = create_trade_record(sh_orderbook, 101, 105000, 100, "B")
        process_one(sh_orderbook, t1)
        # 关键：process_single_market_record 会自动把 first_continuous 置 False

        t2 = create_trade_record(sh_orderbook, 102, 105100, 150, "B")
        process_one(sh_orderbook, t2)

        assert 101 in sh_orderbook.orders  # ✅ 新行为：会被 finalize

    def test_tc_sh_010_complex_trade_order_sequence(self, sh_orderbook):
        """Multi-trade accumulate then order finalizes with correct quantities."""
        # dummy trade：触发 first_continuous 自动关闭
        dummy = create_trade_record(sh_orderbook, 100, 105000, 100, "B")
        process_one(sh_orderbook, dummy)

        # 101 多笔 trade 累积
        t1 = create_trade_record(sh_orderbook, 101, 105000, 50, "B")
        t2 = create_trade_record(sh_orderbook, 101, 105100, 50, "B")
        process_one(sh_orderbook, t1)
        process_one(sh_orderbook, t2)

        # order 102 到来 => finalize 101 + add 102
        o1 = create_order_record(sh_orderbook, 102, 105000, 100, "B")
        process_one(sh_orderbook, o1)

        assert 101 in sh_orderbook.orders
        assert sh_orderbook.orders[101].quantity == 100
        assert sh_orderbook.orders[101].price == 105100
        assert 102 in sh_orderbook.orders


class TestShanghaiEdgeCasesNew:
    """Edge cases for Shanghai mock trade/order handling."""

    def test_tc_sh_edge_001_zero_volume_trade(self, sh_orderbook):
        """Zero-volume trade still registers under trade ID."""
        t = create_trade_record(sh_orderbook, 101, 105000, 0, "B")
        process_one(sh_orderbook, t)
        assert 101 in sh_orderbook.market_trades

    def test_tc_sh_edge_002_same_trade_order_id_partial_fill_total_quantity(
        self, sh_orderbook
    ):
        """
        Trade(50) + Order剩余(50) => 总报单量=100.

        新框架 partial handler：trade_volume += order_volume，
        再 add_order(quantity=总量).
        """
        t = create_trade_record(sh_orderbook, 101, 105000, 50, "B")
        process_one(sh_orderbook, t)

        o = create_order_record(sh_orderbook, 101, 105000, 50, "B", order_type="O")
        process_one(sh_orderbook, o)

        assert 101 in sh_orderbook.orders
        assert sh_orderbook.orders[101].quantity == 100
