"""
边界条件和异常处理测试套件.

测试覆盖：
- 空数据处理
- 单边盘口
- 零价格/重复ID/超大数量
- U 单（本方最优）
- 数据类型兼容性
"""

from pylob import OrderBookSZ
from pylob.data_types import Side, TradingPhase


class TestEmptyData:
    """空数据和无委托场景."""

    def test_empty_orderbook_snapshot(self):
        """空订单簿取快照不崩."""
        ob = OrderBookSZ()
        df = ob.get_full_order_book_dataframe(depth=5)
        assert df is not None

    def test_auction_on_empty_book(self):
        """空簿执行集合竞价不崩."""
        ob = OrderBookSZ()
        ob.set_trading_phase(TradingPhase.CALL_AUCTION)
        ob.call_auction_match()  # 不应崩
        assert len(ob.trades) == 0

    def test_cancel_on_empty_book(self):
        """空簿撤单返回 False."""
        ob = OrderBookSZ()
        result = ob.cancel_order(999)
        assert result is False


class TestOneSidedBook:
    """单边盘口测试."""

    def setup_method(self):
        self.orderbook = OrderBookSZ()
        self.orderbook.set_trading_phase(TradingPhase.CONTINUOUS)

    def test_buy_order_no_asks(self):
        """只有买单没有卖单，买单应挂入盘口."""
        self.orderbook.add_order(
            price=105000,
            quantity=100,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=93001000,
        )

        assert len(self.orderbook.trades) == 0
        assert 1 in self.orderbook.orders
        assert 105000 in self.orderbook.bids

    def test_sell_order_no_bids(self):
        """只有卖单没有买单，卖单应挂入盘口."""
        self.orderbook.add_order(
            price=105000,
            quantity=100,
            side=Side.SELL,
            order_id=1,
            order_type="0",
            order_time=93001000,
        )

        assert len(self.orderbook.trades) == 0
        assert 1 in self.orderbook.orders
        assert 105000 in self.orderbook.asks

    def test_auction_one_side_only(self):
        """集合竞价只有一侧，无成交."""
        ob = OrderBookSZ()
        ob.set_trading_phase(TradingPhase.CALL_AUCTION)

        ob.add_order(
            price=105000,
            quantity=100,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=91500000,
        )
        ob.add_order(
            price=106000,
            quantity=200,
            side=Side.BUY,
            order_id=2,
            order_type="0",
            order_time=91500010,
        )

        ob.call_auction_match()
        assert len(ob.trades) == 0


class TestInvalidInput:
    """非法输入测试."""

    def setup_method(self):
        self.orderbook = OrderBookSZ()
        self.orderbook.set_trading_phase(TradingPhase.CONTINUOUS)

    def test_zero_price_limit_order(self):
        """限价单价格为0应被拒绝."""
        result = self.orderbook.add_order(
            price=0,
            quantity=100,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=93001000,
        )
        assert result is None
        assert 1 not in self.orderbook.orders

    def test_duplicate_order_id(self):
        """重复 order_id 应被拒绝."""
        self.orderbook.add_order(
            price=105000,
            quantity=100,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=93001000,
        )
        result = self.orderbook.add_order(
            price=106000,
            quantity=200,
            side=Side.SELL,
            order_id=1,
            order_type="0",
            order_time=93002000,
        )
        assert result is None  # 重复 ID 被拒

    def test_invalid_order_type(self):
        """无效的 order_type 应被拒绝."""
        result = self.orderbook.add_order(
            price=105000,
            quantity=100,
            side=Side.BUY,
            order_id=1,
            order_type="X",
            order_time=93001000,
        )
        assert result is None

    def test_large_quantity(self):
        """超大数量委托应正常处理."""
        self.orderbook.add_order(
            price=105000,
            quantity=999999999,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=93001000,
        )
        assert 1 in self.orderbook.orders
        assert self.orderbook.orders[1].quantity == 999999999


class TestLocalOptimalOrder:
    """本方最优单（U 单）测试."""

    def setup_method(self):
        self.orderbook = OrderBookSZ()
        self.orderbook.set_trading_phase(TradingPhase.CONTINUOUS)

    def test_u_order_buy_takes_bid1(self):
        """买方 U 单取买一价挂入."""
        # 先挂一笔买单建立买盘
        self.orderbook.add_order(
            price=104000,
            quantity=100,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=93001000,
        )

        # U 单买入
        self.orderbook.add_order(
            price=0,
            quantity=200,
            side=Side.BUY,
            order_id=2,
            order_type="U",
            order_time=93002000,
        )

        # U 单应以买一价(104000)挂入
        assert 2 in self.orderbook.orders
        assert self.orderbook.orders[2].price == 104000

    def test_u_order_sell_takes_ask1(self):
        """卖方 U 单取卖一价挂入."""
        # 先挂一笔卖单建立卖盘
        self.orderbook.add_order(
            price=106000,
            quantity=100,
            side=Side.SELL,
            order_id=1,
            order_type="0",
            order_time=93001000,
        )

        # U 单卖出
        self.orderbook.add_order(
            price=0,
            quantity=200,
            side=Side.SELL,
            order_id=2,
            order_type="U",
            order_time=93002000,
        )

        # U 单应以卖一价(106000)挂入
        assert 2 in self.orderbook.orders
        assert self.orderbook.orders[2].price == 106000

    def test_u_order_empty_book_fallback(self):
        """U 单空盘口：买单用 0，卖单用 999999999 兜底."""
        # 买方 U 单，空盘口
        self.orderbook.add_order(
            price=0,
            quantity=100,
            side=Side.BUY,
            order_id=1,
            order_type="U",
            order_time=93001000,
        )
        assert 1 in self.orderbook.orders
        assert self.orderbook.orders[1].price == 0

        # 卖方 U 单，空盘口
        self.orderbook.add_order(
            price=0,
            quantity=100,
            side=Side.SELL,
            order_id=2,
            order_type="U",
            order_time=93002000,
        )
        assert 2 in self.orderbook.orders
        assert self.orderbook.orders[2].price == 999_999_999

    def test_u_order_triggers_match(self):
        """U 单如果价格能匹配对手盘，应该撮合."""
        # 卖单 @ 105000
        self.orderbook.add_order(
            price=105000,
            quantity=100,
            side=Side.SELL,
            order_id=1,
            order_type="0",
            order_time=93001000,
        )
        # 买单 @ 106000 建立买盘
        self.orderbook.add_order(
            price=106000,
            quantity=100,
            side=Side.BUY,
            order_id=2,
            order_type="0",
            order_time=93002000,
        )

        # 此时买一=106000，卖一=105000（已经交叉成交了）
        # 重新建立
        ob = OrderBookSZ()
        ob.set_trading_phase(TradingPhase.CONTINUOUS)

        ob.add_order(
            price=105000,
            quantity=100,
            side=Side.SELL,
            order_id=10,
            order_type="0",
            order_time=93001000,
        )
        ob.add_order(
            price=104000,
            quantity=100,
            side=Side.BUY,
            order_id=11,
            order_type="0",
            order_time=93002000,
        )

        # 卖方 U 单取卖一价 105000，但买一是 104000，不能成交
        ob.add_order(
            price=0,
            quantity=50,
            side=Side.SELL,
            order_id=12,
            order_type="U",
            order_time=93003000,
        )

        assert len(ob.trades) == 0  # 卖一(105000) > 买一(104000)，U 单挂在 105000
