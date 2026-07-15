"""
集合竞价撮合测试套件.

测试覆盖：
- 基本集合竞价撮合（最大成交量原则）
- 集合竞价期间撤单
- 集合竞价 → 连续竞价阶段切换
- 收盘集合竞价
- 集合竞价期间市价单拒绝
"""

from pylob import OrderBookSZ
from pylob.data_types import Side, TradingPhase


class TestCallAuctionBasic:
    """集合竞价基本撮合测试."""

    def setup_method(self):
        self.orderbook = OrderBookSZ()
        self.orderbook.set_trading_phase(TradingPhase.CALL_AUCTION)

    def test_basic_auction_match(self):
        """基本集合竞价：多笔买卖 → 最大成交量价格 → 统一价格成交."""
        # 买单
        self.orderbook.add_order(
            price=105000,
            quantity=300,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=91500000,
        )
        self.orderbook.add_order(
            price=104000,
            quantity=500,
            side=Side.BUY,
            order_id=2,
            order_type="0",
            order_time=91500010,
        )
        self.orderbook.add_order(
            price=103000,
            quantity=400,
            side=Side.BUY,
            order_id=3,
            order_type="0",
            order_time=91500020,
        )

        # 卖单
        self.orderbook.add_order(
            price=101000,
            quantity=200,
            side=Side.SELL,
            order_id=4,
            order_type="0",
            order_time=91500030,
        )
        self.orderbook.add_order(
            price=102000,
            quantity=300,
            side=Side.SELL,
            order_id=5,
            order_type="0",
            order_time=91500040,
        )
        self.orderbook.add_order(
            price=103000,
            quantity=400,
            side=Side.SELL,
            order_id=6,
            order_type="0",
            order_time=91500050,
        )

        # 集合竞价阶段不撮合，只收集
        assert len(self.orderbook.trades) == 0

        # 执行集合竞价撮合
        self.orderbook.call_auction_match()

        # 验证有成交
        assert len(self.orderbook.trades) > 0

        # 验证所有成交以同一价格
        prices = {t.price for t in self.orderbook.trades}
        assert len(prices) == 1, (
            f"集合竞价应以统一价格成交，实际有 {len(prices)} 个价格"
        )

    def test_auction_no_match(self):
        """集合竞价：买卖价格不交叉，无成交."""
        self.orderbook.add_order(
            price=100000,
            quantity=100,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=91500000,
        )
        self.orderbook.add_order(
            price=110000,
            quantity=100,
            side=Side.SELL,
            order_id=2,
            order_type="0",
            order_time=91500010,
        )

        self.orderbook.call_auction_match()
        assert len(self.orderbook.trades) == 0

    def test_auction_single_price(self):
        """集合竞价：买卖价格相同，直接成交."""
        self.orderbook.add_order(
            price=105000,
            quantity=200,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=91500000,
        )
        self.orderbook.add_order(
            price=105000,
            quantity=300,
            side=Side.SELL,
            order_id=2,
            order_type="0",
            order_time=91500010,
        )

        self.orderbook.call_auction_match()

        assert len(self.orderbook.trades) == 1
        assert self.orderbook.trades[0].price == 105000
        assert self.orderbook.trades[0].quantity == 200  # min(200, 300)

    def test_auction_partial_fill(self):
        """集合竞价：部分成交，剩余留在盘口."""
        self.orderbook.add_order(
            price=105000,
            quantity=500,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=91500000,
        )
        self.orderbook.add_order(
            price=105000,
            quantity=200,
            side=Side.SELL,
            order_id=2,
            order_type="0",
            order_time=91500010,
        )

        self.orderbook.call_auction_match()

        # 卖单完全成交
        assert 2 not in self.orderbook.orders

        # 买单剩余 300 还在盘口
        assert 1 in self.orderbook.orders
        assert self.orderbook.orders[1].quantity == 300

    def test_auction_uniform_trade_time(self):
        """集合竞价：所有成交使用统一时间（最晚委托的时间）."""
        self.orderbook.add_order(
            price=105000,
            quantity=100,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=91500000,
        )
        self.orderbook.add_order(
            price=105000,
            quantity=100,
            side=Side.BUY,
            order_id=2,
            order_type="0",
            order_time=91510000,
        )
        self.orderbook.add_order(
            price=105000,
            quantity=200,
            side=Side.SELL,
            order_id=3,
            order_type="0",
            order_time=91520000,
        )

        self.orderbook.call_auction_match()

        # 所有成交时间应相同
        trade_times = {t.trade_time for t in self.orderbook.trades}
        assert len(trade_times) == 1


class TestCallAuctionCancel:
    """集合竞价期间撤单测试."""

    def setup_method(self):
        self.orderbook = OrderBookSZ()
        self.orderbook.set_trading_phase(TradingPhase.CALL_AUCTION)

    def test_cancel_before_auction(self):
        """撤单后不参与撮合."""
        self.orderbook.add_order(
            price=105000,
            quantity=200,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=91500000,
        )
        self.orderbook.add_order(
            price=105000,
            quantity=200,
            side=Side.SELL,
            order_id=2,
            order_type="0",
            order_time=91500010,
        )

        # 撤掉买单
        self.orderbook.cancel_order(1, cancel_time=91500020)

        # 撮合应无成交
        self.orderbook.call_auction_match()
        assert len(self.orderbook.trades) == 0

    def test_cancel_one_of_multiple(self):
        """撤掉一笔，其他仍参与撮合."""
        self.orderbook.add_order(
            price=105000,
            quantity=100,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=91500000,
        )
        self.orderbook.add_order(
            price=105000,
            quantity=200,
            side=Side.BUY,
            order_id=2,
            order_type="0",
            order_time=91500010,
        )
        self.orderbook.add_order(
            price=105000,
            quantity=300,
            side=Side.SELL,
            order_id=3,
            order_type="0",
            order_time=91500020,
        )

        # 撤掉买单 1
        self.orderbook.cancel_order(1, cancel_time=91500030)

        self.orderbook.call_auction_match()

        # 只有买单 2 (200) 和卖单 3 (300) 参与，成交 200
        total_traded = sum(t.quantity for t in self.orderbook.trades)
        assert total_traded == 200


class TestCallAuctionMarketOrderReject:
    """集合竞价期间市价单拒绝测试."""

    def setup_method(self):
        self.orderbook = OrderBookSZ()
        self.orderbook.set_trading_phase(TradingPhase.CALL_AUCTION)

    def test_reject_market_order(self):
        """集合竞价期间提交市价单应被拒绝."""
        # 市价单处理需要 trade_df_with_c，初始化为空
        import pandas as pd

        self.orderbook.trade_df_with_c = pd.DataFrame()

        result = self.orderbook.add_order(
            price=0,
            quantity=100,
            side=Side.BUY,
            order_id=1,
            order_type="1",
            order_time=91500000,
        )

        assert result is None
        assert 1 not in self.orderbook.orders
        assert len(self.orderbook.bids) == 0


class TestCallAuctionPhaseSwitch:
    """集合竞价 ↔ 连续竞价阶段切换测试."""

    def setup_method(self):
        self.orderbook = OrderBookSZ()
        self.orderbook.set_trading_phase(TradingPhase.CALL_AUCTION)

    def test_switch_triggers_auction_match(self):
        """从集合竞价切到连续竞价时，应先执行集合竞价撮合."""
        # 集合竞价期间挂单
        self.orderbook.add_order(
            price=105000,
            quantity=100,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=91500000,
        )
        self.orderbook.add_order(
            price=105000,
            quantity=100,
            side=Side.SELL,
            order_id=2,
            order_type="0",
            order_time=91500010,
        )

        assert len(self.orderbook.trades) == 0  # 集合竞价不撮合

        # 切换到连续竞价（模拟 _auto_detect_trading_phase 的行为）
        self.orderbook.call_auction_match()
        self.orderbook.set_trading_phase(TradingPhase.CONTINUOUS)

        # 集合竞价撮合应已执行
        assert len(self.orderbook.trades) == 1

    def test_continuous_then_closing_auction(self):
        """连续竞价 → 收盘集合竞价：新委托不立即撮合."""
        self.orderbook.set_trading_phase(TradingPhase.CONTINUOUS)

        # 连续竞价期间挂一笔卖单
        self.orderbook.add_order(
            price=105000,
            quantity=100,
            side=Side.SELL,
            order_id=1,
            order_type="0",
            order_time=140000000,
        )

        # 切到收盘集合竞价
        self.orderbook.set_trading_phase(TradingPhase.CALL_AUCTION)

        # 再挂一笔买单（应入簿不撮合）
        self.orderbook.add_order(
            price=105000,
            quantity=100,
            side=Side.BUY,
            order_id=2,
            order_type="0",
            order_time=145800000,
        )

        assert len(self.orderbook.trades) == 0  # 收盘集合竞价不逐笔撮合

        # 收盘集合竞价撮合
        self.orderbook.call_auction_match()
        assert len(self.orderbook.trades) == 1
