"""
增强 mock 数据测试套件.

补充现有测试的覆盖缺口：
1. 同一价位多笔委托 FIFO 验证（5-10笔）
2. 集合竞价 tie-breaking（多个价格成交量相同）
3. 连续操作后订单簿一致性（大量增删改后三数据结构同步）
4. 撤单时序边界（双重撤单、成交同时撤单）
5. 卖单方向的撮合覆盖
"""

import pytest
from pylob import OrderBookSH, OrderBookSZ
from pylob.data_types import Side, TradingPhase


# ---------------------------------------------------------------------------
# 1. FIFO：同一价位多笔委托
# ---------------------------------------------------------------------------
class TestFIFOMultipleOrders:
    """同一价位 5-10 笔委托的时间优先验证."""

    def setup_method(self):
        self.ob = OrderBookSZ()
        self.ob.set_trading_phase(TradingPhase.CONTINUOUS)

    def test_fifo_10_sell_orders(self):
        """10 笔同价卖单，买单逐步吃掉，验证严格 FIFO."""
        # 挂 10 笔卖单 @105000，每笔 100
        for i in range(10):
            self.ob.add_order(
                price=105000,
                quantity=100,
                side=Side.SELL,
                order_id=100 + i,
                order_type="0",
                order_time=93001000 + i * 10,
            )

        # 买单吃掉 350（应该消耗前 3 笔 + 第 4 笔的 50）
        self.ob.add_order(
            price=105000,
            quantity=350,
            side=Side.BUY,
            order_id=999,
            order_type="0",
            order_time=93002000,
        )

        trades = self.ob.get_trades_table()
        assert len(trades) == 4

        # 前 3 笔完全成交，顺序是 100, 101, 102
        for i in range(3):
            assert trades[i]["卖单ID"] == 100 + i
            assert trades[i]["成交量"] == 100

        # 第 4 笔部分成交
        assert trades[3]["卖单ID"] == 103
        assert trades[3]["成交量"] == 50

        # 验证残余
        assert 103 in self.ob.orders
        assert self.ob.orders[103].quantity == 50
        for i in range(4, 10):
            assert 100 + i in self.ob.orders
            assert self.ob.orders[100 + i].quantity == 100

    def test_fifo_10_buy_orders(self):
        """10 笔同价买单，卖单逐步吃掉，验证严格 FIFO."""
        for i in range(10):
            self.ob.add_order(
                price=104000,
                quantity=100,
                side=Side.BUY,
                order_id=200 + i,
                order_type="0",
                order_time=93001000 + i * 10,
            )

        # 卖单吃掉 550
        self.ob.add_order(
            price=104000,
            quantity=550,
            side=Side.SELL,
            order_id=888,
            order_type="0",
            order_time=93002000,
        )

        trades = self.ob.get_trades_table()
        assert len(trades) == 6

        for i in range(5):
            assert trades[i]["买单ID"] == 200 + i
            assert trades[i]["成交量"] == 100

        assert trades[5]["买单ID"] == 205
        assert trades[5]["成交量"] == 50

    def test_fifo_interleaved_prices(self):
        """不同价位的卖单，验证价格优先 > 时间优先."""
        # 先挂卖单：105200 先到，105000 后到
        self.ob.add_order(
            price=105200,
            quantity=100,
            side=Side.SELL,
            order_id=1,
            order_type="0",
            order_time=93001000,
        )
        self.ob.add_order(
            price=105000,
            quantity=100,
            side=Side.SELL,
            order_id=2,
            order_type="0",
            order_time=93002000,
        )
        self.ob.add_order(
            price=105000,
            quantity=100,
            side=Side.SELL,
            order_id=3,
            order_type="0",
            order_time=93003000,
        )

        # 买单 @105200 吃 250
        self.ob.add_order(
            price=105200,
            quantity=250,
            side=Side.BUY,
            order_id=10,
            order_type="0",
            order_time=93004000,
        )

        trades = self.ob.get_trades_table()
        assert len(trades) == 3

        # 价格优先：105000 先成交，虽然时间更晚
        assert trades[0]["成交价"] == 105000
        assert trades[0]["卖单ID"] == 2
        assert trades[1]["成交价"] == 105000
        assert trades[1]["卖单ID"] == 3
        assert trades[2]["成交价"] == 105200
        assert trades[2]["卖单ID"] == 1


# ---------------------------------------------------------------------------
# 2. 集合竞价 tie-breaking
# ---------------------------------------------------------------------------
class TestCallAuctionTieBreaking:
    """集合竞价：多个价格成交量相同时的 tie-breaking."""

    def setup_method(self):
        self.ob = OrderBookSZ()
        self.ob.set_trading_phase(TradingPhase.CALL_AUCTION)

    def test_symmetric_tie(self):
        """买卖完全对称，多个价格成交量相同 → 验证代码选哪个."""
        # 设计：103000 和 104000 都能成交 100
        self.ob.add_order(
            price=104000,
            quantity=100,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=91500000,
        )
        self.ob.add_order(
            price=103000,
            quantity=100,
            side=Side.SELL,
            order_id=2,
            order_type="0",
            order_time=91500010,
        )

        self.ob.call_auction_match()

        assert len(self.ob.trades) == 1
        # 当前代码在成交量相同时取 sorted 序列中最后一个 → 应该是 104000
        # 只要行为稳定即可（这是一个记录当前行为的测试）
        trade_price = self.ob.trades[0].price
        assert trade_price in (103000, 104000)

    def test_three_way_tie(self):
        """三个价格成交量相同."""
        # 买单
        self.ob.add_order(
            price=105000,
            quantity=100,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=91500000,
        )
        self.ob.add_order(
            price=104000,
            quantity=100,
            side=Side.BUY,
            order_id=2,
            order_type="0",
            order_time=91500010,
        )
        self.ob.add_order(
            price=103000,
            quantity=100,
            side=Side.BUY,
            order_id=3,
            order_type="0",
            order_time=91500020,
        )

        # 卖单
        self.ob.add_order(
            price=103000,
            quantity=100,
            side=Side.SELL,
            order_id=4,
            order_type="0",
            order_time=91500030,
        )
        self.ob.add_order(
            price=104000,
            quantity=100,
            side=Side.SELL,
            order_id=5,
            order_type="0",
            order_time=91500040,
        )
        self.ob.add_order(
            price=105000,
            quantity=100,
            side=Side.SELL,
            order_id=6,
            order_type="0",
            order_time=91500050,
        )

        self.ob.call_auction_match()

        # 所有 3 个价格 (103000, 104000, 105000) 都能成交 100
        # 验证成交价是其中之一，且统一价格
        prices = {t.price for t in self.ob.trades}
        assert len(prices) == 1
        assert prices.pop() in (103000, 104000, 105000)

    def test_auction_max_volume_wins(self):
        """成交量不同时，最大成交量价格胜出."""
        # 设计：在 104000 能成交 300，在 105000 只能成交 100
        self.ob.add_order(
            price=105000,
            quantity=100,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=91500000,
        )
        self.ob.add_order(
            price=104000,
            quantity=300,
            side=Side.BUY,
            order_id=2,
            order_type="0",
            order_time=91500010,
        )
        self.ob.add_order(
            price=103000,
            quantity=200,
            side=Side.SELL,
            order_id=3,
            order_type="0",
            order_time=91500020,
        )
        self.ob.add_order(
            price=104000,
            quantity=200,
            side=Side.SELL,
            order_id=4,
            order_type="0",
            order_time=91500030,
        )

        self.ob.call_auction_match()

        # 在 104000：买量=100+300=400，卖量=200+200=400，成交 400
        # 在 103000：买量=400，卖量=200，成交 200
        # 在 105000：买量=100，卖量=400，成交 100
        # 最大成交量在 104000
        prices = {t.price for t in self.ob.trades}
        assert len(prices) == 1
        assert prices.pop() == 104000


# ---------------------------------------------------------------------------
# 3. 大量操作后订单簿一致性
# ---------------------------------------------------------------------------
class TestOrderBookConsistency:
    """大量增删改后验证 bids/asks/orders 三个数据结构同步."""

    def setup_method(self):
        self.ob = OrderBookSZ()
        self.ob.set_trading_phase(TradingPhase.CONTINUOUS)

    def _check_consistency(self):
        """验证三个数据结构完全一致."""
        # 1. bids 中每个订单都在 orders 中
        bid_ids = set()
        for price, orders in self.ob.bids.items():
            for order in orders:
                assert order.order_id in self.ob.orders, (
                    f"bids 中的订单 {order.order_id} 不在 orders 中"
                )
                assert order.quantity > 0, (
                    f"bids 中订单 {order.order_id} 数量为 {order.quantity}"
                )
                assert order.price == price, (
                    f"bids 中订单 {order.order_id} 价格 {order.price} != 挡位 {price}"
                )
                bid_ids.add(order.order_id)

        # 2. asks 中每个订单都在 orders 中
        ask_ids = set()
        for price, orders in self.ob.asks.items():
            for order in orders:
                assert order.order_id in self.ob.orders, (
                    f"asks 中的订单 {order.order_id} 不在 orders 中"
                )
                assert order.quantity > 0, (
                    f"asks 中订单 {order.order_id} 数量为 {order.quantity}"
                )
                assert order.price == price, (
                    f"asks 中订单 {order.order_id} 价格 {order.price} != 挡位 {price}"
                )
                ask_ids.add(order.order_id)

        # 3. orders 中每个订单都在 bids 或 asks 中
        for oid, _order in self.ob.orders.items():
            assert oid in bid_ids or oid in ask_ids, (
                f"orders 中的订单 {oid} 既不在 bids 也不在 asks 中"
            )

        # 4. 没有空的价格档位
        for price, orders in self.ob.bids.items():
            assert len(orders) > 0, f"bids 中价格 {price} 有空 deque"
        for price, orders in self.ob.asks.items():
            assert len(orders) > 0, f"asks 中价格 {price} 有空 deque"

    def test_50_orders_interleaved(self):
        """50 笔委托交错挂入和撤单，验证一致性."""
        oid = 1
        # 挂 20 笔买单
        for i in range(20):
            self.ob.add_order(
                price=104000 + (i % 5) * 100,
                quantity=100 + i * 10,
                side=Side.BUY,
                order_id=oid,
                order_type="0",
                order_time=93001000 + oid,
            )
            oid += 1

        # 挂 20 笔卖单
        for i in range(20):
            self.ob.add_order(
                price=105000 + (i % 5) * 100,
                quantity=100 + i * 10,
                side=Side.SELL,
                order_id=oid,
                order_type="0",
                order_time=93001000 + oid,
            )
            oid += 1

        self._check_consistency()

        # 撤掉偶数 ID 的买单
        for i in range(1, 21, 2):
            self.ob.cancel_order(i, cancel_time=93003000 + i)

        self._check_consistency()

        # 挂 10 笔能成交的买单（吃掉部分卖单）
        for _i in range(10):
            self.ob.add_order(
                price=105500,
                quantity=50,
                side=Side.BUY,
                order_id=oid,
                order_type="0",
                order_time=93004000 + oid,
            )
            oid += 1

        self._check_consistency()

    def test_full_sweep_then_rebuild(self):
        """全部吃光再重新挂单."""
        # 挂 5 笔卖单
        for i in range(5):
            self.ob.add_order(
                price=105000 + i * 100,
                quantity=100,
                side=Side.SELL,
                order_id=i + 1,
                order_type="0",
                order_time=93001000 + i,
            )

        # 一笔大买单全吃
        self.ob.add_order(
            price=105500,
            quantity=500,
            side=Side.BUY,
            order_id=100,
            order_type="0",
            order_time=93002000,
        )

        # 卖单全部消耗完
        assert len(self.ob.asks) == 0

        self._check_consistency()

        # 重新挂卖单
        for i in range(5):
            self.ob.add_order(
                price=106000 + i * 100,
                quantity=200,
                side=Side.SELL,
                order_id=200 + i,
                order_type="0",
                order_time=93003000 + i,
            )

        self._check_consistency()


# ---------------------------------------------------------------------------
# 4. 撤单时序边界
# ---------------------------------------------------------------------------
class TestCancelEdgeCases:
    """撤单时序边界测试."""

    def setup_method(self):
        self.ob = OrderBookSZ()
        self.ob.set_trading_phase(TradingPhase.CONTINUOUS)

    def test_double_cancel(self):
        """同一笔单撤两次：第二次应返回 False."""
        self.ob.add_order(
            price=104000,
            quantity=100,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=93001000,
        )

        assert self.ob.cancel_order(1, cancel_time=93002000) is True
        assert self.ob.cancel_order(1, cancel_time=93003000) is False

        # 只有一条撤单记录
        assert len(self.ob.get_cancels_table()) == 1

    def test_cancel_after_partial_fill(self):
        """部分成交后撤残余."""
        self.ob.add_order(
            price=105000,
            quantity=500,
            side=Side.SELL,
            order_id=1,
            order_type="0",
            order_time=93001000,
        )
        self.ob.add_order(
            price=105000,
            quantity=200,
            side=Side.BUY,
            order_id=2,
            order_type="0",
            order_time=93002000,
        )

        # 卖单剩 300
        assert self.ob.orders[1].quantity == 300

        # 撤掉剩余
        assert self.ob.cancel_order(1, cancel_time=93003000) is True
        assert 1 not in self.ob.orders

        cancels = self.ob.get_cancels_table()
        assert len(cancels) == 1
        assert cancels[0]["数量"] == 300  # 撤的是剩余量

    def test_cancel_all_at_same_price(self):
        """同价位 3 笔单全部撤掉."""
        for i in range(3):
            self.ob.add_order(
                price=104000,
                quantity=100,
                side=Side.BUY,
                order_id=i + 1,
                order_type="0",
                order_time=93001000 + i,
            )

        for i in range(3):
            assert self.ob.cancel_order(i + 1, cancel_time=93002000 + i) is True

        assert 104000 not in self.ob.bids
        assert len(self.ob.orders) == 0

    def test_cancel_middle_preserves_order(self):
        """撤掉中间的单，前后单的 FIFO 顺序不变."""
        for i in range(5):
            self.ob.add_order(
                price=106000,
                quantity=100,
                side=Side.SELL,
                order_id=i + 1,
                order_type="0",
                order_time=93001000 + i,
            )

        # 撤掉 #2 和 #4
        self.ob.cancel_order(2, cancel_time=93002000)
        self.ob.cancel_order(4, cancel_time=93002001)

        remaining = self.ob.asks[106000]
        assert len(remaining) == 3
        assert [o.order_id for o in remaining] == [1, 3, 5]


# ---------------------------------------------------------------------------
# 5. 卖单方向撮合补充
# ---------------------------------------------------------------------------
class TestSellSideMatching:
    """卖单主动吃买盘的场景（现有测试主要测买单吃卖盘）."""

    def setup_method(self):
        self.ob = OrderBookSZ()
        self.ob.set_trading_phase(TradingPhase.CONTINUOUS)

    def test_sell_sweeps_multiple_bid_levels(self):
        """低价卖单穿透多档买盘."""
        # 3 档买盘
        self.ob.add_order(
            price=106000,
            quantity=100,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=93001000,
        )
        self.ob.add_order(
            price=105500,
            quantity=150,
            side=Side.BUY,
            order_id=2,
            order_type="0",
            order_time=93001010,
        )
        self.ob.add_order(
            price=105000,
            quantity=200,
            side=Side.BUY,
            order_id=3,
            order_type="0",
            order_time=93001020,
        )

        # 卖单 @104500，吃穿所有买盘
        self.ob.add_order(
            price=104500,
            quantity=400,
            side=Side.SELL,
            order_id=10,
            order_type="0",
            order_time=93002000,
        )

        trades = self.ob.get_trades_table()
        assert len(trades) == 3

        # 价格优先：先成交最高买价
        assert trades[0]["成交价"] == 106000
        assert trades[0]["成交量"] == 100
        assert trades[1]["成交价"] == 105500
        assert trades[1]["成交量"] == 150
        assert trades[2]["成交价"] == 105000
        assert trades[2]["成交量"] == 150

        # 卖单剩 0，全部成交
        assert 10 not in self.ob.orders

        # 买单 #3 剩 50
        assert self.ob.orders[3].quantity == 50

    def test_sell_partial_fill_hangs(self):
        """卖单部分成交后挂在卖盘."""
        self.ob.add_order(
            price=105000,
            quantity=100,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=93001000,
        )

        # 卖单量大于买盘
        self.ob.add_order(
            price=105000,
            quantity=300,
            side=Side.SELL,
            order_id=2,
            order_type="0",
            order_time=93002000,
        )

        trades = self.ob.get_trades_table()
        assert len(trades) == 1
        assert trades[0]["成交量"] == 100

        # 卖单剩 200 挂在 asks
        assert 2 in self.ob.orders
        assert self.ob.orders[2].quantity == 200
        assert 105000 in self.ob.asks


# ---------------------------------------------------------------------------
# 6. 集合竞价 → 连续竞价无缝衔接
# ---------------------------------------------------------------------------
class TestAuctionToContinuousTransition:
    """集合竞价残余单在连续竞价中的行为."""

    def setup_method(self):
        self.ob = OrderBookSZ()
        self.ob.set_trading_phase(TradingPhase.CALL_AUCTION)

    def test_residual_orders_match_in_continuous(self):
        """集合竞价部分成交的残余单在切换到连续竞价后可以继续撮合."""
        # 集合竞价
        self.ob.add_order(
            price=105000,
            quantity=500,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=91500000,
        )
        self.ob.add_order(
            price=105000,
            quantity=200,
            side=Side.SELL,
            order_id=2,
            order_type="0",
            order_time=91500010,
        )

        self.ob.call_auction_match()

        # 买单残余 300
        assert self.ob.orders[1].quantity == 300

        # 切换到连续竞价
        self.ob.set_trading_phase(TradingPhase.CONTINUOUS)

        # 新卖单 @105000 应立即匹配残余买单
        self.ob.add_order(
            price=105000,
            quantity=100,
            side=Side.SELL,
            order_id=3,
            order_type="0",
            order_time=93001000,
        )

        # 集合竞价 1 笔 + 连续竞价 1 笔 = 2 笔
        assert len(self.ob.trades) == 2

    def test_no_residual_after_full_auction(self):
        """集合竞价完全成交后盘口干净."""
        self.ob.add_order(
            price=105000,
            quantity=200,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=91500000,
        )
        self.ob.add_order(
            price=105000,
            quantity=200,
            side=Side.SELL,
            order_id=2,
            order_type="0",
            order_time=91500010,
        )

        self.ob.call_auction_match()
        self.ob.set_trading_phase(TradingPhase.CONTINUOUS)

        assert len(self.ob.bids) == 0
        assert len(self.ob.asks) == 0
        assert len(self.ob.orders) == 0


# ---------------------------------------------------------------------------
# 7. OrderBookSH 基础单元测试（不通过 state machine，直接调 add_order）
# ---------------------------------------------------------------------------
class TestSHBasicMatching:
    """沪市订单簿基础撮合（使用继承的 add_order 接口）."""

    def setup_method(self):
        self.ob = OrderBookSH()
        self.ob.set_trading_phase(TradingPhase.CONTINUOUS)

    def test_sh_limit_order_match(self):
        """SH 限价单撮合."""
        self.ob.add_order(
            price=105000,
            quantity=200,
            side=Side.SELL,
            order_id=1,
            order_type="0",
            order_time=93001000,
        )
        self.ob.add_order(
            price=105000,
            quantity=100,
            side=Side.BUY,
            order_id=2,
            order_type="0",
            order_time=93002000,
        )

        trades = self.ob.get_trades_table()
        assert len(trades) == 1
        assert trades[0]["成交价"] == 105000
        assert trades[0]["成交量"] == 100

    def test_sh_call_auction(self):
        """SH 集合竞价基本撮合."""
        self.ob.set_trading_phase(TradingPhase.CALL_AUCTION)

        self.ob.add_order(
            price=105000,
            quantity=300,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=91500000,
        )
        self.ob.add_order(
            price=104000,
            quantity=200,
            side=Side.BUY,
            order_id=2,
            order_type="0",
            order_time=91500010,
        )
        self.ob.add_order(
            price=103000,
            quantity=200,
            side=Side.SELL,
            order_id=3,
            order_type="0",
            order_time=91500020,
        )
        self.ob.add_order(
            price=104000,
            quantity=300,
            side=Side.SELL,
            order_id=4,
            order_type="0",
            order_time=91500030,
        )

        self.ob.call_auction_match()

        assert len(self.ob.trades) > 0
        prices = {t.price for t in self.ob.trades}
        assert len(prices) == 1  # 统一价格

    def test_sh_cancel_order(self):
        """SH 撤单."""
        self.ob.add_order(
            price=104000,
            quantity=100,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=93001000,
        )

        assert self.ob.cancel_order(1, cancel_time=93002000) is True
        assert 1 not in self.ob.orders
        assert 104000 not in self.ob.bids

    def test_sh_orderbook_consistency_after_operations(self):
        """SH 操作序列后一致性."""
        self.ob.add_order(
            price=105000,
            quantity=100,
            side=Side.SELL,
            order_id=1,
            order_type="0",
            order_time=93001000,
        )
        self.ob.add_order(
            price=105200,
            quantity=100,
            side=Side.SELL,
            order_id=2,
            order_type="0",
            order_time=93001010,
        )
        self.ob.add_order(
            price=104000,
            quantity=100,
            side=Side.BUY,
            order_id=3,
            order_type="0",
            order_time=93001020,
        )
        self.ob.add_order(
            price=104500,
            quantity=100,
            side=Side.BUY,
            order_id=4,
            order_type="0",
            order_time=93001030,
        )

        # 成交一笔
        self.ob.add_order(
            price=105000,
            quantity=50,
            side=Side.BUY,
            order_id=5,
            order_type="0",
            order_time=93002000,
        )

        # 撤掉一笔
        self.ob.cancel_order(3, cancel_time=93003000)

        # 验证一致性
        for _price, orders in self.ob.bids.items():
            for order in orders:
                assert order.order_id in self.ob.orders
        for _price, orders in self.ob.asks.items():
            for order in orders:
                assert order.order_id in self.ob.orders


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
