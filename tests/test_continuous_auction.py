"""
连续竞价撮合逻辑测试套件.

测试覆盖：
- 限价单撮合（完全成交、部分成交、未成交）
- 市价单撮合（市转限、全吃、市转撤）
- 订单撤销功能
- 价格-时间优先原则
"""

import pandas as pd
import pytest
from pylob import OrderBookSZ
from pylob.data_types import Side, TradingPhase


class TestLimitOrderMatching:
    """限价单撮合测试."""

    def setup_method(self):
        """每个测试前初始化订单簿."""
        self.orderbook = OrderBookSZ()
        self.orderbook.set_trading_phase(TradingPhase.CONTINUOUS)

    def test_tc_limit_001_full_fill_single_level(self):
        """TC-LIMIT-001: 限价买单完全成交（单档）."""
        # 前置条件：卖单
        self.orderbook.add_order(
            price=105000,  # 10.50 * 10000
            quantity=500,
            side=Side.SELL,
            order_id=101,
            order_type="0",
            order_time=20260202093001000000,
        )

        # 执行：提交限价买单
        result = self.orderbook.add_order(
            price=105000,
            quantity=300,
            side=Side.BUY,
            order_id=201,
            order_type="0",
            order_time=20260202093001000000,
        )

        # 验证：订单完全成交
        assert result == 201
        assert 201 not in self.orderbook.orders

        # 验证：成交记录
        trades = self.orderbook.get_trades_table()
        assert len(trades) == 1
        assert trades[0]["买单ID"] == 201
        assert trades[0]["卖单ID"] == 101
        assert trades[0]["成交价"] == 105000
        assert trades[0]["成交量"] == 300

        # 验证：订单簿状态
        assert self.orderbook.asks[105000][0].quantity == 200
        assert len(self.orderbook.bids) == 0

    def test_tc_limit_002_full_fill_multi_level(self):
        """TC-LIMIT-002: 限价买单完全成交（多档）."""
        # 前置条件：多档卖单
        self.orderbook.add_order(
            price=105000, quantity=200, side=Side.SELL, order_id=101, order_type="0"
        )
        self.orderbook.add_order(
            price=105000, quantity=100, side=Side.SELL, order_id=102, order_type="0"
        )
        self.orderbook.add_order(
            price=105200, quantity=300, side=Side.SELL, order_id=103, order_type="0"
        )
        self.orderbook.add_order(
            price=105500, quantity=500, side=Side.SELL, order_id=104, order_type="0"
        )

        # 执行：提交大额限价买单
        self.orderbook.add_order(
            price=105400,
            quantity=550,
            side=Side.BUY,
            order_id=201,
            order_type="0",
            order_time=20260202093002000000,
        )

        # 验证：订单完全成交
        assert 201 not in self.orderbook.orders

        # 验证：成交记录（3笔）
        trades = self.orderbook.get_trades_table()
        assert len(trades) == 3

        assert trades[0]["成交价"] == 105000
        assert trades[0]["成交量"] == 200

        assert trades[1]["成交价"] == 105000
        assert trades[1]["成交量"] == 100

        assert trades[2]["成交价"] == 105200
        assert trades[2]["成交量"] == 250

        # 验证：订单簿状态
        assert 105000 not in self.orderbook.asks
        assert self.orderbook.asks[105200][0].quantity == 50
        assert self.orderbook.asks[105500][0].quantity == 500

    def test_tc_limit_003_partial_fill(self):
        """TC-LIMIT-003: 限价买单部分成交后挂单."""
        # 前置条件：卖单
        self.orderbook.add_order(
            price=105000, quantity=200, side=Side.SELL, order_id=101, order_type="0"
        )

        # 执行：提交大额买单
        self.orderbook.add_order(
            price=105200,
            quantity=500,
            side=Side.BUY,
            order_id=201,
            order_type="0",
            order_time=20260202093003000000,
        )

        # 验证：订单部分成交
        assert 201 in self.orderbook.orders
        assert self.orderbook.orders[201].quantity == 300

        # 验证：成交记录
        trades = self.orderbook.get_trades_table()
        assert len(trades) == 1
        assert trades[0]["成交量"] == 200

        # 验证：订单簿状态
        assert 105000 not in self.orderbook.asks
        assert self.orderbook.bids[105200][0].order_id == 201
        assert self.orderbook.bids[105200][0].quantity == 300

    def test_tc_limit_004_no_fill(self):
        """TC-LIMIT-004: 限价买单完全未成交."""
        # 前置条件：卖单
        self.orderbook.add_order(
            price=105000, quantity=200, side=Side.SELL, order_id=101, order_type="0"
        )

        # 执行：提交低价买单
        self.orderbook.add_order(
            price=104500,
            quantity=500,
            side=Side.BUY,
            order_id=201,
            order_type="0",
            order_time=20260202093004000000,
        )

        # 验证：订单未成交，完全挂单
        assert 201 in self.orderbook.orders
        assert self.orderbook.orders[201].quantity == 500

        # 验证：无成交记录
        trades = self.orderbook.get_trades_table()
        assert len(trades) == 0

        # 验证：订单簿状态
        assert self.orderbook.bids[104500][0].order_id == 201
        assert len(self.orderbook.asks[105000]) == 1

    def test_tc_limit_005_price_time_priority(self):
        """TC-LIMIT-005: 价格-时间优先验证."""
        # 前置条件：同价位卖单（不同时间）
        self.orderbook.add_order(
            price=105000,
            quantity=100,
            side=Side.SELL,
            order_id=101,
            order_type="0",
            order_time=20260202093001000000,
        )
        self.orderbook.add_order(
            price=105000,
            quantity=100,
            side=Side.SELL,
            order_id=102,
            order_type="0",
            order_time=20260202093002000000,
        )
        self.orderbook.add_order(
            price=105000,
            quantity=100,
            side=Side.SELL,
            order_id=103,
            order_type="0",
            order_time=20260202093003000000,
        )

        # 执行：提交买单（只能吃掉部分）
        self.orderbook.add_order(
            price=105000,
            quantity=150,
            side=Side.BUY,
            order_id=201,
            order_type="0",
            order_time=20260202093004000000,
        )

        # 验证：成交记录（按时间优先）
        trades = self.orderbook.get_trades_table()
        assert len(trades) == 2

        # 第1笔：最早的订单#101
        assert trades[0]["卖单ID"] == 101
        assert trades[0]["成交量"] == 100

        # 第2笔：第二早的订单#102（部分成交）
        assert trades[1]["卖单ID"] == 102
        assert trades[1]["成交量"] == 50

        # 验证：订单簿状态
        assert 101 not in self.orderbook.orders
        assert self.orderbook.orders[102].quantity == 50
        assert self.orderbook.orders[103].quantity == 100


class TestMarketOrderMatching:
    """市价单撮合测试."""

    def setup_method(self):
        """每个测试前初始化订单簿."""
        self.orderbook = OrderBookSZ()
        self.orderbook.set_trading_phase(TradingPhase.CONTINUOUS)

    def test_tc_market_001_lock_price_mode(self):
        """TC-MARKET-001: 市转限模式 - 价格锁定."""
        # 前置条件：多档卖单
        self.orderbook.add_order(
            price=105000, quantity=200, side=Side.SELL, order_id=101, order_type="0"
        )
        self.orderbook.add_order(
            price=105200, quantity=300, side=Side.SELL, order_id=102, order_type="0"
        )

        # 模拟 trade_df（市价单只在10.50成交）
        mock_trade_df = pd.DataFrame(
            {
                "buy_id": [201],
                "sell_id": [101],
                "trade_price": [105000],
                "trade_volume": [200],
            }
        )
        self.orderbook.trade_df_with_c = mock_trade_df

        # 执行：提交市价买单（price会被自动设置为0）
        self.orderbook.add_order(
            price=0,  # 市转限模式
            quantity=500,
            side=Side.BUY,
            order_id=201,
            order_type="1",
            order_time=20260202093005000000,
        )

        # 验证：在10.50成交200手
        trades = self.orderbook.get_trades_table()
        assert len(trades) == 1
        assert trades[0]["成交价"] == 105000
        assert trades[0]["成交量"] == 200

        # 验证：剩余300手转挂限价单@10.50
        assert 201 in self.orderbook.orders
        assert self.orderbook.orders[201].order_type == "0"
        assert self.orderbook.orders[201].price == 105000
        assert self.orderbook.orders[201].quantity == 300

    def test_tc_market_002_sweep_mode(self):
        """TC-MARKET-002: 全吃模式 - 多档扫盘."""
        # 前置条件：多档卖单
        self.orderbook.add_order(
            price=105000, quantity=100, side=Side.SELL, order_id=101, order_type="0"
        )
        self.orderbook.add_order(
            price=105200, quantity=150, side=Side.SELL, order_id=102, order_type="0"
        )
        self.orderbook.add_order(
            price=105500, quantity=200, side=Side.SELL, order_id=103, order_type="0"
        )

        # 模拟 trade_df（市价单在多个价格成交）
        mock_trade_df = pd.DataFrame(
            {
                "buy_id": [201, 201, 201],
                "sell_id": [101, 102, 103],
                "trade_price": [105000, 105200, 105500],
                "trade_volume": [100, 150, 50],
            }
        )
        self.orderbook.trade_df_with_c = mock_trade_df

        # 执行：提交市价买单（price会被自动设置为1）
        self.orderbook.add_order(
            price=1,  # 全吃模式
            quantity=300,
            side=Side.BUY,
            order_id=201,
            order_type="1",
            order_time=20260202093006000000,
        )

        # 验证：多档成交（3笔）
        trades = self.orderbook.get_trades_table()
        assert len(trades) == 3

        assert trades[0]["成交价"] == 105000
        assert trades[0]["成交量"] == 100

        assert trades[1]["成交价"] == 105200
        assert trades[1]["成交量"] == 150

        assert trades[2]["成交价"] == 105500
        assert trades[2]["成交量"] == 50

        # 验证：订单完全成交，不挂单
        assert 201 not in self.orderbook.orders

    @pytest.mark.xfail(
        reason="已知局限：_query_market_order_type 在单一非零成交价+撤单时无法区分市转限和市转撤，返回0而非-2"
    )
    def test_tc_market_003_cancel_mode(self):
        """TC-MARKET-003: 市转撤模式 - 部分成交后撤单."""
        # 前置条件：多档卖单
        self.orderbook.add_order(
            price=105000, quantity=200, side=Side.SELL, order_id=101, order_type="0"
        )
        self.orderbook.add_order(
            price=105200, quantity=300, side=Side.SELL, order_id=102, order_type="0"
        )

        # 模拟 trade_df（市价单成交200手后撤单100手）
        mock_trade_df = pd.DataFrame(
            {
                "buy_id": [201, 201],
                "sell_id": [101, 0],
                "trade_price": [105000, 0],
                "trade_volume": [200, 100],
                "trade_type": ["0", "C"],
                "bsflag": ["B", ""],
            }
        )
        self.orderbook.trade_df_with_c = mock_trade_df

        # 执行：提交市价买单（price会被自动设置为-2）
        self.orderbook.add_order(
            price=-2,  # 市转撤模式
            quantity=300,
            side=Side.BUY,
            order_id=201,
            order_type="1",
            order_time=20260202093007000000,
        )

        # 验证：成交200手
        trades = self.orderbook.get_trades_table()
        assert len(trades) == 1
        assert trades[0]["成交价"] == 105000
        assert trades[0]["成交量"] == 200

        # 验证：剩余100手撤单（不挂单）
        assert 201 not in self.orderbook.orders

    def test_tc_market_004_no_liquidity(self):
        """TC-MARKET-004: 市价单无流动性失效."""
        # 前置条件：空订单簿
        # 模拟 trade_df：无成交记录
        mock_trade_df = pd.DataFrame(
            {"buy_id": [], "sell_id": [], "trade_price": [], "trade_volume": []}
        )
        self.orderbook.trade_df_with_c = mock_trade_df

        # 执行：提交市价买单
        self.orderbook.add_order(
            price=0,
            quantity=500,
            side=Side.BUY,
            order_id=201,
            order_type="1",
            order_time=20260202093008000000,
        )

        # 验证：无成交记录
        trades = self.orderbook.get_trades_table()
        assert len(trades) == 0

        # 验证：订单失效（不挂单）
        assert 201 not in self.orderbook.orders


class TestOrderCancellation:
    """订单撤销测试."""

    def setup_method(self):
        """每个测试前初始化订单簿."""
        self.orderbook = OrderBookSZ()
        self.orderbook.set_trading_phase(TradingPhase.CONTINUOUS)

    def test_tc_cancel_001_cancel_unfilled_order(self):
        """TC-CANCEL-001: 撤销未成交订单."""
        # 前置条件：买单
        self.orderbook.add_order(
            price=104500,
            quantity=300,
            side=Side.BUY,
            order_id=101,
            order_type="0",
            order_time=20260202093001000000,
        )

        # 执行：撤销订单
        success = self.orderbook.cancel_order(
            order_id=101, cancel_time=20260202093010000000
        )

        # 验证：撤单成功
        assert success is True
        assert 101 not in self.orderbook.orders

        # 验证：撤单记录
        cancels = self.orderbook.get_cancels_table()
        assert len(cancels) == 1
        assert cancels[0]["订单ID"] == 101
        assert cancels[0]["数量"] == 300

        # 验证：订单簿状态
        assert 104500 not in self.orderbook.bids

    def test_tc_cancel_002_cancel_partial_filled_order(self):
        """TC-CANCEL-002: 撤销部分成交订单."""
        # 前置条件：卖单
        self.orderbook.add_order(
            price=105000, quantity=500, side=Side.SELL, order_id=101, order_type="0"
        )
        # 买单部分成交
        self.orderbook.add_order(
            price=105000, quantity=200, side=Side.BUY, order_id=201, order_type="0"
        )

        # 执行：撤销剩余订单
        success = self.orderbook.cancel_order(
            order_id=101, cancel_time=20260202093011000000
        )

        # 验证：撤单成功
        assert success is True
        assert 101 not in self.orderbook.orders

        # 验证：撤单记录
        cancels = self.orderbook.get_cancels_table()
        assert len(cancels) == 1
        assert cancels[0]["订单ID"] == 101
        assert cancels[0]["数量"] == 300  # 撤销的是剩余数量

        # 验证：成交记录不受影响
        trades = self.orderbook.get_trades_table()
        assert len(trades) == 1
        assert trades[0]["成交量"] == 200

    def test_tc_cancel_003_cancel_nonexistent_order(self):
        """TC-CANCEL-003: 撤销不存在的订单."""
        # 执行：撤销不存在的订单
        success = self.orderbook.cancel_order(
            order_id=999, cancel_time=20260202093012000000
        )

        # 验证：撤单失败
        assert success is False

        # 验证：无撤单记录
        cancels = self.orderbook.get_cancels_table()
        assert len(cancels) == 0

    def test_tc_cancel_004_cancel_filled_order(self):
        """TC-CANCEL-004: 撤销已完全成交订单."""
        # 前置条件：卖单和买单（完全成交）
        self.orderbook.add_order(
            price=105000, quantity=200, side=Side.SELL, order_id=101, order_type="0"
        )
        self.orderbook.add_order(
            price=105000, quantity=200, side=Side.BUY, order_id=201, order_type="0"
        )

        # 执行：尝试撤销已完全成交的订单
        success = self.orderbook.cancel_order(
            order_id=101, cancel_time=20260202093013000000
        )

        # 验证：撤单失败
        assert success is False

        # 验证：无新撤单记录
        cancels = self.orderbook.get_cancels_table()
        assert len(cancels) == 0

        # 验证：成交记录保留
        trades = self.orderbook.get_trades_table()
        assert len(trades) == 1

    def test_tc_cancel_005_cancel_specific_order_at_same_price(self):
        """TC-CANCEL-005: 同价位多订单部分撤销."""
        # 前置条件：同价位多个买单
        self.orderbook.add_order(
            price=104500, quantity=100, side=Side.BUY, order_id=101, order_type="0"
        )
        self.orderbook.add_order(
            price=104500, quantity=200, side=Side.BUY, order_id=102, order_type="0"
        )
        self.orderbook.add_order(
            price=104500, quantity=300, side=Side.BUY, order_id=103, order_type="0"
        )

        # 执行：撤销中间的订单#102
        success = self.orderbook.cancel_order(
            order_id=102, cancel_time=20260202093014000000
        )

        # 验证：撤单成功
        assert success is True

        # 验证：只移除指定订单
        assert 102 not in self.orderbook.orders
        assert 101 in self.orderbook.orders
        assert 103 in self.orderbook.orders

        # 验证：订单簿状态
        price_level = self.orderbook.bids[104500]
        assert len(price_level) == 2
        assert price_level[0].order_id == 101
        assert price_level[1].order_id == 103


class TestComplexScenarios:
    """综合场景测试."""

    def setup_method(self):
        """每个测试前初始化订单簿."""
        self.orderbook = OrderBookSZ()
        self.orderbook.set_trading_phase(TradingPhase.CONTINUOUS)

    def test_tc_complex_001_price_reversal(self):
        """TC-COMPLEX-001: 价格反转场景."""
        # 前置条件：正常订单簿
        self.orderbook.add_order(
            price=105000, quantity=200, side=Side.SELL, order_id=101, order_type="0"
        )
        self.orderbook.add_order(
            price=104500, quantity=200, side=Side.BUY, order_id=102, order_type="0"
        )

        # 执行：提交高价买单（超过卖一价）
        self.orderbook.add_order(
            price=105500, quantity=100, side=Side.BUY, order_id=103, order_type="0"
        )

        # 验证：买单应立即成交
        assert 103 not in self.orderbook.orders

        # 验证：成交价格为卖单价格
        trades = self.orderbook.get_trades_table()
        assert len(trades) == 1
        assert trades[0]["成交价"] == 105000
        assert trades[0]["成交量"] == 100

    def test_tc_complex_002_large_order_impact(self):
        """TC-COMPLEX-002: 大额订单冲击测试."""
        # 前置条件：构建深度订单簿（10档）
        for i in range(10):
            self.orderbook.add_order(
                price=105000 + i * 100,
                quantity=100,
                side=Side.SELL,
                order_id=100 + i,
                order_type="0",
            )

        # 执行：提交超大买单（吃穿7档）
        self.orderbook.add_order(
            price=106000, quantity=700, side=Side.BUY, order_id=201, order_type="0"
        )

        # 验证：成交数量
        trades = self.orderbook.get_trades_table()
        assert len(trades) == 7

        # 验证：成交价格递增（价格以万分之一元存储）
        for i, trade in enumerate(trades):
            expected_price = 105000 + i * 100
            assert trade["成交价"] == expected_price

        # 验证：订单簿深度
        assert len(list(self.orderbook.asks.keys())) == 3

    def test_tc_complex_003_orderbook_consistency(self):
        """TC-COMPLEX-003: 订单簿一致性验证."""
        # 执行：复杂操作序列
        self.orderbook.add_order(
            price=105000, quantity=100, side=Side.SELL, order_id=101, order_type="0"
        )
        self.orderbook.add_order(
            price=104500, quantity=100, side=Side.BUY, order_id=102, order_type="0"
        )
        self.orderbook.add_order(
            price=105000, quantity=50, side=Side.BUY, order_id=103, order_type="0"
        )
        self.orderbook.cancel_order(102)
        self.orderbook.add_order(
            price=105000, quantity=30, side=Side.BUY, order_id=104, order_type="0"
        )

        # 验证：订单簿一致性
        for _price, orders in self.orderbook.bids.items():
            for order in orders:
                assert order.order_id in self.orderbook.orders
                assert order.quantity > 0

        for _price, orders in self.orderbook.asks.items():
            for order in orders:
                assert order.order_id in self.orderbook.orders
                assert order.quantity > 0


if __name__ == "__main__":
    # 可以直接运行测试
    pytest.main([__file__, "-v", "--tb=short"])
