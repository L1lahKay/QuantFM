"""
上海交易所订单处理逻辑测试套件.

测试重点：
- _process_trade_record: Trade记录处理和累积
- _process_order_record: Order记录处理和完全/部分成交判断
- Trade-Order关联逻辑
- 状态机转换
"""

import numpy as np
import pytest
from pylob import OrderBookSH
from pylob.data_types import TradingPhase


class TestShanghaiTradeOrderProcessing:
    """上海交易所 Trade-Order 处理逻辑测试."""

    def setup_method(self):
        """每个测试前初始化订单簿."""
        self.orderbook = OrderBookSH()
        self.orderbook.set_trading_phase(TradingPhase.CONTINUOUS)

        # 初始化first_continuous标志（模拟连续竞价刚开始）
        self.orderbook.first_continuous = True

        # 设置列索引（与实际数据结构对应）
        self.orderbook.column_indices = {
            "int_time": 8,
            "trade_type": 11,
            "order_type": 19,
            "orderorino": 22,
            "sell_id": 15,
            "buy_id": 16,
            "bsflag": 10,
            "trade_price": 9,
            "order_price": 23,
            "trade_volume": 13,
            "order_volume": 21,
            "type": 18,
            "extime": 4,
            "localtime": 3,
        }

    def _create_trade_record(
        self, trade_id, price, volume, side="B", time=20260202093001000000
    ):
        """创建Trade记录（numpy数组格式）."""
        record = np.zeros(25, dtype=object)
        record[self.orderbook.column_indices["type"]] = "T"
        record[self.orderbook.column_indices["buy_id"]] = trade_id if side == "B" else 0
        record[self.orderbook.column_indices["sell_id"]] = (
            trade_id if side == "S" else 0
        )
        record[self.orderbook.column_indices["trade_price"]] = price
        record[self.orderbook.column_indices["trade_volume"]] = volume
        record[self.orderbook.column_indices["bsflag"]] = side
        record[self.orderbook.column_indices["extime"]] = time
        record[self.orderbook.column_indices["int_time"]] = 93001000
        return record

    def _create_order_record(
        self,
        order_id,
        price,
        volume,
        side="B",
        order_type="O",
        time=20260202093002000000,
    ):
        """创建Order记录（numpy数组格式）."""
        record = np.zeros(25, dtype=object)
        record[self.orderbook.column_indices["type"]] = "O"
        record[self.orderbook.column_indices["orderorino"]] = order_id
        record[self.orderbook.column_indices["order_price"]] = price
        record[self.orderbook.column_indices["order_volume"]] = volume
        record[self.orderbook.column_indices["bsflag"]] = side
        record[self.orderbook.column_indices["order_type"]] = order_type
        record[self.orderbook.column_indices["extime"]] = time
        record[self.orderbook.column_indices["int_time"]] = 93002000
        return record

    def _process_trade(self, trade_record, is_continuous=True):
        """辅助方法：设置row_data并处理Trade记录."""
        self.orderbook.row_data = trade_record
        self.orderbook._process_trade_record(trade_record, is_continuous=is_continuous)

    def _process_order(self, order_record, is_continuous=True):
        """辅助方法：设置row_data并处理Order记录."""
        self.orderbook.row_data = order_record
        self.orderbook._process_order_record(order_record, is_continuous=is_continuous)

    def test_tc_sh_001_single_trade_accumulation(self):
        """TC-SH-001: 单个Trade记录处理."""
        # 处理Trade记录
        trade_record = self._create_trade_record(
            trade_id=101, price=105000, volume=100, side="B"
        )

        self._process_trade(trade_record)

        # 验证：Trade记录已保存
        assert 101 in self.orderbook.market_trades
        assert self.orderbook.current_trade_id == 101
        assert self.orderbook.data_status == "T"

        # 验证：Trade数据正确
        trade = self.orderbook.market_trades[101]
        assert trade[self.orderbook.column_indices["trade_price"]] == 105000
        assert trade[self.orderbook.column_indices["trade_volume"]] == 100

    def test_tc_sh_002_multiple_trades_same_id(self):
        """TC-SH-002: 同一trade_id的多笔Trade记录累积."""
        # 第1笔Trade
        trade1 = self._create_trade_record(
            trade_id=101, price=105000, volume=50, side="B"
        )
        self._process_trade(trade1)

        # 第2笔Trade（同一ID）
        trade2 = self._create_trade_record(
            trade_id=101, price=105100, volume=30, side="B"
        )
        self._process_trade(trade2)

        # 第3笔Trade（同一ID）
        trade3 = self._create_trade_record(
            trade_id=101, price=104900, volume=20, side="B"
        )
        self.orderbook._process_trade_record(trade3, is_continuous=True)

        # 验证：数量累加
        trade = self.orderbook.market_trades[101]
        assert trade[self.orderbook.column_indices["trade_volume"]] == 100  # 50+30+20

        # 验证：买单取最高价
        assert trade[self.orderbook.column_indices["trade_price"]] == 105100

    def test_tc_sh_003_multiple_trades_sell_side_price(self):
        """TC-SH-003: 卖单多笔Trade取最低价."""
        # 第1笔Trade（卖单）
        trade1 = self._create_trade_record(
            trade_id=201, price=105000, volume=50, side="S"
        )
        self._process_trade(trade1)

        # 第2笔Trade（同一ID，更高价）
        trade2 = self._create_trade_record(
            trade_id=201, price=105200, volume=30, side="S"
        )
        self._process_trade(trade2)

        # 第3笔Trade（同一ID，更低价）
        trade3 = self._create_trade_record(
            trade_id=201, price=104800, volume=20, side="S"
        )
        self.orderbook._process_trade_record(trade3, is_continuous=True)

        # 验证：卖单取最低价
        trade = self.orderbook.market_trades[201]
        assert trade[self.orderbook.column_indices["trade_price"]] == 104800
        assert trade[self.orderbook.column_indices["trade_volume"]] == 100

    def test_tc_sh_004_trade_then_order_full_fill(self):
        """TC-SH-004: Trade记录后跟Order记录（完全成交）."""
        # 先解除first_continuous限制
        self.orderbook.first_continuous = False

        # 步骤1: 处理Trade记录
        trade_record = self._create_trade_record(
            trade_id=101, price=105000, volume=100, side="B"
        )
        self._process_trade(trade_record)

        # 步骤2: 处理对应的Order记录（order_id与trade_id不同，说明完全成交）
        order_record = self._create_order_record(
            order_id=102,  # 不同于trade_id，说明是新订单
            price=105000,
            volume=50,
            side="B",
        )
        self._process_order(order_record)

        # 验证：之前的Trade已转为订单并挂单
        assert 101 in self.orderbook.orders
        assert self.orderbook.orders[101].price == 105000
        assert self.orderbook.orders[101].quantity == 100

        # 验证：新订单也已挂单
        assert 102 in self.orderbook.orders

    def test_tc_sh_005_trade_then_same_order_partial_fill(self):
        """TC-SH-005: Trade记录后跟相同ID的Order记录（部分成交）."""
        # 先解除first_continuous限制
        self.orderbook.first_continuous = False

        # 步骤1: 处理Trade记录（成交100手）
        trade_record = self._create_trade_record(
            trade_id=101, price=105000, volume=100, side="B"
        )
        self._process_trade(trade_record)

        # 步骤2: 处理相同ID的Order记录（总量200手，说明部分成交）
        order_record = self._create_order_record(
            order_id=101,  # 与trade_id相同，说明部分成交
            price=105000,
            volume=100,  # 未成交部分
            side="B",
        )
        self._process_order(order_record)

        # 验证：订单总量应该是 trade_volume(100) + order_volume(100) = 200
        assert 101 in self.orderbook.orders
        trade = self.orderbook.market_trades[101]
        assert trade[self.orderbook.column_indices["trade_volume"]] == 200

    def test_tc_sh_006_sequential_different_trades(self):
        """TC-SH-006: 不同trade_id的连续Trade记录."""
        # 第1个Trade（完整）
        trade1 = self._create_trade_record(
            trade_id=101, price=105000, volume=100, side="B"
        )
        self._process_trade(trade1)

        # 第2个Trade（新ID，应触发前一个Trade的finalize）
        trade2 = self._create_trade_record(
            trade_id=102, price=105100, volume=150, side="B"
        )

        # 设置first_continuous为False（模拟非首笔）
        self.orderbook.first_continuous = False
        self._process_trade(trade2)

        # 验证：第1个Trade已转为订单
        assert 101 in self.orderbook.orders

        # 验证：第2个Trade成为当前Trade
        assert self.orderbook.current_trade_id == 102
        assert 102 in self.orderbook.market_trades

    def test_tc_sh_007_order_cancellation(self):
        """TC-SH-007: Order记录为撤单类型（D）."""
        # 前置：先有一个挂单
        order_record1 = self._create_order_record(
            order_id=101, price=105000, volume=100, side="B"
        )
        self.orderbook._process_order_record(order_record1, is_continuous=False)

        # 验证订单已挂单
        assert 101 in self.orderbook.orders

        # 处理撤单记录
        cancel_record = self._create_order_record(
            order_id=101,
            price=105000,
            volume=100,
            side="B",
            order_type="D",  # 撤单类型
        )
        self.orderbook._process_order_record(cancel_record, is_continuous=True)

        # 验证：订单已被撤销
        assert 101 not in self.orderbook.orders

        # 验证：撤单记录已生成
        cancels = self.orderbook.get_cancels_table()
        assert len(cancels) == 1
        assert cancels[0]["订单ID"] == 101

    def test_tc_sh_008_state_machine_transitions(self):
        """TC-SH-008: 状态机转换（O -> T -> O）."""
        # 初始状态
        assert self.orderbook.data_status == "O"

        # O -> T: 处理Trade记录
        trade_record = self._create_trade_record(
            trade_id=101, price=105000, volume=100, side="B"
        )
        self._process_trade(trade_record)
        assert self.orderbook.data_status == "T"

        # T -> O: 处理Order记录
        order_record = self._create_order_record(
            order_id=102, price=105000, volume=50, side="B"
        )
        self._process_order(order_record)
        assert self.orderbook.data_status == "O"

    def test_tc_sh_009_first_continuous_flag(self):
        """TC-SH-009: first_continuous标志的作用."""
        self.orderbook.first_continuous = True

        # 第1笔Trade
        trade1 = self._create_trade_record(
            trade_id=101, price=105000, volume=100, side="B"
        )
        self._process_trade(trade1)

        # 第2笔Trade（新ID）
        trade2 = self._create_trade_record(
            trade_id=102, price=105100, volume=150, side="B"
        )
        self._process_trade(trade2)

        # 验证：first_continuous为True时，不会触发finalize
        # 所以订单101不应该在订单簿中
        assert 101 not in self.orderbook.orders

    def test_tc_sh_010_complex_trade_order_sequence(self):
        """TC-SH-010: 复杂的Trade-Order序列（从第一条数据开始的完整流程）."""
        # 确保从连续竞价第一条数据开始
        assert self.orderbook.first_continuous is True

        # === 阶段1: 处理第一条数据，解除first_continuous限制 ===
        dummy_trade = self._create_trade_record(
            trade_id=100, price=105000, volume=100, side="B"
        )
        self._process_trade(dummy_trade)

        # 模拟 process_single_market_record 中对first_continuous的重置
        # 在真实场景中，这会在处理完第一条数据后自动设置
        if self.orderbook.first_continuous:
            self.orderbook.first_continuous = False

        # 验证：first_continuous已经变为False，可以正常测试后续逻辑
        assert self.orderbook.first_continuous is False

        # === 阶段2: 测试复杂的Trade-Order序列 ===
        # 序列: T(101) -> T(101) -> O(102)

        # 多笔Trade累积
        trade1 = self._create_trade_record(
            trade_id=101, price=105000, volume=50, side="B"
        )
        self._process_trade(trade1)

        trade2 = self._create_trade_record(
            trade_id=101, price=105100, volume=50, side="B"
        )
        self._process_trade(trade2)

        # 新Order触发前Trade的finalize
        order1 = self._create_order_record(
            order_id=102, price=105000, volume=100, side="B"
        )
        self._process_order(order1)

        # 验证：Trade 101已转为订单
        assert 101 in self.orderbook.orders
        assert self.orderbook.orders[101].quantity == 100  # 50+50
        assert self.orderbook.orders[101].price == 105100  # 买单取最高价

        # 验证：Order 102也已挂单
        assert 102 in self.orderbook.orders


class TestShanghaiEdgeCases:
    """上海交易所边界情况测试."""

    def setup_method(self):
        self.orderbook = OrderBookSH()
        self.orderbook.set_trading_phase(TradingPhase.CONTINUOUS)

        # 初始化first_continuous标志（模拟连续竞价刚开始）
        self.orderbook.first_continuous = True

        self.orderbook.column_indices = {
            "int_time": 8,
            "trade_type": 11,
            "order_type": 19,
            "orderorino": 22,
            "sell_id": 15,
            "buy_id": 16,
            "bsflag": 10,
            "trade_price": 9,
            "order_price": 23,
            "trade_volume": 13,
            "order_volume": 21,
            "type": 18,
            "extime": 4,
            "localtime": 3,
        }

    def _process_trade(self, trade_record, is_continuous=True):
        """辅助方法：设置row_data并处理Trade记录."""
        self.orderbook.row_data = trade_record
        self.orderbook._process_trade_record(trade_record, is_continuous=is_continuous)

    def _process_order(self, order_record, is_continuous=True):
        """辅助方法：设置row_data并处理Order记录."""
        self.orderbook.row_data = order_record
        self.orderbook._process_order_record(order_record, is_continuous=is_continuous)

    def test_tc_sh_edge_001_zero_volume_trade(self):
        """TC-SH-EDGE-001: 零成交量的Trade记录."""
        record = np.zeros(25, dtype=object)
        record[self.orderbook.column_indices["type"]] = "T"
        record[self.orderbook.column_indices["buy_id"]] = 101
        record[self.orderbook.column_indices["trade_price"]] = 105000
        record[self.orderbook.column_indices["trade_volume"]] = 0  # 零成交量
        record[self.orderbook.column_indices["bsflag"]] = "B"
        record[self.orderbook.column_indices["extime"]] = 20260202093001000000
        record[self.orderbook.column_indices["int_time"]] = 93001000

        self._process_trade(record)

        # 验证：仍然会记录（实际系统可能需要过滤）
        assert 101 in self.orderbook.market_trades

    def test_tc_sh_edge_002_same_trade_order_id(self):
        """TC-SH-EDGE-002: Trade和Order使用相同ID（部分成交场景）."""
        # Trade记录
        trade_record = np.zeros(25, dtype=object)
        trade_record[self.orderbook.column_indices["type"]] = "T"
        trade_record[self.orderbook.column_indices["buy_id"]] = 101
        trade_record[self.orderbook.column_indices["trade_price"]] = 105000
        trade_record[self.orderbook.column_indices["trade_volume"]] = 50
        trade_record[self.orderbook.column_indices["bsflag"]] = "B"
        trade_record[self.orderbook.column_indices["extime"]] = 20260202093001000000
        trade_record[self.orderbook.column_indices["int_time"]] = 93001000

        self._process_trade(trade_record)

        # Order记录（相同ID）
        order_record = np.zeros(25, dtype=object)
        order_record[self.orderbook.column_indices["type"]] = "O"
        order_record[self.orderbook.column_indices["orderorino"]] = 101
        order_record[self.orderbook.column_indices["order_price"]] = 105000
        order_record[self.orderbook.column_indices["order_volume"]] = 50
        order_record[self.orderbook.column_indices["bsflag"]] = "B"
        order_record[self.orderbook.column_indices["order_type"]] = "O"
        order_record[self.orderbook.column_indices["extime"]] = 20260202093002000000
        order_record[self.orderbook.column_indices["int_time"]] = 93002000

        self._process_order(order_record)

        # 验证：相同ID的情况下，订单已挂单
        assert 101 in self.orderbook.orders
        # order_id与trade_id相同时，表示部分成交
        # order_volume才是剩余的挂单量
        assert self.orderbook.orders[101].quantity == 50


if __name__ == "__main__":
    # 可以直接运行测试
    pytest.main([__file__, "-v", "--tb=short"])
