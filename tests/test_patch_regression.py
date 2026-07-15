"""
已合入补丁的回归测试.

覆盖 3/26 补丁和后续 PR 的关键修复，防止回归：
- c1e2085: int 强转 (price/quantity/order_id)
- 199da16: _parse_side 多格式支持
- PR #4: cancel_order success 标记
- PR #5: _normalize_dtypes
- PR #6: 空数据不崩
"""

import numpy as np
import polars as pl
import pytest
from pylob import OrderBookSH, OrderBookSZ
from pylob.data_types import Side, TradingPhase


class TestIntegerConversion:
    """
    c1e2085: 确保数据解析路径中 numpy 值被转为 int.

    int() 强转发生在 orderbook_builder_sh/sz 的 _process_*_record 里，
    不是在 add_order API 层。这里测试通过 add_order 传入 int 后
    SortedDict key 类型正确。
    """

    def setup_method(self):
        self.ob = OrderBookSZ()
        self.ob.set_trading_phase(TradingPhase.CONTINUOUS)

    def test_int_price_keeps_sorteddict_keys_clean(self):
        """Int 价格作为 SortedDict key 不应变成 float."""
        self.ob.add_order(
            price=105000,
            quantity=100,
            side=Side.SELL,
            order_id=1,
            order_type="0",
            order_time=93001000,
        )
        self.ob.add_order(
            price=104000,
            quantity=100,
            side=Side.BUY,
            order_id=2,
            order_type="0",
            order_time=93001010,
        )

        for price_key in self.ob.bids:
            assert not isinstance(price_key, float), f"bid key {price_key} is float"
        for price_key in self.ob.asks:
            assert not isinstance(price_key, float), f"ask key {price_key} is float"

    def test_numpy_int64_as_order_id(self):
        """np.int64 的 order_id 应正常工作."""
        self.ob.add_order(
            price=105000,
            quantity=100,
            side=Side.BUY,
            order_id=np.int64(42),
            order_type="0",
            order_time=93001000,
        )
        assert 42 in self.ob.orders


class TestParseSide:
    """199da16: _parse_side 支持多种字符串格式."""

    def setup_method(self):
        self.ob = OrderBookSZ()

    def test_buy_variants(self):
        assert self.ob._parse_side("B") == Side.BUY
        assert self.ob._parse_side("b'B'") == Side.BUY
        assert self.ob._parse_side("买入") == Side.BUY

    def test_sell_variants(self):
        assert self.ob._parse_side("S") == Side.SELL
        assert self.ob._parse_side("b'S'") == Side.SELL
        assert self.ob._parse_side("卖出") == Side.SELL

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="未知的交易方向"):
            self.ob._parse_side("X")


class TestCancelOrderSuccessFlag:
    """PR #4: cancel_order 只在实际从 book 移除时返回 True."""

    def setup_method(self):
        self.ob = OrderBookSZ()
        self.ob.set_trading_phase(TradingPhase.CONTINUOUS)

    def test_normal_cancel_returns_true(self):
        """正常撤单返回 True."""
        self.ob.add_order(
            price=105000,
            quantity=100,
            side=Side.BUY,
            order_id=1,
            order_type="0",
            order_time=93001000,
        )
        assert self.ob.cancel_order(1, cancel_time=93002000) is True

    def test_nonexistent_cancel_returns_false(self):
        """不存在的订单撤单返回 False."""
        assert self.ob.cancel_order(999, cancel_time=93002000) is False

    def test_already_filled_cancel_returns_false(self):
        """已完全成交的订单撤单返回 False."""
        self.ob.add_order(
            price=105000,
            quantity=100,
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
        # 卖单 #1 已完全成交，不在 orders 里
        assert self.ob.cancel_order(1, cancel_time=93003000) is False


class TestEmptyDataNocrash:
    """PR #6: 空数据不崩."""

    def test_sz_finalize_empty(self):
        """SZ 空簿 finalize 不报 AttributeError."""
        ob = OrderBookSZ()
        ob.finalize_trading_session()  # 不崩即通过

    def test_sh_finalize_empty(self):
        """SH 空簿 finalize 不报 AttributeError."""
        ob = OrderBookSH()
        ob.finalize_trading_session()


class TestNormalizeDtypes:
    """PR #5: _normalize_dtypes 处理 UInt64 和 Binary 列."""

    def test_uint64_to_int64(self):
        from pylob._utils import normalize_dtypes as _normalize_dtypes

        df = pl.DataFrame(
            {
                "exchange_time": pl.Series([1, 2, 3], dtype=pl.UInt64),
                "normal_col": [4, 5, 6],
            }
        )

        result = _normalize_dtypes(df)
        assert result["exchange_time"].dtype == pl.Int64

    def test_binary_to_utf8(self):
        from pylob._utils import normalize_dtypes as _normalize_dtypes

        df = pl.DataFrame(
            {
                "bsflag": pl.Series([b"B", b"S", b"B"], dtype=pl.Binary),
                "normal_col": ["a", "b", "c"],
            }
        )

        result = _normalize_dtypes(df)
        assert result["bsflag"].dtype == pl.Utf8
        assert result["bsflag"].to_list() == ["B", "S", "B"]

    def test_no_change_needed(self):
        from pylob._utils import normalize_dtypes as _normalize_dtypes

        df = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
            }
        )

        result = _normalize_dtypes(df)
        assert result["a"].dtype == pl.Int64
        assert result["b"].dtype == pl.Utf8

    def test_shared_normalize_dtypes(self):
        """normalize_dtypes 已提取到 _utils 模块."""
        from pylob._utils import normalize_dtypes

        df = pl.DataFrame(
            {
                "col": pl.Series([1, 2], dtype=pl.UInt64),
            }
        )
        result = normalize_dtypes(df)
        assert result["col"].dtype == pl.Int64


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
