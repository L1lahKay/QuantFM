import pytest
from pylob import OrderBookSH, OrderBookSZ
from pylob.data_types import Side, TradingPhase


@pytest.mark.parametrize("book_type", [OrderBookSZ, OrderBookSH])
@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_cancel_physically_removes_order_and_preserves_fifo(book_type, side):
    book = book_type()
    book.set_trading_phase(TradingPhase.CONTINUOUS)
    price = 104_500

    for order_id in (101, 102, 103):
        book.add_order(
            price=price,
            quantity=100,
            side=side,
            order_id=order_id,
            order_type="0",
            order_time=93_001_000 + order_id,
        )

    assert book.cancel_order(102, cancel_time=93_002_000) is True

    side_book = book.bids if side == Side.BUY else book.asks
    assert [order.order_id for order in side_book[price]] == [101, 103]
    assert set(book.orders) == {101, 103}
    assert all(order.quantity > 0 for order in side_book[price])

    assert book.cancel_order(101, cancel_time=93_002_001) is True
    assert book.cancel_order(103, cancel_time=93_002_002) is True
    assert price not in side_book
    assert book.orders == {}
