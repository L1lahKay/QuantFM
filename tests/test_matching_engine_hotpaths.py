from collections import deque

from pylob import OrderBookSZ
from pylob.data_types import Order, Side, TradingPhase
from sortedcontainers import SortedDict


def _brute_force_auction_price(book: OrderBookSZ) -> tuple[float | None, int]:
    prices = sorted(set(book.bids) | set(book.asks))
    best_price = None
    max_volume = 0
    for price in prices:
        buy_volume = sum(
            order.quantity
            for level_price, orders in book.bids.items()
            if level_price >= price
            for order in orders
            if order.quantity > 0
        )
        sell_volume = sum(
            order.quantity
            for level_price, orders in book.asks.items()
            if level_price <= price
            for order in orders
            if order.quantity > 0
        )
        volume = min(buy_volume, sell_volume)
        if volume > max_volume:
            best_price = price
            max_volume = volume
    return best_price, max_volume


def test_linear_auction_scan_matches_brute_force_boundaries_and_ties() -> None:
    book = OrderBookSZ()
    book.set_trading_phase(TradingPhase.CALL_AUCTION)
    order_id = 1
    for price in range(100, 220):
        for side, quantity in ((Side.BUY, price % 17 + 1), (Side.SELL, price % 13 + 1)):
            book.add_order(
                price=price,
                quantity=quantity,
                side=side,
                order_id=order_id,
                order_type="0",
                order_time=91_500_000 + order_id,
            )
            order_id += 1

    assert book._find_auction_price() == _brute_force_auction_price(book)


class _NoIterationSortedDict(SortedDict):
    def __iter__(self):
        message = "continuous matching must not scan every price level"
        raise AssertionError(message)


def test_non_crossing_limit_order_uses_best_price_without_full_book_scan() -> None:
    book = OrderBookSZ()
    book.set_trading_phase(TradingPhase.CONTINUOUS)
    asks = _NoIterationSortedDict()
    for index, price in enumerate(range(10_000, 20_000), start=1):
        maker = Order(
            order_id=index,
            side=Side.SELL,
            price=price,
            quantity=100,
            order_time=93_000_000 + index,
            order_type="0",
        )
        asks[price] = deque([maker])
        book.orders[index] = maker
    book.asks = asks

    book.add_order(
        price=9_999,
        quantity=100,
        side=Side.BUY,
        order_id=20_001,
        order_type="0",
        order_time=94_000_000,
    )

    assert not book.trades
    assert book.bids[9_999][0].order_id == 20_001
