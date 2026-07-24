"""Compact causal snapshots derived from a replayed limit-order book."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping


class OrderLike(Protocol):
    """Structural type required by :func:`snapshot_book_state`."""

    order_id: int
    price: int | float
    quantity: int


class OrderBookLike(Protocol):
    """Minimal interface exposed by the Shanghai and Shenzhen engines."""

    bids: Mapping[int | float, Iterable[OrderLike]]
    asks: Mapping[int | float, Iterable[OrderLike]]
    orders: Mapping[int, OrderLike]


@dataclass(frozen=True, slots=True)
class BookState:
    """
    A compact, immutable top-of-book/depth snapshot.

    ``valid`` deliberately means *two-sided*: both best quotes exist and therefore
    spread and microprice are defined. A one-sided limit-up/limit-down book is not
    discarded; its available depths and imbalances remain populated.
    """

    valid: bool
    bid1: int | None
    ask1: int | None
    bid_qty_1: int
    ask_qty_1: int
    bid_depth_5: int
    ask_depth_5: int
    bid_depth_10: int
    ask_depth_10: int
    spread_ticks: int | None
    imbalance_1: float | None
    imbalance_5: float | None
    imbalance_10: float | None
    microprice_delta_ticks: float | None

    @property
    def empty(self) -> bool:
        """Return whether neither side contains live quantity."""
        return self.bid1 is None and self.ask1 is None


@dataclass(frozen=True, slots=True)
class BookStateTransition:
    """Book snapshots immediately before and after one processed event."""

    pre_event_state: BookState
    post_event_state: BookState

    @property
    def pre(self) -> BookState:
        """Short alias for :attr:`pre_event_state`."""
        return self.pre_event_state

    @property
    def post(self) -> BookState:
        """Short alias for :attr:`post_event_state`."""
        return self.post_event_state


def snapshot_book_state(
    book: OrderBookLike,
    *,
    tick_size: int = 100,
    eps: float = 1e-12,
) -> BookState:
    """
    Build one compact snapshot without mutating the matching engine.

    Parameters
    ----------
    book
        Any object exposing ``bids``, ``asks`` and the active ``orders`` mapping.
        PyLOB stores prices as integer 1/10000 yuan units, so the normal A-share
        one-cent tick is ``100``.
    tick_size
        Price units per tick. Must be positive.
    eps
        Numerical guard used in imbalance and microprice denominators.
    """
    if tick_size <= 0:
        msg = "tick_size must be positive"
        raise ValueError(msg)
    if eps <= 0:
        msg = "eps must be positive"
        raise ValueError(msg)

    active_orders = getattr(book, "orders", None)
    bids = _live_levels(book.bids, active_orders, descending=True)
    asks = _live_levels(book.asks, active_orders, descending=False)

    bid1, bid_qty_1 = bids[0] if bids else (None, 0)
    ask1, ask_qty_1 = asks[0] if asks else (None, 0)
    bid_depth_5 = _depth(bids, 5)
    ask_depth_5 = _depth(asks, 5)
    bid_depth_10 = _depth(bids, 10)
    ask_depth_10 = _depth(asks, 10)

    valid = bid1 is not None and ask1 is not None
    spread_ticks: int | None = None
    microprice_delta_ticks: float | None = None
    if valid:
        spread_ticks = round((ask1 - bid1) / tick_size)
        top_quantity = bid_qty_1 + ask_qty_1
        if top_quantity > 0:
            microprice = (ask1 * bid_qty_1 + bid1 * ask_qty_1) / (top_quantity + eps)
            midpoint = (bid1 + ask1) / 2.0
            microprice_delta_ticks = (microprice - midpoint) / tick_size

    return BookState(
        valid=valid,
        bid1=bid1,
        ask1=ask1,
        bid_qty_1=bid_qty_1,
        ask_qty_1=ask_qty_1,
        bid_depth_5=bid_depth_5,
        ask_depth_5=ask_depth_5,
        bid_depth_10=bid_depth_10,
        ask_depth_10=ask_depth_10,
        spread_ticks=spread_ticks,
        imbalance_1=_imbalance(bid_qty_1, ask_qty_1, eps=eps),
        imbalance_5=_imbalance(bid_depth_5, ask_depth_5, eps=eps),
        imbalance_10=_imbalance(bid_depth_10, ask_depth_10, eps=eps),
        microprice_delta_ticks=microprice_delta_ticks,
    )


def capture_book_transition(
    book: OrderBookLike,
    apply_event: Callable[[], object],
    *,
    tick_size: int = 100,
) -> BookStateTransition:
    """
    Apply exactly one event and capture its causal pre/post snapshots.

    The post-event snapshot is the state that may be attached to event ``t`` when
    predicting event ``t+1``. The callback must process only the current event.
    """
    pre = snapshot_book_state(book, tick_size=tick_size)
    apply_event()
    post = snapshot_book_state(book, tick_size=tick_size)
    return BookStateTransition(pre_event_state=pre, post_event_state=post)


def iter_book_state_transitions[EventT](
    book: OrderBookLike,
    events: Iterable[EventT],
    apply_event: Callable[[EventT], object],
    *,
    tick_size: int = 100,
) -> Iterator[BookStateTransition]:
    """
    Replay an already exchange-ordered event stream as causal snapshots.

    This generator never reads ahead. Callers must first order equal timestamps by
    the exchange sequence number, rather than by feed reception ``local_time``.
    """
    for event in events:
        yield capture_book_transition(
            book,
            lambda event=event: apply_event(event),
            tick_size=tick_size,
        )


def _live_levels(
    side: Mapping[int | float, Iterable[OrderLike]],
    active_orders: Mapping[int, OrderLike] | None,
    *,
    descending: bool,
) -> list[tuple[int, int]]:
    levels: list[tuple[int, int]] = []
    for raw_price, orders in side.items():
        price = _integer_price(raw_price)
        quantity = 0
        for order in orders:
            order_quantity = int(order.quantity)
            if order_quantity <= 0:
                continue
            if (
                active_orders is not None
                and active_orders.get(int(order.order_id)) is not order
            ):
                continue
            quantity += order_quantity
        if quantity > 0:
            levels.append((price, quantity))
    return sorted(levels, key=lambda level: level[0], reverse=descending)


def _integer_price(price: int | float) -> int:
    value = float(price)
    if not math.isfinite(value) or not value.is_integer():
        msg = f"book price must be a finite integer unit, got {price!r}"
        raise ValueError(msg)
    return int(value)


def _depth(levels: list[tuple[int, int]], n_levels: int) -> int:
    return sum(quantity for _, quantity in levels[:n_levels])


def _imbalance(bid_depth: int, ask_depth: int, *, eps: float) -> float | None:
    total = bid_depth + ask_depth
    if total <= 0:
        return None
    return (bid_depth - ask_depth) / (total + eps)


__all__ = [
    "BookState",
    "BookStateTransition",
    "capture_book_transition",
    "iter_book_state_transitions",
    "snapshot_book_state",
]
