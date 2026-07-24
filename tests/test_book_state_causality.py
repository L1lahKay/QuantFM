from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl
import pytest
from pylob.book_state import (
    iter_book_state_transitions,
    snapshot_book_state,
)
from pylob.data_types import Order, Side
from pylob.orderbook_builder_sz import OrderBookSZ
from sortedcontainers import SortedDict

from quant_fm.schema.cn_l2_v2 import (
    CANONICAL_COLUMNS,
    SCHEMA_VERSION,
    canonical_arrow_schema,
    events_to_canonical,
)
from quant_fm.tokenizer.lob_transforms import (
    sort_by_exchange_sequence,
    transitions_to_feature_frame,
)

if TYPE_CHECKING:
    from pylob.book_state import BookStateTransition


@dataclass
class _Book:
    bids: SortedDict = field(default_factory=SortedDict)
    asks: SortedDict = field(default_factory=SortedDict)
    orders: dict[int, Order] = field(default_factory=dict)


@dataclass(frozen=True)
class _Add:
    order_id: int
    side: Side
    price: int
    quantity: int


def _add(book: _Book, event: _Add) -> None:
    order = Order(
        order_id=event.order_id,
        side=event.side,
        price=event.price,
        quantity=event.quantity,
        order_time=93_000_000 + event.order_id,
        order_type="0",
    )
    side = book.bids if event.side == Side.BUY else book.asks
    side.setdefault(event.price, deque()).append(order)
    book.orders[event.order_id] = order


def _replay(events: list[_Add]) -> list[BookStateTransition]:
    book = _Book()
    return list(
        iter_book_state_transitions(book, events, lambda event: _add(book, event))
    )


def test_empty_book_state_is_explicitly_missing() -> None:
    state = snapshot_book_state(_Book())

    assert state.empty
    assert not state.valid
    assert state.bid1 is None
    assert state.ask1 is None
    assert state.bid_depth_10 == 0
    assert state.ask_depth_10 == 0
    assert state.spread_ticks is None
    assert state.imbalance_1 is None
    assert state.microprice_delta_ticks is None


@pytest.mark.parametrize(
    ("side", "expected_imbalance"),
    [(Side.BUY, 1.0), (Side.SELL, -1.0)],
)
def test_one_sided_book_preserves_limit_state(
    side: Side,
    expected_imbalance: float,
) -> None:
    book = _Book()
    _add(book, _Add(1, side, 10_000, 100))
    _add(book, _Add(2, side, 9_900 if side == Side.BUY else 10_100, 250))

    state = snapshot_book_state(book)

    assert not state.valid
    assert not state.empty
    assert state.bid_depth_5 + state.ask_depth_5 == 350
    assert state.imbalance_1 == pytest.approx(expected_imbalance)
    assert state.imbalance_5 == pytest.approx(expected_imbalance)
    assert state.spread_ticks is None
    assert state.microprice_delta_ticks is None


def test_two_sided_depth_imbalance_and_microprice() -> None:
    book = _Book()
    _add(book, _Add(1, Side.BUY, 10_000, 100))
    _add(book, _Add(2, Side.BUY, 9_900, 200))
    _add(book, _Add(3, Side.SELL, 10_100, 300))
    _add(book, _Add(4, Side.SELL, 10_200, 100))

    state = snapshot_book_state(book, tick_size=100)

    assert state.valid
    assert state.bid1 == 10_000
    assert state.ask1 == 10_100
    assert state.bid_qty_1 == 100
    assert state.ask_qty_1 == 300
    assert state.bid_depth_5 == 300
    assert state.ask_depth_5 == 400
    assert state.spread_ticks == 1
    assert state.imbalance_1 == pytest.approx(-0.5)
    assert state.imbalance_5 == pytest.approx(-1 / 7)
    # microprice=10025, midpoint=10050, hence -0.25 tick.
    assert state.microprice_delta_ticks == pytest.approx(-0.25)


def test_snapshot_ignores_tombstones_not_in_active_index() -> None:
    book = _Book()
    _add(book, _Add(1, Side.BUY, 10_000, 100))
    stale = book.orders.pop(1)
    _add(book, _Add(2, Side.BUY, 9_900, 200))

    state = snapshot_book_state(book)

    assert stale in book.bids[10_000]
    assert state.bid1 == 9_900
    assert state.bid_depth_5 == 200


def test_snapshot_works_with_real_matching_engine_without_mutating_it() -> None:
    book = OrderBookSZ()
    book.add_order(10_000, 100, Side.BUY, order_id=1, order_time=92_500_000)
    book.add_order(10_100, 300, Side.SELL, order_id=2, order_time=92_500_001)

    before = snapshot_book_state(book)
    assert before.valid
    assert before.bid_qty_1 == 100
    assert before.ask_qty_1 == 300

    assert book.cancel_order(1, cancel_time=92_500_002)
    after = snapshot_book_state(book)
    assert not after.valid
    assert after.bid1 is None
    assert after.imbalance_1 == pytest.approx(-1.0)


def test_full_replay_equals_every_online_prefix_and_future_is_irrelevant() -> None:
    first_two = [
        _Add(1, Side.BUY, 10_000, 100),
        _Add(2, Side.SELL, 10_200, 300),
    ]
    full = _replay([*first_two, _Add(3, Side.BUY, 10_100, 10_000)])
    changed_future = _replay([*first_two, _Add(3, Side.SELL, 9_900, 99_999)])

    for prefix_length in (1, 2):
        online_prefix = _replay(first_two[:prefix_length])
        assert full[:prefix_length] == online_prefix
        assert changed_future[:prefix_length] == online_prefix

    assert full[0].pre.empty
    assert full[0].post.bid1 == 10_000
    assert full[1].pre == full[0].post
    assert full[1].post.valid
    assert full[2].pre == full[1].post


def test_pre_feature_uses_pre_state_while_post_feature_includes_current_event() -> None:
    events = [
        _Add(1, Side.BUY, 10_000, 100),
        _Add(2, Side.SELL, 10_200, 100),
        _Add(3, Side.SELL, 10_100, 300),
    ]
    transitions = _replay(events)
    features = transitions_to_feature_frame(
        transitions,
        event_prices=[event.price for event in events],
    )

    # Before event 3 the midpoint is 10100, so its distance is exactly zero.
    assert features["event_price_distance_ticks_pre"][2] == pytest.approx(0.0)
    # After event 3 its new ask is visible and the spread has narrowed to one tick.
    assert features["spread_ticks_post"][2] == 1
    assert features["imbalance_l1_post"][2] == pytest.approx(-0.5)
    # Event 2 arrived into a one-sided pre-book, so no midpoint distance was invented.
    assert features["event_price_distance_ticks_pre"][1] is None


def test_equal_timestamps_use_exchange_sequence_not_local_time() -> None:
    events = pl.DataFrame(
        {
            "int_time": [93_000_000, 93_000_000, 93_000_000, 93_000_001],
            "serial": [2, 1, 1, 3],
            "local_time": [10, 30, 20, 5],
            "marker": ["seq2", "tie-first", "tie-second", "next"],
        }
    )

    ordered = sort_by_exchange_sequence(events)

    assert ordered["marker"].to_list() == ["tie-first", "tie-second", "seq2", "next"]


def test_cn_l2_v2_requires_real_aligned_post_state() -> None:
    additions = [
        _Add(1, Side.BUY, 10_000, 100),
        _Add(2, Side.SELL, 10_200, 100),
        _Add(3, Side.SELL, 10_100, 300),
    ]
    transitions = _replay(additions)
    features = transitions_to_feature_frame(
        transitions,
        event_prices=[event.price for event in additions],
    )
    events = _standard_events(additions)

    canonical = events_to_canonical(
        events,
        date="2026-01-05",
        market="SZ",
        book_features=features,
    )

    assert tuple(canonical.columns) == CANONICAL_COLUMNS
    assert canonical["schema_version"].unique().to_list() == [SCHEMA_VERSION]
    assert canonical["exchange_seqnum"].to_list() == [1, 2, 3]
    assert canonical["time_of_day_ms"][0] == 9 * 3_600_000 + 30 * 60_000
    assert canonical["book_valid_post"].to_list() == [False, True, True]
    assert canonical_arrow_schema().names == list(CANONICAL_COLUMNS)

    with pytest.raises(ValueError, match="identical row counts"):
        events_to_canonical(
            events,
            date="2026-01-05",
            market="SZ",
            book_features=features.head(2),
        )


def _standard_events(additions: list[_Add]) -> pl.DataFrame:
    n_rows = len(additions)
    return pl.DataFrame(
        {
            "symbol": ["000001"] * n_rows,
            "market": ["SZ"] * n_rows,
            "event_idx": list(range(n_rows)),
            "int_time": [93_000_000, 93_000_001, 93_000_002],
            "local_time": [93_000_100, 93_000_101, 93_000_102],
            "serial": [1, 2, 3],
            "delta_t": [0, 1, 1],
            "session_phase": ["CONTINUOUS_AM"] * n_rows,
            "event_type": ["ADD"] * n_rows,
            "side": [event.side.name for event in additions],
            "price": [event.price for event in additions],
            "volume": [event.quantity for event in additions],
            "log_volume": [0.0] * n_rows,
            "orderorino": [event.order_id for event in additions],
            "buy_id": [0] * n_rows,
            "sell_id": [0] * n_rows,
        }
    )
