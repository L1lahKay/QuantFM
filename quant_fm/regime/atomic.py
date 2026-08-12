"""在订单簿回放仍驻留内存时生成逐股日 Regime 原子指标。"""

from __future__ import annotations

import math
from datetime import date as calendar_date
from typing import TYPE_CHECKING

import polars as pl

from quant_fm.regime.contract import (
    ATOMIC_ARTIFACT_VERSION,
    ATOMIC_FORMULA_VERSION,
    validate_atomic_frame,
)
from quant_fm.schema.cn_l2_v1 import board_of

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pylob.book_state import BookState, BookStateTransition

_MINUTE_MS = 60_000
CONTINUOUS_SESSIONS_MS: tuple[tuple[int, int], ...] = (
    ((9 * 60 + 30) * _MINUTE_MS, (11 * 60 + 30) * _MINUTE_MS),
    (13 * 60 * _MINUTE_MS, (14 * 60 + 57) * _MINUTE_MS),
)
CONTINUOUS_SESSION_SECONDS = sum(
    end - start for start, end in CONTINUOUS_SESSIONS_MS
) / 1_000.0


def packed_time_to_ms(value: int) -> int:
    """将交易所 ``HHMMSSmmm`` 整数时间转换为自午夜毫秒。"""
    packed = int(value)
    if packed < 0:
        msg = f"exchange time must be non-negative, got {packed}"
        raise ValueError(msg)
    milliseconds = packed % 1_000
    seconds = (packed // 1_000) % 100
    minutes = (packed // 100_000) % 100
    hours = (packed // 10_000_000) % 100
    if hours > 23 or minutes > 59 or seconds > 59:
        msg = f"invalid packed exchange time: {packed}"
        raise ValueError(msg)
    return hours * 3_600_000 + minutes * 60_000 + seconds * 1_000 + milliseconds


def _continuous_interval_end(start_ms: int, next_ms: int | None) -> int | None:
    """返回当前事件状态在同一连续竞价时段内的结束时间。"""
    for session_start, session_end in CONTINUOUS_SESSIONS_MS:
        if session_start <= start_ms < session_end:
            return min(session_end, next_ms) if next_ms is not None else session_end
    return None


def _is_continuous_event(time_ms: int) -> bool:
    """判断事件本身是否落在连续竞价窗口，不依赖其状态持续时间。"""
    return any(start <= time_ms < end for start, end in CONTINUOUS_SESSIONS_MS)


def _side_ofi(
    previous_price: int | None,
    previous_qty: int,
    current_price: int | None,
    current_qty: int,
    *,
    bid: bool,
) -> float | None:
    """计算 Cont OFI 中单侧报价的价格/数量贡献。"""
    if previous_price is None or current_price is None:
        return None
    if bid:
        return float(
            (current_qty if current_price >= previous_price else 0)
            - (previous_qty if current_price <= previous_price else 0)
        )
    return float(
        -(current_qty if current_price <= previous_price else 0)
        + (previous_qty if current_price >= previous_price else 0)
    )


def event_ofi_l1(previous: BookState, current: BookState) -> float | None:
    """从相邻 pre/post L1 状态计算一个事件的标准化前原始 OFI。"""
    bid = _side_ofi(
        previous.bid1,
        previous.bid_qty_1,
        current.bid1,
        current.bid_qty_1,
        bid=True,
    )
    ask = _side_ofi(
        previous.ask1,
        previous.ask_qty_1,
        current.ask1,
        current.ask_qty_1,
        bid=False,
    )
    if bid is None or ask is None:
        return None
    return bid + ask


def build_stock_day_atomic(
    transitions: Sequence[BookStateTransition],
    event_times: Sequence[int],
    *,
    date: str,
    symbol: str,
    market: str,
    event_ordering_version: str,
) -> pl.DataFrame:
    """生成一行时间加权 spread/depth 与事件级 L1 OFI 原子特征。"""
    try:
        calendar_date.fromisoformat(date)
    except ValueError as exc:
        msg = f"date must be canonical YYYY-MM-DD, got {date!r}"
        raise ValueError(msg) from exc
    if not transitions or len(transitions) != len(event_times):
        msg = "transitions and event_times must have the same non-zero length"
        raise ValueError(msg)
    normalized_market = market.upper()
    if normalized_market not in {"SH", "SZ"}:
        msg = "market must be SH or SZ"
        raise ValueError(msg)
    normalized_symbol = str(symbol).zfill(6)
    times_ms = [packed_time_to_ms(int(value)) for value in event_times]
    if times_ms != sorted(times_ms):
        msg = "event_times must be non-decreasing in exchange order"
        raise ValueError(msg)

    observed_ms = valid_ms = 0
    spread_weighted = depth_weighted = 0.0
    ofi_values: list[float] = []
    for index, transition in enumerate(transitions):
        start_ms = times_ms[index]
        post = transition.post_event_state
        if _is_continuous_event(start_ms):
            ofi = event_ofi_l1(transition.pre_event_state, post)
            if ofi is not None and math.isfinite(ofi):
                ofi_values.append(ofi)

        next_ms = times_ms[index + 1] if index + 1 < len(times_ms) else None
        interval_end = _continuous_interval_end(start_ms, next_ms)
        if interval_end is None or interval_end <= start_ms:
            continue
        duration = interval_end - start_ms
        observed_ms += duration
        depth_weighted += math.log1p(post.bid_depth_5 + post.ask_depth_5) * duration
        if post.valid and post.bid1 is not None and post.ask1 is not None:
            midpoint = (post.bid1 + post.ask1) / 2.0
            if midpoint > 0:
                spread_bps = 10_000.0 * (post.ask1 - post.bid1) / midpoint
                spread_weighted += spread_bps * duration
                valid_ms += duration

    spread = spread_weighted / valid_ms if valid_ms else None
    depth = depth_weighted / observed_ms if observed_ms else None
    ofi_denominator = sum(abs(value) for value in ofi_values)
    ofi = (
        sum(ofi_values) / ofi_denominator
        if ofi_values and ofi_denominator > 0
        else (0.0 if ofi_values else None)
    )
    frame = pl.DataFrame(
        {
            "artifact_version": [ATOMIC_ARTIFACT_VERSION],
            "formula_version": [ATOMIC_FORMULA_VERSION],
            "date": [date],
            "symbol": [normalized_symbol],
            "market": [normalized_market],
            "board": [board_of(normalized_symbol, normalized_market)],
            "stock_spread_bps": [spread],
            "stock_depth_l5_log": [depth],
            "stock_ofi_l1": [ofi],
            "book_valid_seconds": [valid_ms / 1_000.0],
            "observed_seconds": [observed_ms / 1_000.0],
            "continuous_session_seconds": [CONTINUOUS_SESSION_SECONDS],
            "book_valid_ratio": [
                (valid_ms / 1_000.0) / CONTINUOUS_SESSION_SECONDS
            ],
            "n_events": [len(transitions)],
            "n_ofi_events": [len(ofi_values)],
            "feature_cutoff_ts": [f"{date}T15:00:00+08:00"],
            "event_ordering_version": [event_ordering_version],
        }
    )
    return validate_atomic_frame(frame, expected_date=date)


__all__ = [
    "CONTINUOUS_SESSIONS_MS",
    "CONTINUOUS_SESSION_SECONDS",
    "build_stock_day_atomic",
    "event_ofi_l1",
    "packed_time_to_ms",
]
