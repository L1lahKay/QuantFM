"""把异步逐笔时间映射到 A 股同步交易时钟。"""

from __future__ import annotations

import numpy as np
import polars as pl

from quant_fm.tokenizer.transforms import int_time_to_ms

_MINUTE_MS = 60_000
_OPEN_CALL_START_MS = (9 * 60 + 15) * _MINUTE_MS
_AM_START_MS = (9 * 60 + 30) * _MINUTE_MS
_AM_END_MS = (11 * 60 + 30) * _MINUTE_MS
_PM_START_MS = 13 * 60 * _MINUTE_MS
_PM_END_MS = 15 * 60 * _MINUTE_MS


def clock_interval_id(
    time_of_day_ms: np.ndarray,
    *,
    minutes: int = 5,
    include_open_call: bool = True,
) -> np.ndarray:
    """
    返回不跨午休的顺序 interval id；非交易时段为 ``-1``。

    开盘集合竞价统一为 interval 0；连续竞价从 1 开始。11:30 和 15:00
    的边界事件归入各自最后一个 bucket，避免交易所收盘事件被丢弃。
    """
    if minutes < 1 or 120 % minutes:
        msg = "minutes must be a positive divisor of the 120-minute sessions"
        raise ValueError(msg)
    values = np.asarray(time_of_day_ms, dtype=np.int64)
    result = np.full(values.shape, -1, dtype=np.int32)
    width = minutes * _MINUTE_MS
    buckets_per_session = 120 // minutes
    continuous_offset = 1 if include_open_call else 0

    if include_open_call:
        opening = (values >= _OPEN_CALL_START_MS) & (values < _AM_START_MS)
        result[opening] = 0

    morning = (values >= _AM_START_MS) & (values <= _AM_END_MS)
    morning_bucket = np.minimum(
        (values[morning] - _AM_START_MS) // width,
        buckets_per_session - 1,
    )
    result[morning] = (continuous_offset + morning_bucket).astype(np.int32)

    afternoon = (values >= _PM_START_MS) & (values <= _PM_END_MS)
    afternoon_bucket = np.minimum(
        (values[afternoon] - _PM_START_MS) // width,
        buckets_per_session - 1,
    )
    result[afternoon] = (
        continuous_offset + buckets_per_session + afternoon_bucket
    ).astype(np.int32)
    return result


def add_clock_interval(
    frame: pl.DataFrame,
    *,
    time_col: str = "int_time",
    output_col: str = "clock_interval",
    packed_int_time: bool = True,
    minutes: int = 5,
) -> pl.DataFrame:
    """为事件表追加确定性的同步 interval id。"""
    raw = frame[time_col].to_numpy()
    elapsed = int_time_to_ms(raw) if packed_int_time else raw.astype(np.int64)
    intervals = clock_interval_id(elapsed, minutes=minutes)
    return frame.with_columns(pl.Series(output_col, intervals))
