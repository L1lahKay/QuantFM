"""cn_l2_v1 规范事件模式及沪/深 -> 抽象空间的映射。"""

from __future__ import annotations

from quant_fm.schema.cn_l2_v1 import (
    CANONICAL_COLUMNS,
    SCHEMA_VERSION,
    board_of,
    canonical_arrow_schema,
    events_to_canonical,
    exchange_of,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "SCHEMA_VERSION",
    "board_of",
    "canonical_arrow_schema",
    "events_to_canonical",
    "exchange_of",
]
