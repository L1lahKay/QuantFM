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
from quant_fm.schema.cn_l2_v2 import (
    BOOK_STATE_TIMING,
)
from quant_fm.schema.cn_l2_v2 import (
    CANONICAL_COLUMNS as V2_CANONICAL_COLUMNS,
)
from quant_fm.schema.cn_l2_v2 import SCHEMA_VERSION as V2_SCHEMA_VERSION
from quant_fm.schema.cn_l2_v2 import (
    canonical_arrow_schema as canonical_arrow_schema_v2,
)
from quant_fm.schema.cn_l2_v2 import events_to_canonical as events_to_canonical_v2

__all__ = [
    "BOOK_STATE_TIMING",
    "CANONICAL_COLUMNS",
    "SCHEMA_VERSION",
    "V2_CANONICAL_COLUMNS",
    "V2_SCHEMA_VERSION",
    "board_of",
    "canonical_arrow_schema",
    "canonical_arrow_schema_v2",
    "events_to_canonical",
    "events_to_canonical_v2",
    "exchange_of",
]
