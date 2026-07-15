"""
cn_l2_v1 事件的全局字段级分词器。

与原先按标的/按日的 ``qcut``（跨股票不可比且会泄漏未来分位数）不同，
此处分箱边界在**训练窗口**上拟合一次并冻结到 ``vocab.json``；验证/测试复用同一组边界。
"""

from __future__ import annotations

from quant_fm.tokenizer.vocab import (
    CONTINUOUS_FIELDS,
    N_SPECIAL,
    PAD_ID,
    Vocab,
    default_vocab,
)

__all__ = [
    "CONTINUOUS_FIELDS",
    "N_SPECIAL",
    "PAD_ID",
    "Vocab",
    "default_vocab",
]
