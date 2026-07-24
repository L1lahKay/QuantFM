"""
cn_l2_v1 事件的全局字段级分词器。

与原先按标的/按日的 ``qcut``（跨股票不可比且会泄漏未来分位数）不同，
此处分箱边界在**训练窗口**上拟合一次并冻结到 ``vocab.json``；验证/测试复用同一组边界。
"""

from __future__ import annotations

from quant_fm.tokenizer.field_spec import (
    BOOK_FIELD_SPECS_V2,
    DEFAULT_FIELD_SPECS_V2,
    FULL_FIELD_SPECS_V2,
    FieldSpec,
)
from quant_fm.tokenizer.vocab import (
    CONTINUOUS_FIELDS,
    N_SPECIAL,
    PAD_ID,
    Vocab,
    default_vocab,
)
from quant_fm.tokenizer.vocab_v2 import (
    V2_BOS_ID,
    V2_EOS_ID,
    V2_N_SPECIAL,
    V2_NA_ID,
    V2_PAD_ID,
    V2_SESSION_BREAK_ID,
    V2_UNK_ID,
    BinnedFieldVocab,
    VocabV2,
    default_vocab_v2,
)

__all__ = [
    "BOOK_FIELD_SPECS_V2",
    "CONTINUOUS_FIELDS",
    "DEFAULT_FIELD_SPECS_V2",
    "FULL_FIELD_SPECS_V2",
    "N_SPECIAL",
    "PAD_ID",
    "V2_BOS_ID",
    "V2_EOS_ID",
    "V2_NA_ID",
    "V2_N_SPECIAL",
    "V2_PAD_ID",
    "V2_SESSION_BREAK_ID",
    "V2_UNK_ID",
    "BinnedFieldVocab",
    "FieldSpec",
    "Vocab",
    "VocabV2",
    "default_vocab",
    "default_vocab_v2",
]
