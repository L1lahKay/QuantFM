"""
对规范事件应用冻结的 :class:`Vocab`，生成 token 分片。

确定性：给定相同 ``vocab.json`` 与输入 parquet，token 输出字节级可复现。
含门控 2 辅助（分箱覆盖率 / UNK 率，以及验证/测试日期未参与分箱拟合的防泄漏断言）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from pylob.event_ordering import validate_order_if_present

from quant_fm.tokenizer.artifact_contract import write_token_contract
from quant_fm.tokenizer.transforms import add_derived_fields
from quant_fm.tokenizer.vocab import CATEGORICAL_SOURCE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_fm.tokenizer.vocab import Vocab

logger = logging.getLogger(__name__)

# [导读] token 列名 -> (类型, 源字段名)
# categorical = 固定类别（如 BUY/SELL）；binned = 连续值分箱后的整数
TOKEN_FIELDS: dict[str, tuple[str, str]] = {
    "tok_evt_type": ("categorical", "evt_type"),
    "tok_side": ("categorical", "side"),
    "tok_session": ("categorical", "session"),
    "tok_board": ("categorical", "board"),
    "tok_order_type": ("categorical", "order_type"),
    "tok_event_source": ("categorical", "event_source"),
    "tok_price_bin": ("binned", "price_rel"),
    "tok_volume_bin": ("binned", "log_volume"),
    "tok_delta_t_bin": ("binned", "log_delta_t"),
}

INDEX_COLUMNS = ("symbol", "security_id", "date", "int_time", "event_idx")


def tokenize_frame(events: pl.DataFrame, vocab: Vocab) -> pl.DataFrame:
    """对单个规范标的日数据帧进行分词。"""
    validate_order_if_present(events, version=vocab.event_ordering_version)
    df = add_derived_fields(
        events,
        transform_version=vocab.feature_transform_version,
    )  # 先算 price_rel、log_volume 等连续特征
    out_cols: dict[str, np.ndarray] = {}

    for tok_name, (kind, src) in TOKEN_FIELDS.items():
        if kind == "categorical":
            src_col = CATEGORICAL_SOURCE[src]
            # [导读] 字符串类别 → 整数 id（如 "EXEC" → 3）
            out_cols[tok_name] = vocab.encode_categorical(src, df[src_col].to_numpy())
        else:
            # [导读] 连续值 → 按 vocab 里冻结的分箱边界 → 整数 bin id
            out_cols[tok_name] = vocab.encode_binned(src, df[src].to_numpy())

    keep = [c for c in INDEX_COLUMNS if c in df.columns]
    token_df = df.select(keep).with_columns(
        [pl.Series(name, arr) for name, arr in out_cols.items()]
    )
    return token_df


def tokenize_path(
    src: Path,
    dst: Path,
    vocab: Vocab,
) -> int:
    """分词单个规范 parquet 并写出 token 分片。"""
    events = pl.read_parquet(src)
    tokens = tokenize_frame(events, vocab)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_suffix(".parquet.tmp")
    tokens.write_parquet(temporary)
    temporary.replace(dst)
    write_token_contract(dst, vocab)
    return tokens.height


@dataclass(slots=True)
class CoverageReport:
    """门控 2 分词器覆盖率统计。"""

    edge_bin_rate: dict[str, float]
    unknown_rate: dict[str, float]
    n_events: int

    def passed(self, *, max_edge_rate: float = 0.5, max_unknown: float = 0.05) -> bool:
        """各字段是否均在配置的覆盖率界限内。"""
        edges_ok = all(v <= max_edge_rate for v in self.edge_bin_rate.values())
        unk_ok = all(v <= max_unknown for v in self.unknown_rate.values())
        return edges_ok and unk_ok


def coverage_report(token_df: pl.DataFrame, vocab: Vocab) -> CoverageReport:
    """统计分箱字段落入极端箱 / UNK 类别的频率。"""
    n = token_df.height
    edge_rate: dict[str, float] = {}
    unk_rate: dict[str, float] = {}

    for tok_name, (kind, src) in TOKEN_FIELDS.items():
        if tok_name not in token_df.columns:
            continue
        vals = token_df[tok_name].to_numpy()
        if kind == "binned":
            n_bins = vocab.n_bins
            first_id, last_id = 1, n_bins  # PAD（id 0）之后
            edge_hits = int(np.sum((vals == first_id) | (vals == last_id)))
            edge_rate[src] = edge_hits / n if n else 0.0
        else:
            cats = vocab.categorical[src]
            if "UNKNOWN" in cats:
                unk_id = cats.index("UNKNOWN") + 1
                unk_rate[src] = float(np.mean(vals == unk_id)) if n else 0.0

    return CoverageReport(edge_bin_rate=edge_rate, unknown_rate=unk_rate, n_events=n)


def assert_no_leakage(
    vocab: Vocab,
    val_dates: Sequence[str],
    test_dates: Sequence[str],
) -> None:
    """若任一验证/测试日期曾用于拟合分箱边界，则显式失败。"""
    fit = set(vocab.fit_dates)
    overlap = fit & (set(val_dates) | set(test_dates))
    if overlap:
        msg = f"leakage: val/test dates used in bin fitting: {sorted(overlap)}"
        raise AssertionError(msg)
    logger.info("no-leakage check passed (%d fit dates)", len(fit))
