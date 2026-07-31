"""用冻结 :class:`VocabV2` 生成 token 与连续双通道。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from pylob.event_ordering import validate_order_if_present

from quant_fm.tokenizer.artifact_contract import write_token_contract
from quant_fm.tokenizer.storage_encoding_v2 import quantize_frame_v2
from quant_fm.tokenizer.transforms import DERIVED_CONTINUOUS, add_derived_fields
from quant_fm.tokenizer.vocab_v2 import N_SPECIAL, NA_ID, UNK_ID

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_fm.tokenizer.field_spec import FieldSpec
    from quant_fm.tokenizer.vocab_v2 import VocabV2

logger = logging.getLogger(__name__)

INDEX_COLUMNS: tuple[str, ...] = (
    "schema_version",
    "symbol",
    "security_id",
    "date",
    "exchange",
    "int_time",
    "source_seqnum",
    "event_idx",
)


def _prepare_frame(events: pl.DataFrame, vocab: VocabV2) -> pl.DataFrame:
    """按需生成兼容派生列，并校验冻结 source schema。"""
    required = {spec.source for spec in vocab.field_specs}
    missing = required - set(events.columns)
    if missing & set(DERIVED_CONTINUOUS):
        events = add_derived_fields(
            events,
            transform_version=vocab.feature_transform_version,
        )
        missing = required - set(events.columns)
    if missing:
        msg = f"events missing frozen v2 sources: {sorted(missing)}"
        raise ValueError(msg)
    return events


def _applicable_mask(frame: pl.DataFrame, spec: FieldSpec) -> np.ndarray:
    """生成字段适用性 mask；空事件列表表示全量适用。"""
    if not spec.applicable_events:
        return np.ones(frame.height, dtype=bool)
    if "evt_type" not in frame.columns:
        msg = f"field {spec.name!r} applicability requires evt_type"
        raise ValueError(msg)
    return np.isin(frame["evt_type"].cast(pl.String).to_numpy(), spec.applicable_events)


def _check_missing_contract(
    spec: FieldSpec, token_values: np.ndarray, applicable: np.ndarray
) -> None:
    """字段声明不允许缺失时，拒绝将有效事件静默编码成 NA。"""
    if spec.missing_token:
        return
    if np.any((token_values == NA_ID) & applicable):
        msg = f"field {spec.name!r} contains NA but missing_token=False"
        raise ValueError(msg)


def tokenize_frame_v2(events: pl.DataFrame, vocab: VocabV2) -> pl.DataFrame:
    """
    对单个规范事件帧生成稳定的 v2 token/scalar 列。

    数值字段同时输出 ``tok_*_bin`` 和冻结标准化的 ``val_*``。缺失标量写 0，
    但对应 token 明确写 ``NA_ID``，因此真实数值 0 不会与缺失混淆。
    """
    validate_order_if_present(events, version=vocab.event_ordering_version)
    frame = _prepare_frame(events, vocab)
    output: dict[str, np.ndarray] = {}

    for spec in vocab.field_specs:
        applicable = _applicable_mask(frame, spec)
        source = frame[spec.source].to_numpy()
        if spec.kind in {"categorical", "context"}:
            tokens = vocab.encode_categorical(spec.name, source)
            _check_missing_contract(spec, tokens, applicable)
            tokens[~applicable] = NA_ID
            output[str(spec.token_column)] = tokens
            continue

        numeric = frame[spec.source].cast(pl.Float64).to_numpy()
        if spec.is_binned:
            tokens = vocab.encode_binned(spec.name, numeric)
            _check_missing_contract(spec, tokens, applicable)
            tokens[~applicable] = NA_ID
            output[str(spec.token_column)] = tokens

        scalars = vocab.encode_scalar(spec.name, numeric)
        scalars[~applicable] = 0.0
        output[str(spec.value_column)] = scalars

    keep = [column for column in INDEX_COLUMNS if column in frame.columns]
    return frame.select(keep).with_columns(
        [pl.Series(name, values) for name, values in output.items()]
    )


def tokenize_path_v2(src: Path, dst: Path, vocab: VocabV2) -> int:
    """分词并用窄 token/Q16 scalar 写出一个 v2 parquet shard。"""
    tokens = tokenize_frame_v2(pl.read_parquet(src), vocab)
    encoded, storage_metadata = quantize_frame_v2(tokens, vocab)
    destination = Path(dst)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.tmp")
    encoded.write_parquet(
        temporary,
        compression="zstd",
        compression_level=3,
        statistics=True,
    )
    temporary.replace(destination)
    write_token_contract(
        destination,
        vocab,
        storage_encoding=storage_metadata.to_dict(),
    )
    return tokens.height


@dataclass(frozen=True, slots=True)
class CoverageReportV2:
    """v2 token 的边缘箱、NA 和 UNK 覆盖率。"""

    edge_bin_rate: dict[str, float]
    missing_rate: dict[str, float]
    unknown_rate: dict[str, float]
    actual_n_bins: dict[str, int]
    n_events: int

    def passed(
        self,
        *,
        max_edge_rate: float = 0.5,
        max_missing_rate: float = 0.5,
        max_unknown_rate: float = 0.05,
    ) -> bool:
        """检查各字段是否位于调用方给定的覆盖率阈值内。"""
        return (
            all(value <= max_edge_rate for value in self.edge_bin_rate.values())
            and all(value <= max_missing_rate for value in self.missing_rate.values())
            and all(value <= max_unknown_rate for value in self.unknown_rate.values())
        )


def coverage_report_v2(token_frame: pl.DataFrame, vocab: VocabV2) -> CoverageReportV2:
    """按字段实际 bin 数统计覆盖率，特殊 token 不算边缘箱。"""
    n_events = token_frame.height
    edge_rate: dict[str, float] = {}
    missing_rate: dict[str, float] = {}
    unknown_rate: dict[str, float] = {}
    actual_n_bins: dict[str, int] = {}

    for spec in vocab.field_specs:
        column = spec.token_column
        if column is None or column not in token_frame.columns:
            continue
        values = token_frame[column].to_numpy()
        missing_rate[spec.name] = float(np.mean(values == NA_ID)) if n_events else 0.0
        if spec.kind in {"categorical", "context"}:
            unknown_rate[spec.name] = (
                float(np.mean(values == UNK_ID)) if n_events else 0.0
            )
            continue

        n_bins = vocab.binned[spec.name].actual_n_bins
        actual_n_bins[spec.name] = n_bins
        finite_tokens = values >= N_SPECIAL
        denominator = int(finite_tokens.sum())
        first_id = N_SPECIAL
        last_id = N_SPECIAL + n_bins - 1
        edge_hits = int(
            np.sum(finite_tokens & ((values == first_id) | (values == last_id)))
        )
        edge_rate[spec.name] = edge_hits / denominator if denominator else 0.0

    return CoverageReportV2(
        edge_bin_rate=edge_rate,
        missing_rate=missing_rate,
        unknown_rate=unknown_rate,
        actual_n_bins=actual_n_bins,
        n_events=n_events,
    )


def assert_no_leakage_v2(
    vocab: VocabV2,
    val_dates: Sequence[str],
    test_dates: Sequence[str],
) -> None:
    """拒绝使用验证/测试日期拟合 v2 边界或 normalizer。"""
    fit = set(vocab.fit_dates)
    overlap = fit & (set(val_dates) | set(test_dates))
    if overlap:
        msg = f"leakage: val/test dates used in v2 vocab fitting: {sorted(overlap)}"
        raise AssertionError(msg)
    logger.info("v2 no-leakage check passed (%d fit dates)", len(fit))
