"""
在完整训练流上拟合可复现、分层的 Tokenizer v2 词表。

采样使用 bottom-k priority reservoir：每行由稳定 shard 身份、event index、字段名
和 seed 生成一个确定性随机优先级，每个 stratum 保留优先级最小的固定配额。它与
经典 reservoir 一样遍历全部数据，但结果不依赖调用方传入的路径顺序。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from quant_fm.tokenizer.field_spec import validate_field_specs
from quant_fm.tokenizer.transforms import DERIVED_CONTINUOUS, add_derived_fields
from quant_fm.tokenizer.vocab_v2 import (
    BinnedFieldVocab,
    ContinuousNormalizer,
    VocabV2,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from quant_fm.tokenizer.field_spec import FieldSpec

logger = logging.getLogger(__name__)

DEFAULT_STRATA_COLUMNS: tuple[str, ...] = (
    "date",
    "exchange",
    "board",
    "evt_type",
)

_TWO_SIDED_SOURCE_MARKERS = (
    "price_rel",
    "price_distance",
    "microprice",
    "imbalance",
    "signed_ofi",
)


@dataclass(slots=True)
class _RunningFieldStats:
    """可批量合并的数值统计量与分层计数。"""

    count: int = 0
    missing: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: float | None = None
    max_value: float | None = None
    strata_counts: dict[str, int] = field(default_factory=dict)

    def update(self, values: np.ndarray, strata: np.ndarray) -> None:
        """合并一个 shard 的有限值统计与各 stratum 样本量。"""
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        self.missing += int((~finite).sum())
        clean = values[finite]
        if clean.size == 0:
            return

        batch_count = int(clean.size)
        batch_mean = float(clean.mean())
        batch_m2 = float(np.square(clean - batch_mean).sum())
        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.mean += delta * batch_count / total
            self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.count += batch_count

        batch_min = float(clean.min())
        batch_max = float(clean.max())
        self.min_value = (
            batch_min if self.min_value is None else min(self.min_value, batch_min)
        )
        self.max_value = (
            batch_max if self.max_value is None else max(self.max_value, batch_max)
        )

        labels, counts = np.unique(strata[finite], return_counts=True)
        for label, count in zip(labels.tolist(), counts.tolist(), strict=True):
            key = str(label)
            self.strata_counts[key] = self.strata_counts.get(key, 0) + int(count)

    @property
    def std(self) -> float:
        """返回 population std；常量/空字段使用 1 保持推理数值稳定。"""
        if self.count == 0:
            return 1.0
        variance = max(self.m2 / self.count, 0.0)
        value = float(np.sqrt(variance))
        return value if value > 0 and np.isfinite(value) else 1.0


@dataclass(slots=True)
class _PriorityReservoir:
    """固定容量的确定性 bottom-k priority reservoir。"""

    capacity: int
    priorities: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint64))
    values: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def add(self, values: np.ndarray, priorities: np.ndarray) -> None:
        """加入一个批次并仅保留全局最小的 ``capacity`` 个优先级。"""
        if self.capacity <= 0 or values.size == 0:
            return
        combined_values = np.concatenate((self.values, values.astype(np.float64)))
        combined_priorities = np.concatenate(
            (self.priorities, priorities.astype(np.uint64))
        )
        if combined_values.size > self.capacity:
            keep = np.argpartition(combined_priorities, self.capacity - 1)[
                : self.capacity
            ]
            combined_values = combined_values[keep]
            combined_priorities = combined_priorities[keep]
        # 固定内部顺序，使 shard 处理顺序不会影响 artifact 字节。
        order = np.lexsort((combined_values, combined_priorities))
        self.values = combined_values[order]
        self.priorities = combined_priorities[order]


def _prepare_frame(
    frame: pl.DataFrame, field_specs: tuple[FieldSpec, ...]
) -> pl.DataFrame:
    """按需生成 v1 兼容派生列，并检查所有 FieldSpec source 均存在。"""
    required = {spec.source for spec in field_specs}
    missing = required - set(frame.columns)
    if missing & set(DERIVED_CONTINUOUS):
        frame = add_derived_fields(frame)
        missing = required - set(frame.columns)
    if missing:
        msg = f"input shard missing FieldSpec sources: {sorted(missing)}"
        raise ValueError(msg)
    return frame


def _strata_labels(frame: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """生成逐行稳定 stratum 标签；不存在的维度显式记为缺失列。"""
    if frame.height == 0:
        return np.empty(0, dtype=object)
    if not columns:
        return np.full(frame.height, "__all__", dtype=object)

    expressions: list[pl.Expr] = []
    for column in columns:
        value = (
            pl.col(column).cast(pl.String).fill_null("<NA>")
            if column in frame.columns
            else pl.lit("<MISSING_COLUMN>")
        )
        expressions.append((pl.lit(f"{column}=") + value).alias(column))
    labels = frame.select(
        pl.concat_str(expressions, separator="\x1f").alias("stratum")
    )["stratum"]
    return labels.to_numpy()


def _applicable_mask(frame: pl.DataFrame, spec: FieldSpec) -> np.ndarray:
    """返回与 tokenizer/loss 一致的字段适用性 mask。"""
    if not spec.applicable_events:
        return np.ones(frame.height, dtype=bool)
    if "evt_type" not in frame.columns:
        msg = f"field {spec.name!r} applicability requires evt_type"
        raise ValueError(msg)
    return np.isin(frame["evt_type"].cast(pl.String).to_numpy(), spec.applicable_events)


def _allocate_quotas(counts: Mapping[str, int], budget: int) -> dict[str, int]:
    """至少覆盖每个非空 stratum，再按剩余容量比例分配。"""
    positive = {key: value for key, value in counts.items() if value > 0}
    if not positive:
        return {}
    if budget < len(positive):
        msg = (
            f"sample budget {budget} is smaller than {len(positive)} non-empty "
            "strata; increase max_samples_per_field or reduce strata_columns"
        )
        raise ValueError(msg)
    total = sum(positive.values())
    if total <= budget:
        return dict(positive)

    quota = dict.fromkeys(positive, 1)
    remaining = budget - len(positive)
    while remaining > 0:
        residual = {key: positive[key] - quota[key] for key in positive}
        eligible = {key: value for key, value in residual.items() if value > 0}
        if not eligible:
            break
        residual_total = sum(eligible.values())
        raw = {
            key: remaining * value / residual_total for key, value in eligible.items()
        }
        additions = {
            key: min(int(np.floor(value)), eligible[key]) for key, value in raw.items()
        }
        allocated = sum(additions.values())
        if allocated:
            for key, value in additions.items():
                quota[key] += value
            remaining -= allocated
            continue

        # remaining 小于 stratum 数时按最大余数法逐个补齐，字符串作为稳定 tie-break。
        order = sorted(eligible, key=lambda key: (-raw[key], key))
        for key in order[:remaining]:
            quota[key] += 1
        remaining = 0
    return quota


def _stable_shard_identity(path: Path, frame: pl.DataFrame) -> str:
    """构造与调用顺序无关、尽可能不依赖绝对根目录的 shard 身份。"""
    parts: list[str] = []
    for column in ("date", "exchange", "market", "board", "symbol", "security_id"):
        if column not in frame.columns or frame.height == 0:
            continue
        unique = frame[column].drop_nulls().unique().sort().head(4).to_list()
        parts.append(f"{column}={','.join(str(value) for value in unique)}")
    # 同一标的日被拆成多个文件时仍需区分 shard。
    parts.append(f"file={path.name}")
    return "|".join(parts)


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """向量化 SplitMix64，用作稳定的无符号优先级哈希。"""
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    with np.errstate(over="ignore"):
        values = (values + np.uint64(0x9E3779B97F4A7C15)) & mask
        values = (
            (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        ) & mask
        values = (
            (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        ) & mask
    return values ^ (values >> np.uint64(31))


def _row_priorities(
    path: Path,
    frame: pl.DataFrame,
    *,
    field_name: str,
    seed: int,
) -> np.ndarray:
    """为 shard 内每行生成与路径输入顺序无关的 deterministic priority。"""
    identity = _stable_shard_identity(path, frame)
    digest = hashlib.blake2b(
        f"{seed}|{field_name}|{identity}".encode(), digest_size=8
    ).digest()
    base = np.uint64(int.from_bytes(digest, "little"))
    if "event_idx" in frame.columns:
        row_id = frame["event_idx"].cast(pl.UInt64).to_numpy()
    else:
        row_id = np.arange(frame.height, dtype=np.uint64)
    # row ordinal 作为防御性 tie-break；规范 shard 中 event_idx 本应唯一。
    ordinal = np.arange(frame.height, dtype=np.uint64)
    with np.errstate(over="ignore"):
        key = row_id ^ (ordinal * np.uint64(0xD6E8FEB86659FD93)) ^ base
    return _splitmix64(key)


def _effective_quantile_edges(
    values: np.ndarray,
    n_bins: int,
    *,
    two_sided: bool,
) -> tuple[float, ...]:
    """计算最多 ``n_bins-1`` 个真实边界，不用 linspace 伪造重复分位点。"""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or n_bins <= 1:
        return ()

    lo_quantile = 0.01 if two_sided else 0.0
    lo = float(np.quantile(finite, lo_quantile))
    hi = float(np.quantile(finite, 0.99))
    if hi <= lo:
        return ()
    clipped = np.clip(finite, lo, hi)
    unique_values = np.unique(clipped)
    if unique_values.size <= 1:
        return ()

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    candidates = np.quantile(clipped, quantiles)
    edges: list[float] = []
    for candidate in candidates.tolist():
        edge = float(candidate)
        if edge <= unique_values[0]:
            edge = float((unique_values[0] + unique_values[1]) / 2.0)
        elif edge >= unique_values[-1]:
            edge = float((unique_values[-2] + unique_values[-1]) / 2.0)
        if unique_values[0] < edge < unique_values[-1]:
            edges.append(edge)
    return tuple(float(value) for value in np.unique(np.asarray(edges)))


def _is_two_sided(spec: FieldSpec) -> bool:
    """基于稳定源字段名选择双尾缩尾。"""
    return any(marker in spec.source for marker in _TWO_SIDED_SOURCE_MARKERS)


def fit_vocab_v2(
    paths: Sequence[Path],
    *,
    field_specs: tuple[FieldSpec, ...],
    max_samples_per_field: int = 5_000_000,
    fit_dates: Sequence[str] = (),
    seed: int = 0,
    strata_columns: Sequence[str] = DEFAULT_STRATA_COLUMNS,
    categorical_values: Mapping[str, Sequence[str]] | None = None,
    normalizer_clip: float = 5.0,
    schema_version: str = "cn_l2_v2",
) -> VocabV2:
    """
    遍历完整训练流并拟合严格版本化的 v2 词表。

    Parameters
    ----------
    paths
        仅属于训练窗口的 parquet shard。函数内部排序，调用顺序不影响结果。
    field_specs
        冻结字段声明；每个字段可拥有不同的请求 bin 数。
    max_samples_per_field
        每字段 priority reservoir 的总容量。
    fit_dates
        显式记录进 artifact 的训练日期；为空时从输入 ``date`` 列收集。
    seed
        稳定 priority hash 的 seed。
    strata_columns
        确定性分层维度。缺失列作为显式常量维度，而非静默删除。
    categorical_values
        可选固定类别表。未提供的类别字段从完整训练流收集并排序。
    normalizer_clip
        连续双通道标准化后的绝对截断值。
    schema_version
        输入 schema 版本，写入 v2 artifact。

    Returns
    -------
    VocabV2
        含实际有效 bins、精确 occupancy、缺失率和 normalizer 的冻结词表。
    """
    validate_field_specs(field_specs)
    if max_samples_per_field < 1:
        msg = "max_samples_per_field must be positive"
        raise ValueError(msg)
    if normalizer_clip <= 0:
        msg = "normalizer_clip must be positive"
        raise ValueError(msg)

    ordered_paths = tuple(sorted((Path(path) for path in paths), key=lambda p: str(p)))
    if not ordered_paths:
        msg = "paths must not be empty"
        raise ValueError(msg)
    strata_tuple = tuple(str(value) for value in strata_columns)
    numeric_specs = tuple(
        spec for spec in field_specs if spec.kind in {"ordinal", "continuous"}
    )
    binned_specs = tuple(spec for spec in numeric_specs if spec.is_binned)
    categorical_specs = tuple(
        spec for spec in field_specs if spec.kind in {"categorical", "context"}
    )
    stats = {spec.name: _RunningFieldStats() for spec in numeric_specs}
    observed_categories: dict[str, set[str]] = {
        spec.name: set() for spec in categorical_specs
    }
    category_counts: dict[str, dict[str, int]] = {
        spec.name: {} for spec in categorical_specs
    }
    category_missing_counts: dict[str, int] = dict.fromkeys(
        (spec.name for spec in categorical_specs), 0
    )
    observed_dates: set[str] = set()
    total_rows = 0

    # Pass 1: 全流统计、类别发现和分层样本计数。
    for path in ordered_paths:
        frame = _prepare_frame(pl.read_parquet(path), field_specs)
        total_rows += frame.height
        if "date" in frame.columns:
            observed_dates.update(str(value) for value in frame["date"].drop_nulls())
        strata = _strata_labels(frame, strata_tuple)
        for spec in numeric_specs:
            applicable = _applicable_mask(frame, spec)
            stats[spec.name].update(
                frame[spec.source].cast(pl.Float64).to_numpy()[applicable],
                strata[applicable],
            )
        for spec in categorical_specs:
            applicable = _applicable_mask(frame, spec)
            selected = frame[spec.source].to_numpy()[applicable]
            missing = np.fromiter(
                (
                    value is None
                    or (
                        isinstance(value, (float, np.floating))
                        and not np.isfinite(value)
                    )
                    for value in selected
                ),
                dtype=bool,
                count=selected.size,
            )
            category_missing_counts[spec.name] += int(missing.sum())
            clean = selected[~missing].astype(str)
            values, counts = np.unique(clean, return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist(), strict=True):
                category = str(value)
                observed_categories[spec.name].add(category)
                category_counts[spec.name][category] = category_counts[spec.name].get(
                    category, 0
                ) + int(count)

    quotas = {
        spec.name: _allocate_quotas(
            stats[spec.name].strata_counts, max_samples_per_field
        )
        for spec in binned_specs
    }
    reservoirs = {
        spec.name: {
            stratum: _PriorityReservoir(capacity=capacity)
            for stratum, capacity in quotas[spec.name].items()
        }
        for spec in binned_specs
    }

    # Pass 2: 对所有 shard 生成稳定优先级；无“预算满后跳过后续文件”的分支。
    for path in ordered_paths:
        frame = _prepare_frame(pl.read_parquet(path), field_specs)
        strata = _strata_labels(frame, strata_tuple)
        for spec in binned_specs:
            values = frame[spec.source].cast(pl.Float64).to_numpy()
            finite = np.isfinite(values) & _applicable_mask(frame, spec)
            priorities = _row_priorities(path, frame, field_name=spec.name, seed=seed)
            for stratum in np.unique(strata[finite]).tolist():
                key = str(stratum)
                mask = finite & (strata == stratum)
                reservoirs[spec.name][key].add(values[mask], priorities[mask])

    edges: dict[str, tuple[float, ...]] = {
        spec.name: () for spec in numeric_specs if not spec.is_binned
    }
    sample_counts: dict[str, int] = {
        spec.name: 0 for spec in numeric_specs if not spec.is_binned
    }
    for spec in binned_specs:
        field_reservoirs = reservoirs[spec.name].values()
        samples = [reservoir.values for reservoir in field_reservoirs]
        pooled = np.concatenate(samples) if samples else np.empty(0, dtype=np.float64)
        edges[spec.name] = _effective_quantile_edges(
            pooled, int(spec.n_bins or 1), two_sided=_is_two_sided(spec)
        )
        sample_counts[spec.name] = int(pooled.size)

    # Pass 3: 在冻结边界上统计完整训练流的精确 occupancy。
    occupancy = {
        spec.name: np.zeros(len(edges[spec.name]) + 1, dtype=np.int64)
        for spec in numeric_specs
    }
    for path in ordered_paths:
        frame = _prepare_frame(pl.read_parquet(path), field_specs)
        for spec in numeric_specs:
            values = frame[spec.source].cast(pl.Float64).to_numpy()
            valid = np.isfinite(values) & _applicable_mask(frame, spec)
            finite_values = values[valid]
            if finite_values.size == 0:
                continue
            bins = np.digitize(
                finite_values,
                np.asarray(edges[spec.name], dtype=np.float64),
                right=False,
            )
            occupancy[spec.name] += np.bincount(
                bins, minlength=occupancy[spec.name].size
            )

    binned_vocab: dict[str, BinnedFieldVocab] = {}
    for spec in numeric_specs:
        field_stats = stats[spec.name]
        binned_vocab[spec.name] = BinnedFieldVocab(
            requested_n_bins=int(spec.n_bins or 1),
            edges=edges[spec.name],
            occupancy=tuple(int(value) for value in occupancy[spec.name]),
            transform=(
                "log_pretransformed" if spec.source.startswith("log_") else "identity"
            ),
            normalizer=ContinuousNormalizer(
                mean=field_stats.mean,
                std=field_stats.std,
                clip=float(normalizer_clip),
                count=field_stats.count,
            ),
            min_value=field_stats.min_value,
            max_value=field_stats.max_value,
            n_observed=field_stats.count,
            n_missing=field_stats.missing,
        )
        logger.info(
            "fitted v2 field %s: %d/%d bins, %d sampled, %d observed",
            spec.name,
            binned_vocab[spec.name].actual_n_bins,
            spec.n_bins,
            sample_counts[spec.name],
            field_stats.count,
        )

    fixed = categorical_values or {}
    categorical_vocab = {
        spec.name: tuple(
            str(value)
            for value in (
                fixed[spec.name]
                if spec.name in fixed
                else sorted(observed_categories[spec.name])
            )
        )
        for spec in categorical_specs
    }
    categorical_occupancy = {
        name: tuple(category_counts[name].get(value, 0) for value in values)
        for name, values in categorical_vocab.items()
    }
    categorical_unknown_counts = {
        name: sum(
            count
            for value, count in category_counts[name].items()
            if value not in set(categorical_vocab[name])
        )
        for name in categorical_vocab
    }
    if not observed_dates:
        msg = "v2 vocab fitting requires a non-null date column for leakage auditing"
        raise ValueError(msg)
    declared_fit_dates = {str(value) for value in fit_dates}
    if declared_fit_dates and declared_fit_dates != observed_dates:
        msg = (
            "fit_dates must exactly match dates observed in the fitted parquet shards; "
            f"declared_only={sorted(declared_fit_dates - observed_dates)}, "
            f"observed_only={sorted(observed_dates - declared_fit_dates)}"
        )
        raise ValueError(msg)
    effective_fit_dates = tuple(sorted(observed_dates))
    return VocabV2(
        field_specs=field_specs,
        categorical=categorical_vocab,
        categorical_occupancy=categorical_occupancy,
        categorical_unknown_counts=categorical_unknown_counts,
        categorical_missing_counts=category_missing_counts,
        binned=binned_vocab,
        schema_version=schema_version,
        fit_dates=effective_fit_dates,
        sampling={
            "method": "deterministic_stratified_priority_reservoir",
            "seed": seed,
            "max_samples_per_field": max_samples_per_field,
            "strata_columns": list(strata_tuple),
            "n_shards": len(ordered_paths),
            "n_rows": total_rows,
            "sample_counts": sample_counts,
            "strata_counts": {
                spec.name: len(stats[spec.name].strata_counts) for spec in numeric_specs
            },
        },
    )
