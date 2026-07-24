"""
Tokenizer v2 的严格版本化字段词表。

本模块刻意不复用或修改 :mod:`quant_fm.tokenizer.vocab` 中 v1 的特殊 token
常量。旧 checkpoint 仍看到 ``PAD=0, N_SPECIAL=1``；只有显式加载
``VocabV2`` 的路径才使用六个 v2 特殊 token。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from quant_fm.schema.cn_l2_v1 import EVT_TYPES, SESSIONS, SIDES
from quant_fm.tokenizer.field_spec import (
    DEFAULT_FIELD_SPECS_V2,
    FieldSpec,
    validate_field_specs,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# v2 独立 id 空间。不要把这些值回写到 tokenizer.vocab。
PAD_ID = 0
UNK_ID = 1
NA_ID = 2
BOS_ID = 3
EOS_ID = 4
SESSION_BREAK_ID = 5
N_SPECIAL = 6

V2_PAD_ID = PAD_ID
V2_UNK_ID = UNK_ID
V2_NA_ID = NA_ID
V2_BOS_ID = BOS_ID
V2_EOS_ID = EOS_ID
V2_SESSION_BREAK_ID = SESSION_BREAK_ID
V2_N_SPECIAL = N_SPECIAL

SPECIAL_IDS: Mapping[str, int] = MappingProxyType(
    {
        "PAD": PAD_ID,
        "UNK": UNK_ID,
        "NA": NA_ID,
        "BOS": BOS_ID,
        "EOS": EOS_ID,
        "SESSION_BREAK": SESSION_BREAK_ID,
    }
)


@dataclass(frozen=True, slots=True)
class ContinuousNormalizer:
    """冻结的训练期连续值标准化统计量。"""

    mean: float = 0.0
    std: float = 1.0
    clip: float = 5.0
    count: int = 0

    def __post_init__(self) -> None:
        """校验统计量可安全用于推理。"""
        if not np.isfinite(self.mean):
            msg = "normalizer mean must be finite"
            raise ValueError(msg)
        if not np.isfinite(self.std) or self.std <= 0:
            msg = "normalizer std must be finite and positive"
            raise ValueError(msg)
        if not np.isfinite(self.clip) or self.clip <= 0:
            msg = "normalizer clip must be finite and positive"
            raise ValueError(msg)
        if self.count < 0:
            msg = "normalizer count must be non-negative"
            raise ValueError(msg)

    def encode(self, values: np.ndarray | Sequence[float]) -> np.ndarray:
        """标准化数值；缺失位置输出 0，由独立 ``NA`` token 表达缺失。"""
        array = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(array)
        out = np.zeros(array.shape, dtype=np.float32)
        if finite.any():
            normalized = (array[finite] - self.mean) / self.std
            out[finite] = np.clip(normalized, -self.clip, self.clip).astype(np.float32)
        return out

    def to_dict(self) -> dict[str, float | int]:
        """转换成 JSON 可序列化字典。"""
        return {
            "mean": self.mean,
            "std": self.std,
            "clip": self.clip,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContinuousNormalizer:
        """从 artifact 字典恢复。"""
        return cls(
            mean=float(data.get("mean", 0.0)),
            std=float(data.get("std", 1.0)),
            clip=float(data.get("clip", 5.0)),
            count=int(data.get("count", 0)),
        )


@dataclass(frozen=True, slots=True)
class BinnedFieldVocab:
    """一个连续/有序字段的实际有效词表。"""

    requested_n_bins: int
    edges: tuple[float, ...] = ()
    occupancy: tuple[int, ...] = (0,)
    special_ids: Mapping[str, int] = field(default_factory=lambda: dict(SPECIAL_IDS))
    transform: str = "identity"
    normalizer: ContinuousNormalizer = field(default_factory=ContinuousNormalizer)
    min_value: float | None = None
    max_value: float | None = None
    n_observed: int = 0
    n_missing: int = 0

    def __post_init__(self) -> None:
        """校验边界严格递增，且 occupancy 与实际 bin 数一致。"""
        if self.requested_n_bins < 1:
            msg = "requested_n_bins must be positive"
            raise ValueError(msg)
        edge_array = np.asarray(self.edges, dtype=np.float64)
        if edge_array.size and (
            not np.isfinite(edge_array).all() or np.any(np.diff(edge_array) <= 0)
        ):
            msg = "bin edges must be finite and strictly increasing"
            raise ValueError(msg)
        if self.actual_n_bins > self.requested_n_bins:
            msg = "actual bins cannot exceed requested bins"
            raise ValueError(msg)
        if len(self.occupancy) != self.actual_n_bins:
            msg = "occupancy length must equal actual_n_bins"
            raise ValueError(msg)
        if any(value < 0 for value in self.occupancy):
            msg = "occupancy counts must be non-negative"
            raise ValueError(msg)
        if dict(self.special_ids) != dict(SPECIAL_IDS):
            msg = "v2 special token ids do not match the frozen contract"
            raise ValueError(msg)
        if self.n_observed < 0 or self.n_missing < 0:
            msg = "observed/missing counts must be non-negative"
            raise ValueError(msg)
        if sum(self.occupancy) != self.n_observed:
            msg = "occupancy must sum to n_observed"
            raise ValueError(msg)
        if (self.min_value is None) != (self.max_value is None):
            msg = "min_value and max_value must either both exist or both be None"
            raise ValueError(msg)
        if self.min_value is not None:
            if not np.isfinite(self.min_value) or not np.isfinite(self.max_value):
                msg = "min/max values must be finite"
                raise ValueError(msg)
            if self.min_value > self.max_value:
                msg = "min_value cannot exceed max_value"
                raise ValueError(msg)

    @property
    def actual_n_bins(self) -> int:
        """实际数值 bin 数；重复分位点合并后可能小于请求值。"""
        return len(self.edges) + 1

    @property
    def size(self) -> int:
        """字段 embedding 的总大小（特殊 token + 实际数值 bin）。"""
        return N_SPECIAL + self.actual_n_bins

    @property
    def missing_rate(self) -> float:
        """训练流中的非有限值占比。"""
        total = self.n_observed + self.n_missing
        return self.n_missing / total if total else 0.0

    def encode(self, values: np.ndarray | Sequence[float]) -> np.ndarray:
        """分箱连续值，非有限值始终映射到专用 ``NA_ID``。"""
        array = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(array)
        out = np.full(array.shape, NA_ID, dtype=np.int64)
        if finite.any():
            bins = np.digitize(
                array[finite], np.asarray(self.edges, dtype=np.float64), right=False
            )
            out[finite] = bins.astype(np.int64) + N_SPECIAL
        return out

    def encode_scalar(self, values: np.ndarray | Sequence[float]) -> np.ndarray:
        """输出冻结标准化后的连续通道。"""
        return self.normalizer.encode(values)

    def to_dict(self) -> dict[str, Any]:
        """转换成 JSON 可序列化字典。"""
        return {
            "requested_n_bins": self.requested_n_bins,
            "actual_n_bins": self.actual_n_bins,
            "edges": list(self.edges),
            "occupancy": list(self.occupancy),
            "special_ids": dict(self.special_ids),
            "transform": self.transform,
            "normalizer": self.normalizer.to_dict(),
            "min_value": self.min_value,
            "max_value": self.max_value,
            "n_observed": self.n_observed,
            "n_missing": self.n_missing,
            "missing_rate": self.missing_rate,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BinnedFieldVocab:
        """从 artifact 字典恢复并重新验证。"""
        vocab = cls(
            requested_n_bins=int(data["requested_n_bins"]),
            edges=tuple(float(v) for v in data.get("edges", [])),
            occupancy=tuple(int(v) for v in data.get("occupancy", [0])),
            special_ids={str(k): int(v) for k, v in data["special_ids"].items()},
            transform=str(data.get("transform", "identity")),
            normalizer=ContinuousNormalizer.from_dict(data.get("normalizer", {})),
            min_value=(
                None if data.get("min_value") is None else float(data["min_value"])
            ),
            max_value=(
                None if data.get("max_value") is None else float(data["max_value"])
            ),
            n_observed=int(data.get("n_observed", 0)),
            n_missing=int(data.get("n_missing", 0)),
        )
        declared = data.get("actual_n_bins")
        if declared is not None and int(declared) != vocab.actual_n_bins:
            msg = "declared actual_n_bins does not match edges"
            raise ValueError(msg)
        return vocab


def _is_missing_category(value: object) -> bool:
    """判断类别值是否为真实缺失，而不是未知字符串类别。"""
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return not np.isfinite(value)
    return False


@dataclass(slots=True)
class VocabV2:
    """字段级 v2 词表；与 v1 artifact 严格隔离。"""

    VOCAB_VERSION: ClassVar[str] = "2.0"

    field_specs: tuple[FieldSpec, ...]
    categorical: dict[str, tuple[str, ...]] = field(default_factory=dict)
    categorical_occupancy: dict[str, tuple[int, ...]] = field(default_factory=dict)
    categorical_unknown_counts: dict[str, int] = field(default_factory=dict)
    categorical_missing_counts: dict[str, int] = field(default_factory=dict)
    binned: dict[str, BinnedFieldVocab] = field(default_factory=dict)
    schema_version: str = "cn_l2_v2"
    fit_dates: tuple[str, ...] = ()
    sampling: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验字段顺序和每类词表均与 FieldSpec 一致。"""
        validate_field_specs(self.field_specs)
        known = {spec.name: spec for spec in self.field_specs}
        unexpected = (set(self.categorical) | set(self.binned)) - set(known)
        if unexpected:
            msg = f"vocab contains fields absent from FieldSpec: {sorted(unexpected)}"
            raise ValueError(msg)

        for spec in self.field_specs:
            if spec.kind in {"categorical", "context"}:
                if spec.name not in self.categorical:
                    msg = f"missing categorical vocab for {spec.name!r}"
                    raise ValueError(msg)
                categories = self.categorical[spec.name]
                if len(set(categories)) != len(categories):
                    msg = f"duplicate categories for {spec.name!r}"
                    raise ValueError(msg)
                occupancy = self.categorical_occupancy.get(
                    spec.name, tuple(0 for _ in categories)
                )
                if len(occupancy) != len(categories):
                    msg = f"categorical occupancy length mismatch for {spec.name!r}"
                    raise ValueError(msg)
                if any(value < 0 for value in occupancy):
                    msg = f"negative categorical occupancy for {spec.name!r}"
                    raise ValueError(msg)
                if self.categorical_unknown_counts.get(spec.name, 0) < 0:
                    msg = f"negative unknown count for {spec.name!r}"
                    raise ValueError(msg)
                if self.categorical_missing_counts.get(spec.name, 0) < 0:
                    msg = f"negative missing count for {spec.name!r}"
                    raise ValueError(msg)
            elif (
                spec.kind in {"ordinal", "continuous"} and spec.name not in self.binned
            ):
                msg = f"missing numeric vocab for {spec.name!r}"
                raise ValueError(msg)

    def spec(self, field_name: str) -> FieldSpec:
        """返回字段声明。"""
        for spec in self.field_specs:
            if spec.name == field_name:
                return spec
        msg = f"unknown field {field_name!r}"
        raise KeyError(msg)

    def size(self, field_name: str) -> int:
        """返回一个 token 字段的实际 id 空间大小。"""
        if field_name in self.categorical:
            return N_SPECIAL + len(self.categorical[field_name])
        if field_name in self.binned:
            return self.binned[field_name].size
        msg = f"field {field_name!r} has no token channel"
        raise KeyError(msg)

    def logical_field_sizes(self) -> dict[str, int]:
        """以 FieldSpec 逻辑名为 key 返回 token id 空间大小。"""
        return {
            spec.name: self.size(spec.name)
            for spec in self.field_specs
            if spec.token_column is not None
        }

    def token_field_sizes(self) -> dict[str, int]:
        """以 token parquet 列名为 key 返回模型 embedding 大小。"""
        return {
            str(spec.token_column): self.size(spec.name)
            for spec in self.field_specs
            if spec.token_column is not None
        }

    def field_sizes(self) -> dict[str, int]:
        """返回训练侧可直接使用的 ``tok_*`` 列名到 embedding 大小映射。"""
        return self.token_field_sizes()

    @property
    def input_token_fields(self) -> tuple[str, ...]:
        """按冻结 FieldSpec 顺序返回允许进入模型的 token 列。"""
        return tuple(
            str(spec.token_column)
            for spec in self.field_specs
            if spec.is_input and spec.token_column is not None
        )

    @property
    def target_token_fields(self) -> tuple[str, ...]:
        """按冻结 FieldSpec 顺序返回 next-event token 目标列。"""
        return tuple(
            str(spec.token_column)
            for spec in self.field_specs
            if spec.is_target and spec.token_column is not None
        )

    @property
    def input_value_fields(self) -> tuple[str, ...]:
        """返回允许进入模型的连续 ``val_*`` 列。"""
        return tuple(
            str(spec.value_column)
            for spec in self.field_specs
            if spec.is_input and spec.value_column is not None
        )

    @property
    def target_value_fields(self) -> tuple[str, ...]:
        """返回用作连续/辅助目标的 ``val_*`` 列。"""
        return tuple(
            str(spec.value_column)
            for spec in self.field_specs
            if spec.is_target and spec.value_column is not None
        )

    @property
    def token_to_value_fields(self) -> dict[str, str]:
        """返回有序双通道的 ``tok_*_bin -> val_*`` 映射。"""
        return {
            str(spec.token_column): str(spec.value_column)
            for spec in self.field_specs
            if spec.token_column is not None and spec.value_column is not None
        }

    def _logical_name(self, field_name: str) -> str:
        """将逻辑字段名或 token parquet 列名统一解析为逻辑名。"""
        if any(spec.name == field_name for spec in self.field_specs):
            return field_name
        for spec in self.field_specs:
            if spec.token_column == field_name:
                return spec.name
        msg = f"unknown logical/token field {field_name!r}"
        raise KeyError(msg)

    def train_entropy(self, field_name: str, *, include_missing: bool = False) -> float:
        """
        返回完整训练流上的离散 token 熵（自然对数，单位 nats）。

        ``field_name`` 可传 FieldSpec 逻辑名或实际 parquet token 列名。默认排除
        Loss applicability mask 会忽略的 ``NA``，但类别 ``UNK`` 是有效预测类，
        因而始终计入。
        """
        logical_name = self._logical_name(field_name)
        if logical_name in self.categorical:
            counts = list(
                self.categorical_occupancy.get(
                    logical_name,
                    tuple(0 for _ in self.categorical[logical_name]),
                )
            )
            counts.append(self.categorical_unknown_counts.get(logical_name, 0))
            if include_missing:
                counts.append(self.categorical_missing_counts.get(logical_name, 0))
        elif logical_name in self.binned:
            numeric_vocab = self.binned[logical_name]
            counts = list(numeric_vocab.occupancy)
            if include_missing:
                counts.append(numeric_vocab.n_missing)
        else:
            msg = f"field {field_name!r} has no discrete token distribution"
            raise KeyError(msg)

        count_array = np.asarray(counts, dtype=np.float64)
        count_array = count_array[count_array > 0]
        if count_array.size <= 1:
            return 0.0
        probabilities = count_array / count_array.sum()
        return float(-np.sum(probabilities * np.log(probabilities)))

    def encode_categorical(
        self, field_name: str, values: np.ndarray | Sequence[object]
    ) -> np.ndarray:
        """类别缺失映射 NA，词表外值映射 UNK，已知值位于特殊区之后。"""
        categories = self.categorical[field_name]
        index = {value: i for i, value in enumerate(categories)}
        flat = np.asarray(values, dtype=object)
        out = np.empty(flat.shape, dtype=np.int64)
        for position, value in np.ndenumerate(flat):
            if _is_missing_category(value):
                out[position] = NA_ID
            else:
                category_index = index.get(str(value))
                out[position] = (
                    UNK_ID if category_index is None else N_SPECIAL + category_index
                )
        return out

    def encode_binned(
        self, field_name: str, values: np.ndarray | Sequence[float]
    ) -> np.ndarray:
        """使用该字段实际边界分箱。"""
        return self.binned[field_name].encode(values)

    def encode_scalar(
        self, field_name: str, values: np.ndarray | Sequence[float]
    ) -> np.ndarray:
        """使用训练期冻结统计量生成连续通道。"""
        return self.binned[field_name].encode_scalar(values)

    def to_json(self) -> str:
        """序列化为稳定、可审计的 v2 JSON。"""
        return json.dumps(
            {
                "vocab_version": self.VOCAB_VERSION,
                "schema_version": self.schema_version,
                "special_ids": dict(SPECIAL_IDS),
                "fit_dates": list(self.fit_dates),
                "field_specs": [spec.to_dict() for spec in self.field_specs],
                "categorical": {
                    name: list(values) for name, values in self.categorical.items()
                },
                "categorical_occupancy": {
                    name: list(values)
                    for name, values in self.categorical_occupancy.items()
                },
                "categorical_unknown_counts": self.categorical_unknown_counts,
                "categorical_missing_counts": self.categorical_missing_counts,
                "binned": {
                    name: vocab.to_dict() for name, vocab in self.binned.items()
                },
                "sampling": self.sampling,
            },
            indent=2,
            sort_keys=True,
        )

    def save(self, path: Path) -> None:
        """原子性由调用方管理；本方法只写入一个明确的 v2 artifact。"""
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> VocabV2:
        """加载 v2 artifact，并拒绝 v1 或特殊 token 不一致的文件。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("vocab_version") != cls.VOCAB_VERSION:
            msg = f"expected vocab_version={cls.VOCAB_VERSION!r}"
            raise ValueError(msg)
        if data.get("special_ids") != dict(SPECIAL_IDS):
            msg = "artifact special token ids do not match tokenizer v2"
            raise ValueError(msg)
        return cls(
            field_specs=tuple(
                FieldSpec.from_dict(spec) for spec in data["field_specs"]
            ),
            categorical={
                name: tuple(str(value) for value in values)
                for name, values in data.get("categorical", {}).items()
            },
            categorical_occupancy={
                name: tuple(int(value) for value in values)
                for name, values in data.get("categorical_occupancy", {}).items()
            },
            categorical_unknown_counts={
                str(name): int(value)
                for name, value in data.get("categorical_unknown_counts", {}).items()
            },
            categorical_missing_counts={
                str(name): int(value)
                for name, value in data.get("categorical_missing_counts", {}).items()
            },
            binned={
                name: BinnedFieldVocab.from_dict(value)
                for name, value in data.get("binned", {}).items()
            },
            schema_version=str(data.get("schema_version", "cn_l2_v2")),
            fit_dates=tuple(str(value) for value in data.get("fit_dates", [])),
            sampling=dict(data.get("sampling", {})),
        )


_DEFAULT_CATEGORIES_V2: dict[str, tuple[str, ...]] = {
    "evt_type": EVT_TYPES,
    "side": SIDES,
    "session": SESSIONS,
}


def default_vocab_v2(
    field_specs: tuple[FieldSpec, ...] = DEFAULT_FIELD_SPECS_V2,
    *,
    categorical: Mapping[str, Sequence[str]] | None = None,
) -> VocabV2:
    """构造未拟合的 v2 词表，主要用于 schema/smoke 测试。"""
    categories = dict(_DEFAULT_CATEGORIES_V2)
    if categorical is not None:
        categories.update(
            {
                name: tuple(str(value) for value in values)
                for name, values in categorical.items()
            }
        )

    categorical_vocab = {
        spec.name: tuple(categories.get(spec.name, ()))
        for spec in field_specs
        if spec.kind in {"categorical", "context"}
    }
    binned_vocab = {
        spec.name: BinnedFieldVocab(requested_n_bins=int(spec.n_bins or 1))
        for spec in field_specs
        if spec.kind in {"ordinal", "continuous"}
    }
    return VocabV2(
        field_specs=field_specs,
        categorical=categorical_vocab,
        categorical_occupancy={
            name: tuple(0 for _ in values) for name, values in categorical_vocab.items()
        },
        categorical_unknown_counts=dict.fromkeys(categorical_vocab, 0),
        categorical_missing_counts=dict.fromkeys(categorical_vocab, 0),
        binned=binned_vocab,
    )
