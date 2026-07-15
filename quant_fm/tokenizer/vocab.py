"""
字段级词表：固定类别 + 冻结的连续分箱边界。

每个字段拥有独立的 id 空间。``PAD`` 在各字段中占用 id 0，以便批处理时填充序列。
不存在巨大的组合词表；模型对各字段分别嵌入，并用独立预测头。

"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from quant_fm.schema.cn_l2_v1 import (
    BOARDS,
    EVENT_SOURCES,
    EVT_TYPES,
    ORDER_TYPES,
    SESSIONS,
    SIDES,
)

PAD_ID = 0
N_SPECIAL = 1  # 预留的前导 id（当前仅 PAD）

# 需要拟合分箱边界的连续字段（见 tokenizer.fit_bins）。
CONTINUOUS_FIELDS = ("price_rel", "log_volume", "log_delta_t")

# 取值空间固定、确定性的类别字段。
CATEGORICAL_FIELDS: dict[str, tuple[str, ...]] = {
    "evt_type": EVT_TYPES,
    "side": SIDES,
    "session": SESSIONS,
    "board": BOARDS,
    "order_type": ORDER_TYPES,
    "event_source": EVENT_SOURCES,
}

# 各类别字段对应的规范列。
CATEGORICAL_SOURCE = {
    "evt_type": "evt_type",
    "side": "side",
    "session": "session",
    "board": "board",
    "order_type": "order_type",
    "event_source": "event_source",
}

_UNKNOWN_FALLBACK = {"side": "N"}


@dataclass(slots=True)
class Vocab:
    """可序列化的字段级词表。"""

    n_bins: int = 32
    categorical: dict[str, tuple[str, ...]] = field(default_factory=dict)
    edges: dict[str, list[float]] = field(default_factory=dict)
    schema_version: str = "cn_l2_v1"
    fit_dates: tuple[str, ...] = ()

    # -- 规模 -----------------------------------------------------------
    def size(self, field_name: str) -> int:
        """返回 ``field_name`` 的 id 空间大小（含 PAD）。"""
        if field_name in self.categorical:
            return N_SPECIAL + len(self.categorical[field_name])
        if field_name in self.edges:
            return N_SPECIAL + self.n_bins
        msg = f"unknown field {field_name!r}"
        raise KeyError(msg)

    def field_sizes(self) -> dict[str, int]:
        """返回各字段的 id 空间大小。"""
        names = list(self.categorical) + list(self.edges)
        return {name: self.size(name) for name in names}

    # -- 编码 --------------------------------------------------------
    def encode_categorical(self, field_name: str, values: np.ndarray) -> np.ndarray:
        """将类别字段的字符串取值映射为 token id。"""
        vocab = self.categorical[field_name]
        index = {v: i for i, v in enumerate(vocab)}
        fallback = _UNKNOWN_FALLBACK.get(field_name, "UNKNOWN")
        default = index.get(fallback, 0)
        out = np.fromiter(
            (index.get(str(v), default) for v in values),
            dtype=np.int64,
            count=len(values),
        )
        return out + N_SPECIAL

    def encode_binned(self, field_name: str, values: np.ndarray) -> np.ndarray:
        """用冻结边界将连续值分箱为 token id。"""
        edges = np.asarray(self.edges[field_name], dtype=np.float64)
        if edges.size == 0:
            # 未拟合字段：全部落入 bin 0。
            return np.full(len(values), N_SPECIAL, dtype=np.int64)
        clean = np.nan_to_num(values, nan=0.0, posinf=edges[-1], neginf=edges[0])
        bins = np.digitize(clean, edges, right=False)
        bins = np.clip(bins, 0, self.n_bins - 1)
        return bins.astype(np.int64) + N_SPECIAL

    # -- 读写 --------------------------------------------------------------
    def to_json(self) -> str:
        """序列化为稳定的 JSON 字符串。"""
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "n_bins": self.n_bins,
                "fit_dates": list(self.fit_dates),
                "categorical": {k: list(v) for k, v in self.categorical.items()},
                "edges": {k: list(v) for k, v in self.edges.items()},
            },
            indent=2,
            sort_keys=True,
        )

    def save(self, path: Path) -> None:
        """将 ``vocab.json`` 写入 ``path``。"""
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Vocab:
        """从 ``vocab.json`` 加载词表。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            n_bins=int(data["n_bins"]),
            categorical={k: tuple(v) for k, v in data["categorical"].items()},
            edges={k: list(v) for k, v in data["edges"].items()},
            schema_version=data.get("schema_version", "cn_l2_v1"),
            fit_dates=tuple(data.get("fit_dates", [])),
        )


def default_vocab(n_bins: int = 32) -> Vocab:
    """返回含固定类别、空（未拟合）边界的词表。"""
    return Vocab(
        n_bins=n_bins,
        categorical=dict(CATEGORICAL_FIELDS),
        edges={f: [] for f in CONTINUOUS_FIELDS},
    )
