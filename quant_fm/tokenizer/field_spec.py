"""
Tokenizer v2 的冻结字段声明。

v1 通过几个全局 tuple 隐式约定输入字段。v2 将字段的来源、语义、分箱数和
输入/目标用途写进 artifact，避免训练与推理因字段顺序或默认值不同而静默漂移。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

FieldKind = Literal["categorical", "ordinal", "continuous", "context"]

_FIELD_KINDS = frozenset({"categorical", "ordinal", "continuous", "context"})


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """
    描述一个 v2 tokenizer 字段。

    Parameters
    ----------
    name
        artifact 内的稳定字段名。token 列由它派生，而不是由源列名隐式决定。
    source
        输入事件表中的源列。
    kind
        字段语义。``ordinal`` 必须分箱；``continuous`` 可仅保留标量，也可同时
        分箱；``categorical``/``context`` 使用冻结类别表。
    n_bins
        请求的最大数值 bin 数。重复分位点会合并，因此实际 bin 数可以更少。
    applicable_events
        此字段有意义的事件类型。空 tuple 表示对所有事件适用。
    is_input
        是否可进入模型输入。
    is_target
        是否作为训练目标。
    missing_token
        是否允许用专用 ``NA`` token 表示缺失。
    """

    name: str
    source: str
    kind: FieldKind
    n_bins: int | None = None
    applicable_events: tuple[str, ...] = ()
    is_input: bool = True
    is_target: bool = False
    missing_token: bool = True

    def __post_init__(self) -> None:
        """拒绝无法稳定序列化或含糊的字段声明。"""
        if not self.name or not self.source:
            msg = "FieldSpec name and source must be non-empty"
            raise ValueError(msg)
        if self.kind not in _FIELD_KINDS:
            msg = f"unsupported field kind: {self.kind!r}"
            raise ValueError(msg)
        if self.kind == "ordinal" and self.n_bins is None:
            msg = f"ordinal field {self.name!r} requires n_bins"
            raise ValueError(msg)
        if self.n_bins is not None and self.n_bins < 1:
            msg = f"n_bins must be positive for {self.name!r}"
            raise ValueError(msg)
        if self.kind in {"categorical", "context"} and self.n_bins is not None:
            msg = f"categorical/context field {self.name!r} cannot define n_bins"
            raise ValueError(msg)
        if not self.is_input and not self.is_target:
            msg = f"field {self.name!r} is neither input nor target"
            raise ValueError(msg)
        if len(set(self.applicable_events)) != len(self.applicable_events):
            msg = f"duplicate applicable_events in field {self.name!r}"
            raise ValueError(msg)

    @property
    def is_binned(self) -> bool:
        """字段是否同时产生有序 bin token。"""
        return self.kind == "ordinal" or (
            self.kind == "continuous" and self.n_bins is not None
        )

    @property
    def token_column(self) -> str | None:
        """返回约定的 token 输出列名。"""
        if self.kind in {"categorical", "context"}:
            return f"tok_{self.name}"
        if self.is_binned:
            return f"tok_{self.name}_bin"
        return None

    @property
    def value_column(self) -> str | None:
        """返回约定的连续标量输出列名。"""
        if self.kind in {"ordinal", "continuous"}:
            return f"val_{self.name}"
        return None

    def to_dict(self) -> dict[str, Any]:
        """转换成 JSON 可序列化字典。"""
        data = asdict(self)
        data["applicable_events"] = list(self.applicable_events)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldSpec:
        """从 artifact 字典恢复字段声明。"""
        return cls(
            name=str(data["name"]),
            source=str(data["source"]),
            kind=data["kind"],
            n_bins=(None if data.get("n_bins") is None else int(data["n_bins"])),
            applicable_events=tuple(str(v) for v in data.get("applicable_events", [])),
            is_input=bool(data.get("is_input", True)),
            is_target=bool(data.get("is_target", False)),
            missing_token=bool(data.get("missing_token", True)),
        )


def validate_field_specs(field_specs: tuple[FieldSpec, ...]) -> None:
    """校验冻结字段列表中的名字、token 列和标量列均唯一。"""
    if not field_specs:
        msg = "field_specs must not be empty"
        raise ValueError(msg)

    names = [spec.name for spec in field_specs]
    if len(set(names)) != len(names):
        msg = "field_specs contain duplicate names"
        raise ValueError(msg)

    output_columns = [
        column
        for spec in field_specs
        for column in (spec.token_column, spec.value_column)
        if column is not None
    ]
    if len(set(output_columns)) != len(output_columns):
        msg = "field_specs generate duplicate output columns"
        raise ValueError(msg)


# 第一阶段保持 v1 可派生字段，去掉 event_source、伪 order_type 和逐事件 board。
# 真正盘口字段由 cn_l2_v2 生成后，以同一 FieldSpec 接口追加。
DEFAULT_FIELD_SPECS_V2: tuple[FieldSpec, ...] = (
    FieldSpec("evt_type", "evt_type", "categorical", is_target=True),
    FieldSpec("side", "side", "categorical", is_target=True),
    FieldSpec("session", "session", "categorical"),
    FieldSpec("price", "price_rel", "ordinal", n_bins=32, is_target=True),
    FieldSpec("volume", "log_volume", "ordinal", n_bins=32, is_target=True),
    FieldSpec("delta_t", "log_delta_t", "ordinal", n_bins=32, is_target=True),
)

# 第一版紧凑因果盘口：所有列名明确标注 pre/post，数值字段保留 bin+scalar。
BOOK_FIELD_SPECS_V2: tuple[FieldSpec, ...] = (
    FieldSpec("book_valid_post", "book_valid_post", "categorical"),
    FieldSpec("spread_ticks_post", "spread_ticks_post", "ordinal", n_bins=16),
    FieldSpec(
        "microprice_delta_ticks_post",
        "microprice_delta_ticks_post",
        "ordinal",
        n_bins=21,
    ),
    FieldSpec("imbalance_l1_post", "imbalance_l1_post", "ordinal", n_bins=21),
    FieldSpec("imbalance_l5_post", "imbalance_l5_post", "ordinal", n_bins=21),
    FieldSpec("imbalance_l10_post", "imbalance_l10_post", "ordinal", n_bins=21),
    FieldSpec(
        "log_bid_depth_l5_post",
        "log_bid_depth_l5_post",
        "ordinal",
        n_bins=32,
    ),
    FieldSpec(
        "log_ask_depth_l5_post",
        "log_ask_depth_l5_post",
        "ordinal",
        n_bins=32,
    ),
    FieldSpec(
        "event_price_distance_ticks_pre",
        "event_price_distance_ticks_pre",
        "ordinal",
        n_bins=32,
    ),
)

FULL_FIELD_SPECS_V2: tuple[FieldSpec, ...] = (
    *DEFAULT_FIELD_SPECS_V2,
    *BOOK_FIELD_SPECS_V2,
)
