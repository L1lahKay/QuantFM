"""
研究收益区间的显式定义。

``score(T)`` 仅在 T 日收盘后可用，因此正式可交易评估应使用 T+1 或更晚的
价格建仓。本模块集中定义信号日、建仓日、退出日和价格字段，避免这些约定
散落在脚本默认值中。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

PriceField = Literal["open", "close", "vwap"]


@dataclass(frozen=True, slots=True)
class ReturnSpec:
    """从信号日映射到可执行收益区间。"""

    name: str
    entry_day_lag: int
    exit_day_lag: int
    entry_price: PriceField
    exit_price: PriceField

    def validate(self) -> None:
        """拒绝倒置区间和无法区分的零持有期。"""
        if self.entry_day_lag < 0 or self.exit_day_lag < 0:
            msg = "entry/exit day lag must be non-negative"
            raise ValueError(msg)
        if self.exit_day_lag < self.entry_day_lag:
            msg = "exit_day_lag must be >= entry_day_lag"
            raise ValueError(msg)
        if (
            self.exit_day_lag == self.entry_day_lag
            and self.entry_price == self.exit_price
        ):
            msg = "same-day return requires different entry and exit price fields"
            raise ValueError(msg)

    def as_dict(self) -> dict[str, str | int]:
        """返回可 JSON 序列化配置。"""
        return asdict(self)


RETURN_SPECS: dict[str, ReturnSpec] = {
    "close_t_close_t1": ReturnSpec(
        name="close_t_close_t1",
        entry_day_lag=0,
        exit_day_lag=1,
        entry_price="close",
        exit_price="close",
    ),
    "vwap_t_vwap_t1": ReturnSpec(
        name="vwap_t_vwap_t1",
        entry_day_lag=0,
        exit_day_lag=1,
        entry_price="vwap",
        exit_price="vwap",
    ),
    "open_t1_close_t1": ReturnSpec(
        name="open_t1_close_t1",
        entry_day_lag=1,
        exit_day_lag=1,
        entry_price="open",
        exit_price="close",
    ),
    "vwap_t1_vwap_t2": ReturnSpec(
        name="vwap_t1_vwap_t2",
        entry_day_lag=1,
        exit_day_lag=2,
        entry_price="vwap",
        exit_price="vwap",
    ),
}


def get_return_spec(name: str) -> ReturnSpec:
    """按稳定名称加载并校验收益定义。"""
    try:
        spec = RETURN_SPECS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(RETURN_SPECS))
        msg = f"unknown return spec {name!r}; choose one of: {choices}"
        raise ValueError(msg) from exc
    spec.validate()
    return spec
