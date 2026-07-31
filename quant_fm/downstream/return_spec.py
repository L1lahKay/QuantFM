"""
研究收益区间的显式定义。

``score(T)`` 仅在 T 日收盘后可用，因此正式可交易评估应使用 T+1 或更晚的
价格建仓。本模块集中定义信号日、建仓日、退出日和价格字段，避免这些约定
散落在脚本默认值中。
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import date as calendar_date
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from os import PathLike

PriceField = Literal["open", "close", "vwap"]

AFTER_CLOSE_AVAILABILITY = "available_after_signal_date_close"
EXECUTION_CONTRACT_VERSION = "calendar_indexed_execution_v1"
_EXECUTION_CONTRACT_COLUMNS = {
    "execution_contract_version",
    "return_spec",
    "signal_availability",
    "entry_day_lag",
    "exit_day_lag",
    "entry_date",
    "exit_date",
    "entry_price_field",
    "exit_price_field",
    "trading_calendar_sha256",
    "calendar_date_count",
    "signal_calendar_index",
    "entry_calendar_index",
    "exit_calendar_index",
}


def normalise_trading_calendar(values: Iterable[str]) -> list[str]:
    """Validate an explicit, unique, strictly increasing ISO trading calendar."""
    calendar = [str(value) for value in values]
    if not calendar:
        msg = "trading_calendar must not be empty"
        raise ValueError(msg)
    if len(calendar) != len(set(calendar)):
        msg = "trading_calendar contains duplicate dates"
        raise ValueError(msg)
    for value in calendar:
        try:
            calendar_date.fromisoformat(value)
        except ValueError as exc:
            msg = f"trading_calendar contains a non-ISO date: {value!r}"
            raise ValueError(msg) from exc
    if calendar != sorted(calendar):
        msg = "trading_calendar must be strictly increasing"
        raise ValueError(msg)
    return calendar


def trading_calendar_sha256(values: Sequence[str]) -> str:
    """Hash the exact ordered trading-day sequence used for label alignment."""
    calendar = normalise_trading_calendar(values)
    payload = ("\n".join(calendar) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def read_trading_calendar(path: str | PathLike[str]) -> list[str]:
    """Read a newline-delimited calendar, ignoring blank lines and comments."""
    values = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return normalise_trading_calendar(values)


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


def validate_execution_panel_contract(
    panel: pl.DataFrame,
    *,
    trading_calendar: Sequence[str] | None = None,
    require_calendar_verification: bool = False,
    allow_legacy: bool = False,
) -> dict[str, str | int | bool]:
    """
    校验收盘后信号使用的可交易收益契约。

    正式 score 契约规定 ``score(T)`` 在 T 日收盘后才可用，因此建仓日期必须
    严格晚于信号日。旧 panel 只有 ``date/symbol/fwd_ret``，无法证明收益区间
    可执行；只有显式传入 ``allow_legacy=True`` 才能用于诊断。
    """
    missing = sorted(_EXECUTION_CONTRACT_COLUMNS - set(panel.columns))
    if missing:
        if allow_legacy:
            return {
                "verified": False,
                "mode": "legacy_unverified",
                "missing_columns": ",".join(missing),
            }
        msg = (
            "strict tradable evaluation requires an execution panel; "
            f"missing columns: {missing}"
        )
        raise ValueError(msg)

    contract_columns = [
        "execution_contract_version",
        "return_spec",
        "signal_availability",
        "entry_day_lag",
        "exit_day_lag",
        "entry_price_field",
        "exit_price_field",
        "trading_calendar_sha256",
        "calendar_date_count",
    ]
    contracts = panel.select(contract_columns).unique()
    if contracts.height != 1:
        msg = "execution panel must contain exactly one return contract"
        raise ValueError(msg)
    contract = contracts.row(0, named=True)
    version = str(contract["execution_contract_version"])
    if version != EXECUTION_CONTRACT_VERSION:
        msg = f"unsupported execution contract version: {version!r}"
        raise ValueError(msg)
    availability = str(contract["signal_availability"])
    if availability != AFTER_CLOSE_AVAILABILITY:
        msg = f"unsupported signal availability: {availability!r}"
        raise ValueError(msg)

    spec = get_return_spec(str(contract["return_spec"]))
    declared = {
        "entry_day_lag": int(contract["entry_day_lag"]),
        "exit_day_lag": int(contract["exit_day_lag"]),
        "entry_price": str(contract["entry_price_field"]),
        "exit_price": str(contract["exit_price_field"]),
    }
    expected = {
        "entry_day_lag": spec.entry_day_lag,
        "exit_day_lag": spec.exit_day_lag,
        "entry_price": spec.entry_price,
        "exit_price": spec.exit_price,
    }
    if declared != expected:
        msg = (
            f"execution panel metadata does not match return spec {spec.name}: "
            f"declared={declared}, expected={expected}"
        )
        raise ValueError(msg)
    if spec.entry_day_lag < 1:
        msg = (
            f"return spec {spec.name!r} enters on T, but score(T) is available "
            "only after the T close"
        )
        raise ValueError(msg)

    index_columns = [
        "signal_calendar_index",
        "entry_calendar_index",
        "exit_calendar_index",
    ]
    index_frame = panel.select(
        pl.col(name).cast(pl.Int64, strict=False).alias(name) for name in index_columns
    )
    invalid_indices = index_frame.filter(
        pl.any_horizontal(pl.col(name).is_null() for name in index_columns)
        | (pl.col("signal_calendar_index") < 0)
        | (
            pl.col("entry_calendar_index") - pl.col("signal_calendar_index")
            != spec.entry_day_lag
        )
        | (
            pl.col("exit_calendar_index") - pl.col("signal_calendar_index")
            != spec.exit_day_lag
        )
    ).height
    if invalid_indices:
        msg = (
            "execution panel contains "
            f"{invalid_indices} rows inconsistent with declared trading-day lags"
        )
        raise ValueError(msg)

    calendar_count = int(contract["calendar_date_count"])
    calendar_hash = str(contract["trading_calendar_sha256"])
    if calendar_count < 1 or len(calendar_hash) != 64:
        msg = "execution panel contains invalid trading-calendar provenance"
        raise ValueError(msg)
    out_of_bounds = index_frame.filter(
        (pl.col("entry_calendar_index") >= calendar_count)
        | (pl.col("exit_calendar_index") >= calendar_count)
    ).height
    if out_of_bounds:
        msg = f"execution panel contains {out_of_bounds} calendar indices out of bounds"
        raise ValueError(msg)

    invalid_timing = panel.filter(
        pl.col("entry_date").is_null()
        | pl.col("exit_date").is_null()
        | (pl.col("entry_date").cast(pl.Utf8) <= pl.col("date").cast(pl.Utf8))
        | (pl.col("exit_date").cast(pl.Utf8) < pl.col("entry_date").cast(pl.Utf8))
    ).height
    if invalid_timing:
        msg = f"execution panel contains {invalid_timing} invalid entry/exit rows"
        raise ValueError(msg)

    date_mappings = panel.select(
        "date",
        "entry_date",
        "exit_date",
        *index_columns,
    ).unique()
    inconsistent_signal_dates = (
        date_mappings.group_by("date").len().filter(pl.col("len") != 1).height
    )
    if inconsistent_signal_dates:
        msg = (
            "execution panel maps one signal date to multiple calendar positions: "
            f"{inconsistent_signal_dates} dates"
        )
        raise ValueError(msg)

    verified_calendar = False
    if trading_calendar is not None:
        calendar = normalise_trading_calendar(trading_calendar)
        observed_hash = trading_calendar_sha256(calendar)
        if observed_hash != calendar_hash or len(calendar) != calendar_count:
            msg = (
                "execution panel trading calendar does not match the supplied calendar: "
                f"panel_sha256={calendar_hash}, supplied_sha256={observed_hash}"
            )
            raise ValueError(msg)
        positions = {value: index for index, value in enumerate(calendar)}
        mismatch = 0
        for row in date_mappings.iter_rows(named=True):
            signal_date = str(row["date"])
            signal_index = positions.get(signal_date)
            if signal_index is None:
                mismatch += 1
                continue
            entry_index = signal_index + spec.entry_day_lag
            exit_index = signal_index + spec.exit_day_lag
            if (
                entry_index >= len(calendar)
                or exit_index >= len(calendar)
                or int(row["signal_calendar_index"]) != signal_index
                or int(row["entry_calendar_index"]) != entry_index
                or int(row["exit_calendar_index"]) != exit_index
                or str(row["entry_date"]) != calendar[entry_index]
                or str(row["exit_date"]) != calendar[exit_index]
            ):
                mismatch += 1
        if mismatch:
            msg = (
                "execution panel contains "
                f"{mismatch} signal-date mappings that are not exact T+1/T+2 "
                "positions in the supplied trading calendar"
            )
            raise ValueError(msg)
        verified_calendar = True
    elif require_calendar_verification:
        msg = (
            "strict execution-panel validation requires the original trading calendar; "
            "embedded lag declarations alone are not sufficient"
        )
        raise ValueError(msg)

    return {
        "verified": True,
        "mode": "strict_execution_panel",
        "execution_contract_version": version,
        "return_spec": spec.name,
        "signal_availability": availability,
        "entry_day_lag": spec.entry_day_lag,
        "exit_day_lag": spec.exit_day_lag,
        "entry_price_field": spec.entry_price,
        "exit_price_field": spec.exit_price,
        "trading_calendar_sha256": calendar_hash,
        "calendar_date_count": calendar_count,
        "calendar_reverified": verified_calendar,
    }
