"""从连续 EOD、PIT 股票池和日级 atomic 生成最终 Level-2 Regime 表。"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from quant_fm.downstream.return_spec import (
    normalise_trading_calendar,
    read_trading_calendar,
    trading_calendar_sha256,
)
from quant_fm.downstream.universe import validate_pit_universe
from quant_fm.regime.contract import (
    ATOMIC_ARTIFACT_VERSION,
    ATOMIC_FEATURE_COLUMNS,
    ATOMIC_FORMULA_VERSION,
    L2_FEATURE_COLUMNS,
    REGIME_ARTIFACT_VERSION,
    REGIME_FORMULA_VERSION,
    atomic_manifest_path,
    regime_manifest_path,
    sha256_file,
    sha256_paths,
    validate_atomic_frame,
    write_json_atomic,
)
from quant_fm.schema.cn_l2_v1 import board_of

if TYPE_CHECKING:
    from collections.abc import Sequence

_EOD_REQUIRED = {"date", "symbol", "market", "close", "pre_close"}


def _finite_float(value: object) -> float | None:
    """把可用数值规范为有限 float，其他值返回 ``None``。"""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _compound(values: Sequence[float]) -> float:
    """稳定计算一段简单收益的复合收益。"""
    return float(np.prod(1.0 + np.asarray(values, dtype=np.float64)) - 1.0)


def _normalize_eod(
    eod: pl.DataFrame,
    *,
    amount_column: str,
    calendar: list[str],
) -> pl.DataFrame:
    """校验并规范最终特征使用的连续 EOD 原子表。"""
    required = {*_EOD_REQUIRED, amount_column}
    missing = sorted(required - set(eod.columns))
    if missing:
        msg = f"Regime EOD frame is missing columns: {missing}"
        raise ValueError(msg)
    frame = eod.select(
        pl.col("date").cast(pl.Utf8, strict=False),
        pl.col("symbol").cast(pl.Utf8, strict=False).str.zfill(6),
        pl.col("market").cast(pl.Utf8, strict=False).str.to_uppercase(),
        pl.col("close").cast(pl.Float64, strict=False),
        pl.col("pre_close").cast(pl.Float64, strict=False),
        pl.col(amount_column)
        .cast(pl.Float64, strict=False)
        .alias("_total_notional"),
    )
    if frame.select(pl.struct(["date", "symbol"]).is_duplicated().any()).item():
        msg = "Regime EOD frame contains duplicate (date, symbol) keys"
        raise ValueError(msg)
    if frame.filter(
        pl.col("date").is_null()
        | pl.col("symbol").is_null()
        | ~pl.col("market").is_in(["SH", "SZ"])
    ).height:
        msg = "Regime EOD frame contains invalid keys or market values"
        raise ValueError(msg)
    extra_dates = sorted(set(frame["date"].unique()) - set(calendar))
    if extra_dates:
        msg = f"Regime EOD dates are absent from the trading calendar: {extra_dates[:5]}"
        raise ValueError(msg)
    return frame.with_columns(
        pl.when(
            pl.col("close").is_finite()
            & pl.col("pre_close").is_finite()
            & (pl.col("close") > 0)
            & (pl.col("pre_close") > 0)
        )
        .then(pl.col("close") / pl.col("pre_close") - 1.0)
        .otherwise(None)
        .alias("_stock_return_1d")
    )


def _daily_state(
    joined: pl.DataFrame,
    *,
    calendar: list[str],
    min_eod_coverage: float,
    min_board_names: int,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[tuple[str, str], float],
]:
    """计算每个连续交易日的市场收益、宽度、成交额和板块收益。"""
    if not 0 < min_eod_coverage <= 1:
        msg = "min_eod_coverage must be in (0, 1]"
        raise ValueError(msg)
    if min_board_names < 1:
        msg = "min_board_names must be positive"
        raise ValueError(msg)

    rows_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined.iter_rows(named=True):
        rows_by_date[str(row["date"])].append(row)

    market_returns: dict[str, float] = {}
    breadth: dict[str, float] = {}
    total_amount: dict[str, float] = {}
    board_returns: dict[tuple[str, str], float] = {}
    for date in calendar:
        rows = rows_by_date.get(date, [])
        if not rows:
            msg = f"PIT universe/EOD join has no rows for trading date {date}"
            raise ValueError(msg)
        returns = [
            value
            for row in rows
            if (value := _finite_float(row["_stock_return_1d"])) is not None
        ]
        amounts = [
            value
            for row in rows
            if (value := _finite_float(row["_total_notional"])) is not None
            and value >= 0
        ]
        return_coverage = len(returns) / len(rows)
        amount_coverage = len(amounts) / len(rows)
        if min(return_coverage, amount_coverage) < min_eod_coverage:
            msg = (
                f"Regime EOD coverage below threshold on {date}: "
                f"return={return_coverage:.4f}, amount={amount_coverage:.4f}, "
                f"required={min_eod_coverage:.4f}"
            )
            raise ValueError(msg)
        market_returns[date] = float(np.mean(returns))
        breadth[date] = float(np.sign(np.asarray(returns)).mean())
        total_amount[date] = float(np.sum(amounts))
        if total_amount[date] <= 0:
            msg = f"market total notional must be positive on {date}"
            raise ValueError(msg)

        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            value = _finite_float(row["_stock_return_1d"])
            if value is not None:
                grouped[str(row["_board"])].append(value)
        for board, values in grouped.items():
            if len(values) >= min_board_names:
                board_returns[(date, board)] = float(np.mean(values))
    return market_returns, breadth, total_amount, board_returns


def build_l2_regime_features(
    atomic: pl.DataFrame,
    eod: pl.DataFrame,
    universe: pl.DataFrame,
    trading_calendar: Sequence[str],
    *,
    amount_column: str = "total_notional",
    min_eod_coverage: float = 0.98,
    min_board_names: int = 1,
    min_book_valid_ratio: float = 0.0,
) -> pl.DataFrame:
    """构建纯 Level-2 九特征表；20日窗口严格使用连续交易日历。"""
    calendar = normalise_trading_calendar(trading_calendar)
    atomic_frame = validate_atomic_frame(atomic)
    signal_dates = sorted(atomic_frame["date"].unique().to_list())
    unknown_signal_dates = sorted(set(signal_dates) - set(calendar))
    if unknown_signal_dates:
        msg = f"Regime atomic dates are absent from calendar: {unknown_signal_dates}"
        raise ValueError(msg)
    eod_frame = _normalize_eod(eod, amount_column=amount_column, calendar=calendar)
    pit_universe, _contract = validate_pit_universe(
        universe,
        required_dates=calendar,
        min_names_per_day=1,
        context="Regime PIT universe",
        require_pit_metadata=True,
    )
    keyed_universe = pit_universe.select("date", "symbol")
    joined = keyed_universe.join(eod_frame, on=["date", "symbol"], how="left")
    joined = joined.with_columns(
        pl.struct(["symbol", "market"])
        .map_elements(
            lambda value: board_of(value["symbol"], value["market"])
            if value["market"] is not None
            else "UNKNOWN",
            return_dtype=pl.Utf8,
        )
        .alias("_board")
    )
    market_returns, daily_breadth, daily_amount, board_returns = _daily_state(
        joined,
        calendar=calendar,
        min_eod_coverage=min_eod_coverage,
        min_board_names=min_board_names,
    )
    eod_identity = {
        (str(row["date"]), str(row["symbol"])): (
            str(row["market"]),
            str(row["_board"]),
        )
        for row in joined.iter_rows(named=True)
        if row["market"] is not None
    }

    calendar_index = {date: index for index, date in enumerate(calendar)}
    market_features: dict[str, dict[str, float]] = {}
    board_strength: dict[tuple[str, str], float] = {}
    boards = sorted(set(atomic_frame["board"].to_list()))
    for date in signal_dates:
        index = calendar_index[date]
        if index < 19:
            msg = (
                f"Regime feature date {date} lacks 20 consecutive trading-day "
                "warm-up in the supplied calendar"
            )
            raise ValueError(msg)
        dates_5 = calendar[index - 4 : index + 1]
        dates_20 = calendar[index - 19 : index + 1]
        market_5 = [market_returns[value] for value in dates_5]
        market_20 = [market_returns[value] for value in dates_20]
        amount_20 = [daily_amount[value] for value in dates_20]
        market_return_20d = _compound(market_20)
        amount_mean = float(np.mean(amount_20))
        market_features[date] = {
            "market_return_5d": _compound(market_5),
            "market_return_20d": market_return_20d,
            "market_realized_vol_20d": float(
                np.std(np.asarray(market_20), ddof=1) * math.sqrt(252.0)
            ),
            "market_breadth": daily_breadth[date],
            "market_amount_ratio_20d": daily_amount[date] / amount_mean,
        }
        for board in boards:
            values: list[float] = []
            for value in dates_20:
                daily_value = board_returns.get((value, board))
                if daily_value is None:
                    break
                values.append(daily_value)
            if len(values) == 20:
                board_strength[(date, board)] = _compound(values) - market_return_20d

    universe_keys = set(keyed_universe.iter_rows())
    output_rows: list[dict[str, object]] = []
    invalid: list[tuple[str, str, list[str]]] = []
    for row in atomic_frame.iter_rows(named=True):
        date = str(row["date"])
        symbol = str(row["symbol"])
        if (date, symbol) not in universe_keys:
            msg = f"Regime atomic key is absent from PIT universe: {(date, symbol)}"
            raise ValueError(msg)
        expected_identity = eod_identity.get((date, symbol))
        if expected_identity is None:
            msg = f"Regime atomic key lacks target-date EOD identity: {(date, symbol)}"
            raise ValueError(msg)
        actual_identity = (str(row["market"]), str(row["board"]))
        if actual_identity != expected_identity:
            msg = (
                "Regime atomic market/board disagrees with target-date EOD: "
                f"key={(date, symbol)} atomic={actual_identity} "
                f"eod={expected_identity}"
            )
            raise ValueError(msg)
        feature_row: dict[str, object] = {
            "date": date,
            "symbol": symbol,
            **market_features[date],
            "board_relative_strength_20d": board_strength.get(
                (date, str(row["board"]))
            ),
            **{name: row[name] for name in ATOMIC_FEATURE_COLUMNS},
            "asof_date": date,
        }
        bad = [
            name
            for name in L2_FEATURE_COLUMNS
            if _finite_float(feature_row.get(name)) is None
        ]
        ratio = _finite_float(row["book_valid_ratio"])
        if ratio is None or ratio < min_book_valid_ratio:
            bad.append("book_valid_ratio")
        if bad:
            invalid.append((date, symbol, sorted(set(bad))))
        output_rows.append(feature_row)
    if invalid:
        msg = f"Regime final features are incomplete/non-finite: {invalid[:5]}"
        raise ValueError(msg)

    output = pl.DataFrame(output_rows).select(
        pl.col("date").cast(pl.Utf8),
        pl.col("symbol").cast(pl.Utf8).str.zfill(6),
        *(pl.col(name).cast(pl.Float64) for name in L2_FEATURE_COLUMNS),
        pl.col("asof_date").cast(pl.Utf8),
    )
    if output.select(pl.struct(["date", "symbol"]).is_duplicated().any()).item():
        msg = "final Regime features contain duplicate (date, symbol) keys"
        raise ValueError(msg)
    return output.sort(["date", "symbol"])


def _load_atomic_archives(
    atomic_dir: Path,
    *,
    signal_dates: Sequence[str] | None = None,
) -> tuple[pl.DataFrame, list[Path]]:
    """加载并校验所有日级 archive 及其 manifest。"""
    parquet_paths = sorted(Path(atomic_dir).glob("*.parquet"))
    if not parquet_paths:
        msg = f"no daily Regime atomic archives under {atomic_dir}"
        raise FileNotFoundError(msg)
    if signal_dates is not None:
        selected_dates = normalise_trading_calendar(signal_dates)
        by_date = {path.stem: path for path in parquet_paths}
        missing = sorted(set(selected_dates) - set(by_date))
        if missing:
            msg = f"missing requested daily Regime atomic archives: {missing[:8]}"
            raise FileNotFoundError(msg)
        parquet_paths = [by_date[date] for date in selected_dates]
    frames: list[pl.DataFrame] = []
    inputs: list[Path] = []
    for path in parquet_paths:
        manifest_path = atomic_manifest_path(path)
        if not manifest_path.is_file():
            msg = f"daily Regime atomic archive lacks manifest: {manifest_path}"
            raise FileNotFoundError(msg)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            payload.get("artifact_version") != ATOMIC_ARTIFACT_VERSION
            or payload.get("formula_version") != ATOMIC_FORMULA_VERSION
            or payload.get("date") != path.stem
            or payload.get("parquet_sha256") != sha256_file(path)
        ):
            msg = f"daily Regime atomic manifest mismatch: {manifest_path}"
            raise ValueError(msg)
        frames.append(
            validate_atomic_frame(pl.read_parquet(path), expected_date=path.stem)
        )
        inputs.extend((path, manifest_path))
    return validate_atomic_frame(pl.concat(frames, how="vertical_relaxed")), inputs


def finalize_l2_regime_features(
    *,
    atomic_dir: Path,
    eod_path: Path,
    universe_path: Path,
    calendar_path: Path,
    output_path: Path,
    amount_column: str = "total_notional",
    min_eod_coverage: float = 0.98,
    min_board_names: int = 1,
    min_book_valid_ratio: float = 0.0,
    signal_dates: Sequence[str] | None = None,
) -> tuple[Path, Path]:
    """读取正式输入，写出最终 parquet 和完整血缘 manifest。"""
    atomic, atomic_inputs = _load_atomic_archives(
        atomic_dir,
        signal_dates=signal_dates,
    )
    calendar = read_trading_calendar(calendar_path)
    output = build_l2_regime_features(
        atomic,
        pl.read_parquet(eod_path),
        pl.read_parquet(universe_path),
        calendar,
        amount_column=amount_column,
        min_eod_coverage=min_eod_coverage,
        min_board_names=min_board_names,
        min_book_valid_ratio=min_book_valid_ratio,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    output.write_parquet(
        temporary,
        compression="zstd",
        compression_level=3,
        statistics=True,
    )
    temporary.replace(destination)
    counts = output.group_by("date").len().sort("date")
    manifest = regime_manifest_path(destination)
    write_json_atomic(
        manifest,
        {
            "artifact_version": REGIME_ARTIFACT_VERSION,
            "formula_version": REGIME_FORMULA_VERSION,
            "availability": "available_after_signal_date_close",
            "asof_rule": "asof_date_equals_signal_date",
            "features": list(L2_FEATURE_COLUMNS),
            "rows": output.height,
            "dates": output["date"].n_unique(),
            "date_min": str(output["date"].min()),
            "date_max": str(output["date"].max()),
            "signal_dates_sha256": trading_calendar_sha256(
                sorted(output["date"].unique().to_list())
            ),
            "names_per_date": [
                {"date": str(date), "names": int(names)}
                for date, names in counts.iter_rows()
            ],
            "trading_calendar_sha256": trading_calendar_sha256(calendar),
            "calendar_date_count": len(calendar),
            "atomic_archives_sha256": sha256_paths(atomic_inputs),
            "eod_file": Path(eod_path).name,
            "eod_sha256": sha256_file(eod_path),
            "universe_file": Path(universe_path).name,
            "universe_sha256": sha256_file(universe_path),
            "amount_column": amount_column,
            "min_eod_coverage": min_eod_coverage,
            "min_board_names": min_board_names,
            "min_book_valid_ratio": min_book_valid_ratio,
            "parquet_file": destination.name,
            "parquet_sha256": sha256_file(destination),
        },
    )
    return destination, manifest


__all__ = ["build_l2_regime_features", "finalize_l2_regime_features"]
