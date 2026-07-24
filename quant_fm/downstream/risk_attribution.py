"""组合风险暴露与横截面收益中性化。"""

from __future__ import annotations

import numpy as np
import polars as pl


def portfolio_exposures(
    holdings: pl.DataFrame,
    factors: pl.DataFrame,
    *,
    exposure_cols: list[str] | None = None,
) -> pl.DataFrame:
    """按持仓权重聚合每日数值因子暴露。"""
    if holdings.is_empty():
        return pl.DataFrame()
    columns = exposure_cols or [
        name for name in factors.columns if name.startswith("factor_")
    ]
    missing = set(columns) - set(factors.columns)
    if missing:
        msg = f"factor table missing exposure columns: {sorted(missing)}"
        raise ValueError(msg)
    joined = holdings.select(["date", "symbol", "weight"]).join(
        factors.select(["date", "symbol", *columns]),
        on=["date", "symbol"],
        how="left",
    )
    expressions = [
        (
            (pl.col(name) * pl.col("weight")).sum()
            / pl.when(pl.col(name).is_not_null())
            .then(pl.col("weight"))
            .otherwise(0.0)
            .sum()
        ).alias(name)
        for name in columns
    ]
    return joined.group_by("date").agg(expressions).sort("date")


def residualize_returns(
    panel: pl.DataFrame,
    factors: pl.DataFrame,
    *,
    exposure_cols: list[str] | None = None,
    ret_col: str = "fwd_ret",
    output_col: str = "neutralized_ret",
) -> pl.DataFrame:
    """逐日 OLS 去除指定因子暴露，返回与 panel 同键的残差收益。"""
    columns = exposure_cols or [
        name for name in factors.columns if name.startswith("factor_")
    ]
    if not columns:
        msg = "at least one factor exposure is required"
        raise ValueError(msg)
    joined = panel.select(["date", "symbol", ret_col]).join(
        factors.select(["date", "symbol", *columns]),
        on=["date", "symbol"],
        how="inner",
    )
    rows: list[pl.DataFrame] = []
    for (_date,), day in joined.group_by(["date"], maintain_order=True):
        clean = day.drop_nulls([ret_col, *columns])
        if clean.height <= len(columns) + 1:
            continue
        x = clean.select(columns).to_numpy().astype(np.float64)
        y = clean[ret_col].to_numpy().astype(np.float64)
        finite = np.isfinite(x).all(axis=1) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if y.size <= len(columns) + 1:
            continue
        design = np.column_stack([np.ones(y.size), x])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coefficients
        rows.append(
            clean.filter(pl.Series(finite))
            .select(["date", "symbol"])
            .with_columns(pl.Series(output_col, residual))
        )
    if not rows:
        return pl.DataFrame(
            schema={"date": pl.Utf8, "symbol": pl.Utf8, output_col: pl.Float64}
        )
    return pl.concat(rows).sort(["date", "symbol"])
