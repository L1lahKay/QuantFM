"""
传统因子基线：动量 / 反转 / 简易 OFI，供与 FM embedding 对比。

因子列统一 ``factor_*`` 前缀，可直接传给 :func:`quant_fm.downstream.make_features.build_features`。

* ``factor_ret_oc``：当日 open-to-close 近似（``close/pre_close - 1``）
* ``factor_mom_1``：相对面板上一交易日的收益（稀疏日历上的跨日动量）
* ``factor_rev_1``：``-factor_mom_1``（短周期反转）
* ``factor_ofi``：可选；从 tokens 按股日聚合 signed side（B=+1, S=-1）均值

用法::

    python -m quant_fm.downstream.baselines \\
      --panel quant_fm/runs/medium_300m/panel/daily_panel.parquet \\
      --out quant_fm/runs/medium_300m/panel/factors.parquet \\
      --tokens-dir quant_fm/runs/medium_300m/tokens \\
      --max-ofi-shards 2000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from quant_fm.schema.cn_l2_v1 import SIDES
from quant_fm.tokenizer.vocab import N_SPECIAL

logger = logging.getLogger(__name__)

# vocab: PAD=0, then SIDES order → B=1, S=2, N=3
_SIDE_B = N_SPECIAL + SIDES.index("B")
_SIDE_S = N_SPECIAL + SIDES.index("S")


def panel_momentum_factors(panel: pl.DataFrame) -> pl.DataFrame:
    """从日频面板构造动量 / 反转因子（不依赖 tokens）。"""
    df = panel.select(
        ["date", "symbol"]
        + [c for c in ("close", "pre_close", "vwap") if c in panel.columns]
    ).sort(["symbol", "date"])

    exprs: list[pl.Expr] = []
    if "close" in df.columns and "pre_close" in df.columns:
        exprs.append(
            (
                pl.when(pl.col("pre_close") > 0)
                .then(pl.col("close") / pl.col("pre_close") - 1.0)
                .otherwise(None)
            ).alias("factor_ret_oc")
        )
    if "close" in df.columns:
        exprs.append(
            (
                pl.when(pl.col("close").shift(1).over("symbol") > 0)
                .then(pl.col("close") / pl.col("close").shift(1).over("symbol") - 1.0)
                .otherwise(None)
            ).alias("factor_mom_1")
        )

    out = df.with_columns(exprs) if exprs else df
    if "factor_mom_1" in out.columns:
        out = out.with_columns((-pl.col("factor_mom_1")).alias("factor_rev_1"))
    keep = ["date", "symbol"] + [c for c in out.columns if c.startswith("factor_")]
    return out.select(keep)


def ofi_from_tokens(
    tokens_dir: Path,
    *,
    max_shards: int | None = None,
) -> pl.DataFrame:
    """
    扫描 tokens parquet，按股日聚合简易 OFI = mean(sign(side))。

    ``tok_side``：B → +1，S → -1，其余 0。
    """
    paths = sorted(tokens_dir.rglob("*.parquet"))
    if max_shards is not None:
        paths = paths[: max(0, max_shards)]
    if not paths:
        logger.warning("no token shards under %s", tokens_dir)
        return pl.DataFrame(
            schema={
                "date": pl.Utf8,
                "symbol": pl.Utf8,
                "factor_ofi": pl.Float64,
                "ofi_n_events": pl.Int64,
            }
        )

    logger.info("OFI: scanning %d shards under %s", len(paths), tokens_dir)
    frames: list[pl.DataFrame] = []
    for i, path in enumerate(paths):
        # path: .../tokens/{market}/{symbol}/{date}.parquet
        parts = path.parts
        try:
            date = path.stem
            symbol = parts[-2]
        except Exception:
            continue
        try:
            side = pl.scan_parquet(path).select("tok_side").collect()["tok_side"]
        except Exception:
            logger.exception("skip %s", path)
            continue
        s = side.to_numpy()
        signed = (s == _SIDE_B).astype("float64") - (s == _SIDE_S).astype("float64")
        frames.append(
            pl.DataFrame(
                {
                    "date": [date],
                    "symbol": [str(symbol).zfill(6)],
                    "factor_ofi": [float(signed.mean()) if len(signed) else None],
                    "ofi_n_events": [len(signed)],
                }
            )
        )
        if (i + 1) % 500 == 0:
            logger.info("OFI progress %d/%d", i + 1, len(paths))

    if not frames:
        return pl.DataFrame(
            schema={
                "date": pl.Utf8,
                "symbol": pl.Utf8,
                "factor_ofi": pl.Float64,
                "ofi_n_events": pl.Int64,
            }
        )
    return pl.concat(frames, how="vertical_relaxed")


def build_baselines(
    panel: pl.DataFrame,
    *,
    tokens_dir: Path | None = None,
    max_ofi_shards: int | None = None,
) -> pl.DataFrame:
    """合并面板动量因子与可选 OFI。"""
    factors = panel_momentum_factors(panel)
    if tokens_dir is not None and Path(tokens_dir).is_dir():
        ofi = ofi_from_tokens(Path(tokens_dir), max_shards=max_ofi_shards)
        if ofi.height > 0:
            factors = factors.join(
                ofi.select(["date", "symbol", "factor_ofi"]),
                on=["date", "symbol"],
                how="left",
            )
    return factors.sort(["date", "symbol"])


def main() -> None:
    """CLI：从 panel（+可选 tokens）写出 factor parquet。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens-dir", type=Path, default=None)
    parser.add_argument(
        "--max-ofi-shards",
        type=int,
        default=None,
        help="限制 OFI 扫描的 shard 数（全市场很大；试跑可设 2000）",
    )
    parser.add_argument("--skip-ofi", action="store_true")
    args = parser.parse_args()

    panel = pl.read_parquet(args.panel)
    tokens = None if args.skip_ofi else args.tokens_dir
    factors = build_baselines(
        panel,
        tokens_dir=tokens,
        max_ofi_shards=args.max_ofi_shards,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    factors.write_parquet(args.out)
    fac_cols = [c for c in factors.columns if c.startswith("factor_")]
    logger.info(
        "wrote %s rows=%d factors=%s",
        args.out,
        factors.height,
        fac_cols,
    )
    # 与 fwd_ret 的简易相关（sanity）
    if "fwd_ret" in panel.columns:
        joined = factors.join(
            panel.select(["date", "symbol", "fwd_ret"]),
            on=["date", "symbol"],
            how="inner",
        ).filter(pl.col("fwd_ret").is_not_null())
        for c in fac_cols:
            sub = joined.select([c, "fwd_ret"]).drop_nulls()
            if sub.height < 100:
                continue
            corr = sub.select(pl.corr(c, "fwd_ret")).item()
            logger.info("corr(%s, fwd_ret)=%.4f (n=%d)", c, corr or float("nan"), sub.height)


if __name__ == "__main__":
    main()
