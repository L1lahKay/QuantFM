r"""
下游裁判：embedding + 日频面板 → Ranker → RankIC / 回测，并**每次跑完持久化**结果。

输出目录默认 ``{workdir}/downstream/``::

    runs/
      {timestamp}_{ckpt_stem}.json   # 每次完整报告
    latest.json                      # 指向最近一次的副本
    history.jsonl                    # 追加一行摘要（便于扫历史）

用法::

    python -m quant_fm.downstream.run_judge \\
      --workdir quant_fm/runs/medium_try \\
      --checkpoint quant_fm/runs/medium_try/run/best.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from quant_fm.downstream.backtest_topk import backtest_topk
from quant_fm.downstream.evaluate import (
    cpcv_splits,
    deflated_sharpe_ratio,
    group_monotonicity,
    rank_ic,
    rank_icir,
)
from quant_fm.downstream.make_features import build_features
from quant_fm.downstream.train_ranker import fit_ranker, predict, train_ranker

logger = logging.getLogger(__name__)


def _run_cpcv(
    feat_all: pl.DataFrame,
    panel: pl.DataFrame,
    *,
    device: str,
    epochs: int,
    seed: int,
    top_k: int | None,
) -> dict:
    """
    组合式 purged CV。

    每个 fold 仅用 train 日期拟合 Ranker，再在 test 块上计算 RankIC。
    """
    dates = sorted(str(d) for d in feat_all["date"].unique().to_list())
    n = len(dates)
    if n < 4:
        return {
            "skipped": True,
            "reason": f"need >= 4 dates for CPCV, got {n}",
            "n_dates": n,
        }

    n_groups = min(6, n)
    n_test_groups = 1 if n_groups <= 4 else 2
    purge = 1 if n >= 8 else 0
    embargo = 1 if n >= 8 else 0
    splits = cpcv_splits(
        dates,
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        purge=purge,
        embargo=embargo,
    )

    fold_rows: list[dict] = []
    fold_epochs = max(5, min(epochs, 15))
    for i, (train_dates, test_dates) in enumerate(splits):
        train_f = feat_all.filter(pl.col("date").is_in(train_dates))
        test_f = feat_all.filter(pl.col("date").is_in(test_dates))
        if train_f.is_empty() or test_f.is_empty():
            continue
        if train_f["date"].n_unique() < 2 or test_f["date"].n_unique() < 1:
            continue
        model, history = train_ranker(
            train_f,
            epochs=fold_epochs,
            device=device,
            seed=seed + i,
        )
        preds = predict(model, test_f, device=device)
        pan = panel.filter(pl.col("date").is_in(test_f["date"].unique().to_list()))
        ic = rank_ic(preds, pan)
        mean_ic = float(np.nanmean(ic["ic"].to_numpy()))
        n_per_day = max(test_f.height // max(test_f["date"].n_unique(), 1), 1)
        k = top_k if top_k is not None else min(10, max(3, n_per_day // 5))
        bt = backtest_topk(preds, pan, top_k=k, long_short=False, cost_bps=15.0)
        fold_rows.append(
            {
                "fold": i,
                "n_train_days": len(train_dates),
                "n_test_days": len(test_dates),
                "mean_rank_ic": mean_ic,
                "final_train_ic": history[-1] if history else None,
                "backtest_long_only_sharpe_daily": bt.sharpe_daily,
                "backtest_long_only_cum_return": bt.cum_return,
                "backtest_long_only_hit_rate": bt.hit_rate,
                "backtest_long_only_sharpe": bt.sharpe,
                "backtest_long_only_ann_return": bt.ann_return,
                "backtest_reliable": bt.reliable,
            }
        )

    if not fold_rows:
        return {
            "skipped": True,
            "reason": "no valid CPCV folds after filtering",
            "n_dates": n,
            "n_groups": n_groups,
        }

    ics = np.array([r["mean_rank_ic"] for r in fold_rows], dtype=np.float64)
    return {
        "skipped": False,
        "n_dates": n,
        "n_groups": n_groups,
        "n_test_groups": n_test_groups,
        "purge": purge,
        "embargo": embargo,
        "n_folds": len(fold_rows),
        "mean_rank_ic": float(np.nanmean(ics)),
        "std_rank_ic": float(np.nanstd(ics, ddof=1)) if len(ics) > 1 else None,
        "folds": fold_rows,
    }


def _file_meta(path: Path) -> dict:
    path = Path(path)
    st = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {
        "path": str(path.resolve()),
        "name": path.name,
        "stem": path.stem,
        "size_bytes": st.st_size,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=UTC).isoformat(),
        "sha256": h.hexdigest()[:16],
        "kind": (
            "best"
            if path.stem == "best"
            else "final"
            if path.stem == "final"
            else "step"
            if path.stem.startswith("step")
            else "other"
        ),
    }


def _load_emb(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    drop = [c for c in ("split",) if c in df.columns]
    return df.drop(drop) if drop else df


def _eval_split(
    name: str,
    model,
    feat: pl.DataFrame,
    panel: pl.DataFrame,
    *,
    device: str,
    top_k: int | None,
) -> dict:
    if feat.is_empty() or feat["date"].n_unique() == 0:
        return {"split": name, "error": "empty"}
    preds = predict(model, feat, device=device)
    pan = panel.filter(pl.col("date").is_in(feat["date"].unique().to_list()))
    ic = rank_ic(preds, pan)
    icir = rank_icir(ic)
    mono = group_monotonicity(preds, pan, n_groups=5)
    n_per_day = max(feat.height // max(feat["date"].n_unique(), 1), 1)
    k = top_k if top_k is not None else min(10, max(3, n_per_day // 5))
    bt_ls = backtest_topk(preds, pan, top_k=k, long_short=True, cost_bps=15.0)
    bt_lo = backtest_topk(preds, pan, top_k=k, long_short=False, cost_bps=15.0)
    dsr = deflated_sharpe_ratio(
        bt_lo.sharpe_daily,
        n_trials=5,
        n_obs=max(len(bt_lo.dates), 2),
        sr_variance=0.25,
    )
    mean_ic = float(np.nanmean(ic["ic"].to_numpy()))
    return {
        "split": name,
        "n_days": int(feat["date"].n_unique()),
        "n_rows": int(feat.height),
        "top_k": k,
        "mean_rank_ic": mean_ic,
        "icir": float(icir) if icir == icir else None,
        "daily_ic": [
            {
                "date": r["date"],
                "ic": None if r["ic"] != r["ic"] else float(r["ic"]),
            }
            for r in ic.iter_rows(named=True)
        ],
        "group_mean_fwd_ret": [None if x != x else float(x) for x in mono],
        "backtest_long_short": bt_ls.as_dict(),
        "backtest_long_only": bt_lo.as_dict(),
        "deflated_sharpe_long_only": float(dsr) if dsr == dsr else None,
    }


def run_judge(
    *,
    workdir: Path,
    checkpoint: Path,
    panel_path: Path | None = None,
    emb_dir: Path | None = None,
    epochs: int = 30,
    device: str = "cuda:0",
    top_k: int | None = None,
    min_names_per_day: int = 20,
    seed: int = 42,
) -> Path:
    """跑下游裁判并把完整报告写入 ``workdir/downstream/``，返回报告路径。"""
    workdir = Path(workdir)
    emb_dir = Path(emb_dir) if emb_dir else workdir / "embeddings"
    panel_path = (
        Path(panel_path) if panel_path else workdir / "panel/daily_panel.parquet"
    )
    out_dir = workdir / "downstream"
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    ckpt_meta = _file_meta(checkpoint)
    panel = pl.read_parquet(panel_path).filter(pl.col("fwd_ret").is_not_null())

    train_feat = build_features(
        _load_emb(emb_dir / "train.parquet"),
        panel,
        min_names_per_day=min_names_per_day,
    )
    val_feat = build_features(
        _load_emb(emb_dir / "val.parquet"),
        panel,
        min_names_per_day=min_names_per_day,
    )
    test_feat = build_features(
        _load_emb(emb_dir / "test.parquet"),
        panel,
        min_names_per_day=min_names_per_day,
    )

    training = fit_ranker(
        train_feat,
        val_features=val_feat,
        epochs=epochs,
        device=device,
        seed=seed,
    )
    model = training.model
    history = [float(row["train_ic"] or 0.0) for row in training.history]

    rng = np.random.default_rng(0)
    test_pan = panel.filter(pl.col("date").is_in(test_feat["date"].unique().to_list()))
    rand_preds = test_feat.select(["date", "symbol"]).with_columns(
        pl.Series("score", rng.normal(size=test_feat.height))
    )
    rand_ic = rank_ic(rand_preds, test_pan)

    feat_all = pl.concat(
        [df for df in (train_feat, val_feat, test_feat) if not df.is_empty()],
        how="vertical_relaxed",
    )
    cpcv_report = _run_cpcv(
        feat_all,
        panel,
        device=device,
        epochs=epochs,
        seed=seed,
        top_k=top_k,
    )

    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "created_utc": ts,
        "checkpoint": ckpt_meta,
        "panel": _file_meta(panel_path),
        "embeddings_dir": str(emb_dir.resolve()),
        "workdir": str(workdir.resolve()),
        "ranker": {
            "epochs": epochs,
            "device": device,
            "seed": seed,
            "train_history_ic": history,
            "final_train_ic": history[-1] if history else None,
            "best_epoch": training.best_epoch,
            "best_val_ic": training.best_val_ic,
            "stopped_early": training.stopped_early,
            "training_history": training.history,
        },
        "in_sample_train": _eval_split(
            "train", model, train_feat, panel, device=device, top_k=top_k
        ),
        "val": _eval_split("val", model, val_feat, panel, device=device, top_k=top_k),
        "test": _eval_split(
            "test", model, test_feat, panel, device=device, top_k=top_k
        ),
        "cpcv": cpcv_report,
        "test_random_baseline_mean_ic": float(np.nanmean(rand_ic["ic"].to_numpy())),
    }

    run_path = runs_dir / f"{ts}_{ckpt_meta['stem']}.json"
    run_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest = out_dir / "latest.json"
    shutil.copyfile(run_path, latest)
    # 兼容旧文件名
    shutil.copyfile(run_path, out_dir / "judge_report.json")

    summary = {
        "created_utc": ts,
        "checkpoint_kind": ckpt_meta["kind"],
        "checkpoint_path": ckpt_meta["path"],
        "checkpoint_sha256_16": ckpt_meta["sha256"],
        "train_ic": report["ranker"]["final_train_ic"],
        "val_mean_rank_ic": report["val"].get("mean_rank_ic"),
        "test_mean_rank_ic": report["test"].get("mean_rank_ic"),
        "cpcv_mean_rank_ic": cpcv_report.get("mean_rank_ic"),
        "test_random_baseline_mean_ic": report["test_random_baseline_mean_ic"],
        "report_path": str(run_path.resolve()),
    }
    history_path = out_dir / "history.jsonl"
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    logger.info(
        "persisted judge report → %s (checkpoint=%s kind=%s)",
        run_path,
        ckpt_meta["path"],
        ckpt_meta["kind"],
    )
    logger.info(
        "summary train_ic=%.4f val_ic=%s test_ic=%s",
        summary["train_ic"] or 0.0,
        summary["val_mean_rank_ic"],
        summary["test_mean_rank_ic"],
    )
    return run_path


def main() -> None:
    """Run downstream evaluation and persist the resulting report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="默认 {workdir}/run/best.pt",
    )
    parser.add_argument("--panel", type=Path, default=None)
    parser.add_argument("--emb-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt = args.checkpoint or (args.workdir / "run" / "best.pt")
    path = run_judge(
        workdir=args.workdir,
        checkpoint=ckpt,
        panel_path=args.panel,
        emb_dir=args.emb_dir,
        epochs=args.epochs,
        device=args.device,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(path)


if __name__ == "__main__":
    main()
