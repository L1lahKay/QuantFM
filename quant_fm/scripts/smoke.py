"""
在合成数据上的端到端冒烟测试（无需 MinIO、无需 GPU）。

串联流水线各阶段，使 ``make smoke`` 在任一环节断裂时立即失败：
合成规范事件 -> 拟合分箱 -> 分词 -> 覆盖率/泄漏门控 -> 清单 ->
小规模预训练 -> 嵌入提取 -> 特征矩阵 -> 横截面排序器 ->
Top-K 回测 -> RankIC / DSR 门控。

运行::

    python -m quant_fm.scripts.smoke --workdir quant_fm/runs/smoke

【新手】这是整个项目最好的入门文件：按顺序读完 run() 函数，
就等于走通了一遍真实 pipeline（只是数据是随机合成的）。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl  # [导读] polars：类似 pandas 的表格库，读写 parquet 很快
import yaml  # [导读] 读取/写出 YAML 配置（人类可读的字典）

from quant_fm.schema.cn_l2_v1 import CANONICAL_COLUMNS
from quant_fm.tokenizer.fit_bins import fit_bins
from quant_fm.tokenizer.tokenize_events import (
    assert_no_leakage,
    coverage_report,
    tokenize_path,
)

logger = logging.getLogger(__name__)

# [导读] 列表推导式：[表达式 for 变量 in 范围] 快速生成列表
_DATES = [f"2024-01-{d:02d}" for d in range(2, 12)]  # 10 个交易日
# [导读] 字典：键 -> 值；这里按市场分组的示例股票代码
_SYMBOLS = {"SZ": ["000001", "000002", "300750"], "SH": ["600519", "688981"]}


def _synth_canonical(symbol: str, market: str, date: str, seed: int) -> pl.DataFrame:
    """为单股单日构建合理的合成 cn_l2_v1 事件表。"""
    # [导读] 用固定 seed 的随机数生成器，保证每次运行合成数据一致（可复现）
    rng = np.random.default_rng(seed)
    n = int(rng.integers(400, 900))  # 这一天大约有多少个事件
    base_px = float(rng.uniform(8, 60))
    steps = np.cumsum(rng.normal(0, base_px * 3e-4, size=n))
    price = np.clip(base_px + steps, 0.5, None)
    qty = rng.integers(1, 50, size=n).astype(float) * 100
    # [导读] evt_type：挂单 ADD / 撤单 CANCEL / 成交 EXEC
    evt = rng.choice(["ADD", "CANCEL", "EXEC"], size=n, p=[0.55, 0.2, 0.25])
    side = rng.choice(["B", "S"], size=n)
    # 09:30:00.000 .. 14:56:59 打包 HHMMSSmmm，单调递增
    secs = np.linspace(9 * 3600 + 30 * 60, 14 * 3600 + 56 * 60, n)
    hh = (secs // 3600).astype(int)
    mm = ((secs % 3600) // 60).astype(int)
    ss = (secs % 60).astype(int)
    ms = rng.integers(0, 1000, size=n)
    int_time = hh * 10_000_000 + mm * 100_000 + ss * 1000 + ms

    board = "STAR" if symbol.startswith("688") else "MAIN"
    frame = pl.DataFrame(
        {
            "schema_version": ["cn_l2_v1"] * n,
            "date": [date] * n,
            "exchange": ["XSHG" if market == "SH" else "XSHE"] * n,
            "market": ["A_SHARE"] * n,
            "symbol": [f"{symbol}.{market}"] * n,
            "security_id": [symbol] * n,
            "board": [board] * n,
            "session": ["CONT_AM"] * n,
            "event_source": np.where(evt == "EXEC", "TRADE", "ORDER"),
            "evt_type": evt,
            "side": side,
            "price": price,
            "qty": qty,
            "amount": price * qty,
            "order_type": np.where(evt == "CANCEL", "CANCEL", "LIMIT"),
            "level": np.zeros(n, dtype=np.int32),
            "delta_t": np.diff(int_time, prepend=int_time[0]).clip(0),
            "int_time": int_time,
            "local_time": int_time,
            "source_seqnum": np.arange(n, dtype=np.int64),
            "event_idx": np.arange(n, dtype=np.int64),
            "quality_flag": np.zeros(n, dtype=np.int32),
        }
    ).select(CANONICAL_COLUMNS)
    # [导读] pl.DataFrame 类似表格；.select 只保留规范列，顺序固定
    return frame


def _stage_events(events_dir: Path) -> list[Path]:
    """把合成事件写到 events/{市场}/{股票}/{日期}.parquet。"""
    paths = []
    seed = 0
    # [导读] 三层嵌套循环：每个市场 × 每只股票 × 每个交易日 → 一个文件
    for market, syms in _SYMBOLS.items():
        for symbol in syms:
            for date in _DATES:
                seed += 1
                frame = _synth_canonical(symbol, market, date, seed)
                dst = events_dir / market / symbol / f"{date}.parquet"
                dst.parent.mkdir(parents=True, exist_ok=True)
                frame.write_parquet(dst)
                paths.append(dst)
    logger.info("synth: wrote %d canonical shards", len(paths))
    return paths


def run(workdir: Path) -> None:
    """运行完整合成流水线并断言各阶段成功。"""
    workdir = Path(workdir)
    events_dir = workdir / "events"  # 规范事件 parquet
    tokens_dir = workdir / "tokens"  # 分词后的整数列
    data_dir = workdir / "data"  # vocab.json + manifest.json
    data_dir.mkdir(parents=True, exist_ok=True)

    # ========== 阶段 1：造数据（真实流程里由 pylob + export_events 完成）==========
    _stage_events(events_dir)

    # [导读] 按日期切 train/val/test，模拟真实训练的时间切分（防止用未来数据训练）
    train_dates = _DATES[:6]
    val_dates = _DATES[6:8]
    test_dates = _DATES[8:]

    # ========== 阶段 2：拟合词表（只用训练日期！）==========
    train_paths = [p for p in events_dir.rglob("*.parquet") if p.stem in train_dates]
    # [导读] rglob("*.parquet")：递归找所有 parquet；p.stem 是文件名不含后缀，即日期
    vocab = fit_bins(train_paths, n_bins=16, fit_dates=train_dates, seed=0)
    vocab_path = data_dir / "vocab.json"
    vocab.save(vocab_path)
    assert_no_leakage(vocab, val_dates, test_dates)  # 闸门：val/test 日期不能参与拟合

    # ========== 阶段 3：用冻结词表把事件变成 tok_* 整数列 ==========
    for p in events_dir.rglob("*.parquet"):
        rel = p.relative_to(events_dir)
        tokenize_path(p, tokens_dir / rel, vocab)
    sample = pl.read_parquet(next(tokens_dir.rglob("*.parquet")))
    cov = coverage_report(sample, vocab)
    logger.info("coverage edge=%s unk=%s", cov.edge_bin_rate, cov.unknown_rate)

    # ========== 阶段 4：生成 manifest（训练器的「目录索引」）==========
    from quant_fm.manifest.build_manifest import build_manifest

    manifest = build_manifest(
        tokens_dir,
        train_end=train_dates[-1],  # 此日期及之前 → train
        val_end=val_dates[-1],  # 之后到 val_end → val，再往后 → test
        vocab_path=str(vocab_path),
    )
    manifest_path = data_dir / "manifest.json"
    manifest.save(manifest_path)
    assert manifest.split("train"), "empty train split"
    assert manifest.split("test"), "empty test split"

    # ========== 阶段 5：预训练（调用 train.py，与真实训练同一套代码）==========
    run_dir = workdir / "run"
    cfg = {
        "seed": 0,
        "data": {
            "manifest": str(manifest_path),
            "vocab": str(vocab_path),
            "context": 256,
            "stride": 256,
            "min_len": 8,
            "cache_size": 4,
            "num_workers": 0,
        },
        "model": {
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "ffn_mult": 4.0,
            "dropout": 0.1,
            "max_seq_len": 512,
            "rope_theta": 10000.0,
        },
        "optim": {
            "lr": 1.0e-3,
            "weight_decay": 0.1,
            "betas": [0.9, 0.95],
            "grad_clip": 1.0,
            "warmup_steps": 5,
            "max_steps": 30,
            "batch_size": 4,
            "grad_accum": 1,
            "precision": "fp32",
        },
        "runtime": {
            "out_dir": str(run_dir),
            "log_every": 10,
            "eval_every": 20,
            "ckpt_every": 1000,
            "fsdp": False,
            "device": "cpu",
        },
    }
    cfg_path = workdir / "smoke_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    from quant_fm.pretrain.train import load_checkpoint, resolve_device, train

    train(cfg_path)

    # ========== 阶段 6：用训好的模型抽 embedding ==========
    import torch

    from quant_fm.embedding.extract_hidden import extract_stock_day_embeddings

    device = resolve_device("cpu")
    model = load_checkpoint(run_dir / "final.pt", device)
    # [导读] torch.no_grad()：推理时不算梯度，省内存
    with torch.no_grad():
        embeddings = extract_stock_day_embeddings(
            model,
            manifest.split("test") or manifest.split("train"),
            device,
            context=256,
        )
    assert embeddings.height > 0, "no embeddings produced"

    # ========== 阶段 7：下游选股演示（合成标签，仅验证代码能跑）==========
    _downstream(embeddings, workdir)

    logger.info("SMOKE OK: all stages passed")


def _downstream(embeddings: pl.DataFrame, workdir: Path) -> None:
    """构建合成标签面板并运行排序器/回测/门控。"""
    from quant_fm.downstream.backtest_topk import backtest_topk
    from quant_fm.downstream.evaluate import (
        deflated_sharpe_ratio,
        rank_ic,
        rank_icir,
    )
    from quant_fm.downstream.make_features import build_features
    from quant_fm.downstream.train_ranker import predict, train_ranker

    rng = np.random.default_rng(1)
    emb_cols = [c for c in embeddings.columns if c.startswith("emb_")]
    signal = embeddings[emb_cols[0]].to_numpy()
    fwd = 0.5 * (signal - signal.mean()) / (signal.std() + 1e-9) * 0.01
    fwd = fwd + rng.normal(0, 0.02, size=len(fwd))
    panel = embeddings.select(["date", "symbol"]).with_columns(
        pl.Series("fwd_ret", fwd),
        pl.lit(False).alias("is_st"),
        pl.lit(False).alias("is_halt"),
        pl.lit(False).alias("is_new"),
        pl.lit(False).alias("limit_locked"),
    )

    features = build_features(embeddings, panel, min_names_per_day=2)
    model, history = train_ranker(features, epochs=5, use_attention=True, device="cpu")
    preds = predict(model, features, device="cpu")

    ic = rank_ic(preds, panel)
    icir = rank_icir(ic)
    result = backtest_topk(preds, panel, top_k=2, long_short=True)
    dsr = deflated_sharpe_ratio(
        result.sharpe / np.sqrt(244) if result.sharpe else 0.0,
        n_trials=10,
        n_obs=max(len(result.dates), 2),
        sr_variance=0.25,
    )
    logger.info(
        "downstream: train_ic=%.3f icir=%.3f sharpe=%.2f dsr=%.3f",
        history[-1],
        icir,
        result.sharpe,
        dsr,
    )


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("quant_fm/runs/smoke"))
    args = parser.parse_args()
    run(args.workdir)


if __name__ == "__main__":
    main()
