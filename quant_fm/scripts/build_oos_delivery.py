"""
跨期 OOS 打分与交付：**在 A 期数据训练 Ranker，对严格晚于 FM 训练期的 B 期打分**。

动机：300M FM 的预训练日横跨整个 2025（最后一天 2025-12-31），因此只有 2026 才
「完全晚于」训练数据。本脚本用 2025 的 (embedding, label) 训练截面 Ranker，再对 2026
的 embedding 前向出 ``score`` —— FM、Ranker、vocab 全部 ≤2025，2026 的表征与标签都
从未被任何组件见过，是零泄漏的时序 OOS 信号。

产出（默认 ``<test-workdir>/delivery_oos/``）：
* ``scores.parquet`` —— ``date, symbol, score``（B 期全部信号日）
* ``signal_manifest.json`` —— 稳定信号契约与版本信息

用法::

    uv run python -m quant_fm.scripts.build_oos_delivery \
        --train-emb-dir quant_fm/runs/medium_300m/embeddings \
        --train-panel   quant_fm/runs/medium_300m/panel/daily_panel.parquet \
        --test-emb       quant_fm/runs/oos2026/embeddings/all.parquet \
        --out-dir       quant_fm/runs/oos2026/delivery_oos
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import torch

from quant_fm.downstream.make_features import (
    build_scoring_features,
    build_training_features,
)
from quant_fm.downstream.train_ranker import (
    CrossSectionalRanker,
    RankerConfig,
    predict,
    train_ranker,
)
from quant_fm.signal.schema import validate_scores

logger = logging.getLogger(__name__)

_RANKER_CHECKPOINT = "ranker_checkpoint.pt"
_RANKER_METADATA = "ranker_metadata.json"
_SCORE_STATE = "score_state.json"


def _sha256_16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _load_emb(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path).with_columns(
        pl.col("symbol").cast(pl.Utf8).str.zfill(6)
    )


def _load_train_emb(emb_dir: Path) -> pl.DataFrame:
    """合并 A 期 train/val/test（或 all）embedding 作为训练特征来源。"""
    if (emb_dir / "all.parquet").exists():
        return _load_emb(emb_dir / "all.parquet")
    frames = [
        _load_emb(emb_dir / f"{s}.parquet")
        for s in ("train", "val", "test")
        if (emb_dir / f"{s}.parquet").exists()
    ]
    if not frames:
        msg = f"no embeddings under {emb_dir}"
        raise RuntimeError(msg)
    return pl.concat(frames, how="vertical_relaxed")


def _file_fingerprint(path: Path) -> dict[str, Any]:
    """Cheap local identity used to skip unchanged incremental inputs."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _train_embedding_files(emb_dir: Path) -> list[Path]:
    all_path = emb_dir / "all.parquet"
    if all_path.exists():
        return [all_path]
    return [
        emb_dir / f"{split}.parquet"
        for split in ("train", "val", "test")
        if (emb_dir / f"{split}.parquet").exists()
    ]


def _stable_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def _ranker_cache_spec(
    *,
    train_emb_dir: Path,
    train_panel: Path,
    epochs: int,
    seed: int,
    min_names_per_day: int,
) -> dict[str, Any]:
    emb_files = _train_embedding_files(train_emb_dir)
    if not emb_files:
        msg = f"no embeddings under {train_emb_dir}"
        raise RuntimeError(msg)
    return {
        "train_embeddings": [_file_fingerprint(path) for path in emb_files],
        "train_panel": _file_fingerprint(train_panel),
        "epochs": epochs,
        "seed": seed,
        "min_names_per_day": min_names_per_day,
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _load_cached_ranker(
    *,
    out_dir: Path,
    cache_key: str,
    device: str,
) -> tuple[CrossSectionalRanker, list[float]] | None:
    metadata = _load_json(out_dir / _RANKER_METADATA)
    checkpoint = out_dir / _RANKER_CHECKPOINT
    if (
        metadata is None
        or metadata.get("cache_key") != cache_key
        or not checkpoint.exists()
    ):
        return None
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = RankerConfig(**payload["config"])
    model = CrossSectionalRanker(cfg).to(torch.device(device))
    model.load_state_dict(payload["state_dict"])
    history = [float(value) for value in payload.get("history", [])]
    logger.info("复用冻结 Ranker checkpoint: %s", checkpoint)
    return model, history


def _save_ranker(
    *,
    model: CrossSectionalRanker,
    history: list[float],
    out_dir: Path,
    cache_key: str,
    cache_spec: dict[str, Any],
    feature_columns: list[str],
    training_end_date: str,
) -> None:
    cfg = RankerConfig(
        in_dim=model.proj.in_features,
        hidden=model.proj.out_features,
        depth=sum(1 for layer in model.layers if layer.__class__.__name__ == "_RowMLP"),
        n_heads=next(
            (layer.attn.num_heads for layer in model.layers if hasattr(layer, "attn")),
            4,
        ),
        dropout=next(
            (float(layer.net[2].p) for layer in model.layers if hasattr(layer, "net")),
            0.3,
        ),
        use_attention=any(hasattr(layer, "attn") for layer in model.layers),
    )
    checkpoint = out_dir / _RANKER_CHECKPOINT
    tmp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    torch.save(
        {
            "config": asdict(cfg),
            "state_dict": model.state_dict(),
            "history": history,
            "feature_columns": feature_columns,
        },
        tmp,
    )
    tmp.replace(checkpoint)
    _atomic_json(
        out_dir / _RANKER_METADATA,
        {
            "cache_key": cache_key,
            "cache_spec": cache_spec,
            "feature_columns": feature_columns,
            "training_end_date": training_end_date,
        },
    )
    logger.info("缓存冻结 Ranker → %s", checkpoint)


def _prune_consumed_tokens(test_emb: Path, embedded_dates: list[str]) -> None:
    """Remove token shards only after their durable embeddings have been scored."""
    try:
        workdir = test_emb.resolve().parents[2]
    except IndexError:
        return
    marker = workdir / "data" / ".prune_embedded_tokens"
    tokens_dir = workdir / "tokens"
    if not marker.exists() or not tokens_dir.is_dir():
        return

    receipts_dir = workdir / "data" / "pruned_token_receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    token_root = tokens_dir.resolve()
    for date in embedded_dates:
        receipt = receipts_dir / f"{date}.json"
        if receipt.exists():
            continue
        shards = sorted(tokens_dir.rglob(f"{date}.parquet"))
        if not shards:
            continue
        total_bytes = 0
        safe_shards: list[Path] = []
        for shard in shards:
            resolved = shard.resolve()
            if not resolved.is_relative_to(token_root):
                msg = f"refuse to prune token outside {token_root}: {resolved}"
                raise RuntimeError(msg)
            total_bytes += resolved.stat().st_size
            safe_shards.append(resolved)

        pending_receipt = receipt.with_suffix(".json.pending")
        _atomic_json(
            pending_receipt,
            {
                "date": date,
                "embedding": _file_fingerprint(test_emb),
                "token_shards": len(safe_shards),
                "token_bytes": total_bytes,
            },
        )
        for shard in safe_shards:
            shard.unlink(missing_ok=True)
        pending_receipt.replace(receipt)
        logger.info(
            "已释放 embedding 消费完毕的 tokens: date=%s shards=%d size=%.2fGiB",
            date,
            len(safe_shards),
            total_bytes / (1 << 30),
        )

    # Best-effort cleanup of now-empty market/symbol directories.
    for path in sorted(tokens_dir.glob("*/*")):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def _build_oos_delivery_locked(
    *,
    train_emb_dir: Path,
    train_panel: Path,
    test_emb: Path,
    out_dir: Path,
    test_panel: Path | None = None,
    epochs: int = 30,
    seed: int = 42,
    device: str = "cuda:0",
    min_names_per_day: int = 20,
) -> Path:
    """训练(A期) → 打分(B期) → 落盘 OOS 交付，返回交付目录。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_spec = _ranker_cache_spec(
        train_emb_dir=train_emb_dir,
        train_panel=train_panel,
        epochs=epochs,
        seed=seed,
        min_names_per_day=min_names_per_day,
    )
    ranker_cache_key = _stable_key(cache_spec)
    test_emb_fingerprint = _file_fingerprint(test_emb)
    score_state_path = out_dir / _SCORE_STATE
    prior_state = _load_json(score_state_path)
    scores_path = out_dir / "scores.parquet"

    # 编排器即使在没有新日期时每轮调用，也应是 O(1) no-op。
    if (
        prior_state is not None
        and prior_state.get("ranker_cache_key") == ranker_cache_key
        and prior_state.get("test_emb") == test_emb_fingerprint
        and scores_path.exists()
        and (out_dir / "signal_manifest.json").exists()
    ):
        logger.info("OOS embedding 无变化，跳过重复训练与打分。")
        return out_dir

    cached = _load_cached_ranker(
        out_dir=out_dir,
        cache_key=ranker_cache_key,
        device=device,
    )
    if cached is None:
        # A 期固定训练集只训练一次；后续 OOS 增量仅加载 checkpoint。
        pan_a = pl.read_parquet(train_panel).filter(pl.col("fwd_ret").is_not_null())
        feat_a = build_training_features(
            _load_train_emb(train_emb_dir),
            pan_a,
            min_names_per_day=min_names_per_day,
        )
        if feat_a.is_empty():
            msg = "训练期特征为空"
            raise RuntimeError(msg)
        logger.info(
            "train(A) features: %d rows, %d days",
            feat_a.height,
            feat_a["date"].n_unique(),
        )
        model, history = train_ranker(feat_a, epochs=epochs, device=device, seed=seed)
        ranker_columns = [
            column
            for column in feat_a.columns
            if column.startswith(("emb_", "factor_"))
        ]
        _save_ranker(
            model=model,
            history=history,
            out_dir=out_dir,
            cache_key=ranker_cache_key,
            cache_spec=cache_spec,
            feature_columns=ranker_columns,
            training_end_date=max(feat_a["date"].to_list()),
        )
    else:
        model, history = cached
        ranker_metadata = _load_json(out_dir / _RANKER_METADATA) or {}
        ranker_columns = list(ranker_metadata.get("feature_columns", []))
        if not ranker_columns:
            msg = "cached ranker metadata is missing feature_columns"
            raise RuntimeError(msg)

    # B 期（OOS）只对尚未处理的新日期打分。
    emb_b = _load_emb(test_emb)
    all_embedding_dates = sorted(str(value) for value in emb_b["date"].unique())
    state_compatible = (
        prior_state is not None
        and prior_state.get("ranker_cache_key") == ranker_cache_key
        and scores_path.exists()
    )
    processed_dates = (
        {str(value) for value in prior_state.get("processed_embedding_dates", [])}
        if state_compatible and prior_state is not None
        else set()
    )
    new_dates = [date for date in all_embedding_dates if date not in processed_dates]
    existing_scores = pl.read_parquet(scores_path) if state_compatible else None

    if new_dates:
        feat_b_new = build_scoring_features(
            emb_b.filter(pl.col("date").is_in(new_dates)),
            min_names_per_day=min_names_per_day,
        )
        if feat_b_new.is_empty() and existing_scores is None:
            msg = "OOS 期打分特征为空"
            raise RuntimeError(msg)
        new_scores = (
            predict(
                model,
                feat_b_new,
                device=device,
                expected_columns=ranker_columns,
            )
            if not feat_b_new.is_empty()
            else None
        )
    else:
        new_scores = None

    score_frames = [
        frame for frame in (existing_scores, new_scores) if frame is not None
    ]
    if not score_frames:
        msg = "没有可交付的 OOS score"
        raise RuntimeError(msg)
    scores = pl.concat(score_frames, how="vertical_relaxed")
    scores = scores.unique(subset=["date", "symbol"], keep="last").sort(
        ["date", "symbol"]
    )
    scores = validate_scores(
        scores.select(
            pl.col("date").cast(pl.Utf8),
            pl.col("symbol").cast(pl.Utf8).str.zfill(6),
            pl.col("score").cast(pl.Float64),
        )
    )
    scores_tmp = scores_path.with_suffix(".parquet.tmp")
    scores.write_parquet(scores_tmp)
    scores_tmp.replace(scores_path)
    logger.info(
        "scores.parquet rows=%d days=%d new_embedding_days=%d",
        scores.height,
        scores["date"].n_unique(),
        len(new_dates),
    )

    manifest = {
        "format_version": "1.0",
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "score_semantics": {
            "direction": "higher_is_more_bullish",
            "availability": "available_after_signal_date_close",
            "comparability": "cross_sectional_within_date",
        },
        "data": {
            "file": "scores.parquet",
            "schema": {"date": "string", "symbol": "string", "score": "float64"},
            "primary_key": ["date", "symbol"],
            "rows": scores.height,
            "dates": scores["date"].n_unique(),
            "date_min": scores["date"].min(),
            "date_max": scores["date"].max(),
        },
        "ranker": {
            "epochs": epochs,
            "seed": seed,
            "device": device,
            "min_names_per_day": min_names_per_day,
            "cache_key": ranker_cache_key,
            "checkpoint": _RANKER_CHECKPOINT,
        },
        "note": "score 越大越看多；score(T) 仅 T 收盘后可用；2026 全部晚于 FM 训练期。",
    }
    _atomic_json(out_dir / "signal_manifest.json", manifest)
    _atomic_json(
        score_state_path,
        {
            "ranker_cache_key": ranker_cache_key,
            "test_emb": test_emb_fingerprint,
            "processed_embedding_dates": all_embedding_dates,
        },
    )
    _prune_consumed_tokens(test_emb, all_embedding_dates)
    logger.info("OOS 信号清单 → %s", out_dir / "signal_manifest.json")
    return out_dir


def build_oos_delivery(
    *,
    train_emb_dir: Path,
    train_panel: Path,
    test_emb: Path,
    out_dir: Path,
    test_panel: Path | None = None,
    epochs: int = 30,
    seed: int = 42,
    device: str = "cuda:0",
    min_names_per_day: int = 20,
) -> Path:
    """串行 Ranker 打分，并保证成功或异常路径都会释放 GPU 锁。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    score_lock = out_dir / ".score.lock"
    score_lock.write_text(str(os.getpid()), encoding="utf-8")
    try:
        return _build_oos_delivery_locked(
            train_emb_dir=train_emb_dir,
            train_panel=train_panel,
            test_emb=test_emb,
            test_panel=test_panel,
            out_dir=out_dir,
            epochs=epochs,
            seed=seed,
            device=device,
            min_names_per_day=min_names_per_day,
        )
    finally:
        score_lock.unlink(missing_ok=True)


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-emb-dir", type=Path, required=True)
    p.add_argument("--train-panel", type=Path, required=True)
    p.add_argument("--test-emb", type=Path, required=True)
    p.add_argument("--test-panel", type=Path, help="已弃用；生产打分不读取未来标签")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--min-names-per-day", type=int, default=20)
    args = p.parse_args()

    build_oos_delivery(
        train_emb_dir=args.train_emb_dir,
        train_panel=args.train_panel,
        test_emb=args.test_emb,
        test_panel=args.test_panel,
        out_dir=args.out_dir,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        min_names_per_day=args.min_names_per_day,
    )


if __name__ == "__main__":
    main()
