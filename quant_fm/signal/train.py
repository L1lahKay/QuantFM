"""离线训练并冻结生产 Ranker。"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import polars as pl

from quant_fm.downstream.make_features import build_training_features
from quant_fm.downstream.train_ranker import feature_columns, train_ranker
from quant_fm.signal.artifact import save_ranker_artifact

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_signal_ranker(
    *,
    embeddings_path: Path,
    panel_path: Path,
    out_dir: Path,
    epochs: int = 30,
    seed: int = 42,
    device: str = "cpu",
    min_names_per_day: int = 20,
) -> tuple[Path, Path]:
    """从历史标签训练 Ranker；该函数是唯一允许读取 ``fwd_ret`` 的信号步骤。"""
    embeddings = pl.read_parquet(embeddings_path)
    panel = pl.read_parquet(panel_path).filter(pl.col("fwd_ret").is_not_null())
    features = build_training_features(
        embeddings, panel, min_names_per_day=min_names_per_day
    )
    if features.is_empty():
        msg = "training features are empty"
        raise RuntimeError(msg)
    model, history = train_ranker(features, epochs=epochs, seed=seed, device=device)
    checkpoint = out_dir / "ranker.pt"
    metadata = out_dir / "ranker_metadata.json"
    save_ranker_artifact(
        model,
        checkpoint,
        metadata,
        feature_columns=feature_columns(features),
        training_end_date=max(features["date"].to_list()),
        seed=seed,
        history=history,
        provenance={
            "training_embeddings_sha256": _sha256(embeddings_path),
            "training_panel_sha256": _sha256(panel_path),
        },
    )
    logger.info("frozen ranker → %s", checkpoint)
    return checkpoint, metadata


def main() -> None:
    """运行离线 Ranker 训练。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-names-per-day", type=int, default=20)
    args = parser.parse_args()
    train_signal_ranker(
        embeddings_path=args.embeddings,
        panel_path=args.panel,
        out_dir=args.out_dir,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        min_names_per_day=args.min_names_per_day,
    )


if __name__ == "__main__":
    main()
