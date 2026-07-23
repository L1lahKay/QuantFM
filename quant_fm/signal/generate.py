"""从信号日 embedding 和冻结 Ranker 生成唯一生产交付。"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from quant_fm.downstream.make_features import build_scoring_features
from quant_fm.downstream.train_ranker import predict
from quant_fm.signal.artifact import load_ranker_artifact
from quant_fm.signal.schema import validate_scores

logger = logging.getLogger(__name__)


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_scores(
    *,
    embeddings_path: Path,
    ranker_path: Path,
    ranker_metadata_path: Path,
    out_dir: Path,
    device: str = "cpu",
    fm_checkpoint_path: Path | None = None,
    vocab_path: Path | None = None,
    universe_path: Path | None = None,
    allow_in_sample: bool = False,
) -> Path:
    """生成无标签 score，并原子写入 parquet 和 manifest。"""
    embeddings = pl.read_parquet(embeddings_path)
    universe = pl.read_parquet(universe_path) if universe_path else None
    features = build_scoring_features(embeddings, universe=universe)
    model, metadata = load_ranker_artifact(
        ranker_path, ranker_metadata_path, device=device
    )
    training_end = str(metadata.get("training_end_date", ""))
    signal_dates = sorted(str(value) for value in features["date"].unique())
    if not allow_in_sample and training_end and signal_dates[0] <= training_end:
        msg = (
            "signal dates must be strictly after ranker training_end_date; "
            "use --allow-in-sample only for research/smoke"
        )
        raise ValueError(msg)
    scores = predict(
        model,
        features,
        device=device,
        expected_columns=list(metadata["feature_columns"]),
    ).select(["date", "symbol", "score"])
    scores = validate_scores(scores)

    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "scores.parquet"
    scores_tmp = out_dir / "scores.parquet.tmp"
    scores.write_parquet(scores_tmp)
    scores_tmp.replace(scores_path)
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
        "artifacts": {
            "fm_checkpoint_sha256": _sha256(fm_checkpoint_path),
            "vocab_sha256": _sha256(vocab_path),
            "ranker_checkpoint_sha256": _sha256(ranker_path),
        },
    }
    manifest_tmp = out_dir / "signal_manifest.json.tmp"
    manifest_tmp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest_tmp.replace(out_dir / "signal_manifest.json")
    logger.info(
        "score signal → %s rows=%d dates=%d",
        scores_path,
        scores.height,
        scores["date"].n_unique(),
    )
    return scores_path


def main() -> None:
    """运行生产 score 生成器。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument("--ranker-metadata", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fm-checkpoint", type=Path)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--universe", type=Path)
    parser.add_argument("--allow-in-sample", action="store_true")
    args = parser.parse_args()
    generate_scores(
        embeddings_path=args.embeddings,
        ranker_path=args.ranker,
        ranker_metadata_path=args.ranker_metadata,
        out_dir=args.out_dir,
        device=args.device,
        fm_checkpoint_path=args.fm_checkpoint,
        vocab_path=args.vocab,
        universe_path=args.universe,
        allow_in_sample=args.allow_in_sample,
    )


if __name__ == "__main__":
    main()
