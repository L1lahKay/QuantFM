"""从信号日 embedding 和冻结 Ranker 生成唯一生产交付。"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from datetime import date as calendar_date
from pathlib import Path

import polars as pl

from quant_fm.downstream.make_features import build_scoring_features
from quant_fm.downstream.representation import validate_strict_topk_representation
from quant_fm.downstream.train_ranker import predict
from quant_fm.downstream.universe import (
    cross_section_stats,
    validate_pit_universe,
    validate_universe_alignment,
)
from quant_fm.embedding.contract import (
    EmbeddingContract,
    assert_embedding_contract_compatible,
    load_embedding_contract,
    validate_embedding_columns,
)
from quant_fm.signal.artifact import load_ranker_artifact
from quant_fm.signal.schema import validate_scores

logger = logging.getLogger(__name__)


def _canonical_iso_date(value: object, *, context: str) -> str:
    text = str(value)
    try:
        parsed = calendar_date.fromisoformat(text)
    except ValueError as exc:
        msg = f"{context} must contain canonical ISO dates; got {text!r}"
        raise ValueError(msg) from exc
    if parsed.isoformat() != text:
        msg = f"{context} must contain canonical YYYY-MM-DD dates; got {text!r}"
        raise ValueError(msg)
    return text


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
    allow_legacy_embedding_contract: bool = False,
    allow_legacy_training_contract: bool = False,
    require_causal_representation: bool = True,
) -> Path:
    """生成无标签 score，并原子写入 parquet 和 manifest。"""
    embeddings = pl.read_parquet(embeddings_path)
    scoring_embedding_contract = load_embedding_contract(
        embeddings_path,
        required=not allow_legacy_embedding_contract,
        require_vocab=not allow_legacy_embedding_contract,
    )
    representation_gate: dict[str, str | int | bool] | None = None
    if scoring_embedding_contract is not None:
        validate_embedding_columns(
            embeddings.columns,
            scoring_embedding_contract,
            context=str(embeddings_path),
        )
        if require_causal_representation:
            representation_gate = validate_strict_topk_representation(
                scoring_embedding_contract,
                context="ranker scoring embeddings",
            )
    model, metadata = load_ranker_artifact(
        ranker_path,
        ranker_metadata_path,
        device=device,
        allow_legacy_embedding_contract=allow_legacy_embedding_contract,
        allow_legacy_training_contract=allow_legacy_training_contract,
    )
    if not allow_legacy_training_contract:
        if scoring_embedding_contract is None:  # pragma: no cover - loader guards it
            msg = "strict score generation requires a scoring embedding contract"
            raise ValueError(msg)
        identity_inputs = {
            "--fm-checkpoint": (
                fm_checkpoint_path,
                scoring_embedding_contract.fm_checkpoint_sha256,
            ),
            "--vocab": (vocab_path, scoring_embedding_contract.vocab_sha256),
        }
        for flag, (path, expected_hash) in identity_inputs.items():
            if path is None:
                msg = (
                    "strict score generation requires explicit --fm-checkpoint and "
                    f"--vocab identity inputs; missing {flag}"
                )
                raise ValueError(msg)
            observed_hash = _sha256(path)
            if expected_hash is None or observed_hash != expected_hash:
                msg = (
                    f"{flag} SHA-256 does not match the scoring embedding contract: "
                    f"expected={expected_hash!r}, observed={observed_hash!r}"
                )
                raise ValueError(msg)
    training_contract = metadata.get("training_contract", {})
    training_universe = training_contract.get("universe", {})
    strict_universe = training_universe.get("mode") == "daily_pit_file"
    if strict_universe and universe_path is None:
        msg = (
            "ranker was trained with a daily PIT universe; scoring requires an "
            "explicit daily PIT --universe for the same policy"
        )
        raise ValueError(msg)
    objective = metadata.get("objective", {})
    ndcg_ks = [int(value) for value in objective.get("ndcg_ks", [1])]
    required_names = max(ndcg_ks, default=1)
    scoring_universe_contract: dict[str, object] = {
        "format_version": "legacy_unverified",
        "verified": False,
    }
    universe = pl.read_parquet(universe_path) if universe_path else None
    if universe is not None:
        universe, scoring_universe_contract = validate_pit_universe(
            universe,
            required_dates=(str(value) for value in embeddings["date"].unique()),
            min_names_per_day=required_names if strict_universe else 1,
            context="scoring PIT universe",
            require_pit_metadata=strict_universe,
        )
    features = build_scoring_features(
        embeddings,
        universe=universe,
        min_names_per_day=required_names if strict_universe else 1,
    )
    scoring_feature_stats = cross_section_stats(features)
    universe_alignment: dict[str, float | str] | None = None
    if strict_universe:
        stored_contract = training_universe.get("contract")
        retained_stats = training_universe.get("retained_training_features")
        if not isinstance(stored_contract, dict) or not stored_contract.get("verified"):
            msg = "strict ranker metadata is missing a verified PIT universe contract"
            raise ValueError(msg)
        if not isinstance(retained_stats, dict):
            msg = "strict ranker metadata is missing retained training universe widths"
            raise ValueError(msg)
        train_alignment_contract = {**stored_contract, "stats": retained_stats}
        score_alignment_contract = {
            **scoring_universe_contract,
            "stats": scoring_feature_stats,
        }
        universe_alignment = validate_universe_alignment(
            train_alignment_contract,
            score_alignment_contract,
        )
    training_embedding_payload = metadata.get("embedding_contract")
    training_embedding_contract = (
        EmbeddingContract.from_dict(training_embedding_payload, require_vocab=True)
        if training_embedding_payload is not None
        else None
    )
    if (
        training_embedding_contract is not None
        and scoring_embedding_contract is not None
    ):
        assert_embedding_contract_compatible(
            training_embedding_contract,
            scoring_embedding_contract,
            context="ranker training vs scoring",
        )
    training_end = str(metadata.get("training_end_date", ""))
    label_end = str(metadata.get("label_end_date", training_end))
    signal_dates = sorted(
        _canonical_iso_date(value, context="scoring embeddings date")
        for value in features["date"].unique()
    )
    parsed_signal_start = calendar_date.fromisoformat(signal_dates[0])
    parsed_label_end = (
        calendar_date.fromisoformat(
            _canonical_iso_date(label_end, context="ranker label_end_date")
        )
        if label_end
        else None
    )
    if (
        not allow_in_sample
        and parsed_label_end is not None
        and parsed_signal_start <= parsed_label_end
    ):
        msg = (
            "signal dates must be strictly after ranker label_end_date; "
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
        "ranker": {
            "artifact_version": metadata.get("artifact_version"),
            "training_end_date": training_end,
            "label_end_date": label_end,
            "objective": metadata.get("objective"),
            "training_contract": metadata.get("training_contract"),
        },
        "scoring_universe": {
            "file_sha256": _sha256(universe_path),
            "contract": scoring_universe_contract,
            "retained_scoring_features": scoring_feature_stats,
            "alignment": universe_alignment,
        },
        "embedding_representation": (
            scoring_embedding_contract.to_dict()
            if scoring_embedding_contract is not None
            else {"mode": "legacy_unverified"}
        ),
        "strict_representation_gate": representation_gate,
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
    parser.add_argument(
        "--allow-legacy-embedding-contract",
        action="store_true",
        help="仅迁移/诊断：允许缺少 FM/vocab/pooling 表征 sidecar 的旧 embedding",
    )
    parser.add_argument(
        "--allow-legacy-training-contract",
        action="store_true",
        help=("仅研究/迁移诊断：允许非生产训练契约，并允许缺省 FM/vocab 身份文件"),
    )
    parser.add_argument(
        "--allow-noncausal-representation",
        action="store_false",
        dest="require_causal_representation",
        help="仅迁移/诊断：允许旧排序、非因果填充、非重叠 chunk 或旧 pooling",
    )
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
        allow_legacy_embedding_contract=args.allow_legacy_embedding_contract,
        allow_legacy_training_contract=args.allow_legacy_training_contract,
        require_causal_representation=args.require_causal_representation,
    )


if __name__ == "__main__":
    main()
