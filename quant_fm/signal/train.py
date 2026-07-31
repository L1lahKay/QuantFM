"""离线训练并冻结生产 Ranker。"""

from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import asdict
from pathlib import Path

import polars as pl

from quant_fm.downstream.make_features import build_training_features
from quant_fm.downstream.representation import validate_strict_topk_representation
from quant_fm.downstream.return_spec import (
    read_trading_calendar,
    validate_execution_panel_contract,
)
from quant_fm.downstream.train_ranker import (
    RankerObjectiveConfig,
    chronological_ranker_split,
    feature_columns,
    fit_ranker,
)
from quant_fm.downstream.universe import (
    cross_section_stats,
    validate_pit_universe,
)
from quant_fm.embedding.contract import (
    load_embedding_contract,
    validate_embedding_columns,
)
from quant_fm.signal.artifact import (
    SIGNAL_FEATURE_TARGET_SPEC_VERSION,
    save_ranker_artifact,
)

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
    universe_path: Path | None,
    calendar_path: Path | None,
    out_dir: Path,
    epochs: int = 30,
    seed: int = 42,
    device: str = "cpu",
    min_names_per_day: int = 20,
    val_days: int = 10,
    purge_days: int = 2,
    patience: int = 8,
    require_validation: bool = True,
    allow_legacy_panel: bool = False,
    allow_legacy_embedding_contract: bool = False,
    allow_legacy_training_contract: bool = False,
    require_causal_representation: bool = True,
    objective: RankerObjectiveConfig | None = None,
) -> tuple[Path, Path]:
    """从历史标签训练 Ranker；该函数是唯一允许读取 ``fwd_ret`` 的信号步骤。"""
    embeddings = pl.read_parquet(embeddings_path)
    embedding_contract = load_embedding_contract(
        embeddings_path,
        required=not allow_legacy_embedding_contract,
        require_vocab=not allow_legacy_embedding_contract,
    )
    representation_gate: dict[str, str | int | bool] | None = None
    if embedding_contract is not None:
        validate_embedding_columns(
            embeddings.columns,
            embedding_contract,
            context=str(embeddings_path),
        )
        if require_causal_representation:
            representation_gate = validate_strict_topk_representation(
                embedding_contract,
                context="ranker training embeddings",
            )
    panel = pl.read_parquet(panel_path)
    trading_calendar = (
        read_trading_calendar(calendar_path) if calendar_path is not None else None
    )
    execution_contract = validate_execution_panel_contract(
        panel,
        trading_calendar=trading_calendar,
        require_calendar_verification=not allow_legacy_panel,
        allow_legacy=allow_legacy_panel,
    )
    objective_cfg = objective or RankerObjectiveConfig()
    objective_cfg.validate()
    if not allow_legacy_panel:
        if universe_path is None:
            msg = "strict Top-K ranker training requires a daily PIT universe"
            raise ValueError(msg)
        if calendar_path is None:
            msg = "strict Top-K ranker training requires the original trading calendar"
            raise ValueError(msg)
        exit_lag = int(execution_contract["exit_day_lag"])
        if purge_days < exit_lag:
            msg = (
                "purge_days must cover the execution label horizon: "
                f"purge_days={purge_days}, exit_day_lag={exit_lag}"
            )
            raise ValueError(msg)
    universe = pl.read_parquet(universe_path) if universe_path is not None else None
    universe_contract: dict[str, object] = {
        "format_version": "legacy_unverified",
        "verified": False,
    }
    if universe is not None:
        candidate_dates = sorted(
            {str(value) for value in embeddings["date"].unique()}
            & {str(value) for value in panel["date"].unique()}
        )
        universe, universe_contract = validate_pit_universe(
            universe,
            required_dates=candidate_dates,
            min_names_per_day=(
                max(objective_cfg.ndcg_ks) if not allow_legacy_panel else 1
            ),
            context="training PIT universe",
            require_pit_metadata=not allow_legacy_panel,
        )
    features = build_training_features(
        embeddings,
        panel,
        universe=universe,
        min_names_per_day=(min_names_per_day if allow_legacy_panel else 1),
    )
    if features.is_empty():
        msg = "training features are empty"
        raise RuntimeError(msg)
    daily_min = int(features.group_by("date").len()["len"].min())
    retained_universe_stats = cross_section_stats(features)
    required_daily_names = max(max(objective_cfg.ndcg_ks), min_names_per_day)
    if not allow_legacy_panel and daily_min < required_daily_names:
        msg = (
            "strict Top-K ranker training has an undersized daily universe: "
            f"minimum={daily_min}, required={required_daily_names}"
        )
        raise ValueError(msg)
    train_features, val_features, time_split = chronological_ranker_split(
        features,
        val_days=val_days,
        purge_days=purge_days,
        require_validation=require_validation,
    )
    training = fit_ranker(
        train_features,
        val_features=val_features,
        epochs=epochs,
        patience=patience,
        seed=seed,
        device=device,
        objective=objective_cfg,
    )
    model = training.model
    checkpoint = out_dir / "ranker.pt"
    metadata = out_dir / "ranker_metadata.json"
    feature_dates = features["date"].unique().to_list()
    label_end_date = (
        str(
            panel.filter(pl.col("date").cast(pl.Utf8).is_in(feature_dates))[
                "exit_date"
            ].max()
        )
        if "exit_date" in panel.columns
        else str(max(feature_dates))
    )
    save_ranker_artifact(
        model,
        checkpoint,
        metadata,
        feature_columns=feature_columns(features),
        training_end_date=max(features["date"].to_list()),
        label_end_date=label_end_date,
        seed=seed,
        objective=objective_cfg,
        embedding_contract=embedding_contract,
        allow_legacy_embedding_contract=allow_legacy_embedding_contract,
        allow_legacy_training_contract=allow_legacy_training_contract,
        history=training.history,
        training_contract={
            "execution_contract": execution_contract,
            "representation_gate": representation_gate,
            "time_split": time_split,
            "feature_target_spec_version": SIGNAL_FEATURE_TARGET_SPEC_VERSION,
            "universe": {
                "mode": (
                    "daily_pit_file"
                    if universe_path is not None
                    else "legacy_unverified"
                ),
                "sha256": _sha256(universe_path) if universe_path is not None else None,
                "daily_names_min": daily_min,
                "contract": universe_contract,
                "retained_training_features": retained_universe_stats,
            },
            "selection": {
                "best_epoch": training.best_epoch,
                "best_val_ic": training.best_val_ic,
                "best_val_ndcg": training.best_val_ndcg,
                "best_selection_score": training.best_selection_score,
                "stopped_early": training.stopped_early,
            },
            "objective": asdict(objective_cfg),
        },
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
    parser.add_argument("--universe", type=Path)
    parser.add_argument(
        "--calendar",
        type=Path,
        help="生成 execution panel 所用的完整交易日历；严格训练必填",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-names-per-day", type=int, default=20)
    parser.add_argument("--val-days", type=int, default=10)
    parser.add_argument("--purge-days", type=int, default=2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--allow-no-validation",
        action="store_false",
        dest="require_validation",
        help="仅小样本诊断：日期不足时允许不留验证集",
    )
    parser.add_argument(
        "--allow-legacy-panel",
        action="store_true",
        help="仅诊断：允许缺少严格 execution contract/PIT 股票池",
    )
    parser.add_argument(
        "--allow-legacy-embedding-contract",
        action="store_true",
        help="仅迁移/诊断：允许缺少 FM/vocab/pooling 表征 sidecar 的旧 embedding",
    )
    parser.add_argument(
        "--allow-legacy-training-contract",
        action="store_true",
        help=(
            "仅研究/迁移诊断：允许非生产 objective、无验证集或不完整的 "
            "execution/representation/PIT 训练契约"
        ),
    )
    parser.add_argument(
        "--allow-noncausal-representation",
        action="store_false",
        dest="require_causal_representation",
        help="仅迁移/诊断：允许旧排序、非因果填充、非重叠 chunk 或旧 pooling",
    )
    args = parser.parse_args()
    train_signal_ranker(
        embeddings_path=args.embeddings,
        panel_path=args.panel,
        universe_path=args.universe,
        calendar_path=args.calendar,
        out_dir=args.out_dir,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        min_names_per_day=args.min_names_per_day,
        val_days=args.val_days,
        purge_days=args.purge_days,
        patience=args.patience,
        require_validation=args.require_validation,
        allow_legacy_panel=args.allow_legacy_panel,
        allow_legacy_embedding_contract=args.allow_legacy_embedding_contract,
        allow_legacy_training_contract=args.allow_legacy_training_contract,
        require_causal_representation=args.require_causal_representation,
    )


if __name__ == "__main__":
    main()
