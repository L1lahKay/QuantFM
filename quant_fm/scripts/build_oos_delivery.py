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
        --train-universe quant_fm/runs/medium_300m/panel/pit_universe.parquet \
        --train-calendar quant_fm/data/trading_calendar_2025_plus_t2.txt \
        --test-emb       quant_fm/runs/oos2026/embeddings/all.parquet \
        --test-universe  quant_fm/runs/oos2026/panel/pit_universe.parquet \
        --pretrain-acceptance quant_fm/runs/medium_300m/pretrain_acceptance.json \
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
from datetime import date as calendar_date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import torch

from quant_fm.downstream.make_features import (
    build_scoring_features,
    build_training_features,
)
from quant_fm.downstream.representation import validate_strict_topk_representation
from quant_fm.downstream.return_spec import (
    read_trading_calendar,
    validate_execution_panel_contract,
)
from quant_fm.downstream.train_ranker import (
    CrossSectionalRanker,
    RankerConfig,
    RankerObjectiveConfig,
    chronological_ranker_split,
    fit_ranker,
    predict,
)
from quant_fm.downstream.universe import (
    cross_section_stats as _cross_section_stats,
)
from quant_fm.downstream.universe import (
    validate_pit_universe,
    validate_universe_alignment,
)
from quant_fm.embedding.contract import (
    assert_embedding_contract_compatible,
    load_compatible_embedding_contracts,
    load_embedding_contract,
    validate_embedding_columns,
)
from quant_fm.scripts.validate_pretrain_lineage import validate_pretrain_lineage
from quant_fm.signal.schema import validate_scores

if TYPE_CHECKING:
    from quant_fm.embedding.contract import EmbeddingContract

logger = logging.getLogger(__name__)

_RANKER_CHECKPOINT = "ranker_checkpoint.pt"
_RANKER_METADATA = "ranker_metadata.json"
_SCORE_STATE = "score_state.json"
_RANKER_CHECKPOINT_VERSION = "multi_lambda_ndcg_v1"
_FEATURE_TARGET_SPEC_VERSION = "strict_exec_percentile_mad_head_gain_v1"
_LABEL_HORIZON_CANDIDATES = (
    "label_availability_date",
    "label_available_date",
    "label_availability",
    "label_end_date",
    "next_date",
    "exit_date",
)
_LABEL_CUTOFF_POLICY_VERSION = "strict_oos_label_horizon_v1"
_PRODUCTION_RETURN_SPEC = "vwap_t1_vwap_t2"
_FROZEN_PRODUCTION_OBJECTIVE = RankerObjectiveConfig(
    ndcg_ks=(50, 300, 350),
    ndcg_k_weights=(0.20, 0.60, 0.20),
    head_weight=1.0,
    global_ic_weight=0.30,
    aux_huber_weight=0.05,
    aux_huber_beta=0.5,
    pair_samples_per_day=8192,
    hard_pair_fraction=0.75,
    min_label_rank_gap=0.02,
    score_temperature=1.0,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_16(path: Path) -> str:
    return _sha256(path)[:16]


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


def _embedding_representation_spec(
    train_emb_dir: Path,
    test_emb: Path,
    *,
    allow_legacy: bool,
) -> tuple[EmbeddingContract | None, dict[str, Any]]:
    """读取训练/评分 sidecar，并拒绝同维度但语义不同的 embedding。"""
    train_paths = _train_embedding_files(train_emb_dir)
    train_contract = load_compatible_embedding_contracts(
        train_paths,
        required=not allow_legacy,
        require_vocab=not allow_legacy,
        context="ranker training embeddings",
    )
    scoring_contract = load_embedding_contract(
        test_emb,
        required=not allow_legacy,
        require_vocab=not allow_legacy,
    )
    if scoring_contract is not None:
        validate_embedding_columns(
            list(pl.read_parquet_schema(test_emb).names()),
            scoring_contract,
            context=str(test_emb),
        )
    if train_contract is not None and scoring_contract is not None:
        assert_embedding_contract_compatible(
            train_contract,
            scoring_contract,
            context="ranker training vs OOS scoring",
        )
        strict_gate = None
        if not allow_legacy:
            strict_gate = validate_strict_topk_representation(
                train_contract,
                context="ranker training/OOS embeddings",
            )
        return train_contract, {
            "mode": "verified",
            "fingerprint": train_contract.fingerprint(),
            "contract": train_contract.to_dict(),
            "strict_topk_gate": strict_gate,
        }
    logger.warning(
        "legacy embedding compatibility override: FM/vocab/pooling identity unverified"
    )
    return None, {"mode": "legacy_unverified", "contract": None}


def _stable_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def _stable_sha256(payload: dict[str, Any]) -> str:
    """Return a full, deterministic SHA-256 for a JSON lineage payload."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    """解析逗号分隔的正整数。"""
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        msg = f"invalid comma-separated integer list: {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if not parsed:
        msg = "integer list must not be empty"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    """解析逗号分隔的浮点数。"""
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        msg = f"invalid comma-separated float list: {value!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    if not parsed:
        msg = "float list must not be empty"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _json_compatible(value: Any) -> Any:
    """把 tuple 等配置值转换为写入 JSON 后仍可稳定比较的形式。"""
    return json.loads(json.dumps(value, sort_keys=True))


def _parse_canonical_iso_date(value: str, *, field: str) -> calendar_date:
    """Require the canonical ``YYYY-MM-DD`` spelling before date comparisons."""
    if not isinstance(value, str):
        msg = f"{field} must be a YYYY-MM-DD string"
        raise TypeError(msg)
    try:
        parsed = calendar_date.fromisoformat(value)
    except ValueError as exc:
        msg = f"{field} must be a valid YYYY-MM-DD date, got {value!r}"
        raise ValueError(msg) from exc
    if parsed.isoformat() != value:
        msg = f"{field} must use canonical YYYY-MM-DD format, got {value!r}"
        raise ValueError(msg)
    return parsed


def _validate_objective_policy(
    objective: RankerObjectiveConfig,
    *,
    allow_research_objective_return_spec_override: bool,
) -> None:
    """Keep the production objective frozen independently of legacy input handling."""
    objective.validate()
    if (
        objective != _FROZEN_PRODUCTION_OBJECTIVE
        and not allow_research_objective_return_spec_override
    ):
        msg = (
            "production OOS delivery requires the frozen RankerObjectiveConfig; "
            "custom objectives require "
            "allow_research_objective_return_spec_override=True"
        )
        raise ValueError(msg)


def _delivery_policy(
    *,
    execution_contract: dict[str, Any],
    objective: RankerObjectiveConfig,
    allow_legacy_training_panel: bool,
    allow_research_objective_return_spec_override: bool,
) -> dict[str, Any]:
    """Validate production invariants and describe any non-production override."""
    observed_return_spec = execution_contract.get("return_spec")
    verified_contract = execution_contract.get("verified") is True
    if (
        (not allow_legacy_training_panel or verified_contract)
        and observed_return_spec != _PRODUCTION_RETURN_SPEC
        and not allow_research_objective_return_spec_override
    ):
        msg = (
            "production OOS delivery requires "
            f"execution_contract.return_spec={_PRODUCTION_RETURN_SPEC!r}; "
            "alternate return specs require "
            "allow_research_objective_return_spec_override=True"
        )
        raise ValueError(msg)

    objective_matches = objective == _FROZEN_PRODUCTION_OBJECTIVE
    return_spec_matches = observed_return_spec == _PRODUCTION_RETURN_SPEC
    if allow_legacy_training_panel and allow_research_objective_return_spec_override:
        mode = "legacy_research_override"
    elif allow_legacy_training_panel:
        mode = "legacy_diagnostic"
    elif allow_research_objective_return_spec_override:
        mode = "research_override"
    else:
        mode = "strict_production"
    override_reasons: list[str] = []
    if allow_legacy_training_panel:
        override_reasons.append("legacy_input_contracts_allowed")
    if not objective_matches:
        override_reasons.append("custom_ranker_objective")
    if observed_return_spec is not None and not return_spec_matches:
        override_reasons.append("alternate_return_spec")
    if allow_research_objective_return_spec_override and not override_reasons:
        override_reasons.append("explicit_research_override_enabled")
    return {
        "mode": mode,
        "production_eligible": mode == "strict_production",
        "frozen_return_spec": _PRODUCTION_RETURN_SPEC,
        "observed_return_spec": observed_return_spec,
        "return_spec_matches_frozen": return_spec_matches,
        "frozen_objective": _json_compatible(asdict(_FROZEN_PRODUCTION_OBJECTIVE)),
        "objective_matches_frozen": objective_matches,
        "allow_legacy_training_panel": allow_legacy_training_panel,
        "allow_research_objective_return_spec_override": (
            allow_research_objective_return_spec_override
        ),
        "override_reasons": override_reasons,
    }


def _foundation_model_provenance(
    *,
    lineage_report: dict[str, Any] | None,
    parsed_training_end: calendar_date | None,
    parsed_oos_start: calendar_date,
) -> dict[str, Any]:
    """Normalize the FM cutoff and bind strict deliveries to immutable hashes."""
    if lineage_report is None:
        return {
            "lineage_mode": "legacy_manual_unverified",
            "training_end_date": (
                parsed_training_end.isoformat()
                if parsed_training_end is not None
                else None
            ),
            "oos_start_date": parsed_oos_start.isoformat(),
            "verified_before_oos": (
                parsed_training_end is not None
                and parsed_training_end < parsed_oos_start
            ),
            "lineage_report_sha256": None,
            "pretrain_acceptance_sha256": None,
            "fm_checkpoint_sha256": None,
            "lineage_report": None,
        }

    normalized = _json_compatible(lineage_report)
    if normalized.get("status") != "verified":
        msg = "strict FM lineage report must have status='verified'"
        raise ValueError(msg)
    derived_end = _parse_canonical_iso_date(
        normalized.get("effective_training_end"),
        field="lineage effective_training_end",
    )
    acceptance = normalized.get("acceptance")
    checkpoint = normalized.get("fm_checkpoint")
    if not isinstance(acceptance, dict) or not isinstance(checkpoint, dict):
        msg = "strict FM lineage report is missing acceptance/fm_checkpoint identity"
        raise TypeError(msg)
    acceptance_sha256 = acceptance.get("sha256")
    checkpoint_sha256 = checkpoint.get("sha256")
    for field, value in (
        ("acceptance.sha256", acceptance_sha256),
        ("fm_checkpoint.sha256", checkpoint_sha256),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            msg = f"strict FM lineage {field} must be a lowercase SHA-256"
            raise ValueError(msg)
    if parsed_training_end is not None and parsed_training_end != derived_end:
        msg = (
            "declared FM training end does not match verified lineage: "
            f"declared={parsed_training_end.isoformat()}, "
            f"derived={derived_end.isoformat()}"
        )
        raise ValueError(msg)
    if derived_end >= parsed_oos_start:
        msg = (
            "FM training period overlaps OOS: "
            f"training_end={derived_end.isoformat()}, "
            f"oos_start={parsed_oos_start.isoformat()}"
        )
        raise ValueError(msg)
    return {
        "lineage_mode": "verified_pretrain_lineage",
        "training_end_date": derived_end.isoformat(),
        "oos_start_date": parsed_oos_start.isoformat(),
        "verified_before_oos": True,
        "lineage_report_sha256": _stable_sha256(normalized),
        "pretrain_acceptance_sha256": acceptance_sha256,
        "fm_checkpoint_sha256": checkpoint_sha256,
        "lineage_report": normalized,
    }


def _oos_start_date(test_emb: Path) -> str:
    """只读取得 OOS embedding 的最早信号日。"""
    schema = pl.read_parquet_schema(test_emb)
    if "date" not in schema:
        msg = f"OOS embedding is missing date column: {test_emb}"
        raise RuntimeError(msg)
    value = (
        pl.scan_parquet(test_emb)
        .select(pl.col("date").cast(pl.Utf8, strict=False).min())
        .collect()
        .item()
    )
    if value is None:
        msg = f"OOS embedding contains no dates: {test_emb}"
        raise RuntimeError(msg)
    return str(value)


def _label_cutoff_policy(train_panel: Path, oos_start_date: str) -> dict[str, Any]:
    """描述训练标签相对 OOS 起点的严格可用性规则。"""
    schema = pl.read_parquet_schema(train_panel)
    horizon_columns = [
        name
        for name in schema.names()
        if name in _LABEL_HORIZON_CANDIDATES or name.startswith("label_availability")
    ]
    return {
        "policy_version": _LABEL_CUTOFF_POLICY_VERSION,
        "oos_start_date": oos_start_date,
        "comparison": "strictly_before_oos_start",
        "label_horizon_columns": horizon_columns,
        "fallback_when_horizon_missing": "signal_date_strictly_before_oos_start",
    }


def _filter_training_panel_for_oos(
    panel: pl.DataFrame,
    *,
    label_cutoff: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """剔除在 OOS 起点当日或之后才可确定的训练标签。"""
    required = {"date", "symbol", "fwd_ret"}
    missing = sorted(required - set(panel.columns))
    if missing:
        msg = f"training panel is missing columns: {missing}"
        raise RuntimeError(msg)

    oos_start = str(label_cutoff["oos_start_date"])
    horizon_columns = list(label_cutoff["label_horizon_columns"])
    numeric_return = pl.col("fwd_ret").cast(pl.Float64, strict=False)
    labeled = panel.filter(numeric_return.is_finite().fill_null(False))
    cutoff_exprs = [
        pl.col("date").cast(pl.Utf8, strict=False).lt(oos_start).fill_null(False)
    ]
    cutoff_exprs.extend(
        pl.col(column).cast(pl.Utf8, strict=False).lt(oos_start).fill_null(False)
        for column in horizon_columns
    )
    filtered = labeled.filter(pl.all_horizontal(cutoff_exprs))
    result = {
        **label_cutoff,
        "input_rows": panel.height,
        "finite_label_rows": labeled.height,
        "retained_label_rows": filtered.height,
        "excluded_label_rows": labeled.height - filtered.height,
        "retained_date_min": filtered["date"].min()
        if not filtered.is_empty()
        else None,
        "retained_date_max": filtered["date"].max()
        if not filtered.is_empty()
        else None,
    }
    if filtered.is_empty():
        horizon_description = ",".join(horizon_columns) or "date fallback"
        msg = (
            "no finite training labels are strictly available before OOS start "
            f"{oos_start} using {horizon_description}"
        )
        raise RuntimeError(msg)
    return filtered, result


def _ranker_cache_spec(
    *,
    train_emb_dir: Path,
    train_panel: Path,
    train_universe: Path | None,
    train_calendar: Path | None,
    epochs: int,
    seed: int,
    min_names_per_day: int,
    label_cutoff: dict[str, Any],
    execution_contract: dict[str, Any],
    embedding_representation: dict[str, Any],
    foundation_model: dict[str, Any],
    training_config: dict[str, Any],
) -> dict[str, Any]:
    emb_files = _train_embedding_files(train_emb_dir)
    if not emb_files:
        msg = f"no embeddings under {train_emb_dir}"
        raise RuntimeError(msg)
    return {
        "checkpoint_version": _RANKER_CHECKPOINT_VERSION,
        "feature_target_spec_version": _FEATURE_TARGET_SPEC_VERSION,
        "train_embeddings": [_file_fingerprint(path) for path in emb_files],
        "train_panel": _file_fingerprint(train_panel),
        "train_universe": (
            _file_fingerprint(train_universe) if train_universe is not None else None
        ),
        "train_calendar": (
            _file_fingerprint(train_calendar) if train_calendar is not None else None
        ),
        "epochs": epochs,
        "seed": seed,
        "min_names_per_day": min_names_per_day,
        "label_cutoff": label_cutoff,
        "execution_contract": execution_contract,
        "embedding_representation": embedding_representation,
        "foundation_model": foundation_model,
        "training_config": training_config,
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
) -> tuple[CrossSectionalRanker, list[dict[str, float | int | None]]] | None:
    metadata = _load_json(out_dir / _RANKER_METADATA)
    checkpoint = out_dir / _RANKER_CHECKPOINT
    if (
        metadata is None
        or metadata.get("cache_key") != cache_key
        or not checkpoint.exists()
    ):
        return None
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("checkpoint_version") != _RANKER_CHECKPOINT_VERSION:
        logger.info("忽略旧代 Ranker checkpoint，将按新 Top-K 目标重训。")
        return None
    if payload.get("cache_key") != cache_key:
        msg = "ranker checkpoint cache key does not match sidecar metadata"
        raise RuntimeError(msg)
    for field in (
        "config",
        "feature_columns",
        "objective",
        "embedding_representation",
    ):
        if _json_compatible(payload.get(field)) != _json_compatible(
            metadata.get(field)
        ):
            msg = f"ranker checkpoint {field} does not match sidecar metadata"
            raise RuntimeError(msg)
    cfg = RankerConfig(**payload["config"])
    model = CrossSectionalRanker(cfg).to(torch.device(device))
    model.load_state_dict(payload["state_dict"], strict=True)
    raw_history = payload.get("history", [])
    history = [
        value
        if isinstance(value, dict)
        else {"epoch": index, "train_ic": float(value), "val_ic": None}
        for index, value in enumerate(raw_history)
    ]
    logger.info("复用冻结 Ranker checkpoint: %s", checkpoint)
    return model, history


def _save_ranker(
    *,
    model: CrossSectionalRanker,
    history: list[dict[str, float | int | None]],
    out_dir: Path,
    cache_key: str,
    cache_spec: dict[str, Any],
    feature_columns: list[str],
    training_end_date: str,
    objective: RankerObjectiveConfig,
    delivery_policy: dict[str, Any],
    label_cutoff: dict[str, Any],
    execution_contract: dict[str, Any],
    embedding_representation: dict[str, Any],
    foundation_model: dict[str, Any],
    universe_spec: dict[str, Any],
    time_split: dict[str, Any],
    training_selection: dict[str, Any],
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
    objective_payload = _json_compatible(asdict(objective))
    config_payload = _json_compatible(asdict(cfg))
    torch.save(
        {
            "checkpoint_version": _RANKER_CHECKPOINT_VERSION,
            "cache_key": cache_key,
            "config": config_payload,
            "objective": objective_payload,
            "embedding_representation": embedding_representation,
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
            "checkpoint_version": _RANKER_CHECKPOINT_VERSION,
            "cache_key": cache_key,
            "cache_spec": cache_spec,
            "config": config_payload,
            "objective": objective_payload,
            "delivery_policy": delivery_policy,
            "feature_columns": feature_columns,
            "training_end_date": training_end_date,
            "label_cutoff": label_cutoff,
            "execution_contract": execution_contract,
            "embedding_representation": embedding_representation,
            "foundation_model": foundation_model,
            "universe_spec": universe_spec,
            "time_split": time_split,
            "training_selection": training_selection,
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
    train_universe: Path | None,
    train_calendar: Path | None,
    test_emb: Path,
    test_universe: Path | None,
    out_dir: Path,
    test_panel: Path | None = None,
    epochs: int = 30,
    seed: int = 42,
    device: str = "cuda:0",
    min_names_per_day: int = 20,
    ranker_val_days: int = 10,
    ranker_purge_days: int = 2,
    ranker_patience: int = 8,
    require_ranker_validation: bool = True,
    objective: RankerObjectiveConfig | None = None,
    ranker_lr: float = 1e-3,
    ranker_weight_decay: float = 1e-2,
    ranker_hidden: int = 128,
    ranker_depth: int = 2,
    ranker_dropout: float = 0.3,
    ranker_use_attention: bool = True,
    allow_legacy_training_panel: bool = False,
    allow_research_objective_return_spec_override: bool = False,
    pretrain_acceptance_path: Path | None = None,
    fm_training_end_date: str | None = None,
) -> Path:
    """训练(A期) → 打分(B期) → 落盘 OOS 交付，返回交付目录。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    objective_cfg = objective or _FROZEN_PRODUCTION_OBJECTIVE
    _validate_objective_policy(
        objective_cfg,
        allow_research_objective_return_spec_override=(
            allow_research_objective_return_spec_override
        ),
    )
    parsed_fm_training_end = (
        _parse_canonical_iso_date(
            fm_training_end_date,
            field="fm_training_end_date",
        )
        if fm_training_end_date is not None
        else None
    )
    train_panel_frame = pl.read_parquet(train_panel)
    trading_calendar = (
        read_trading_calendar(train_calendar) if train_calendar is not None else None
    )
    execution_contract = validate_execution_panel_contract(
        train_panel_frame,
        trading_calendar=trading_calendar,
        require_calendar_verification=not allow_legacy_training_panel,
        allow_legacy=allow_legacy_training_panel,
    )
    delivery_policy = _delivery_policy(
        execution_contract=execution_contract,
        objective=objective_cfg,
        allow_legacy_training_panel=allow_legacy_training_panel,
        allow_research_objective_return_spec_override=(
            allow_research_objective_return_spec_override
        ),
    )
    lineage_report: dict[str, Any] | None = None
    if not allow_legacy_training_panel:
        if pretrain_acceptance_path is None:
            msg = "strict OOS delivery requires --pretrain-acceptance"
            raise ValueError(msg)
        lineage_report = validate_pretrain_lineage(
            acceptance_path=pretrain_acceptance_path,
            train_embeddings=train_emb_dir / "all.parquet",
            oos_embeddings=test_emb,
            expected_training_end=fm_training_end_date,
        )
    emb_b = _load_emb(test_emb)
    required_topk_names = max(objective_cfg.ndcg_ks)
    if not allow_legacy_training_panel:
        if train_universe is None:
            msg = (
                "strict Top-K training requires a daily PIT --train-universe; "
                "eligible_at_signal is not a liquidity universe"
            )
            raise ValueError(msg)
        if test_universe is None:
            msg = (
                "strict Top-K scoring requires a daily PIT --test-universe; "
                "test embeddings are observations, not a membership definition"
            )
            raise ValueError(msg)
        if train_calendar is None:
            msg = (
                "strict Top-K training requires --train-calendar to reverify every "
                "T+1/T+2 label mapping"
            )
            raise ValueError(msg)

    scoring_universe_frame = (
        pl.read_parquet(test_universe) if test_universe is not None else None
    )
    scoring_universe_contract: dict[str, Any] = {
        "format_version": "legacy_unverified",
        "verified": False,
    }
    if scoring_universe_frame is not None:
        scoring_universe_frame, scoring_universe_contract = validate_pit_universe(
            scoring_universe_frame,
            required_dates=(str(value) for value in emb_b["date"].unique()),
            min_names_per_day=(
                required_topk_names if not allow_legacy_training_panel else 1
            ),
            context="OOS scoring PIT universe",
            require_pit_metadata=not allow_legacy_training_panel,
        )
    scoring_features = build_scoring_features(
        emb_b,
        universe=scoring_universe_frame,
        min_names_per_day=(
            required_topk_names
            if not allow_legacy_training_panel
            else min_names_per_day
        ),
    )
    scoring_universe_stats = _cross_section_stats(scoring_features)
    scoring_alignment_contract = {
        **scoring_universe_contract,
        "stats": scoring_universe_stats,
    }
    parsed_oos_start = _parse_canonical_iso_date(
        _oos_start_date(test_emb),
        field="OOS embedding start date",
    )
    oos_start = parsed_oos_start.isoformat()
    label_cutoff = _label_cutoff_policy(train_panel, oos_start)
    _embedding_contract, embedding_representation = _embedding_representation_spec(
        train_emb_dir,
        test_emb,
        allow_legacy=allow_legacy_training_panel,
    )
    if not allow_legacy_training_panel:
        exit_lag = int(execution_contract["exit_day_lag"])
        if ranker_purge_days < exit_lag:
            msg = (
                "ranker purge must cover the execution label horizon: "
                f"purge_days={ranker_purge_days}, exit_day_lag={exit_lag}"
            )
            raise ValueError(msg)
        if int(scoring_universe_stats["names_min"] or 0) < required_topk_names:
            msg = (
                "strict Top-K scoring requires every day to contain at least "
                f"max(ndcg_ks)={required_topk_names} names after PIT/embedding join; "
                f"observed minimum={scoring_universe_stats['names_min']}"
            )
            raise ValueError(msg)
    if (
        allow_legacy_training_panel
        and parsed_fm_training_end is not None
        and parsed_fm_training_end >= parsed_oos_start
    ):
        msg = (
            "FM training period overlaps OOS: "
            f"training_end={parsed_fm_training_end.isoformat()}, oos_start={oos_start}"
        )
        raise ValueError(msg)
    fm_provenance = _foundation_model_provenance(
        lineage_report=lineage_report,
        parsed_training_end=parsed_fm_training_end,
        parsed_oos_start=parsed_oos_start,
    )
    canonical_fm_training_end = fm_provenance["training_end_date"]
    training_config = {
        "checkpoint_version": _RANKER_CHECKPOINT_VERSION,
        "feature_target_spec_version": _FEATURE_TARGET_SPEC_VERSION,
        "ranker_val_days": ranker_val_days,
        "ranker_purge_days": ranker_purge_days,
        "ranker_patience": ranker_patience,
        "require_ranker_validation": require_ranker_validation,
        "objective": _json_compatible(asdict(objective_cfg)),
        "model": {
            "hidden": ranker_hidden,
            "depth": ranker_depth,
            "n_heads": 4,
            "dropout": ranker_dropout,
            "use_attention": ranker_use_attention,
        },
        "optimizer": {
            "name": "AdamW",
            "lr": ranker_lr,
            "weight_decay": ranker_weight_decay,
        },
        "delivery_policy": delivery_policy,
        "allow_legacy_training_panel": allow_legacy_training_panel,
        "allow_research_objective_return_spec_override": (
            allow_research_objective_return_spec_override
        ),
        "fm_training_end_date": canonical_fm_training_end,
    }
    cache_spec = _ranker_cache_spec(
        train_emb_dir=train_emb_dir,
        train_panel=train_panel,
        train_universe=train_universe,
        train_calendar=train_calendar,
        epochs=epochs,
        seed=seed,
        min_names_per_day=min_names_per_day,
        label_cutoff=label_cutoff,
        execution_contract=execution_contract,
        embedding_representation=embedding_representation,
        foundation_model=fm_provenance,
        training_config=training_config,
    )
    ranker_cache_key = _stable_key(cache_spec)
    test_emb_fingerprint = _file_fingerprint(test_emb)
    test_universe_fingerprint = (
        _file_fingerprint(test_universe) if test_universe is not None else None
    )
    score_state_path = out_dir / _SCORE_STATE
    prior_state = _load_json(score_state_path)
    scores_path = out_dir / "scores.parquet"

    # 编排器即使在没有新日期时每轮调用，也应是 O(1) no-op。
    if (
        prior_state is not None
        and prior_state.get("ranker_cache_key") == ranker_cache_key
        and prior_state.get("test_emb") == test_emb_fingerprint
        and prior_state.get("test_universe") == test_universe_fingerprint
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
        pan_a, label_cutoff_metadata = _filter_training_panel_for_oos(
            train_panel_frame,
            label_cutoff=label_cutoff,
        )
        logger.info(
            "label cutoff: OOS start=%s horizons=%s retained=%d excluded=%d",
            oos_start,
            label_cutoff_metadata["label_horizon_columns"],
            label_cutoff_metadata["retained_label_rows"],
            label_cutoff_metadata["excluded_label_rows"],
        )
        train_embeddings = _load_train_emb(train_emb_dir)
        train_universe_frame = (
            pl.read_parquet(train_universe) if train_universe is not None else None
        )
        train_universe_contract: dict[str, Any] = {
            "format_version": "legacy_unverified",
            "verified": False,
        }
        if train_universe_frame is not None:
            retained_signal_dates = sorted(
                {str(value) for value in train_embeddings["date"].unique()}
                & {str(value) for value in pan_a["date"].unique()}
            )
            train_universe_frame, train_universe_contract = validate_pit_universe(
                train_universe_frame,
                required_dates=retained_signal_dates,
                min_names_per_day=(
                    required_topk_names if not allow_legacy_training_panel else 1
                ),
                context="ranker training PIT universe",
                require_pit_metadata=not allow_legacy_training_panel,
            )
        feat_a = build_training_features(
            train_embeddings,
            pan_a,
            universe=train_universe_frame,
            min_names_per_day=(min_names_per_day if allow_legacy_training_panel else 1),
        )
        if feat_a.is_empty():
            msg = "训练期特征为空"
            raise RuntimeError(msg)
        logger.info(
            "train(A) features: %d rows, %d days",
            feat_a.height,
            feat_a["date"].n_unique(),
        )
        feature_universe_stats = _cross_section_stats(feat_a)
        required_daily_names = max(max(objective_cfg.ndcg_ks), min_names_per_day)
        if (
            not allow_legacy_training_panel
            and int(feature_universe_stats["names_min"] or 0) < required_daily_names
        ):
            msg = (
                "strict Top-K training requires every retained day to contain at "
                f"least {required_daily_names} names; "
                f"observed minimum={feature_universe_stats['names_min']}"
            )
            raise ValueError(msg)
        universe_alignment = None
        if not allow_legacy_training_panel:
            universe_alignment = validate_universe_alignment(
                {**train_universe_contract, "stats": feature_universe_stats},
                scoring_alignment_contract,
            )
        universe_spec = {
            "mode": (
                "daily_pit_file" if train_universe is not None else "legacy_unverified"
            ),
            "file": (
                _file_fingerprint(train_universe)
                if train_universe is not None
                else None
            ),
            "input": (
                _cross_section_stats(train_universe_frame)
                if train_universe_frame is not None
                else None
            ),
            "contract": train_universe_contract,
            "retained_training_features": feature_universe_stats,
            "train_vs_scoring_alignment": universe_alignment,
            "pit_requirement": "membership must be known on each signal date",
        }
        train_features, val_features, time_split_metadata = chronological_ranker_split(
            feat_a,
            val_days=ranker_val_days,
            purge_days=ranker_purge_days,
            require_validation=require_ranker_validation,
        )
        training = fit_ranker(
            train_features,
            val_features=val_features,
            epochs=epochs,
            patience=ranker_patience,
            lr=ranker_lr,
            weight_decay=ranker_weight_decay,
            hidden=ranker_hidden,
            depth=ranker_depth,
            dropout=ranker_dropout,
            use_attention=ranker_use_attention,
            device=device,
            seed=seed,
            objective=objective_cfg,
        )
        model = training.model
        history = training.history
        training_selection_metadata = {
            "best_epoch": training.best_epoch,
            "best_val_ic": training.best_val_ic,
            "best_val_ndcg": training.best_val_ndcg,
            "best_val_top_spread": training.best_val_top_spread,
            "best_selection_score": training.best_selection_score,
            "stopped_early": training.stopped_early,
            "objective": _json_compatible(asdict(objective_cfg)),
        }
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
            objective=objective_cfg,
            delivery_policy=delivery_policy,
            label_cutoff=label_cutoff_metadata,
            execution_contract=execution_contract,
            embedding_representation=embedding_representation,
            foundation_model=fm_provenance,
            universe_spec=universe_spec,
            time_split=time_split_metadata,
            training_selection=training_selection_metadata,
        )
    else:
        model, history = cached
        ranker_metadata = _load_json(out_dir / _RANKER_METADATA) or {}
        label_cutoff_metadata = dict(ranker_metadata.get("label_cutoff", label_cutoff))
        time_split_metadata = dict(ranker_metadata.get("time_split", {}))
        training_selection_metadata = dict(
            ranker_metadata.get("training_selection", {})
        )
        universe_spec = dict(ranker_metadata.get("universe_spec", {}))
        ranker_columns = list(ranker_metadata.get("feature_columns", []))
        if not ranker_columns:
            msg = "cached ranker metadata is missing feature_columns"
            raise RuntimeError(msg)

    if not allow_legacy_training_panel:
        stored_universe_contract = universe_spec.get("contract")
        retained_training_stats = universe_spec.get("retained_training_features")
        if (
            not isinstance(stored_universe_contract, dict)
            or not stored_universe_contract.get("verified")
            or not isinstance(retained_training_stats, dict)
        ):
            msg = "cached strict ranker is missing its verified PIT universe contract"
            raise ValueError(msg)
        universe_alignment = validate_universe_alignment(
            {**stored_universe_contract, "stats": retained_training_stats},
            scoring_alignment_contract,
        )
    else:
        universe_alignment = None

    # B 期（OOS）只对尚未处理的新日期打分。
    all_embedding_dates = sorted(str(value) for value in emb_b["date"].unique())
    state_compatible = (
        prior_state is not None
        and prior_state.get("ranker_cache_key") == ranker_cache_key
        and prior_state.get("test_universe") == test_universe_fingerprint
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
        feat_b_new = scoring_features.filter(pl.col("date").is_in(new_dates))
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
            "file_sha256": _sha256(scores_path),
            "schema": {"date": "string", "symbol": "string", "score": "float64"},
            "primary_key": ["date", "symbol"],
            "rows": scores.height,
            "dates": scores["date"].n_unique(),
            "date_min": scores["date"].min(),
            "date_max": scores["date"].max(),
        },
        "ranker": {
            "checkpoint_version": _RANKER_CHECKPOINT_VERSION,
            "feature_target_spec_version": _FEATURE_TARGET_SPEC_VERSION,
            "epochs": epochs,
            "seed": seed,
            "device": device,
            "min_names_per_day": min_names_per_day,
            "cache_key": ranker_cache_key,
            "checkpoint": _RANKER_CHECKPOINT,
            "label_cutoff": label_cutoff_metadata,
            "execution_contract": execution_contract,
            "time_split": time_split_metadata,
            "training_selection": training_selection_metadata,
            "objective": _json_compatible(asdict(objective_cfg)),
            "delivery_policy": delivery_policy,
            "training_universe": universe_spec,
            "scoring_universe": {
                "file": test_universe_fingerprint,
                "contract": scoring_universe_contract,
                "retained_scoring_features": scoring_universe_stats,
                "train_vs_scoring_alignment": universe_alignment,
            },
            "embedding_representation": embedding_representation,
        },
        "foundation_model": fm_provenance,
        "note": "score 越大越看多；score(T) 仅 T 收盘后可用；2026 全部晚于 FM 训练期。",
    }
    _atomic_json(out_dir / "signal_manifest.json", manifest)
    _atomic_json(
        score_state_path,
        {
            "ranker_cache_key": ranker_cache_key,
            "test_emb": test_emb_fingerprint,
            "test_universe": test_universe_fingerprint,
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
    train_universe: Path | None = None,
    train_calendar: Path | None = None,
    test_emb: Path,
    test_universe: Path | None = None,
    out_dir: Path,
    test_panel: Path | None = None,
    epochs: int = 30,
    seed: int = 42,
    device: str = "cuda:0",
    min_names_per_day: int = 20,
    ranker_val_days: int = 10,
    ranker_purge_days: int = 2,
    ranker_patience: int = 8,
    require_ranker_validation: bool = True,
    objective: RankerObjectiveConfig | None = None,
    ranker_lr: float = 1e-3,
    ranker_weight_decay: float = 1e-2,
    ranker_hidden: int = 128,
    ranker_depth: int = 2,
    ranker_dropout: float = 0.3,
    ranker_use_attention: bool = True,
    allow_legacy_training_panel: bool = False,
    allow_research_objective_return_spec_override: bool = False,
    pretrain_acceptance_path: Path | None = None,
    fm_training_end_date: str | None = None,
) -> Path:
    """串行 Ranker 打分，并保证成功或异常路径都会释放 GPU 锁。"""
    objective_cfg = objective or _FROZEN_PRODUCTION_OBJECTIVE
    _validate_objective_policy(
        objective_cfg,
        allow_research_objective_return_spec_override=(
            allow_research_objective_return_spec_override
        ),
    )
    if fm_training_end_date is not None:
        _parse_canonical_iso_date(
            fm_training_end_date,
            field="fm_training_end_date",
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    score_lock = out_dir / ".score.lock"
    score_lock.write_text(str(os.getpid()), encoding="utf-8")
    try:
        return _build_oos_delivery_locked(
            train_emb_dir=train_emb_dir,
            train_panel=train_panel,
            train_universe=train_universe,
            train_calendar=train_calendar,
            test_emb=test_emb,
            test_universe=test_universe,
            test_panel=test_panel,
            out_dir=out_dir,
            epochs=epochs,
            seed=seed,
            device=device,
            min_names_per_day=min_names_per_day,
            ranker_val_days=ranker_val_days,
            ranker_purge_days=ranker_purge_days,
            ranker_patience=ranker_patience,
            require_ranker_validation=require_ranker_validation,
            objective=objective_cfg,
            ranker_lr=ranker_lr,
            ranker_weight_decay=ranker_weight_decay,
            ranker_hidden=ranker_hidden,
            ranker_depth=ranker_depth,
            ranker_dropout=ranker_dropout,
            ranker_use_attention=ranker_use_attention,
            allow_legacy_training_panel=allow_legacy_training_panel,
            allow_research_objective_return_spec_override=(
                allow_research_objective_return_spec_override
            ),
            pretrain_acceptance_path=pretrain_acceptance_path,
            fm_training_end_date=fm_training_end_date,
        )
    finally:
        score_lock.unlink(missing_ok=True)


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-emb-dir", type=Path, required=True)
    p.add_argument("--train-panel", type=Path, required=True)
    p.add_argument(
        "--train-universe",
        type=Path,
        help=(
            "逐日 PIT (date,symbol,asof_date,universe_policy) 股票池；"
            "严格 Top-K 训练必填"
        ),
    )
    p.add_argument(
        "--train-calendar",
        type=Path,
        help="生成训练 execution panel 所用完整交易日历；严格模式必填",
    )
    p.add_argument("--test-emb", type=Path, required=True)
    p.add_argument(
        "--test-universe",
        type=Path,
        help=(
            "评分期逐日 PIT (date,symbol,asof_date,universe_policy) 股票池；"
            "严格模式必填且 policy 必须与训练一致"
        ),
    )
    p.add_argument("--test-panel", type=Path, help="已弃用；生产打分不读取未来标签")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--min-names-per-day", type=int, default=20)
    p.add_argument("--ranker-val-days", type=int, default=10)
    p.add_argument("--ranker-purge-days", type=int, default=2)
    p.add_argument("--ranker-patience", type=int, default=8)
    p.add_argument(
        "--allow-no-ranker-validation",
        action="store_false",
        dest="require_ranker_validation",
        help="仅小样本诊断：训练日期不足时允许不留验证集",
    )
    p.add_argument(
        "--ndcg-ks",
        type=_parse_int_tuple,
        default=_FROZEN_PRODUCTION_OBJECTIVE.ndcg_ks,
    )
    p.add_argument(
        "--ndcg-k-weights",
        type=_parse_float_tuple,
        default=_FROZEN_PRODUCTION_OBJECTIVE.ndcg_k_weights,
    )
    p.add_argument(
        "--head-loss-weight",
        type=float,
        default=_FROZEN_PRODUCTION_OBJECTIVE.head_weight,
    )
    p.add_argument(
        "--global-ic-weight",
        type=float,
        default=_FROZEN_PRODUCTION_OBJECTIVE.global_ic_weight,
    )
    p.add_argument(
        "--aux-huber-weight",
        type=float,
        default=_FROZEN_PRODUCTION_OBJECTIVE.aux_huber_weight,
    )
    p.add_argument(
        "--aux-huber-beta",
        type=float,
        default=_FROZEN_PRODUCTION_OBJECTIVE.aux_huber_beta,
    )
    p.add_argument(
        "--pair-samples-per-day",
        type=int,
        default=_FROZEN_PRODUCTION_OBJECTIVE.pair_samples_per_day,
    )
    p.add_argument(
        "--hard-pair-fraction",
        type=float,
        default=_FROZEN_PRODUCTION_OBJECTIVE.hard_pair_fraction,
    )
    p.add_argument(
        "--min-label-rank-gap",
        type=float,
        default=_FROZEN_PRODUCTION_OBJECTIVE.min_label_rank_gap,
    )
    p.add_argument(
        "--score-temperature",
        type=float,
        default=_FROZEN_PRODUCTION_OBJECTIVE.score_temperature,
    )
    p.add_argument("--ranker-lr", type=float, default=1e-3)
    p.add_argument("--ranker-weight-decay", type=float, default=1e-2)
    p.add_argument("--ranker-hidden", type=int, default=128)
    p.add_argument("--ranker-depth", type=int, default=2)
    p.add_argument("--ranker-dropout", type=float, default=0.3)
    p.add_argument(
        "--no-ranker-attention",
        action="store_false",
        dest="ranker_use_attention",
    )
    p.add_argument(
        "--allow-legacy-training-panel",
        action="store_true",
        help=(
            "旧数据兼容诊断：允许未验证 panel/embedding/PIT 输入，产物标为 "
            "legacy_diagnostic、不得生产交付；不会授权自定义 objective/return spec"
        ),
    )
    p.add_argument(
        "--allow-research-objective-return-spec-override",
        action="store_true",
        help=(
            "研究专用：允许偏离冻结 Ranker objective 或 vwap_t1_vwap_t2；"
            "不放宽 calendar/PIT/embedding 校验，产物明确标为 research override"
        ),
    )
    p.add_argument(
        "--pretrain-acceptance",
        type=Path,
        help=(
            "通过验证的预训练 acceptance JSON；所有非 legacy 输入模式必填，"
            "用于推导而非手填 FM cutoff"
        ),
    )
    p.add_argument(
        "--fm-training-end-date",
        help="可选 YYYY-MM-DD 断言；strict 模式的真实 cutoff 由 lineage 推导",
    )
    args = p.parse_args()

    objective = RankerObjectiveConfig(
        ndcg_ks=args.ndcg_ks,
        ndcg_k_weights=args.ndcg_k_weights,
        head_weight=args.head_loss_weight,
        global_ic_weight=args.global_ic_weight,
        aux_huber_weight=args.aux_huber_weight,
        aux_huber_beta=args.aux_huber_beta,
        pair_samples_per_day=args.pair_samples_per_day,
        hard_pair_fraction=args.hard_pair_fraction,
        min_label_rank_gap=args.min_label_rank_gap,
        score_temperature=args.score_temperature,
    )
    objective.validate()

    build_oos_delivery(
        train_emb_dir=args.train_emb_dir,
        train_panel=args.train_panel,
        train_universe=args.train_universe,
        train_calendar=args.train_calendar,
        test_emb=args.test_emb,
        test_universe=args.test_universe,
        test_panel=args.test_panel,
        out_dir=args.out_dir,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        min_names_per_day=args.min_names_per_day,
        ranker_val_days=args.ranker_val_days,
        ranker_purge_days=args.ranker_purge_days,
        ranker_patience=args.ranker_patience,
        require_ranker_validation=args.require_ranker_validation,
        objective=objective,
        ranker_lr=args.ranker_lr,
        ranker_weight_decay=args.ranker_weight_decay,
        ranker_hidden=args.ranker_hidden,
        ranker_depth=args.ranker_depth,
        ranker_dropout=args.ranker_dropout,
        ranker_use_attention=args.ranker_use_attention,
        allow_legacy_training_panel=args.allow_legacy_training_panel,
        allow_research_objective_return_spec_override=(
            args.allow_research_objective_return_spec_override
        ),
        pretrain_acceptance_path=args.pretrain_acceptance,
        fm_training_end_date=args.fm_training_end_date,
    )


if __name__ == "__main__":
    main()
