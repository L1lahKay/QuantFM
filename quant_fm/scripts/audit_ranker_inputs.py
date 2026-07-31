"""
只读审计 Ranker 的历史训练输入与 OOS embedding。

审计不会加载模型或写回输入文件。它验证 embedding/panel 的键、数值、覆盖率、
特征维度与严格时序切分，并同时生成机器可读 JSON 和便于评审的 Markdown。
当 OOS embedding 尚未生成时，历史输入仍会完整检查，但总状态明确为
``preflight``，不会误报为可开始 OOS 打分。
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from typing import Any

_KEY_COLUMNS = ("date", "symbol")
_FORBIDDEN_OOS_COLUMNS = {
    "fwd_ret",
    "label",
    "xs_ret",
    "target_return",
    "aux_target",
    "head_gain",
}
_EMBEDDING_COLUMN = re.compile(r"^emb_(\d+)$")
_LABEL_HORIZON_CANDIDATES = (
    "label_availability_date",
    "label_available_date",
    "label_availability",
    "label_end_date",
    "next_date",
    "exit_date",
)


def _issue(
    scope: str,
    severity: str,
    code: str,
    message: str,
) -> dict[str, str]:
    return {
        "scope": scope,
        "severity": severity,
        "code": code,
        "message": message,
    }


def _path_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _normalised_keys(path: Path) -> pl.LazyFrame:
    return (
        pl.scan_parquet(path)
        .select(
            pl.col("date").cast(pl.Utf8, strict=False).alias("date"),
            pl.col("symbol").cast(pl.Utf8, strict=False).str.zfill(6).alias("symbol"),
        )
        .filter(pl.all_horizontal(pl.all().is_not_null()))
        .unique()
    )


def _embedding_column_issues(
    columns: list[str],
    *,
    scope: str,
) -> tuple[list[str], list[dict[str, str]]]:
    feature_columns = [name for name in columns if name.startswith("emb_")]
    issues: list[dict[str, str]] = []
    if not feature_columns:
        issues.append(
            _issue(
                scope,
                "critical",
                "embedding_columns_missing",
                "未找到 emb_* 特征列",
            )
        )
        return feature_columns, issues

    indices: list[int] = []
    invalid_names: list[str] = []
    for name in feature_columns:
        match = _EMBEDDING_COLUMN.fullmatch(name)
        if match is None:
            invalid_names.append(name)
        else:
            indices.append(int(match.group(1)))
    if invalid_names:
        issues.append(
            _issue(
                scope,
                "critical",
                "embedding_column_names_invalid",
                f"embedding 列名不是 emb_<整数>: {invalid_names}",
            )
        )
    if not invalid_names and indices != list(range(len(feature_columns))):
        issues.append(
            _issue(
                scope,
                "critical",
                "embedding_columns_not_contiguous",
                "embedding 列必须按 emb_0..emb_N 连续有序",
            )
        )
    return feature_columns, issues


def _inspect_embeddings(
    path: Path,
    *,
    scope: str,
    issues: list[dict[str, str]],
) -> dict[str, Any] | None:
    info = _path_info(path)
    if not path.is_file():
        issues.append(_issue(scope, "critical", f"{scope}_missing", f"缺少 {path}"))
        return info
    try:
        schema = pl.read_parquet_schema(path)
        columns = list(schema.names())
        info["schema"] = {name: str(dtype) for name, dtype in schema.items()}
        missing_keys = sorted(set(_KEY_COLUMNS) - set(columns))
        if missing_keys:
            issues.append(
                _issue(
                    scope,
                    "critical",
                    "embedding_key_columns_missing",
                    f"缺少主键列: {missing_keys}",
                )
            )
            return info

        feature_columns, column_issues = _embedding_column_issues(columns, scope=scope)
        issues.extend(column_issues)
        metric_exprs: list[pl.Expr] = [
            pl.len().alias("rows"),
            pl.struct(
                pl.col("date").cast(pl.Utf8, strict=False).alias("date"),
                pl.col("symbol")
                .cast(pl.Utf8, strict=False)
                .str.zfill(6)
                .alias("symbol"),
            )
            .n_unique()
            .alias("unique_keys"),
            pl.col("date").is_null().sum().alias("null_dates"),
            pl.col("symbol").is_null().sum().alias("null_symbols"),
            pl.col("date").cast(pl.Utf8, strict=False).min().alias("date_min"),
            pl.col("date").cast(pl.Utf8, strict=False).max().alias("date_max"),
            pl.col("date").n_unique().alias("dates"),
            pl.col("symbol").n_unique().alias("symbols"),
        ]
        for index, name in enumerate(feature_columns):
            metric_exprs.append(
                pl.col(name)
                .cast(pl.Float64, strict=False)
                .is_finite()
                .fill_null(False)
                .not_()
                .sum()
                .alias(f"invalid_feature_{index}")
            )
        metrics = (
            pl.scan_parquet(path).select(metric_exprs).collect().row(0, named=True)
        )
        invalid_features = {
            name: int(metrics[f"invalid_feature_{index}"])
            for index, name in enumerate(feature_columns)
            if int(metrics[f"invalid_feature_{index}"]) > 0
        }
        rows = int(metrics["rows"])
        unique_keys = int(metrics["unique_keys"])
        info.update(
            {
                "rows": rows,
                "unique_keys": unique_keys,
                "duplicate_key_rows": rows - unique_keys,
                "null_keys": {
                    "date": int(metrics["null_dates"]),
                    "symbol": int(metrics["null_symbols"]),
                },
                "dates": int(metrics["dates"]),
                "symbols": int(metrics["symbols"]),
                "date_min": metrics["date_min"],
                "date_max": metrics["date_max"],
                "embedding_dim": len(feature_columns),
                "feature_columns": feature_columns,
                "invalid_feature_cells": sum(invalid_features.values()),
                "invalid_features": invalid_features,
            }
        )
        if rows == 0:
            issues.append(
                _issue(scope, "critical", "embeddings_empty", "embedding 表为空")
            )
        if rows - unique_keys > 0:
            issues.append(
                _issue(
                    scope,
                    "critical",
                    "embedding_duplicate_keys",
                    f"规范化后存在 {rows - unique_keys} 个重复 (date, symbol) 行",
                )
            )
        null_key_count = int(metrics["null_dates"]) + int(metrics["null_symbols"])
        if null_key_count:
            issues.append(
                _issue(
                    scope,
                    "critical",
                    "embedding_null_keys",
                    f"主键空值单元格数: {null_key_count}",
                )
            )
        if invalid_features:
            issues.append(
                _issue(
                    scope,
                    "critical",
                    "embedding_values_invalid",
                    f"embedding 含 null/NaN/Inf 或不可转为浮点的值: {invalid_features}",
                )
            )
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        issues.append(
            _issue(
                scope,
                "critical",
                "embeddings_unreadable",
                f"无法读取 {path}: {exc}",
            )
        )
        info["read_error"] = str(exc)
    return info


def _inspect_panel(
    path: Path,
    *,
    issues: list[dict[str, str]],
) -> dict[str, Any] | None:
    scope = "train_panel"
    info = _path_info(path)
    if not path.is_file():
        issues.append(_issue(scope, "critical", "train_panel_missing", f"缺少 {path}"))
        return info
    try:
        schema = pl.read_parquet_schema(path)
        columns = list(schema.names())
        info["schema"] = {name: str(dtype) for name, dtype in schema.items()}
        missing = sorted({*_KEY_COLUMNS, "fwd_ret"} - set(columns))
        if missing:
            issues.append(
                _issue(
                    scope,
                    "critical",
                    "panel_columns_missing",
                    f"缺少列: {missing}",
                )
            )
            return info

        numeric_return = pl.col("fwd_ret").cast(pl.Float64, strict=False)
        usable_return = numeric_return.is_finite().fill_null(False)
        horizon_columns = [
            name
            for name in columns
            if name in _LABEL_HORIZON_CANDIDATES
            or name.startswith("label_availability")
        ]
        metric_exprs: list[pl.Expr] = [
            pl.len().alias("rows"),
            pl.struct(
                pl.col("date").cast(pl.Utf8, strict=False).alias("date"),
                pl.col("symbol")
                .cast(pl.Utf8, strict=False)
                .str.zfill(6)
                .alias("symbol"),
            )
            .n_unique()
            .alias("unique_keys"),
            pl.col("date").is_null().sum().alias("null_dates"),
            pl.col("symbol").is_null().sum().alias("null_symbols"),
            pl.col("fwd_ret").is_null().sum().alias("null_returns"),
            (pl.col("fwd_ret").is_not_null() & usable_return.not_())
            .sum()
            .alias("nonfinite_returns"),
            usable_return.sum().alias("usable_returns"),
            pl.col("date").cast(pl.Utf8, strict=False).min().alias("date_min"),
            pl.col("date").cast(pl.Utf8, strict=False).max().alias("date_max"),
            pl.col("date").n_unique().alias("dates"),
            pl.col("symbol").n_unique().alias("symbols"),
        ]
        for index, column in enumerate(horizon_columns):
            horizon = pl.col(column).cast(pl.Utf8, strict=False)
            metric_exprs.extend(
                [
                    horizon.filter(usable_return).min().alias(f"horizon_min_{index}"),
                    horizon.filter(usable_return).max().alias(f"horizon_max_{index}"),
                    (usable_return & horizon.is_null())
                    .sum()
                    .alias(f"horizon_null_{index}"),
                ]
            )
        if horizon_columns:
            metric_exprs.append(
                (
                    usable_return
                    & pl.any_horizontal(
                        pl.col(column).cast(pl.Utf8, strict=False).is_null()
                        for column in horizon_columns
                    )
                )
                .sum()
                .alias("horizon_missing_rows")
            )
        metrics = (
            pl.scan_parquet(path).select(metric_exprs).collect().row(0, named=True)
        )
        rows = int(metrics["rows"])
        unique_keys = int(metrics["unique_keys"])
        null_key_count = int(metrics["null_dates"]) + int(metrics["null_symbols"])
        horizon_details = {
            column: {
                "date_min": metrics[f"horizon_min_{index}"],
                "date_max": metrics[f"horizon_max_{index}"],
                "missing_for_usable_labels": int(metrics[f"horizon_null_{index}"]),
            }
            for index, column in enumerate(horizon_columns)
        }
        horizon_min_values = [
            str(item["date_min"])
            for item in horizon_details.values()
            if item["date_min"] is not None
        ]
        horizon_max_values = [
            str(item["date_max"])
            for item in horizon_details.values()
            if item["date_max"] is not None
        ]
        missing_horizon_cells = sum(
            int(item["missing_for_usable_labels"]) for item in horizon_details.values()
        )
        missing_horizon_rows = (
            int(metrics["horizon_missing_rows"]) if horizon_columns else 0
        )
        label_horizon = {
            "available": bool(horizon_columns),
            "columns": horizon_columns,
            "date_min": min(horizon_min_values) if horizon_min_values else None,
            "date_max": max(horizon_max_values) if horizon_max_values else None,
            "missing_rows_for_usable_labels": missing_horizon_rows,
            "missing_cells_for_usable_labels": missing_horizon_cells,
            "by_column": horizon_details,
        }
        info.update(
            {
                "rows": rows,
                "unique_keys": unique_keys,
                "duplicate_key_rows": rows - unique_keys,
                "null_keys": {
                    "date": int(metrics["null_dates"]),
                    "symbol": int(metrics["null_symbols"]),
                },
                "dates": int(metrics["dates"]),
                "symbols": int(metrics["symbols"]),
                "date_min": metrics["date_min"],
                "date_max": metrics["date_max"],
                "null_fwd_ret": int(metrics["null_returns"]),
                "nonfinite_fwd_ret": int(metrics["nonfinite_returns"]),
                "usable_fwd_ret": int(metrics["usable_returns"]),
                "label_horizon": label_horizon,
            }
        )
        if rows == 0:
            issues.append(_issue(scope, "critical", "panel_empty", "训练 panel 为空"))
        if rows - unique_keys > 0:
            issues.append(
                _issue(
                    scope,
                    "critical",
                    "panel_duplicate_keys",
                    f"规范化后存在 {rows - unique_keys} 个重复 (date, symbol) 行",
                )
            )
        if null_key_count:
            issues.append(
                _issue(
                    scope,
                    "critical",
                    "panel_null_keys",
                    f"主键空值单元格数: {null_key_count}",
                )
            )
        if int(metrics["nonfinite_returns"]):
            issues.append(
                _issue(
                    scope,
                    "critical",
                    "panel_returns_nonfinite",
                    f"fwd_ret 含 {int(metrics['nonfinite_returns'])} 个 NaN/Inf/非法值",
                )
            )
        if int(metrics["null_returns"]):
            issues.append(
                _issue(
                    scope,
                    "warning",
                    "panel_returns_null",
                    f"{int(metrics['null_returns'])} 行 fwd_ret 为空，Ranker 训练会跳过这些行",
                )
            )
        if missing_horizon_rows:
            issues.append(
                _issue(
                    scope,
                    "warning",
                    "panel_label_horizon_missing",
                    f"{missing_horizon_rows} 个可用标签缺少 next/exit/availability 日期；"
                    "严格 OOS 训练会剔除这些行",
                )
            )
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        issues.append(
            _issue(
                scope,
                "critical",
                "panel_unreadable",
                f"无法读取 {path}: {exc}",
            )
        )
        info["read_error"] = str(exc)
    return info


def _coverage_metrics(
    train_embeddings: Path,
    train_panel: Path,
    *,
    threshold: float,
    issues: list[dict[str, str]],
) -> dict[str, Any] | None:
    try:
        train_keys = _normalised_keys(train_embeddings)
        panel_scan = pl.scan_parquet(train_panel)
        panel_keys = _normalised_keys(train_panel)
        labeled_panel_keys = (
            panel_scan.filter(
                pl.col("fwd_ret")
                .cast(pl.Float64, strict=False)
                .is_finite()
                .fill_null(False)
            )
            .select(
                pl.col("date").cast(pl.Utf8, strict=False).alias("date"),
                pl.col("symbol")
                .cast(pl.Utf8, strict=False)
                .str.zfill(6)
                .alias("symbol"),
            )
            .filter(pl.all_horizontal(pl.all().is_not_null()))
            .unique()
        )
        total = int(train_keys.select(pl.len()).collect().item())
        matched = int(
            train_keys.join(panel_keys, on=list(_KEY_COLUMNS), how="inner")
            .select(pl.len())
            .collect()
            .item()
        )
        labeled = train_keys.join(
            labeled_panel_keys, on=list(_KEY_COLUMNS), how="inner"
        )
        labeled_summary = (
            labeled.select(
                pl.len().alias("rows"),
                pl.col("date").min().alias("date_min"),
                pl.col("date").max().alias("date_max"),
                pl.col("date").n_unique().alias("dates"),
            )
            .collect()
            .row(0, named=True)
        )
        labeled_count = int(labeled_summary["rows"])
        key_coverage = matched / total if total else 0.0
        label_coverage = labeled_count / total if total else 0.0
        result = {
            "train_embedding_keys": total,
            "panel_matched_keys": matched,
            "labeled_matched_keys": labeled_count,
            "key_coverage": key_coverage,
            "label_coverage": label_coverage,
            "minimum_required_coverage": threshold,
            "effective_training_date_min": labeled_summary["date_min"],
            "effective_training_date_max": labeled_summary["date_max"],
            "effective_training_dates": int(labeled_summary["dates"]),
        }
        if key_coverage < threshold:
            issues.append(
                _issue(
                    "train_join",
                    "critical",
                    "train_panel_key_coverage_below_threshold",
                    f"panel 键覆盖率 {key_coverage:.2%} < {threshold:.2%}",
                )
            )
        if label_coverage < threshold:
            issues.append(
                _issue(
                    "train_join",
                    "critical",
                    "train_label_coverage_below_threshold",
                    f"可用标签覆盖率 {label_coverage:.2%} < {threshold:.2%}",
                )
            )
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        issues.append(
            _issue(
                "train_join",
                "critical",
                "train_coverage_unavailable",
                f"无法计算 embedding-panel 覆盖率: {exc}",
            )
        )
        return None
    return result


def _temporal_metrics(
    train_embeddings: Path,
    oos_embeddings: Path,
    *,
    training_end: str | None,
    label_horizon_end: str | None,
    issues: list[dict[str, str]],
) -> dict[str, Any] | None:
    try:
        train_keys = _normalised_keys(train_embeddings)
        oos_keys = _normalised_keys(oos_embeddings)
        overlap = train_keys.join(oos_keys, on=list(_KEY_COLUMNS), how="inner")
        overlap_summary = (
            overlap.select(
                pl.len().alias("keys"),
                pl.col("date").n_unique().alias("dates"),
            )
            .collect()
            .row(0, named=True)
        )
        oos_dates = (
            oos_keys.select("date")
            .unique()
            .select(
                pl.col("date").min().alias("date_min"),
                pl.col("date").max().alias("date_max"),
                pl.len().alias("dates"),
            )
            .collect()
            .row(0, named=True)
        )
        oos_min = oos_dates["date_min"]
        strictly_after = bool(training_end and oos_min and oos_min > training_end)
        result = {
            "effective_training_end_date": training_end,
            "label_horizon_end_date": label_horizon_end,
            "oos_date_min": oos_min,
            "oos_date_max": oos_dates["date_max"],
            "oos_dates": int(oos_dates["dates"]),
            "overlapping_keys": int(overlap_summary["keys"]),
            "overlapping_dates": int(overlap_summary["dates"]),
            "oos_strictly_after_training": strictly_after,
        }
        if int(overlap_summary["keys"]):
            issues.append(
                _issue(
                    "temporal_split",
                    "critical",
                    "train_oos_key_overlap",
                    f"训练与 OOS 重叠 {int(overlap_summary['keys'])} 个主键",
                )
            )
        if not strictly_after:
            issues.append(
                _issue(
                    "temporal_split",
                    "critical",
                    "oos_not_strictly_after_training",
                    f"OOS 最早日期 {oos_min} 必须严格晚于训练/标签信息边界 "
                    f"{training_end}",
                )
            )
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        issues.append(
            _issue(
                "temporal_split",
                "critical",
                "temporal_check_unavailable",
                f"无法计算训练/OOS 时序隔离: {exc}",
            )
        )
        return None
    return result


def audit_ranker_inputs(
    train_embeddings: Path,
    train_panel: Path,
    oos_embeddings: Path | None = None,
    *,
    min_train_coverage: float = 0.95,
) -> dict[str, Any]:
    """
    执行 Ranker 输入审计并返回可序列化报告。

    Parameters
    ----------
    train_embeddings
        历史期股日 embedding parquet。
    train_panel
        含 ``fwd_ret`` 的历史训练 panel parquet。
    oos_embeddings
        可选 OOS embedding parquet；缺失时返回 ``preflight``。
    min_train_coverage
        历史 embedding 主键被 panel 与有效标签覆盖的最低比例。
    """
    if not 0.0 <= min_train_coverage <= 1.0:
        msg = "min_train_coverage must be in [0, 1]"
        raise ValueError(msg)

    train_embeddings = Path(train_embeddings)
    train_panel = Path(train_panel)
    oos_embeddings = Path(oos_embeddings) if oos_embeddings is not None else None
    issues: list[dict[str, str]] = []
    train_info = _inspect_embeddings(
        train_embeddings, scope="train_embeddings", issues=issues
    )
    panel_info = _inspect_panel(train_panel, issues=issues)

    coverage = None
    if (
        train_embeddings.is_file()
        and train_panel.is_file()
        and train_info is not None
        and "feature_columns" in train_info
        and panel_info is not None
        and "usable_fwd_ret" in panel_info
    ):
        coverage = _coverage_metrics(
            train_embeddings,
            train_panel,
            threshold=min_train_coverage,
            issues=issues,
        )

    oos_pending = oos_embeddings is None or not oos_embeddings.is_file()
    oos_info: dict[str, Any] | None
    temporal = None
    compatibility = None
    if oos_pending:
        oos_info = None if oos_embeddings is None else _path_info(oos_embeddings)
        pending_path = "未指定" if oos_embeddings is None else str(oos_embeddings)
        issues.append(
            _issue(
                "oos_embeddings",
                "pending",
                "oos_embeddings_pending",
                f"OOS embedding 尚未就绪: {pending_path}",
            )
        )
    else:
        oos_info = _inspect_embeddings(
            oos_embeddings, scope="oos_embeddings", issues=issues
        )
        if oos_info is not None and "schema" in oos_info:
            forbidden = sorted(
                name
                for name in oos_info["schema"]
                if name in _FORBIDDEN_OOS_COLUMNS or name.startswith("target_")
            )
            if forbidden:
                issues.append(
                    _issue(
                        "oos_embeddings",
                        "critical",
                        "oos_future_columns_present",
                        f"OOS embedding 含禁止的未来信息列: {forbidden}",
                    )
                )
        if (
            train_info is not None
            and oos_info is not None
            and "feature_columns" in train_info
            and "feature_columns" in oos_info
        ):
            expected = train_info["feature_columns"]
            actual = oos_info["feature_columns"]
            compatibility = {
                "compatible": actual == expected,
                "train_embedding_dim": len(expected),
                "oos_embedding_dim": len(actual),
                "missing_in_oos": [name for name in expected if name not in actual],
                "extra_in_oos": [name for name in actual if name not in expected],
                "ordered_columns_match": actual == expected,
            }
            if actual != expected:
                issues.append(
                    _issue(
                        "feature_compatibility",
                        "critical",
                        "train_oos_feature_mismatch",
                        f"训练/OOS embedding 特征不一致: train={len(expected)}, "
                        f"oos={len(actual)}",
                    )
                )
        if oos_info is not None and "feature_columns" in oos_info:
            label_horizon = (
                panel_info.get("label_horizon", {}) if panel_info is not None else {}
            )
            label_horizon_end = label_horizon.get("date_max")
            boundary_candidates = [
                value
                for value in (
                    train_info.get("date_max") if train_info is not None else None,
                    coverage.get("effective_training_date_max") if coverage else None,
                    label_horizon_end,
                )
                if value is not None
            ]
            training_end = (
                max(str(value) for value in boundary_candidates)
                if boundary_candidates
                else None
            )
            temporal = _temporal_metrics(
                train_embeddings,
                oos_embeddings,
                training_end=training_end,
                label_horizon_end=label_horizon_end,
                issues=issues,
            )

    training_scopes = {"train_embeddings", "train_panel", "train_join"}
    training_critical = any(
        item["severity"] == "critical" and item["scope"] in training_scopes
        for item in issues
    )
    critical = any(item["severity"] == "critical" for item in issues)
    status = "fail" if critical else "preflight" if oos_pending else "pass"
    return {
        "audit_version": "1.0",
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "status": status,
        "ready_for_ranker_training": not training_critical,
        "ready_for_oos_scoring": status == "pass",
        "inputs": {
            "train_embeddings": train_info,
            "train_panel": panel_info,
            "oos_embeddings": oos_info,
        },
        "checks": {
            "train_panel_coverage": coverage,
            "label_horizon": (
                panel_info.get("label_horizon") if panel_info is not None else None
            ),
            "feature_compatibility": compatibility,
            "temporal_split": temporal,
        },
        "issues": issues,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """将审计结果渲染为稳定、便于人工审批的 Markdown。"""

    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "# Ranker 输入审计",
        "",
        f"- 状态：`{str(report['status']).upper()}`",
        f"- Ranker 训练就绪：`{report['ready_for_ranker_training']}`",
        f"- OOS 打分就绪：`{report['ready_for_oos_scoring']}`",
        f"- 生成时间：`{report['created_utc']}`",
        "",
        "## 输入摘要",
        "",
        "| 输入 | 文件 | 行数 | 日期范围 | embedding 维度 |",
        "|---|---|---:|---|---:|",
    ]
    for name in ("train_embeddings", "train_panel", "oos_embeddings"):
        info = report["inputs"].get(name)
        if info is None:
            lines.append(f"| {name} | 待生成 | - | - | - |")
            continue
        date_range = (
            f"{info.get('date_min')} → {info.get('date_max')}"
            if info.get("date_min") is not None
            else "-"
        )
        lines.append(
            f"| {name} | {cell(info.get('path'))} | {cell(info.get('rows', '-'))} "
            f"| {cell(date_range)} | {cell(info.get('embedding_dim', '-'))} |"
        )

    coverage = report["checks"].get("train_panel_coverage")
    lines.extend(["", "## 关键检查", ""])
    if coverage is not None:
        lines.extend(
            [
                f"- panel 键覆盖率：`{coverage['key_coverage']:.2%}`",
                f"- 可用标签覆盖率：`{coverage['label_coverage']:.2%}`",
                "- 有效训练日期："
                f"`{coverage['effective_training_date_min']} → "
                f"{coverage['effective_training_date_max']}`",
            ]
        )
    compatibility = report["checks"].get("feature_compatibility")
    label_horizon = report["checks"].get("label_horizon")
    if label_horizon is not None:
        lines.append(
            "- 标签信息视界："
            f"`{label_horizon.get('date_min')} → {label_horizon.get('date_max')}` "
            f"(列: {', '.join(label_horizon.get('columns', [])) or '未提供'})"
        )
    if compatibility is not None:
        lines.append(
            "- 训练/OOS 维度一致："
            f"`{compatibility['compatible']}` "
            f"({compatibility['train_embedding_dim']} / "
            f"{compatibility['oos_embedding_dim']})"
        )
    temporal = report["checks"].get("temporal_split")
    if temporal is not None:
        lines.append(
            "- OOS 严格晚于训练："
            f"`{temporal['oos_strictly_after_training']}` "
            f"({temporal['effective_training_end_date']} → "
            f"{temporal['oos_date_min']})"
        )

    lines.extend(
        [
            "",
            "## 问题与待办",
            "",
            "| 级别 | 范围 | 代码 | 说明 |",
            "|---|---|---|---|",
        ]
    )
    if report["issues"]:
        for item in report["issues"]:
            lines.append(
                f"| {cell(item['severity'])} | {cell(item['scope'])} | "
                f"{cell(item['code'])} | {cell(item['message'])} |"
            )
    else:
        lines.append("| - | - | - | 未发现问题 |")
    return "\n".join(lines) + "\n"


def write_audit_reports(
    report: dict[str, Any],
    out_dir: Path,
) -> tuple[Path, Path]:
    """原子写入 ``ranker_input_audit.json`` 与同名 Markdown。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ranker_input_audit.json"
    markdown_path = out_dir / "ranker_input_audit.md"
    json_tmp = json_path.with_suffix(".json.tmp")
    markdown_tmp = markdown_path.with_suffix(".md.tmp")
    json_tmp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_tmp.write_text(render_markdown(report), encoding="utf-8")
    json_tmp.replace(json_path)
    markdown_tmp.replace(markdown_path)
    return json_path, markdown_path


def main() -> None:
    """运行只读审计；仅发现 critical 问题时返回退出码 2。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-embeddings", type=Path, required=True)
    parser.add_argument("--train-panel", type=Path, required=True)
    parser.add_argument("--oos-embeddings", type=Path)
    parser.add_argument("--min-train-coverage", type=float, default=0.95)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit_ranker_inputs(
        args.train_embeddings,
        args.train_panel,
        args.oos_embeddings,
        min_train_coverage=args.min_train_coverage,
    )
    json_path, markdown_path = write_audit_reports(report, args.out_dir)
    print(f"status={report['status']}")
    print(json_path)
    print(markdown_path)
    if report["status"] == "fail":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
