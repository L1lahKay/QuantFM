"""
对 ``scores.parquet`` 执行独立、只读的交付质量门禁。

脚本不修改信号文件，只在指定目录生成 ``signal_quality.json`` 与
``signal_quality.md``。结构、主键、有限值、日期范围、逐日覆盖和常数截面属于硬
门禁；标签、基线与性能阈值均为可选项。

示例::

    uv run python -m quant_fm.scripts.signal_quality_gate \
      --scores quant_fm/runs/oos2026_dense230/delivery_oos/scores.parquet \
      --manifest quant_fm/runs/oos2026_dense230/delivery_oos/signal_manifest.json \
      --expected-dates quant_fm/runs/oos2026_dense230/data/dates.txt \
      --out-dir quant_fm/runs/oos2026_dense230/delivery_oos/quality
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from quant_fm.downstream.evaluate import ic_statistics, rank_ic

if TYPE_CHECKING:
    from collections.abc import Iterable

_SCORE_COLUMNS = ["date", "symbol", "score"]
_BASELINE_CANDIDATES = (
    "factor_mom_1",
    "factor_ret_oc",
    "factor_rev_1",
    "factor_ofi",
)


def _json_safe(value: object) -> object:
    """递归转换 numpy 标量，并用 ``null`` 代替非有限浮点数。"""
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _add_check(
    report: dict[str, Any],
    *,
    check_id: str,
    passed: bool,
    message: str,
    severity: str = "error",
    metrics: dict[str, Any] | None = None,
) -> None:
    status = "pass" if passed else ("warn" if severity == "warning" else "fail")
    report["checks"].append(
        {
            "id": check_id,
            "severity": severity,
            "status": status,
            "message": message,
            "metrics": metrics or {},
        }
    )


def _frame_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _normalise_scores(frame: pl.DataFrame, *, score_col: str = "score") -> pl.DataFrame:
    return frame.select(
        pl.col("date").cast(pl.Utf8, strict=False),
        pl.col("symbol").cast(pl.Utf8, strict=False),
        pl.col(score_col).cast(pl.Float64, strict=False).alias("score"),
    )


def _valid_date_expr() -> pl.Expr:
    return (
        pl.col("date").is_not_null()
        & pl.col("date").str.contains(r"^\d{4}-\d{2}-\d{2}$").fill_null(False)
        & pl.col("date").str.to_date("%Y-%m-%d", strict=False).is_not_null()
    )


def _valid_symbol_expr() -> pl.Expr:
    return pl.col("symbol").is_not_null() & pl.col("symbol").str.contains(
        r"^\d{6}$"
    ).fill_null(False)


def _valid_score_expr() -> pl.Expr:
    return pl.col("score").is_not_null() & pl.col("score").is_finite().fill_null(False)


def _score_distribution(scores: pl.DataFrame) -> tuple[list[dict[str, Any]], list[str]]:
    daily = (
        scores.group_by("date")
        .agg(
            pl.len().alias("rows"),
            pl.col("symbol").n_unique().alias("symbols"),
            pl.col("score").n_unique().alias("unique_scores"),
            pl.col("score").mean().alias("mean"),
            pl.col("score").std(ddof=0).alias("std"),
            pl.col("score").min().alias("min"),
            pl.col("score").quantile(0.01).alias("p01"),
            pl.col("score").quantile(0.25).alias("p25"),
            pl.col("score").median().alias("p50"),
            pl.col("score").quantile(0.75).alias("p75"),
            pl.col("score").quantile(0.99).alias("p99"),
            pl.col("score").max().alias("max"),
        )
        .with_columns((pl.col("unique_scores") / pl.col("rows")).alias("unique_ratio"))
        .sort("date")
    )
    rows = daily.to_dicts()
    constant_dates = [
        str(row["date"])
        for row in rows
        if int(row["unique_scores"]) <= 1
        or row["std"] is None
        or float(row["std"]) <= 1e-12
    ]
    return rows, constant_dates


def _turnover_metrics(scores: pl.DataFrame, *, top_k: int) -> dict[str, Any]:
    """计算相邻信号日的截面秩变化和 top-k 单边换手。"""
    prior_ranks: dict[str, float] | None = None
    prior_top: set[str] | None = None
    rank_changes: list[float] = []
    top_turnovers: list[float] = []
    common_coverages: list[float] = []

    for _, daily in scores.sort(["date", "symbol"]).group_by(
        "date", maintain_order=True
    ):
        ranked = daily.with_columns(
            (pl.col("score").rank(method="average", descending=True) / pl.len()).alias(
                "_rank_pct"
            )
        )
        current_ranks = dict(
            zip(
                ranked["symbol"].to_list(),
                ranked["_rank_pct"].to_list(),
                strict=True,
            )
        )
        k = min(max(top_k, 1), ranked.height)
        current_top = set(
            ranked.sort(["score", "symbol"], descending=[True, False])
            .head(k)["symbol"]
            .to_list()
        )
        if prior_ranks is not None and prior_top is not None:
            common = sorted(set(prior_ranks) & set(current_ranks))
            if common:
                rank_changes.append(
                    float(
                        np.mean(
                            [
                                abs(current_ranks[symbol] - prior_ranks[symbol])
                                for symbol in common
                            ]
                        )
                    )
                )
            common_coverages.append(len(common) / max(len(prior_ranks), 1))
            top_turnovers.append(1.0 - len(current_top & prior_top) / len(prior_top))
        prior_ranks = current_ranks
        prior_top = current_top

    return {
        "top_k": top_k,
        "transitions": len(top_turnovers),
        "mean_rank_change": (float(np.mean(rank_changes)) if rank_changes else None),
        "mean_top_k_turnover": (
            float(np.mean(top_turnovers)) if top_turnovers else None
        ),
        "max_top_k_turnover": (float(np.max(top_turnovers)) if top_turnovers else None),
        "mean_common_symbol_coverage": (
            float(np.mean(common_coverages)) if common_coverages else None
        ),
    }


def _ic_report(
    scores: pl.DataFrame,
    panel: pl.DataFrame,
    *,
    return_col: str,
) -> dict[str, Any]:
    daily = rank_ic(scores, panel, ret_col=return_col)
    stats = asdict(ic_statistics(daily))
    return {
        "statistics": stats,
        "daily": daily.to_dicts(),
    }


def _load_expected_dates(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    dates = [line.strip() for line in raw.splitlines() if line.strip()]
    if not dates:
        msg = f"expected date file is empty: {path}"
        raise ValueError(msg)
    return sorted(set(dates))


def _manifest_checks(
    report: dict[str, Any],
    *,
    manifest_path: Path,
    scores_path: Path,
    rows: int,
    dates: list[str],
) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _add_check(
            report,
            check_id="manifest_readable",
            passed=False,
            message=f"cannot read signal manifest: {exc}",
        )
        return
    report["inputs"]["manifest"] = _frame_info(manifest_path)
    data = manifest.get("data", {}) if isinstance(manifest, dict) else {}
    expected_data = {
        "file": scores_path.name,
        "schema": {"date": "string", "symbol": "string", "score": "float64"},
        "primary_key": ["date", "symbol"],
        "rows": rows,
        "dates": len(dates),
        "date_min": min(dates) if dates else None,
        "date_max": max(dates) if dates else None,
    }
    mismatches = {
        key: {"declared": data.get(key), "actual": actual}
        for key, actual in expected_data.items()
        if data.get(key) != actual
    }
    _add_check(
        report,
        check_id="manifest_contract",
        passed=not mismatches,
        message=(
            "manifest matches the score artifact"
            if not mismatches
            else f"manifest mismatches: {sorted(mismatches)}"
        ),
        metrics={"mismatches": mismatches},
    )
    semantics = manifest.get("score_semantics", {})
    semantics_ok = (
        semantics.get("direction") == "higher_is_more_bullish"
        and semantics.get("availability") == "available_after_signal_date_close"
        and semantics.get("comparability") == "cross_sectional_within_date"
    )
    _add_check(
        report,
        check_id="manifest_score_semantics",
        passed=semantics_ok,
        message=(
            "score direction, availability and comparability are declared"
            if semantics_ok
            else "manifest has missing or unexpected score semantics"
        ),
        metrics={"score_semantics": semantics},
    )


def _load_baseline_scores(
    report: dict[str, Any],
    *,
    path: Path,
) -> pl.DataFrame | None:
    try:
        raw = pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError) as exc:
        _add_check(
            report,
            check_id="baseline_readable",
            passed=False,
            message=f"cannot read baseline scores: {exc}",
        )
        return None
    missing = set(_SCORE_COLUMNS) - set(raw.columns)
    if missing:
        _add_check(
            report,
            check_id="baseline_schema",
            passed=False,
            message=f"baseline is missing columns: {sorted(missing)}",
        )
        return None
    baseline = _normalise_scores(raw)
    invalid = baseline.filter(
        ~(_valid_date_expr() & _valid_symbol_expr() & _valid_score_expr())
    ).height
    duplicates = baseline.select(
        pl.struct(["date", "symbol"]).is_duplicated().sum()
    ).item()
    valid = invalid == 0 and duplicates == 0 and baseline.height > 0
    _add_check(
        report,
        check_id="baseline_values",
        passed=valid,
        message=(
            "baseline scores are valid"
            if valid
            else f"baseline invalid rows={invalid}, duplicate rows={duplicates}"
        ),
        metrics={
            "rows": baseline.height,
            "invalid_rows": invalid,
            "duplicates": duplicates,
        },
    )
    if not valid:
        return None
    report["inputs"]["baseline_scores"] = _frame_info(path)
    return baseline.sort(["date", "symbol"])


def _coverage_and_labels(
    report: dict[str, Any],
    *,
    scores: pl.DataFrame,
    panel_path: Path,
    return_col: str,
    min_daily_coverage: float,
    min_label_coverage: float,
    require_complete_labels: bool,
) -> pl.DataFrame | None:
    try:
        panel_raw = pl.read_parquet(panel_path)
    except (OSError, pl.exceptions.PolarsError) as exc:
        _add_check(
            report,
            check_id="panel_readable",
            passed=False,
            message=f"cannot read panel: {exc}",
        )
        return None
    report["inputs"]["panel"] = _frame_info(panel_path)
    missing = {"date", "symbol"} - set(panel_raw.columns)
    if missing:
        _add_check(
            report,
            check_id="panel_schema",
            passed=False,
            message=f"panel is missing columns: {sorted(missing)}",
        )
        return None

    expressions = [
        pl.col("date").cast(pl.Utf8, strict=False),
        pl.col("symbol").cast(pl.Utf8, strict=False),
    ]
    for column in panel_raw.columns:
        if column not in {"date", "symbol"}:
            expressions.append(pl.col(column))
    panel = panel_raw.select(expressions)
    duplicate_rows = panel.select(
        pl.struct(["date", "symbol"]).is_duplicated().sum()
    ).item()
    _add_check(
        report,
        check_id="panel_primary_key",
        passed=duplicate_rows == 0,
        message=f"panel duplicate key rows={duplicate_rows}",
        metrics={"duplicate_rows": duplicate_rows},
    )
    if duplicate_rows:
        panel = panel.unique(subset=["date", "symbol"], keep="first")

    score_dates = scores["date"].unique().to_list()
    expected_keys = (
        panel.filter(pl.col("date").is_in(score_dates))
        .select(["date", "symbol"])
        .unique()
    )
    score_keys = scores.select(["date", "symbol"]).unique()
    matched = expected_keys.join(score_keys, on=["date", "symbol"], how="inner")
    extra = score_keys.join(expected_keys, on=["date", "symbol"], how="anti")
    missing_scores = expected_keys.join(score_keys, on=["date", "symbol"], how="anti")

    expected_daily = expected_keys.group_by("date").agg(pl.len().alias("expected_rows"))
    matched_daily = matched.group_by("date").agg(pl.len().alias("matched_rows"))
    coverage_daily = (
        pl.DataFrame({"date": score_dates})
        .join(expected_daily, on="date", how="left")
        .join(matched_daily, on="date", how="left")
        .with_columns(
            pl.col("expected_rows").fill_null(0),
            pl.col("matched_rows").fill_null(0),
        )
        .with_columns(
            pl.when(pl.col("expected_rows") > 0)
            .then(pl.col("matched_rows") / pl.col("expected_rows"))
            .otherwise(0.0)
            .alias("coverage")
        )
        .sort("date")
    )
    daily_rows = coverage_daily.to_dicts()
    min_coverage = (
        float(coverage_daily["coverage"].min()) if coverage_daily.height else 0.0
    )
    coverage_ok = (
        expected_keys.height > 0
        and min_coverage >= min_daily_coverage
        and extra.height == 0
    )
    report["coverage"] = {
        "panel_rows_on_signal_dates": expected_keys.height,
        "matched_rows": matched.height,
        "missing_score_rows": missing_scores.height,
        "extra_score_rows": extra.height,
        "overall": matched.height / expected_keys.height
        if expected_keys.height
        else 0.0,
        "minimum_daily": min_coverage,
        "daily": daily_rows,
    }
    _add_check(
        report,
        check_id="panel_key_coverage",
        passed=coverage_ok,
        message=(
            f"minimum daily universe coverage={min_coverage:.4f}, "
            f"extra score keys={extra.height}"
        ),
        metrics={
            "threshold": min_daily_coverage,
            "missing_score_rows": missing_scores.height,
            "extra_score_rows": extra.height,
        },
    )

    if return_col not in panel.columns:
        _add_check(
            report,
            check_id="forward_return_available",
            passed=False,
            message=f"panel has no optional forward return column {return_col!r}",
            severity="warning",
        )
        return panel

    panel = panel.with_columns(
        pl.col(return_col).cast(pl.Float64, strict=False).alias(return_col)
    )
    labelled = scores.join(panel, on=["date", "symbol"], how="inner")
    if "eligible_at_signal" in labelled.columns:
        labelled = labelled.filter(pl.col("eligible_at_signal").fill_null(False))
    valid_labels = labelled.filter(
        pl.col(return_col).is_not_null()
        & pl.col(return_col).is_finite().fill_null(False)
    )
    label_coverage = valid_labels.height / labelled.height if labelled.height else 0.0
    labels_ok = label_coverage >= min_label_coverage
    _add_check(
        report,
        check_id="forward_return_coverage",
        passed=labels_ok,
        message=(
            f"finite {return_col} coverage={valid_labels.height}/{labelled.height} "
            f"({label_coverage:.4f})"
        ),
        severity="error" if require_complete_labels else "warning",
        metrics={"coverage": label_coverage, "threshold": min_label_coverage},
    )
    report["evaluation"] = {
        "return_col": return_col,
        "eligible_rows": labelled.height,
        "finite_label_rows": valid_labels.height,
        "label_coverage": label_coverage,
        "candidate_ic": _ic_report(scores, panel, return_col=return_col),
        "zero_skill_baseline": {"mean_ic": 0.0},
    }
    return panel


def _baseline_from_panel(
    panel: pl.DataFrame,
    *,
    baseline_col: str | None,
) -> tuple[pl.DataFrame | None, str | None]:
    column = baseline_col
    if column is None:
        column = next(
            (
                candidate
                for candidate in _BASELINE_CANDIDATES
                if candidate in panel.columns
            ),
            None,
        )
    if column is None or column not in panel.columns:
        return None, column
    frame = panel.select(
        "date",
        "symbol",
        pl.col(column).cast(pl.Float64, strict=False).alias("score"),
    ).filter(_valid_score_expr())
    if "eligible_at_signal" in panel.columns:
        eligible = panel.select("date", "symbol", "eligible_at_signal")
        frame = (
            frame.join(eligible, on=["date", "symbol"], how="left")
            .filter(pl.col("eligible_at_signal").fill_null(False))
            .drop("eligible_at_signal")
        )
    return frame, column


def _write_reports(
    report: dict[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    report["status"] = (
        "fail"
        if any(check["status"] == "fail" for check in report["checks"])
        else "pass"
    )
    report["summary"] = {
        "passed": sum(check["status"] == "pass" for check in report["checks"]),
        "warnings": sum(check["status"] == "warn" for check in report["checks"]),
        "failed": sum(check["status"] == "fail" for check in report["checks"]),
    }
    safe_report = _json_safe(report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    json_tmp.write_text(
        json.dumps(safe_report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    json_tmp.replace(json_path)

    lines = [
        "# Signal quality gate",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Scores: `{report['inputs'].get('scores', {}).get('path', 'unavailable')}`",
        f"- Created (UTC): `{report['created_utc']}`",
        "",
        "## Checks",
        "",
        "| Check | Severity | Status | Detail |",
        "|---|---|---|---|",
    ]
    for check in report["checks"]:
        detail = str(check["message"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{check['id']}` | {check['severity']} | {check['status']} | {detail} |"
        )
    signal = report.get("signal", {})
    if signal:
        lines.extend(
            [
                "",
                "## Signal summary",
                "",
                f"- Rows / dates / symbols: {signal.get('rows')} / "
                f"{signal.get('dates')} / {signal.get('symbols')}",
                f"- Date range: `{signal.get('date_min')}` to `{signal.get('date_max')}`",
                f"- Daily names min / median / max: {signal.get('daily_names_min')} / "
                f"{signal.get('daily_names_median')} / {signal.get('daily_names_max')}",
            ]
        )
    coverage = report.get("coverage", {})
    if coverage:
        lines.extend(
            [
                "",
                "## Coverage",
                "",
                f"- Overall / minimum daily: {coverage.get('overall', 0):.4f} / "
                f"{coverage.get('minimum_daily', 0):.4f}",
                f"- Missing / extra score keys: {coverage.get('missing_score_rows')} / "
                f"{coverage.get('extra_score_rows')}",
            ]
        )
    evaluation = report.get("evaluation", {})
    if evaluation:
        candidate = evaluation.get("candidate_ic", {}).get("statistics", {})
        lines.extend(
            [
                "",
                "## IC and baseline",
                "",
                f"- Candidate mean IC / ICIR / periods: {candidate.get('mean_ic')} / "
                f"{candidate.get('icir')} / {candidate.get('n_periods')}",
                f"- Label coverage: {evaluation.get('label_coverage')}",
                f"- Baseline: {evaluation.get('baseline', evaluation.get('zero_skill_baseline'))}",
            ]
        )
    turnover = report.get("turnover", {})
    if turnover:
        lines.extend(
            [
                "",
                "## Turnover",
                "",
                f"- Mean top-k turnover: {turnover.get('mean_top_k_turnover')}",
                f"- Mean percentile-rank change: {turnover.get('mean_rank_change')}",
            ]
        )
    markdown_tmp = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    markdown_tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    markdown_tmp.replace(markdown_path)


def evaluate_signal_quality(
    *,
    scores_path: Path,
    json_path: Path,
    markdown_path: Path,
    manifest_path: Path | None = None,
    panel_path: Path | None = None,
    baseline_scores_path: Path | None = None,
    baseline_col: str | None = None,
    expected_dates: Iterable[str] | None = None,
    expected_start: str | None = None,
    expected_end: str | None = None,
    min_names_per_day: int = 20,
    min_daily_coverage: float = 0.95,
    return_col: str = "fwd_ret",
    min_label_coverage: float = 0.95,
    require_complete_labels: bool = True,
    top_k: int = 50,
    max_top_k_turnover: float | None = None,
    min_mean_ic: float | None = None,
    min_baseline_ic_delta: float | None = None,
) -> dict[str, Any]:
    """
    执行信号门禁并写出 JSON 与 Markdown 报告。

    返回的 ``status`` 为 ``pass`` 或 ``fail``。函数本身不因质量失败抛异常，便于编排
    端读取完整报告；CLI 在失败时使用退出码 2。
    """
    report: dict[str, Any] = {
        "format_version": "1.0",
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "status": "fail",
        "inputs": {},
        "checks": [],
    }
    try:
        raw = pl.read_parquet(scores_path)
        report["inputs"]["scores"] = _frame_info(scores_path)
    except (OSError, pl.exceptions.PolarsError) as exc:
        _add_check(
            report,
            check_id="scores_readable",
            passed=False,
            message=f"cannot read scores: {exc}",
        )
        _write_reports(report, json_path=json_path, markdown_path=markdown_path)
        return report

    exact_schema = (
        raw.columns == _SCORE_COLUMNS
        and raw.schema.get("date") == pl.Utf8
        and raw.schema.get("symbol") == pl.Utf8
        and raw.schema.get("score") == pl.Float64
    )
    _add_check(
        report,
        check_id="score_schema",
        passed=exact_schema,
        message=f"actual schema={dict(raw.schema)}",
        metrics={
            "expected_columns": _SCORE_COLUMNS,
            "expected_types": {
                "date": "String",
                "symbol": "String",
                "score": "Float64",
            },
        },
    )
    missing = set(_SCORE_COLUMNS) - set(raw.columns)
    if missing:
        _add_check(
            report,
            check_id="required_columns",
            passed=False,
            message=f"missing required columns: {sorted(missing)}",
        )
        _write_reports(report, json_path=json_path, markdown_path=markdown_path)
        return report

    scores = _normalise_scores(raw)
    _add_check(
        report,
        check_id="non_empty",
        passed=scores.height > 0,
        message=f"score rows={scores.height}",
    )
    invalid_dates = scores.filter(~_valid_date_expr()).height
    invalid_symbols = scores.filter(~_valid_symbol_expr()).height
    invalid_scores = scores.filter(~_valid_score_expr()).height
    duplicate_rows = scores.select(
        pl.struct(["date", "symbol"]).is_duplicated().sum()
    ).item()
    _add_check(
        report,
        check_id="date_values",
        passed=invalid_dates == 0,
        message=f"invalid or null YYYY-MM-DD dates={invalid_dates}",
        metrics={"invalid_rows": invalid_dates},
    )
    _add_check(
        report,
        check_id="symbol_values",
        passed=invalid_symbols == 0,
        message=f"invalid or null six-digit symbols={invalid_symbols}",
        metrics={"invalid_rows": invalid_symbols},
    )
    _add_check(
        report,
        check_id="finite_scores",
        passed=invalid_scores == 0,
        message=f"null, non-numeric or non-finite scores={invalid_scores}",
        metrics={"invalid_rows": invalid_scores},
    )
    _add_check(
        report,
        check_id="primary_key",
        passed=duplicate_rows == 0,
        message=f"duplicate (date, symbol) rows={duplicate_rows}",
        metrics={"duplicate_rows": duplicate_rows},
    )

    clean = scores.filter(
        _valid_date_expr() & _valid_symbol_expr() & _valid_score_expr()
    )
    if clean.is_empty():
        _add_check(
            report,
            check_id="valid_rows_available",
            passed=False,
            message="no valid rows remain for distribution checks",
        )
        _write_reports(report, json_path=json_path, markdown_path=markdown_path)
        return report
    clean = clean.unique(subset=["date", "symbol"], keep="first").sort(
        ["date", "symbol"]
    )
    dates = sorted(str(value) for value in clean["date"].unique())
    daily_counts = clean.group_by("date").len().sort("date")["len"]
    signal_summary = {
        "rows": clean.height,
        "dates": len(dates),
        "symbols": clean["symbol"].n_unique(),
        "date_min": min(dates),
        "date_max": max(dates),
        "daily_names_min": int(daily_counts.min()),
        "daily_names_median": float(daily_counts.median()),
        "daily_names_max": int(daily_counts.max()),
    }
    report["signal"] = signal_summary
    names_ok = int(daily_counts.min()) >= min_names_per_day
    _add_check(
        report,
        check_id="daily_cross_section_size",
        passed=names_ok,
        message=(
            f"daily names min/median/max={signal_summary['daily_names_min']}/"
            f"{signal_summary['daily_names_median']}/"
            f"{signal_summary['daily_names_max']}"
        ),
        metrics={"threshold": min_names_per_day},
    )

    distribution, constant_dates = _score_distribution(clean)
    report["distribution_by_date"] = distribution
    _add_check(
        report,
        check_id="non_constant_cross_sections",
        passed=not constant_dates,
        message=(
            "all daily score cross-sections have non-zero dispersion"
            if not constant_dates
            else f"constant or near-constant score dates={constant_dates[:10]}"
        ),
        metrics={"constant_dates": constant_dates, "epsilon": 1e-12},
    )
    tied_dates = [
        str(row["date"]) for row in distribution if float(row["unique_ratio"]) < 0.1
    ]
    _add_check(
        report,
        check_id="score_tie_concentration",
        passed=not tied_dates,
        message=(
            "daily unique-score ratio is at least 10%"
            if not tied_dates
            else f"heavy score ties on dates={tied_dates[:10]}"
        ),
        severity="warning",
        metrics={"dates": tied_dates, "threshold": 0.1},
    )

    requested_dates = sorted(set(expected_dates or []))
    if requested_dates:
        missing_dates = sorted(set(requested_dates) - set(dates))
        extra_dates = sorted(set(dates) - set(requested_dates))
        _add_check(
            report,
            check_id="expected_dates",
            passed=not missing_dates and not extra_dates,
            message=(
                f"expected/actual dates={len(requested_dates)}/{len(dates)}, "
                f"missing={missing_dates[:10]}, extra={extra_dates[:10]}"
            ),
            metrics={"missing": missing_dates, "extra": extra_dates},
        )
    if expected_start is not None or expected_end is not None:
        start_ok = expected_start is None or min(dates) == expected_start
        end_ok = expected_end is None or max(dates) == expected_end
        _add_check(
            report,
            check_id="expected_date_range",
            passed=start_ok and end_ok,
            message=(
                f"actual={min(dates)}..{max(dates)}, "
                f"expected={expected_start or '*'}..{expected_end or '*'}"
            ),
        )

    if manifest_path is not None:
        _manifest_checks(
            report,
            manifest_path=manifest_path,
            scores_path=scores_path,
            rows=clean.height,
            dates=dates,
        )

    panel = None
    if panel_path is not None:
        panel = _coverage_and_labels(
            report,
            scores=clean,
            panel_path=panel_path,
            return_col=return_col,
            min_daily_coverage=min_daily_coverage,
            min_label_coverage=min_label_coverage,
            require_complete_labels=require_complete_labels,
        )

    report["turnover"] = _turnover_metrics(clean, top_k=top_k)
    if max_top_k_turnover is not None:
        observed_turnover = report["turnover"]["mean_top_k_turnover"]
        turnover_ok = (
            observed_turnover is not None
            and float(observed_turnover) <= max_top_k_turnover
        )
        _add_check(
            report,
            check_id="top_k_turnover_limit",
            passed=turnover_ok,
            message=(
                f"mean top-{top_k} turnover={observed_turnover}, "
                f"maximum={max_top_k_turnover}"
            ),
        )

    baseline = None
    baseline_name = None
    if baseline_scores_path is not None:
        baseline = _load_baseline_scores(report, path=baseline_scores_path)
        baseline_name = str(baseline_scores_path.resolve())
    elif panel is not None:
        baseline, selected_col = _baseline_from_panel(panel, baseline_col=baseline_col)
        baseline_name = f"panel:{selected_col}" if selected_col is not None else None
        if baseline_col is not None and baseline is None:
            _add_check(
                report,
                check_id="baseline_column_available",
                passed=False,
                message=f"panel has no usable baseline column {baseline_col!r}",
            )

    if baseline is not None:
        baseline_turnover = _turnover_metrics(baseline, top_k=top_k)
        report["baseline"] = {
            "name": baseline_name,
            "rows": baseline.height,
            "turnover": baseline_turnover,
        }
        if panel is not None and return_col in panel.columns:
            baseline_ic = _ic_report(baseline, panel, return_col=return_col)
            report["baseline"]["ic"] = baseline_ic
            if "evaluation" in report:
                candidate_mean = report["evaluation"]["candidate_ic"]["statistics"][
                    "mean_ic"
                ]
                baseline_mean = baseline_ic["statistics"]["mean_ic"]
                delta = (
                    float(candidate_mean) - float(baseline_mean)
                    if math.isfinite(float(candidate_mean))
                    and math.isfinite(float(baseline_mean))
                    else None
                )
                report["evaluation"]["baseline"] = {
                    "name": baseline_name,
                    "mean_ic": baseline_mean,
                    "candidate_minus_baseline_mean_ic": delta,
                }

    if min_mean_ic is not None:
        candidate_stats = (
            report.get("evaluation", {}).get("candidate_ic", {}).get("statistics", {})
        )
        observed_ic = candidate_stats.get("mean_ic")
        ic_ok = (
            isinstance(observed_ic, int | float)
            and math.isfinite(float(observed_ic))
            and float(observed_ic) >= min_mean_ic
        )
        _add_check(
            report,
            check_id="minimum_mean_ic",
            passed=ic_ok,
            message=f"candidate mean IC={observed_ic}, minimum={min_mean_ic}",
        )
    if min_baseline_ic_delta is not None:
        observed_delta = (
            report.get("evaluation", {})
            .get("baseline", {})
            .get("candidate_minus_baseline_mean_ic")
        )
        delta_ok = (
            isinstance(observed_delta, int | float)
            and math.isfinite(float(observed_delta))
            and float(observed_delta) >= min_baseline_ic_delta
        )
        _add_check(
            report,
            check_id="minimum_baseline_ic_delta",
            passed=delta_ok,
            message=(
                f"candidate-baseline mean IC={observed_delta}, "
                f"minimum={min_baseline_ic_delta}"
            ),
        )

    _write_reports(report, json_path=json_path, markdown_path=markdown_path)
    return report


def _validate_fraction(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        msg = f"{name} must be between 0 and 1"
        raise ValueError(msg)
    return value


def main() -> None:
    """CLI 入口；严重门禁失败时返回退出码 2。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--baseline-scores", type=Path)
    parser.add_argument("--baseline-col")
    parser.add_argument("--expected-dates", type=Path)
    parser.add_argument("--expected-start")
    parser.add_argument("--expected-end")
    parser.add_argument("--min-names-per-day", type=int, default=20)
    parser.add_argument("--min-daily-coverage", type=float, default=0.95)
    parser.add_argument("--return-col", default="fwd_ret")
    parser.add_argument("--min-label-coverage", type=float, default=0.95)
    parser.add_argument("--allow-incomplete-labels", action="store_true")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-top-k-turnover", type=float)
    parser.add_argument("--min-mean-ic", type=float)
    parser.add_argument("--min-baseline-ic-delta", type=float)
    args = parser.parse_args()

    min_daily_coverage = _validate_fraction(
        "min_daily_coverage", args.min_daily_coverage
    )
    min_label_coverage = _validate_fraction(
        "min_label_coverage", args.min_label_coverage
    )
    if args.min_names_per_day < 1 or args.top_k < 1:
        parser.error("--min-names-per-day and --top-k must be positive")
    if args.max_top_k_turnover is not None:
        _validate_fraction("max_top_k_turnover", args.max_top_k_turnover)

    expected_dates = (
        _load_expected_dates(args.expected_dates) if args.expected_dates else None
    )
    manifest = args.manifest
    if manifest is None:
        adjacent_manifest = args.scores.with_name("signal_manifest.json")
        if adjacent_manifest.exists():
            manifest = adjacent_manifest

    report = evaluate_signal_quality(
        scores_path=args.scores,
        json_path=args.out_dir / "signal_quality.json",
        markdown_path=args.out_dir / "signal_quality.md",
        manifest_path=manifest,
        panel_path=args.panel,
        baseline_scores_path=args.baseline_scores,
        baseline_col=args.baseline_col,
        expected_dates=expected_dates,
        expected_start=args.expected_start,
        expected_end=args.expected_end,
        min_names_per_day=args.min_names_per_day,
        min_daily_coverage=min_daily_coverage,
        return_col=args.return_col,
        min_label_coverage=min_label_coverage,
        require_complete_labels=not args.allow_incomplete_labels,
        top_k=args.top_k,
        max_top_k_turnover=args.max_top_k_turnover,
        min_mean_ic=args.min_mean_ic,
        min_baseline_ic_delta=args.min_baseline_ic_delta,
    )
    print(args.out_dir / "signal_quality.json")
    if report["status"] == "fail":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
