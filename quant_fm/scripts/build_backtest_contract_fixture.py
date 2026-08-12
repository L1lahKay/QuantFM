"""
Build a synthetic, production-shaped package for backtest integration tests.

The fixture exercises only the stable hand-off contract.  It is deliberately
independent of model artifacts and must never be interpreted as a model result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from quant_fm.scripts.package_signal_delivery import package_signal_delivery
from quant_fm.signal.schema import validate_scores

_SOURCE_DIR = "source_signal"
_DELIVERY_DIR = "backtest_delivery"
_ARCHIVE = "backtest_delivery.tar.gz"

_SEMANTICS = {
    "direction": "higher_is_more_bullish",
    "availability": "available_after_signal_date_close",
    "comparability": "cross_sectional_within_date",
}
_SCHEMA = {"date": "string", "symbol": "string", "score": "float64"}


def _synthetic_scores() -> pl.DataFrame:
    """Return deterministic scores covering two dates and both exchanges."""
    return validate_scores(
        pl.DataFrame(
            {
                "date": [
                    "2026-01-05",
                    "2026-01-05",
                    "2026-01-05",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-06",
                    "2026-01-06",
                    "2026-01-06",
                ],
                "symbol": [
                    "000001",
                    "000002",
                    "600000",
                    "688001",
                    "000001",
                    "300001",
                    "600000",
                    "688001",
                ],
                "score": [0.82, -0.31, 0.24, 0.05, -0.18, 0.73, 0.11, -0.42],
            },
            schema={"date": pl.String, "symbol": pl.String, "score": pl.Float64},
        )
    )


def _signal_manifest(
    scores: pl.DataFrame,
    *,
    scores_path: Path,
    created_utc: str,
) -> dict[str, object]:
    """Create the required production manifest fields plus a fixture warning."""
    return {
        "format_version": "1.0",
        "created_utc": created_utc,
        "score_semantics": _SEMANTICS,
        "data": {
            "file": "scores.parquet",
            "file_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
            "schema": _SCHEMA,
            "primary_key": ["date", "symbol"],
            "rows": scores.height,
            "dates": scores["date"].n_unique(),
            "date_min": scores["date"].min(),
            "date_max": scores["date"].max(),
        },
        "artifacts": {
            "fm_checkpoint_sha256": None,
            "vocab_sha256": None,
            "ranker_checkpoint_sha256": None,
        },
        "fixture": {
            "synthetic": True,
            "purpose": "backtest_interface_integration_only",
            "must_not_be_used_for_research_or_trading": True,
        },
        "note": (
            "SYNTHETIC FIXTURE: validates file/schema/time semantics only; "
            "these scores are not Dense230M predictions."
        ),
    }


def _sample_warning() -> str:
    return """# SAMPLE ONLY / 仅供接口联调

本包全部 score 为人工构造值，不是 Dense230M 或任何模型的预测，禁止用于研究结论、
绩效评估、交易或生产发布。

本样本只验证以下稳定接口：

- `scores.parquet` 的列、类型、主键与缺失值约束；
- `score` 越大越看多，且只在同一日截面内可比；
- `score(T)` 在 T 日收盘后可用，回测最早 T+1 执行；
- manifest 元数据与包内 SHA-256 完整性。
"""


def _root_readme() -> str:
    return f"""# QuantFM 回测接口联调样本

这是合成联调样本，不是模型信号。

- `{_SOURCE_DIR}/`：与生产生成器一致的两文件源目录；
- `{_DELIVERY_DIR}/`：冻结、带完整性清单的回测交付目录；
- `{_ARCHIVE}`：便于传输的同内容压缩包。

接收端应优先使用 `{_DELIVERY_DIR}/` 或压缩包进行联调。
"""


def build_backtest_contract_fixture(
    out_root: Path,
    *,
    created_utc: str | None = None,
) -> Path:
    """
    Atomically write a synthetic source signal and frozen delivery package.

    Parameters
    ----------
    out_root
        New, independent output root. Existing paths are never overwritten.
    created_utc
        Optional ISO-8601 timestamp used for reproducible tests.

    Returns
    -------
    Path
        The resolved fixture root.
    """
    out_root = Path(out_root).resolve()
    if out_root.exists():
        msg = f"refusing to overwrite existing fixture root: {out_root}"
        raise FileExistsError(msg)
    out_root.parent.mkdir(parents=True, exist_ok=True)
    stage = out_root.with_name(f".{out_root.name}.tmp-{os.getpid()}")
    if stage.exists():
        msg = f"temporary fixture path already exists: {stage}"
        raise FileExistsError(msg)

    timestamp = created_utc or datetime.now(tz=UTC).isoformat()
    try:
        source_dir = stage / _SOURCE_DIR
        source_dir.mkdir(parents=True)
        scores = _synthetic_scores()
        scores_path = source_dir / "scores.parquet"
        scores.write_parquet(scores_path)
        (source_dir / "signal_manifest.json").write_text(
            json.dumps(
                _signal_manifest(
                    scores,
                    scores_path=scores_path,
                    created_utc=timestamp,
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        warning = stage / "SAMPLE_ONLY.md"
        warning.write_text(_sample_warning(), encoding="utf-8")
        package_signal_delivery(
            source_dir=source_dir,
            out_dir=stage / _DELIVERY_DIR,
            reports=(warning,),
            archive_path=stage / _ARCHIVE,
        )
        (stage / "README.md").write_text(_root_readme(), encoding="utf-8")
        stage.replace(out_root)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return out_root


def main() -> None:
    """Run the fixture builder CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="New independent output root; existing paths are never overwritten.",
    )
    parser.add_argument(
        "--created-utc",
        help="Optional ISO-8601 timestamp for reproducible fixtures.",
    )
    args = parser.parse_args()
    root = build_backtest_contract_fixture(
        args.out_root,
        created_utc=args.created_utc,
    )
    print(f"fixture_root={root}")
    print(f"delivery_dir={root / _DELIVERY_DIR}")
    print(f"archive={root / _ARCHIVE}")


if __name__ == "__main__":
    main()
