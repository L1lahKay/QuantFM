from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "quant_fm/scripts/run_dense230m_strict_oos.sh"


def _entrypoint_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    train_dir = tmp_path / "train_embeddings"
    train_dir.mkdir()
    train_embeddings = train_dir / "all.parquet"
    oos_embeddings = tmp_path / "oos.parquet"
    train_calendar = tmp_path / "train_calendar.txt"
    oos_calendar = tmp_path / "oos_calendar.txt"
    train_universe = tmp_path / "train_universe.parquet"
    oos_universe = tmp_path / "oos_universe.parquet"
    acceptance = tmp_path / "pretrain_acceptance.json"
    for path in (
        train_embeddings,
        oos_embeddings,
        train_calendar,
        oos_calendar,
        train_universe,
        oos_universe,
        acceptance,
    ):
        path.touch()

    calls = tmp_path / "python_calls.txt"
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "$STRICT_OOS_CALL_LOG"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": str(fake_python),
            "STRICT_OOS_CALL_LOG": str(calls),
            "WORKDIR": str(tmp_path / "workdir"),
            "STRICT_DIR": str(tmp_path / "strict"),
            "TRAIN_EMB_DIR": str(train_dir),
            "TRAIN_EMBEDDINGS": str(train_embeddings),
            "OOS_EMBEDDINGS": str(oos_embeddings),
            "TRAIN_CALENDAR": str(train_calendar),
            "OOS_CALENDAR": str(oos_calendar),
            "TRAIN_UNIVERSE": str(train_universe),
            "OOS_UNIVERSE": str(oos_universe),
            "RETURN_SPEC": "vwap_t1_vwap_t2",
            "FM_TRAINING_END_DATE": "2025-12-31",
            "PRETRAIN_ACCEPTANCE": str(acceptance),
            "MARKET_BENCHMARK_PANEL": "",
        }
    )
    return env, calls


def _run_entrypoint(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(ENTRYPOINT)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_strict_entrypoint_passes_frozen_objective_and_oos_calendar(
    tmp_path: Path,
) -> None:
    env, calls_path = _entrypoint_env(tmp_path)
    result = _run_entrypoint(env)
    assert result.returncode == 0, result.stderr
    calls = [shlex.split(line) for line in calls_path.read_text().splitlines()]

    acceptance = next(
        call
        for call in calls
        if "quant_fm.scripts.validate_pretrain_acceptance" in call
    )
    assert acceptance[acceptance.index("--path") + 1] == env["PRETRAIN_ACCEPTANCE"]

    lineage = next(
        call for call in calls if "quant_fm.scripts.validate_pretrain_lineage" in call
    )
    assert lineage[lineage.index("--acceptance") + 1] == env["PRETRAIN_ACCEPTANCE"]
    assert lineage[lineage.index("--train-embeddings") + 1] == env["TRAIN_EMBEDDINGS"]
    assert lineage[lineage.index("--oos-embeddings") + 1] == env["OOS_EMBEDDINGS"]
    assert (
        lineage[lineage.index("--expected-training-end") + 1]
        == env["FM_TRAINING_END_DATE"]
    )

    delivery = next(
        call for call in calls if "quant_fm.scripts.build_oos_delivery" in call
    )
    assert (
        delivery[delivery.index("--pretrain-acceptance") + 1]
        == env["PRETRAIN_ACCEPTANCE"]
    )
    expected_objective = {
        "--ndcg-ks": "50,300,350",
        "--ndcg-k-weights": "0.20,0.60,0.20",
        "--head-loss-weight": "1.0",
        "--global-ic-weight": "0.30",
        "--aux-huber-weight": "0.05",
        "--aux-huber-beta": "0.5",
        "--pair-samples-per-day": "8192",
        "--hard-pair-fraction": "0.75",
        "--min-label-rank-gap": "0.02",
        "--score-temperature": "1.0",
    }
    for flag, value in expected_objective.items():
        assert delivery[delivery.index(flag) + 1] == value

    evaluation = next(
        call for call in calls if "quant_fm.downstream.run_score_evaluation" in call
    )
    assert evaluation[evaluation.index("--calendar") + 1] == env["OOS_CALENDAR"]


def test_strict_entrypoint_derives_fm_cutoff_when_no_manual_assertion(
    tmp_path: Path,
) -> None:
    env, calls_path = _entrypoint_env(tmp_path)
    env["FM_TRAINING_END_DATE"] = ""
    result = _run_entrypoint(env)
    assert result.returncode == 0, result.stderr
    calls = [shlex.split(line) for line in calls_path.read_text().splitlines()]
    lineage = next(
        call for call in calls if "quant_fm.scripts.validate_pretrain_lineage" in call
    )
    delivery = next(
        call for call in calls if "quant_fm.scripts.build_oos_delivery" in call
    )
    assert "--expected-training-end" not in lineage
    assert "--fm-training-end-date" not in delivery


def test_strict_entrypoint_locks_return_spec_and_training_embedding_file(
    tmp_path: Path,
) -> None:
    env, _ = _entrypoint_env(tmp_path)
    env["RETURN_SPEC"] = "open_t1_close_t1"
    result = _run_entrypoint(env)
    assert result.returncode == 2
    assert "只允许 RETURN_SPEC=vwap_t1_vwap_t2" in result.stderr

    env["RETURN_SPEC"] = "vwap_t1_vwap_t2"
    other_embeddings = tmp_path / "different_train.parquet"
    other_embeddings.touch()
    env["TRAIN_EMBEDDINGS"] = str(other_embeddings)
    result = _run_entrypoint(env)
    assert result.returncode == 2
    assert "TRAIN_EMBEDDINGS 必须解析为 TRAIN_EMB_DIR/all.parquet" in result.stderr


def test_strict_entrypoint_cannot_bypass_pretrain_acceptance(tmp_path: Path) -> None:
    env, _ = _entrypoint_env(tmp_path)
    env["REQUIRE_PRETRAIN_ACCEPTANCE"] = "0"
    env["PRETRAIN_ACCEPTANCE"] = ""
    result = _run_entrypoint(env)
    assert result.returncode == 2
    assert "严格入口必须设置新版 FM 的 PRETRAIN_ACCEPTANCE" in result.stderr


def test_strict_entrypoint_parses_required_acceptance_json(tmp_path: Path) -> None:
    env, _ = _entrypoint_env(tmp_path)
    acceptance = tmp_path / "pretrain_acceptance.json"
    acceptance.write_text('{"accepted": true, "decision": "PASS"}', encoding="utf-8")
    env.update(
        {
            "PYTHON": sys.executable,
            "PRETRAIN_ACCEPTANCE": str(acceptance),
        }
    )
    result = _run_entrypoint(env)
    assert result.returncode == 2
    assert "missing required fields" in result.stderr
