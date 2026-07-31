from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from quant_fm.pretrain.eval import evaluation_batch_size
from quant_fm.scripts import posttrain_evaluation


def _config(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "manifest": str(tmp_path / "manifest.json"),
                    "vocab": str(tmp_path / "vocab.json"),
                    "context": 2048,
                },
                "model": {},
                "optim": {"micro_batch_size": 2},
                "runtime": {"out_dir": str(run_dir)},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_evaluation_batch_size_accepts_new_and_legacy_keys() -> None:
    assert (
        evaluation_batch_size({"optim": {"micro_batch_size": 2}}, torch.device("cuda"))
        == 2
    )
    assert (
        evaluation_batch_size({"optim": {"batch_size": 8}}, torch.device("cuda")) == 8
    )
    assert (
        evaluation_batch_size({"optim": {"micro_batch_size": 2}}, torch.device("cpu"))
        == 1
    )


def test_posttrain_plan_is_blocked_until_completion_markers_exist(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    blocked = posttrain_evaluation.build_evaluation_plan(
        config,
        device="cuda",
        max_batches=200,
        unigram_max_batches=200,
        gradient_norm_batches=1,
        python_executable="python",
    )
    assert blocked["runnable"] is False

    run_dir = tmp_path / "run"
    for name in ("best.pt", "final.pt", "final_resume.pt"):
        (run_dir / name).write_bytes(b"placeholder")
    ready = posttrain_evaluation.build_evaluation_plan(
        config,
        device="cuda",
        max_batches=200,
        unigram_max_batches=200,
        gradient_norm_batches=1,
        python_executable="python",
    )
    assert ready["runnable"] is True
    assert [job["split"] for job in ready["jobs"]] == ["val", "test"]
    assert ready["jobs"][0]["command"][:3] == [
        "python",
        "-m",
        "quant_fm.pretrain.eval",
    ]


def test_posttrain_execution_is_serial_and_persisted(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "run"
    for name in ("best.pt", "final.pt", "final_resume.pt"):
        (run_dir / name).write_bytes(b"placeholder")
    plan = posttrain_evaluation.build_evaluation_plan(
        config,
        device="cuda",
        max_batches=2,
        unigram_max_batches=2,
        gradient_norm_batches=0,
        python_executable="python",
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(posttrain_evaluation, "training_process_alive", lambda _: False)

    def fake_run(command, *, check):
        calls.append(command)
        assert check is False
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(posttrain_evaluation.subprocess, "run", fake_run)
    out = run_dir / "posttrain_evaluation_plan.json"
    result = posttrain_evaluation.execute_evaluation_plan(
        plan, config_path=config, plan_path=out
    )
    assert result["state"] == "complete"
    assert [job["state"] for job in result["jobs"]] == ["complete", "complete"]
    assert len(calls) == 2
    assert out.is_file()


def test_posttrain_plan_places_baseline_gate_before_test(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_dir = tmp_path / "run"
    for name in ("best.pt", "final.pt", "final_resume.pt"):
        (run_dir / name).write_bytes(b"placeholder")
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"placeholder")
    plan = posttrain_evaluation.build_evaluation_plan(
        config,
        device="cuda",
        max_batches=2,
        unigram_max_batches=2,
        gradient_norm_batches=0,
        baseline_checkpoint=baseline,
        python_executable="python",
    )
    assert [job["name"] for job in plan["jobs"]] == [
        "candidate_pretrain_val",
        "baseline_pretrain_val",
        "pretrain_noninferiority_gate",
        "candidate_pretrain_test",
    ]
