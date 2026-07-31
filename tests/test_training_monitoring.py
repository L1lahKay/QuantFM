import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from quant_fm.monitoring.training import (
    CheckpointRegistry,
    build_run_metadata,
    collect_training_status,
    parse_training_log,
    render_training_report,
)


def _write_config(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "cn_l2_v1",
                "shards": [
                    {
                        "market": "SZ",
                        "symbol": "000001",
                        "date": "2025-01-02",
                        "path": "/unused/train.parquet",
                        "rows": 100,
                        "sha256": "a",
                        "split": "train",
                    },
                    {
                        "market": "SZ",
                        "symbol": "000001",
                        "date": "2025-01-03",
                        "path": "/unused/val.parquet",
                        "rows": 50,
                        "sha256": "b",
                        "split": "val",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    vocab = tmp_path / "vocab.json"
    vocab.write_text('{"schema_version":"cn_l2_v1"}', encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "data": {
                    "manifest": str(manifest),
                    "vocab": str(vocab),
                    "context": 2048,
                },
                "model": {
                    "d_model": 1024,
                    "n_layers": 18,
                    "backbone_moe": {"enabled": False},
                },
                "optim": {
                    "micro_batch_size": 2,
                    "grad_accum": 8,
                    "max_update_steps": 50_000,
                },
                "runtime": {
                    "out_dir": str(run_dir),
                    "log_every": 25,
                    "ckpt_every": 2_000,
                },
            }
        ),
        encoding="utf-8",
    )
    return config


def test_parse_training_log_tracks_progress_validation_and_errors() -> None:
    parsed = parse_training_log(
        "\n".join(
            [
                "INFO update 25 micro 200 tokens 1000 lr 2.5e-6 loss 11.8 aux 0.0",
                "INFO update 1000 val_loss 6.2",
                "INFO update 50 micro 400 tokens 2000 lr 5e-6 loss 9.7 aux 0.0",
                "INFO update 2000 val_loss 5.8",
                "CUDA out of memory",
            ]
        )
    )
    assert parsed["first_update"]["update"] == 25
    assert parsed["latest_update"]["loss"] == 9.7
    assert parsed["best_validation"] == {"update": 2000, "val_loss": 5.8}
    assert parsed["errors"][0]["code"] == "cuda_oom"


def test_checkpoint_registry_requires_two_stable_observations(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "step2000.pt").write_bytes(b"checkpoint")
    registry = CheckpointRegistry(run_dir / "checkpoint_registry.json")
    first = registry.update(run_dir, now=datetime(2026, 7, 24, tzinfo=UTC))
    second = registry.update(
        run_dir, now=datetime(2026, 7, 24, tzinfo=UTC) + timedelta(minutes=5)
    )
    assert first["checkpoints"][0]["stable"] is False
    assert second["checkpoints"][0]["stable"] is True
    assert second["checkpoints"][0]["resumable"] is True
    assert second["checkpoints"][0]["step"] == 2000


def test_status_metadata_and_report_are_low_cost(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    log = tmp_path / "train.log"
    log.write_text(
        "INFO update 25 micro 200 tokens 6426776 lr 2.5e-6 loss 11.8 aux 0.0\n",
        encoding="utf-8",
    )
    status = collect_training_status(
        config,
        log_path=log,
        process_alive=True,
        tmux_alive=True,
        gpu_rows=[
            {
                "index": 0,
                "memory_used_mib": 10_000,
                "utilization_percent": 90,
                "temperature_c": 70,
            }
        ],
        disk_warning_percent=0,
    )
    metadata = build_run_metadata(config, world_size=8, repo_root=tmp_path)
    report = render_training_report(status, metadata)
    assert status["state"] == "running"
    assert status["progress_fraction"] == 25 / 50_000
    assert metadata["budget"]["effective_sequences_per_update"] == 128
    assert metadata["budget"]["scheduled_tokens"] == 13_107_200_000
    assert metadata["data_splits"]["train"]["events"] == 100
    assert "Gate 1 运行健康：**PASS**" in report


def test_status_marks_non_finite_loss_critical(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    log = tmp_path / "train.log"
    log.write_text("INFO update 1 loss nan\n", encoding="utf-8")
    status = collect_training_status(
        config,
        log_path=log,
        process_alive=True,
        tmux_alive=True,
        gpu_rows=[],
        disk_warning_percent=0,
    )
    assert status["state"] == "critical"
    assert any(alert["code"] == "non_finite_loss" for alert in status["alerts"])
