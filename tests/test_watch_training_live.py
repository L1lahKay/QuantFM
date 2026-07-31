import os
from pathlib import Path

from quant_fm.scripts.watch_training_live import (
    active_error_codes,
    checkpoint_speed,
    parse_log_series,
    progress_bar,
    runtime_speed,
    sparkline,
)


def test_parse_log_series_tracks_updates_and_validation() -> None:
    updates, validations = parse_log_series(
        "\n".join(
            [
                "INFO update 25 micro 200 tokens 1000 lr 2.5e-6 loss 8.0 aux 0.04",
                "INFO update 1000 val_loss 6.2",
                "INFO update 50 micro 400 tokens 2000 lr 5e-6 loss 7.0 aux 0.04",
                "INFO update 2000 val_loss 5.8",
            ]
        )
    )
    assert updates[-1]["update"] == 50
    assert updates[-1]["loss"] == 7.0
    assert validations == [
        {"update": 1000, "val_loss": 6.2},
        {"update": 2000, "val_loss": 5.8},
    ]


def test_active_errors_ignore_failure_before_a_restarted_run() -> None:
    text = "\n".join(
        [
            "Traceback (most recent call last):",
            "INFO update 25 micro 200 tokens 1000 lr 2.5e-6 loss 8.0 aux 0.04",
        ]
    )
    assert active_error_codes(text) == []
    assert active_error_codes(text + "\nCUDA out of memory") == ["cuda_oom"]


def test_terminal_helpers() -> None:
    assert "50.00%" in progress_bar(50, 100, width=10)
    assert len(sparkline([6.2, 6.0, 5.8])) == 3
    assert runtime_speed([(0.0, 100), (120.0, 150)]) == 25.0
    assert runtime_speed([(0.0, 100)]) is None


def test_checkpoint_speed_uses_recent_checkpoint_mtimes(tmp_path: Path) -> None:
    for step, timestamp in ((2000, 1000), (4000, 6000), (6000, 11000)):
        path = tmp_path / f"step{step}.pt"
        path.touch()
        path.chmod(0o644)
        path_stat = (timestamp, timestamp)
        path.touch()
        os.utime(path, path_stat)
    assert checkpoint_speed(tmp_path) == 24.0
