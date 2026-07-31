from __future__ import annotations

import os
import subprocess
from pathlib import Path

from quant_fm.scripts.k8s_dense230m_embeddings import _worker_command

ROOT = Path(__file__).resolve().parents[1]
PARALLEL_DRIVER = ROOT / "quant_fm/scripts/extract_embeddings_parallel.sh"


def test_k8s_worker_uses_checkpoint_representation_by_default(tmp_path: Path) -> None:
    command = _worker_command(
        checkpoint=tmp_path / "model.pt",
        manifest=tmp_path / "manifest.json",
        split="test",
        out=tmp_path / "test.parquet",
        batch_size=4,
        num_gpus=2,
        gpu=1,
        dtype="bf16",
        context=None,
        pooling=None,
        stride=None,
    )

    assert "--context" not in command
    assert "--pooling" not in command
    assert "--stride" not in command


def test_k8s_worker_preserves_explicit_representation_overrides(
    tmp_path: Path,
) -> None:
    command = _worker_command(
        checkpoint=tmp_path / "model.pt",
        manifest=tmp_path / "manifest.json",
        split="test",
        out=tmp_path / "test.parquet",
        batch_size=4,
        num_gpus=2,
        gpu=1,
        dtype="fp32",
        context=1024,
        pooling="lastk_mean",
        stride=256,
    )

    assert command[command.index("--context") + 1] == "1024"
    assert command[command.index("--pooling") + 1] == "lastk_mean"
    assert command[command.index("--stride") + 1] == "256"


def _run_parallel_driver(
    tmp_path: Path,
    *,
    overrides: dict[str, str] | None = None,
) -> list[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "uv-calls.txt"
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "$FAKE_UV_CALLS"\n',
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    env = os.environ.copy()
    for name in ("CONTEXT", "POOLING", "STRIDE"):
        env.pop(name, None)
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_UV_CALLS": str(calls),
            "WORKDIR": str(tmp_path / "work"),
            "CKPT": str(checkpoint),
            "MANIFEST": str(tmp_path / "manifest.json"),
            "EMB_DIR": str(tmp_path / "embeddings"),
            "NPROC": "1",
            "BATCH": "1",
            "MIN_FREE_MEM_GB": "0",
            "RESUME": "0",
            "SCORE_LOCK": str(tmp_path / "score.lock"),
        }
    )
    env.update(overrides or {})
    result = subprocess.run(
        ["bash", str(PARALLEL_DRIVER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return calls.read_text(encoding="utf-8").splitlines()


def test_parallel_driver_does_not_inject_representation_defaults(
    tmp_path: Path,
) -> None:
    calls = _run_parallel_driver(tmp_path)
    extraction = next(
        call for call in calls if "quant_fm.embedding.extract_hidden" in call
    )

    assert "--context" not in extraction
    assert "--pooling" not in extraction
    assert "--stride" not in extraction


def test_parallel_driver_forwards_explicit_representation_overrides(
    tmp_path: Path,
) -> None:
    calls = _run_parallel_driver(
        tmp_path,
        overrides={"CONTEXT": "1024", "POOLING": "last", "STRIDE": "256"},
    )
    extraction = next(
        call for call in calls if "quant_fm.embedding.extract_hidden" in call
    )

    assert "--context 1024" in extraction
    assert "--pooling last" in extraction
    assert "--stride 256" in extraction
