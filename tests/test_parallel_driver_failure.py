from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "quant_fm/scripts/run_medium_parallel_days.sh"


def _run_driver(
    tmp_path: Path, *, group_exit: int = 0, manifest_exit: int = 0
) -> tuple[subprocess.CompletedProcess[str], Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
if [[ "$*" == *"quant_fm.scripts.run_medium"* ]]; then
  exit "${FAKE_GROUP_EXIT:-0}"
fi
: > "$FAKE_MANIFEST_CALLED"
exit "${FAKE_MANIFEST_EXIT:-0}"
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    dates = tmp_path / "dates.txt"
    dates.write_text("2026-01-05\n2026-01-06\n", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    manifest_called = tmp_path / "manifest-called"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "HOME": str(home),
            "WORKDIR": str(tmp_path / "work"),
            "DATES_FILE": str(dates),
            "VOCAB": str(tmp_path / "vocab.json"),
            "SZ_FILE": str(tmp_path / "sz.txt"),
            "SH_FILE": str(tmp_path / "sh.txt"),
            "NGROUPS": "2",
            "CLEAN_WORKERS": "1",
            "TOKENIZE_WORKERS": "1",
            "FAKE_GROUP_EXIT": str(group_exit),
            "FAKE_MANIFEST_EXIT": str(manifest_exit),
            "FAKE_MANIFEST_CALLED": str(manifest_called),
        }
    )
    result = subprocess.run(
        ["bash", str(DRIVER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return result, manifest_called


def test_parallel_driver_does_not_build_manifest_after_group_failure(
    tmp_path: Path,
) -> None:
    result, manifest_called = _run_driver(tmp_path, group_exit=17)

    assert result.returncode == 1
    assert not manifest_called.exists()
    assert "不构建 manifest" in result.stdout


def test_parallel_driver_propagates_manifest_builder_failure(tmp_path: Path) -> None:
    result, manifest_called = _run_driver(tmp_path, manifest_exit=19)

    assert result.returncode == 1
    assert manifest_called.exists()
    assert "manifest 构建或完整性校验失败" in result.stdout
