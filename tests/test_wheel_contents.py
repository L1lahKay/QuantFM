import os
import subprocess
import tomllib
import zipfile
from pathlib import Path


def test_wheel_excludes_runtime_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    configuration = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    backend = configuration["tool"]["uv"]["build-backend"]
    assert "quant_fm/runs" in backend["source-exclude"]
    assert "quant_fm/runs" in backend["wheel-exclude"]

    destination = tmp_path / "dist"
    environment = {
        **os.environ,
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
    }
    subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(destination),
            "--no-create-gitignore",
        ],
        cwd=root,
        env=environment,
        check=True,
        timeout=120,
    )
    wheels = list(destination.glob("*.whl"))
    assert len(wheels) == 1

    forbidden_suffixes = (
        ".arrow",
        ".ckpt",
        ".parquet",
        ".pt",
        ".pth",
        ".safetensors",
    )
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
    assert "quant_fm/__init__.py" in names
    assert not any(name.startswith("quant_fm/runs/") for name in names)
    assert not any(name.endswith(forbidden_suffixes) for name in names)
    assert not any(Path(name).name.startswith("events.out.tfevents") for name in names)
