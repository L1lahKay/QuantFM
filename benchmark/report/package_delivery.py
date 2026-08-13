#!/usr/bin/env python3
"""Create a credential-free, portable scheduler-evaluation delivery archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parent
DEFAULT_HTML = (
    REPO_ROOT
    / "docs"
    / "exports"
    / "GPU-Training-Scheduler-Evaluation-with-Images-2026-08-03.html"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "exports"
    / "GPU-Training-Scheduler-Evaluation-2026-08-03.tar.gz"
)
MANIFEST = REPORT_DIR / "delivery_manifest.json"


INCLUDE = (
    REPO_ROOT / "AGENTS.md",
    REPO_ROOT / "benchmark" / "README.md",
    REPO_ROOT / "benchmark" / "config",
    REPO_ROOT / "benchmark" / "deploy",
    REPO_ROOT / "benchmark" / "lgb",
    REPO_ROOT / "benchmark" / "manifests",
    REPO_ROOT / "benchmark" / "metrics",
    REPO_ROOT / "benchmark" / "nn",
    REPO_ROOT / "benchmark" / "report",
    REPO_ROOT / "benchmark" / "results" / "orchestration",
    REPO_ROOT / "benchmark" / "results" / "raw",
    REPO_ROOT / "benchmark" / "scripts",
    REPO_ROOT / "benchmark" / "tests",
    REPO_ROOT / "benchmark" / "transformer",
    REPO_ROOT
    / "docs"
    / "assets"
    / "gpu-scheduler-evaluation"
    / "current-bare-results.json",
    REPO_ROOT
    / "docs"
    / "assets"
    / "gpu-scheduler-evaluation"
    / "current-kueue-results.json",
    REPO_ROOT / "docs" / "assets" / "gpu-scheduler-evaluation" / "volcano-results.json",
    REPO_ROOT / "docs" / "assets" / "gpu-scheduler-evaluation" / "screenshots",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allowed(path: Path, output: Path) -> bool:
    relative = path.relative_to(REPO_ROOT)
    if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
        return False
    if path == output or path.name.endswith(".tar.gz"):
        return False
    # Credentials are never valid delivery inputs, even if one is accidentally
    # placed under the repository later.
    lowered = path.name.lower()
    if lowered in {"kubeconfig", "config-gpu", "config-volcano-admin", "k3s.yaml"}:
        raise ValueError(f"refusing credential-like file: {path}")
    return path.is_file()


def collect(html_path: Path, output: Path) -> list[Path]:
    candidates: set[Path] = {html_path.resolve()}
    for root in INCLUDE:
        if root.is_file():
            candidates.add(root.resolve())
        elif root.is_dir():
            candidates.update(item.resolve() for item in root.rglob("*"))
        else:
            raise FileNotFoundError(f"required delivery input is missing: {root}")
    return sorted(path for path in candidates if allowed(path, output.resolve()))


def write_manifest(files: list[Path]) -> None:
    entries = [
        {
            "path": str(path.relative_to(REPO_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
        if path != MANIFEST
    ]
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "credential_files_included": False,
        "file_count_excluding_manifest": len(entries),
        "files": entries,
    }
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def create_archive(files: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz", compresslevel=9) as archive:
        for path in files:
            info = archive.gettarinfo(
                str(path), arcname=str(path.relative_to(REPO_ROOT))
            )
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with path.open("rb") as handle:
                archive.addfile(info, handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    html_path = args.html.resolve()
    output = args.output.resolve()
    try:
        html_path.relative_to(REPO_ROOT)
        output.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(
            "HTML and archive paths must remain inside the repository"
        ) from exc
    if not html_path.is_file():
        raise FileNotFoundError(
            f"offline HTML is missing; run benchmark/report/export_self_contained.py first: {html_path}"
        )

    initial = collect(html_path, output)
    write_manifest(initial)
    final = collect(html_path, output)
    create_archive(final, output)
    archive_sha256 = sha256(output)
    checksum_path = Path(str(output) + ".sha256")
    checksum_path.write_text(f"{archive_sha256}  {output.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "archive": str(output),
                "files": len(final),
                "sha256": archive_sha256,
                "checksum": str(checksum_path),
            }
        )
    )


if __name__ == "__main__":
    main()
