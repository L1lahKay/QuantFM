"""
从 MinIO model-cache 下载 quant_fm 产物（tokens / vocab / manifest）到本地 workdir。

与 ``upload_to_minio`` 对称，供「先写回 MinIO，再在本机训练」或断点续训时恢复数据。
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

from quant_fm.scripts.minio_config import (
    load_write_config,
    output_bucket,
    output_prefix,
)
from quant_fm.scripts.upload_to_minio import _ensure_mc_alias, remote_uri

logger = logging.getLogger(__name__)


def download_workdir(
    workdir: Path,
    *,
    tag: str,
    include_events: bool = False,
    dry_run: bool = False,
) -> str:
    """
    Download tokens and data metadata from model-cache into ``workdir``.

    Returns remote ``s3://...`` prefix.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "data").mkdir(parents=True, exist_ok=True)
    (workdir / "tokens").mkdir(parents=True, exist_ok=True)

    alias = _ensure_mc_alias()
    bucket = output_bucket()
    prefix = output_prefix(tag)
    remote = f"{alias}/{bucket}/{prefix}"

    cmds: list[list[str]] = [
        ["mc", "cp", "--recursive", f"{remote}/tokens/", str(workdir / "tokens") + "/"],
        ["mc", "cp", f"{remote}/data/vocab.json", str(workdir / "data" / "vocab.json")],
        [
            "mc",
            "cp",
            f"{remote}/data/manifest.json",
            str(workdir / "data" / "manifest.json"),
        ],
    ]
    if include_events:
        (workdir / "events").mkdir(parents=True, exist_ok=True)
        cmds.insert(
            0,
            [
                "mc",
                "cp",
                "--recursive",
                f"{remote}/events/",
                str(workdir / "events") + "/",
            ],
        )

    for cmd in cmds:
        logger.info("download: %s", " ".join(cmd))
        if not dry_run:
            subprocess.run(cmd, check=True)

    uri = remote_uri(tag)
    logger.info("downloaded ← %s → %s", uri, workdir)
    return uri


def remote_ready(tag: str) -> bool:
    """True if remote vocab + at least one parquet token exist."""
    alias = _ensure_mc_alias()
    bucket = output_bucket()
    prefix = output_prefix(tag)
    remote = f"{alias}/{bucket}/{prefix}"
    vocab = subprocess.run(
        ["mc", "stat", f"{remote}/data/vocab.json"],
        capture_output=True,
        text=True,
    )
    if vocab.returncode != 0:
        return False
    found = subprocess.run(
        ["mc", "find", f"{remote}/tokens", "--name", "*.parquet"],
        capture_output=True,
        text=True,
    )
    return found.returncode == 0 and bool(found.stdout.strip())


def main() -> None:
    """Download or inspect a remote experiment data snapshot."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("quant_fm/runs/medium"))
    parser.add_argument("--tag", default="medium")
    parser.add_argument("--include-events", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only report whether remote tag is ready; exit 0/1",
    )
    args = parser.parse_args()

    # Touch config once so credentials resolve (logs endpoint)
    cfg = load_write_config()
    logger.info("write endpoint=%s bucket=%s", cfg.endpoint, output_bucket())

    if args.check_only:
        ok = remote_ready(args.tag)
        print(f"remote {remote_uri(args.tag)} ready={ok}")
        raise SystemExit(0 if ok else 1)

    download_workdir(
        args.workdir,
        tag=args.tag,
        include_events=args.include_events,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
