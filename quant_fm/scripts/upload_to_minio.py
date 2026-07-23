"""
上传 quant_fm 产物到 MinIO model-cache（写 endpoint :9100）。

Also used by ``run_medium.py --upload-minio``.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

from quant_fm.scripts.minio_config import (
    load_write_config,
    output_bucket,
    output_prefix,
)

logger = logging.getLogger(__name__)


def _ensure_mc_alias(name: str = "fm_upload") -> str:
    cfg = load_write_config()
    scheme = "https" if cfg.secure else "http"
    url = f"{scheme}://{cfg.endpoint}"
    try:
        subprocess.run(
            ["mc", "alias", "set", name, url, cfg.access_key, cfg.secret_key],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        # CalledProcessError renders the full argv, including credentials.
        msg = f"failed to configure MinIO alias {name!r} for {url}"
        raise RuntimeError(msg) from None
    return name


def remote_uri(tag: str) -> str:
    """``s3://model-cache/{prefix}/{tag}/``"""
    bucket = output_bucket()
    prefix = output_prefix(tag)
    return f"s3://{bucket}/{prefix}/"


def upload_workdir(
    workdir: Path,
    *,
    tag: str,
    include_events: bool = False,
    delete_local: bool = False,
    dry_run: bool = False,
) -> str:
    """
    Upload artifacts under ``workdir`` to MinIO.

    Returns remote ``s3://...`` prefix.
    """
    workdir = Path(workdir)
    bucket = output_bucket()
    prefix = output_prefix(tag)
    dest = f"{bucket}/{prefix}"

    tokens = workdir / "tokens"
    events = workdir / "events"
    vocab = workdir / "data" / "vocab.json"
    manifest = workdir / "data" / "manifest.json"

    required = [tokens, vocab, manifest]
    missing = [p for p in required if not p.exists()]
    if missing:
        msg = f"missing: {', '.join(str(p) for p in missing)}; run data pipeline first"
        raise FileNotFoundError(msg)

    # dry-run 只打印计划命令，不要求本机已配置 mc alias
    alias = "fm_upload"
    if not dry_run:
        alias = _ensure_mc_alias()
    remote = f"{alias}/{dest}"
    cmds: list[list[str]] = [
        ["mc", "cp", "--recursive", str(tokens) + "/", f"{remote}/tokens/"],
        ["mc", "cp", str(vocab), f"{remote}/data/vocab.json"],
        ["mc", "cp", str(manifest), f"{remote}/data/manifest.json"],
    ]
    if include_events and events.is_dir():
        cmds.insert(
            0, ["mc", "cp", "--recursive", str(events) + "/", f"{remote}/events/"]
        )

    for cmd in cmds:
        logger.info("upload%s: %s", " (dry-run)" if dry_run else "", " ".join(cmd))
        if not dry_run:
            subprocess.run(cmd, check=True)

    uri = remote_uri(tag)
    logger.info("uploaded → %s", uri)

    if delete_local and not dry_run:
        for path in (tokens, events if include_events else None):
            if path and path.exists():
                shutil.rmtree(path)
                logger.info("deleted local %s", path)

    return uri


def verify_upload(tag: str) -> int:
    """List remote object count via mc; return file count (approx)."""
    alias = _ensure_mc_alias()
    bucket = output_bucket()
    prefix = output_prefix(tag)
    remote = f"{alias}/{bucket}/{prefix}"
    result = subprocess.run(
        ["mc", "find", remote, "--name", "*.parquet"],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    uri = remote_uri(tag)
    logger.info("remote parquet files under %s: %d", uri, len(lines))
    for path in lines[:5]:
        logger.info("  sample: %s", path)
    return len(lines)


def main() -> None:
    """Upload an experiment data snapshot or verify a remote tag."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("quant_fm/runs/pilot"))
    parser.add_argument("--tag", default="pilot")
    parser.add_argument("--include-events", action="store_true")
    parser.add_argument("--delete-local", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只检查远端 tag，不执行上传",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.verify_only:
        upload_workdir(
            args.workdir,
            tag=args.tag,
            include_events=args.include_events,
            delete_local=args.delete_local,
            dry_run=args.dry_run,
        )
    if (args.verify or args.verify_only) and not args.dry_run:
        verify_upload(args.tag)


if __name__ == "__main__":
    main()
