"""
从 MinIO model-cache 下载 quant_fm 产物（tokens / vocab / manifest）到本地 workdir。

与 ``upload_to_minio`` 对称，供「先写回 MinIO，再在本机训练」或断点续训时恢复数据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from quant_fm.data_coverage import coverage_set_sha256
from quant_fm.manifest.validation import sha256_file
from quant_fm.scripts.minio_config import (
    load_write_config,
    output_bucket,
    output_prefix,
)
from quant_fm.scripts.upload_to_minio import (
    _atomic_json,
    _ensure_mc_alias,
    _generation_remote,
    _read_remote_pointer,
    _validate_local_generation,
    _verify_remote_commit_receipt,
    remote_uri,
)

logger = logging.getLogger(__name__)


def _validate_pointer_files(
    workdir: Path,
    pointer: dict[str, Any],
    *,
    vocab_name: str,
    data_version: str,
) -> None:
    """Verify immutable pointer hashes before rebasing the downloaded manifest."""
    root = Path(workdir)
    expected = {
        "manifest_sha256": root / "data" / "manifest.json",
        "vocab_sha256": root / "data" / vocab_name,
    }
    if data_version == "v2":
        if pointer.get("audit_sha256") is None:
            msg = "remote V2 generation pointer does not bind artifact_audit.json"
            raise ValueError(msg)
        expected["audit_sha256"] = root / "artifact_audit.json"
    for field, path in expected.items():
        if not path.is_file():
            msg = f"downloaded generation is missing {path}"
            raise FileNotFoundError(msg)
        actual = sha256_file(path)
        if actual != pointer.get(field):
            msg = (
                f"downloaded generation {field} mismatch for {path}: "
                f"pointer={pointer.get(field)}, actual={actual}"
            )
            raise ValueError(msg)
    if data_version == "v2":
        coverage_dir = root / "data" / "coverage"
        coverage_files = sorted(
            path for path in coverage_dir.rglob("*") if path.is_file()
        )
        if len(coverage_files) != pointer["coverage_file_count"]:
            msg = (
                "downloaded coverage receipt count mismatch: "
                f"pointer={pointer['coverage_file_count']}, "
                f"actual={len(coverage_files)}"
            )
            raise ValueError(msg)
        actual_coverage_hash = coverage_set_sha256(root)
        if actual_coverage_hash != pointer["coverage_sha256"]:
            msg = (
                "downloaded coverage generation mismatch: "
                f"pointer={pointer['coverage_sha256']}, "
                f"actual={actual_coverage_hash}"
            )
            raise ValueError(msg)


def _rebase_downloaded_manifest(
    workdir: Path,
    vocab_name: str,
    *,
    artifact_root: Path | None = None,
) -> None:
    """Point a downloaded manifest at this local artifact root."""
    from quant_fm.manifest.build_manifest import Manifest

    root = Path(workdir).resolve()
    final_root = Path(artifact_root).resolve() if artifact_root is not None else root
    path = root / "data" / "manifest.json"
    manifest = Manifest.load(path)
    for shard in manifest.shards:
        shard.path = str(
            final_root
            / "tokens"
            / shard.market
            / shard.symbol
            / f"{shard.date}.parquet"
        )
    manifest.vocab_path = str(final_root / "data" / vocab_name)
    temporary = path.with_name(f".{path.name}.tmp")
    manifest.save(temporary)
    temporary.replace(path)


def _rebase_downloaded_audit(
    staging_root: Path,
    destination_root: Path,
) -> None:
    """Rebind a completed full audit after its verified tree is atomically moved."""
    staging = Path(staging_root).resolve()
    destination = Path(destination_root).resolve()
    audit_path = staging / "artifact_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["root"] = str(destination)
    sampled = audit.get("sampled_shards")
    if isinstance(sampled, list):
        for item in sampled:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                continue
            path = Path(item["path"])
            try:
                relative = path.resolve().relative_to(staging)
            except ValueError:
                continue
            item["path"] = str(destination / relative)
    manifest_hash = sha256_file(staging / "data" / "manifest.json")
    audit["manifest_sha256"] = manifest_hash
    audit_inputs = {
        "manifest_sha256": manifest_hash,
        "vocab_file_sha256": audit["vocab_file_sha256"],
        "coverage_sha256": audit["coverage_sha256"],
    }
    audit["audit_input_sha256"] = hashlib.sha256(
        json.dumps(
            audit_inputs,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _atomic_json(audit_path, audit)


def _publish_download_staging(
    staging: Path,
    destination: Path,
    *,
    destination_existed: bool,
) -> None:
    """Atomically replace an absent/empty destination with a verified staging tree."""
    if destination_existed:
        if not destination.is_dir() or any(destination.iterdir()):
            msg = f"download destination changed while data was staged: {destination}"
            raise FileExistsError(msg)
        # POSIX rename atomically replaces an existing empty directory on the same
        # filesystem.  The sibling staging directory guarantees that condition.
        staging.replace(destination)
        return
    if destination.exists():
        msg = f"download destination appeared while data was staged: {destination}"
        raise FileExistsError(msg)
    staging.rename(destination)


def download_workdir(
    workdir: Path,
    *,
    tag: str,
    include_events: bool = False,
    dry_run: bool = False,
    data_version: str = "v2",
) -> str:
    """
    Download tokens and data metadata from model-cache into ``workdir``.

    Returns remote ``s3://...`` prefix.
    """
    if data_version not in {"v1", "v2"}:
        msg = f"data_version must be v1 or v2, got {data_version!r}"
        raise ValueError(msg)
    workdir = Path(workdir)
    destination_existed = workdir.is_dir()
    if not dry_run:
        if workdir.exists() and not workdir.is_dir():
            msg = f"download destination is not a directory: {workdir}"
            raise NotADirectoryError(msg)
        if workdir.is_dir() and any(workdir.iterdir()):
            msg = (
                "download destination must be absent or empty to prevent mixing "
                f"artifact generations: {workdir}"
            )
            raise FileExistsError(msg)

    alias = "fm_upload" if dry_run else _ensure_mc_alias()
    bucket = output_bucket()
    prefix = output_prefix(tag)
    remote = f"{alias}/{bucket}/{prefix}"
    pointer = None if dry_run else _read_remote_pointer(remote)
    if dry_run:
        source = f"{remote}/generations/<current-generation>"
    elif pointer is not None:
        source = _generation_remote(remote, pointer)
    else:
        source = remote
    if pointer is not None:
        _verify_remote_commit_receipt(source, pointer)
    vocab_name = "vocab_v2.json" if data_version == "v2" else "vocab.json"
    if not dry_run:
        workdir.parent.mkdir(parents=True, exist_ok=True)
    staging_context = (
        tempfile.TemporaryDirectory(
            prefix=f".{workdir.name}.quantfm-download-",
            dir=workdir.parent,
        )
        if not dry_run
        else None
    )
    try:
        target_root = Path(staging_context.name) if staging_context else workdir
        if not dry_run:
            (target_root / "data").mkdir(parents=True)
            (target_root / "tokens").mkdir(parents=True)
            if include_events:
                (target_root / "events").mkdir(parents=True)
        cmds: list[list[str]] = [
            [
                "mc",
                "cp",
                "--recursive",
                f"{source}/tokens/",
                str(target_root / "tokens") + "/",
            ],
            [
                "mc",
                "cp",
                f"{source}/data/{vocab_name}",
                str(target_root / "data" / vocab_name),
            ],
            [
                "mc",
                "cp",
                f"{source}/data/manifest.json",
                str(target_root / "data" / "manifest.json"),
            ],
        ]
        if data_version == "v2":
            cmds.extend(
                [
                    [
                        "mc",
                        "cp",
                        "--recursive",
                        f"{source}/data/coverage/",
                        str(target_root / "data" / "coverage") + "/",
                    ],
                    [
                        "mc",
                        "cp",
                        f"{source}/artifact_audit.json",
                        str(target_root / "artifact_audit.json"),
                    ],
                ]
            )
        if include_events:
            cmds.insert(
                0,
                [
                    "mc",
                    "cp",
                    "--recursive",
                    f"{source}/events/",
                    str(target_root / "events") + "/",
                ],
            )

        for cmd in cmds:
            logger.info("download: %s", " ".join(cmd))
            if not dry_run:
                subprocess.run(cmd, check=True)
        if dry_run:
            uri = remote_uri(tag)
            logger.info("downloaded ← %s → %s", uri, workdir)
            return uri
        if pointer is not None:
            _validate_pointer_files(
                target_root,
                pointer,
                vocab_name=vocab_name,
                data_version=data_version,
            )
        _rebase_downloaded_manifest(target_root, vocab_name)
        if data_version == "v2":
            from quant_fm.scripts.audit_v2_artifacts import audit_v2_artifacts

            audit = audit_v2_artifacts(target_root, full_path_check=True)
            if audit.get("contract_ready") is not True:
                msg = "downloaded V2 generation failed a fresh full-path audit"
                raise RuntimeError(msg)
            _atomic_json(target_root / "artifact_audit.json", audit)
        validated = _validate_local_generation(
            target_root,
            include_events=include_events,
            run_live_audit=False,
        )
        expected_generation_id = (
            pointer["generation_id"]
            if pointer is not None and include_events
            else pointer["core_generation_id"]
            if pointer is not None
            else None
        )
        if (
            expected_generation_id is not None
            and validated.generation_id != expected_generation_id
        ):
            msg = (
                "downloaded artifacts do not match the committed generation: "
                f"pointer={expected_generation_id}, actual={validated.generation_id}"
            )
            raise ValueError(msg)
        if pointer is not None:
            _atomic_json(target_root / "generation.json", pointer)
        _rebase_downloaded_manifest(
            target_root,
            vocab_name,
            artifact_root=workdir,
        )
        if data_version == "v2":
            _rebase_downloaded_audit(target_root, workdir)
        _publish_download_staging(
            target_root,
            workdir,
            destination_existed=destination_existed,
        )
    finally:
        if staging_context is not None:
            staging_context.cleanup()

    uri = remote_uri(tag)
    logger.info("downloaded ← %s → %s", uri, workdir)
    return uri


def remote_ready(tag: str, *, data_version: str = "v2") -> bool:
    """True if remote vocab + at least one parquet token exist."""
    if data_version not in {"v1", "v2"}:
        return False
    alias = _ensure_mc_alias()
    bucket = output_bucket()
    prefix = output_prefix(tag)
    remote = f"{alias}/{bucket}/{prefix}"
    pointer = _read_remote_pointer(remote)
    source = _generation_remote(remote, pointer) if pointer is not None else remote
    if pointer is not None:
        try:
            _verify_remote_commit_receipt(source, pointer)
        except (OSError, RuntimeError, ValueError):
            return False
    vocab_name = "vocab_v2.json" if data_version == "v2" else "vocab.json"
    required = [f"{source}/data/{vocab_name}", f"{source}/data/manifest.json"]
    if data_version == "v2":
        required.append(f"{source}/artifact_audit.json")
    for target in required:
        result = subprocess.run(
            ["mc", "stat", target],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return False
    found = subprocess.run(
        ["mc", "find", f"{source}/tokens", "--name", "*.parquet"],
        capture_output=True,
        text=True,
        check=False,
    )
    if found.returncode != 0:
        return False
    shards = [line for line in found.stdout.splitlines() if line.strip()]
    coverage_receipts: list[str] | None = None
    if data_version == "v2":
        coverage = subprocess.run(
            ["mc", "find", f"{source}/data/coverage", "--name", "*.json"],
            capture_output=True,
            text=True,
            check=False,
        )
        if coverage.returncode != 0:
            return False
        coverage_receipts = [
            line for line in coverage.stdout.splitlines() if line.strip()
        ]
    if pointer is not None:
        if len(shards) != pointer["shard_count"]:
            return False
        if data_version == "v2":
            assert coverage_receipts is not None
            return len(coverage_receipts) == pointer["coverage_file_count"]
        return True
    return bool(shards) and (data_version != "v2" or bool(coverage_receipts))


def main() -> None:
    """Download or inspect a remote experiment data snapshot."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("quant_fm/runs/v2_shared"))
    parser.add_argument("--tag", default="v2_shared")
    parser.add_argument("--data-version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--include-events", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only report whether remote tag is ready; exit 0/1",
    )
    args = parser.parse_args()

    if not args.dry_run:
        # Resolve credentials only for commands that will contact MinIO.
        cfg = load_write_config()
        logger.info("write endpoint=%s bucket=%s", cfg.endpoint, output_bucket())

    if args.check_only:
        ok = remote_ready(args.tag, data_version=args.data_version)
        print(f"remote {remote_uri(args.tag)} ready={ok}")
        raise SystemExit(0 if ok else 1)

    download_workdir(
        args.workdir,
        tag=args.tag,
        include_events=args.include_events,
        dry_run=args.dry_run,
        data_version=args.data_version,
    )


if __name__ == "__main__":
    main()
