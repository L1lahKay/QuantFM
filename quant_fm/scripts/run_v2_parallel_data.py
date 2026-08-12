"""
Build one formal V2 dataset with parallel dates and one global vocabulary.

The preparation phase runs independent date groups through fast cleaning and
canonical event export only.  After every expected date is present, one final
process fits the vocabulary on the complete training stream, tokenizes all
splits, builds the manifest, and runs the V2 artifact audit.  A group can never
publish a private vocabulary, which prevents silently mixing incompatible V2
generations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from quant_fm.data_coverage import (
    coverage_set_sha256,
    expected_symbol_keys,
    load_coverage_receipt,
    verify_dataset_coverage,
)
from quant_fm.manifest.build_manifest import Manifest
from quant_fm.manifest.validation import sha256_file, validate_manifest_shard_paths
from quant_fm.scripts.run_medium import (
    DEFAULT_DATES,
    DEFAULT_SH,
    DEFAULT_SZ,
    _split_dates,
)
from quant_fm.tokenizer.artifact_contract import token_contract_path

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def split_dates_round_robin(dates: list[str], groups: int) -> list[list[str]]:
    """Split ordered dates across non-empty, load-balanced groups."""
    if groups < 1 or groups > 16:
        msg = f"groups must be in [1, 16], got {groups}"
        raise ValueError(msg)
    if groups > len(dates):
        msg = f"groups={groups} exceeds date count={len(dates)}"
        raise ValueError(msg)
    chunks = [[] for _ in range(groups)]
    for index, date in enumerate(dates):
        chunks[index % groups].append(date)
    return chunks


def validate_dates(dates: list[str]) -> None:
    """Reject duplicate or non-chronological formal split input."""
    if len(dates) < 3:
        msg = f"formal V2 data requires at least 3 dates, got {len(dates)}"
        raise ValueError(msg)
    if dates != sorted(set(dates)):
        msg = "formal V2 dates must be unique and strictly chronological"
        raise ValueError(msg)


def validate_worker_budget(
    groups: int,
    clean_workers: int,
    *,
    cpu_count: int | None = None,
) -> None:
    """Fail closed instead of silently oversubscribing replay workers."""
    available = cpu_count or os.cpu_count() or 1
    requested = groups * clean_workers
    if clean_workers < 1:
        msg = "clean_workers must be positive"
        raise ValueError(msg)
    if requested > available:
        msg = (
            f"parallel replay requests {groups}*{clean_workers}={requested} "
            f"workers but only {available} CPUs are available"
        )
        raise ValueError(msg)


def write_date_chunks(workdir: Path, chunks: list[list[str]]) -> list[Path]:
    """Atomically persist reproducible date-group inputs."""
    split_dir = Path(workdir) / "data" / "parallel_v2"
    split_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, chunk in enumerate(chunks):
        path = split_dir / f"dates.g{index}.txt"
        temporary = path.with_suffix(".txt.tmp")
        temporary.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        temporary.replace(path)
        paths.append(path)
    return paths


def build_prepare_command(
    *,
    dates_file: Path,
    workdir: Path,
    symbols_sz_file: Path,
    symbols_sh_file: Path,
    train_end: str,
    val_end: str,
) -> list[str]:
    """Build one phase-one command that cannot fit or publish a vocab."""
    return [
        sys.executable,
        "-m",
        "quant_fm.scripts.run_medium",
        "--data-version",
        "v2",
        "--dates-file",
        str(dates_file),
        "--symbols-sz-file",
        str(symbols_sz_file),
        "--symbols-sh-file",
        str(symbols_sh_file),
        "--workdir",
        str(workdir),
        "--train-end",
        train_end,
        "--val-end",
        val_end,
        "--fast-clean",
        "--events-only",
        "--drop-clean",
        "--resume",
    ]


def build_finalize_command(
    *,
    dates_file: Path,
    workdir: Path,
    symbols_sz_file: Path,
    symbols_sh_file: Path,
    train_end: str,
    val_end: str,
    n_bins: int,
    max_samples_per_field: int,
    seed: int,
    drop_events: bool,
) -> list[str]:
    """Build the only command allowed to fit the global V2 vocabulary."""
    command = [
        sys.executable,
        "-m",
        "quant_fm.scripts.run_medium",
        "--data-version",
        "v2",
        "--dates-file",
        str(dates_file),
        "--symbols-sz-file",
        str(symbols_sz_file),
        "--symbols-sh-file",
        str(symbols_sh_file),
        "--workdir",
        str(workdir),
        "--train-end",
        train_end,
        "--val-end",
        val_end,
        "--n-bins",
        str(n_bins),
        "--v2-max-samples-per-field",
        str(max_samples_per_field),
        "--v2-seed",
        str(seed),
        "--skip-clean",
        "--drop-clean",
        "--resume",
        "--v2-full-audit",
    ]
    if drop_events:
        command.append("--drop-events")
    return command


def canonical_event_dates(events_dir: Path) -> set[str]:
    """Collect dates that have at least one canonical event shard."""
    if not events_dir.is_dir():
        return set()
    return {path.stem for path in events_dir.rglob("*.parquet")}


def _load_symbol_file(path: Path) -> tuple[str, ...]:
    return tuple(
        value.strip()
        for value in Path(path).read_text(encoding="utf-8").splitlines()
        if value.strip()
    )


def _canonical_event_keys(events_dir: Path, date: str) -> set[str]:
    keys: set[str] = set()
    for path in Path(events_dir).rglob(f"{date}.parquet"):
        market = path.parent.parent.name
        symbol = path.parent.name
        keys.add(f"{market}:{symbol}")
    return keys


def verify_canonical_events(
    events_dir: Path,
    expected_dates: list[str],
    *,
    failure_dir: Path | None = None,
    expected_keys: tuple[str, ...] | None = None,
) -> None:
    """Require every requested date and no recorded replay failures."""
    actual = canonical_event_dates(events_dir)
    missing = sorted(set(expected_dates) - actual)
    if missing:
        preview = ", ".join(missing[:8])
        msg = f"canonical event preparation is incomplete; missing dates: {preview}"
        raise RuntimeError(msg)
    if failure_dir is None:
        failure_dir = Path(events_dir).parent / "data" / ".failed"
    failed_dates = sorted(
        path.stem
        for path in Path(failure_dir).glob("*.json")
        if path.stem in set(expected_dates)
    )
    if failed_dates:
        preview = ", ".join(failed_dates[:8])
        msg = f"canonical event preparation has recorded symbol gaps: {preview}"
        raise RuntimeError(msg)
    if expected_keys is not None:
        workdir = Path(events_dir).parent
        try:
            verify_dataset_coverage(
                workdir,
                expected_dates=expected_dates,
                expected_keys=expected_keys,
            )
            for date in expected_dates:
                receipt = load_coverage_receipt(
                    workdir / "data" / "coverage" / f"{date}.json"
                )
                actual_keys = _canonical_event_keys(events_dir, date)
                if actual_keys != set(receipt["materialized"]):
                    msg = (
                        "canonical event symbol coverage disagrees with the clean "
                        f"receipt for {date}"
                    )
                    raise RuntimeError(msg)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            msg = f"canonical event coverage is invalid: {exc}"
            raise RuntimeError(msg) from exc


def artifacts_ready(
    workdir: Path,
    *,
    expected_dates: list[str] | None = None,
    train_end: str | None = None,
    val_end: str | None = None,
) -> bool:
    """Return whether complete artifacts match the requested formal generation."""
    root = Path(workdir)
    required = [
        root / "tokens",
        root / "data" / "vocab_v2.json",
        root / "data" / "manifest.json",
        root / "artifact_audit.json",
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        audit = json.loads(required[-1].read_text(encoding="utf-8"))
        manifest = Manifest.load(root / "data" / "manifest.json")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    manifest_path = root / "data" / "manifest.json"
    vocab_path = root / "data" / "vocab_v2.json"
    if not (
        audit.get("contract_ready") is True
        and audit.get("checked_all_paths") is True
        and audit.get("audit_version") == "2.0"
        and audit.get("coverage_sha256") == coverage_set_sha256(root)
        and audit.get("manifest_sha256") == sha256_file(manifest_path)
        and audit.get("vocab_file_sha256") == sha256_file(vocab_path)
        and manifest.shards
    ):
        return False
    try:
        validate_manifest_shard_paths(
            manifest,
            context="formal V2 readiness",
            expected_tokens_root=root / "tokens",
        )
    except ValueError:
        return False
    if expected_dates is not None:
        actual_dates = sorted({shard.date for shard in manifest.shards})
        if actual_dates != list(expected_dates):
            return False
    if train_end is not None and manifest.train_end != train_end:
        return False
    if val_end is not None and manifest.val_end != val_end:
        return False
    for shard in manifest.shards:
        path = Path(shard.path)
        if not path.is_file() or not shard.sha256:
            return False
        if sha256_file(path) != shard.sha256:
            return False
        sidecar = token_contract_path(path)
        if not sidecar.is_file() or not shard.data_contract_sha256:
            return False
        if sha256_file(sidecar) != shard.data_contract_sha256:
            return False
    if expected_dates is not None:
        try:
            verify_dataset_coverage(
                root,
                expected_dates=expected_dates,
                manifest=manifest,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
    return True


def _run_logged(command: list[str], *, log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        msg = f"pipeline subprocess failed rc={result.returncode}; see {log_path}"
        raise RuntimeError(msg)


def run(
    *,
    dates_file: Path,
    workdir: Path,
    symbols_sz_file: Path,
    symbols_sh_file: Path,
    groups: int,
    clean_workers: int,
    canon_workers: int,
    tokenize_workers: int,
    train_end: str | None,
    val_end: str | None,
    n_bins: int,
    max_samples_per_field: int,
    seed: int,
    drop_events: bool,
) -> None:
    """Execute parallel canonical preparation and single-generation finalize."""
    dates = [
        line.strip()
        for line in Path(dates_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_dates(dates)
    symbols_sz = _load_symbol_file(symbols_sz_file)
    symbols_sh = _load_symbol_file(symbols_sh_file)
    requested_keys = expected_symbol_keys(symbols_sz, symbols_sh)
    validate_worker_budget(groups, clean_workers)
    if min(canon_workers, tokenize_workers, n_bins, max_samples_per_field) < 1:
        msg = "worker, bin, and sample counts must be positive"
        raise ValueError(msg)
    effective_train_end, effective_val_end = _split_dates(dates, train_end, val_end)
    workdir = Path(workdir)
    if artifacts_ready(
        workdir,
        expected_dates=dates,
        train_end=effective_train_end,
        val_end=effective_val_end,
    ):
        logger.info(
            "formal V2 artifacts already audit-ready; nothing to do: %s", workdir
        )
        return

    chunks = split_dates_round_robin(dates, groups)
    chunk_paths = write_date_chunks(workdir, chunks)
    base_env = os.environ.copy()
    base_env["CLEAN_WORKERS"] = str(clean_workers)
    base_env["CANON_WORKERS"] = str(canon_workers)

    started = time.perf_counter()
    logger.info(
        "V2 prepare start dates=%d groups=%d clean_workers=%d train_end=%s val_end=%s",
        len(dates),
        groups,
        clean_workers,
        effective_train_end,
        effective_val_end,
    )
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=groups) as pool:
        futures = {}
        for index, chunk_path in enumerate(chunk_paths):
            command = build_prepare_command(
                dates_file=chunk_path,
                workdir=workdir,
                symbols_sz_file=symbols_sz_file,
                symbols_sh_file=symbols_sh_file,
                train_end=effective_train_end,
                val_end=effective_val_end,
            )
            future = pool.submit(
                _run_logged,
                command,
                log_path=workdir / "parallel_v2" / f"prepare.g{index}.log",
                env=base_env,
            )
            futures[future] = index
        for future in as_completed(futures):
            index = futures[future]
            try:
                future.result()
            except RuntimeError as exc:
                errors.append(f"group {index}: {exc}")
    logger.info(
        "stage timing stage=parallel_prepare date=- elapsed_s=%.3f status=%s",
        time.perf_counter() - started,
        "error" if errors else "ok",
    )
    if errors:
        msg = f"parallel V2 preparation failed: {errors}"
        raise RuntimeError(msg)

    verify_canonical_events(
        workdir / "events",
        dates,
        expected_keys=requested_keys,
    )

    final_env = os.environ.copy()
    final_env["TOKENIZE_WORKERS"] = str(tokenize_workers)
    final_env["CANON_WORKERS"] = str(canon_workers)
    final_command = build_finalize_command(
        dates_file=dates_file,
        workdir=workdir,
        symbols_sz_file=symbols_sz_file,
        symbols_sh_file=symbols_sh_file,
        train_end=effective_train_end,
        val_end=effective_val_end,
        n_bins=n_bins,
        max_samples_per_field=max_samples_per_field,
        seed=seed,
        drop_events=drop_events,
    )
    started = time.perf_counter()
    _run_logged(
        final_command,
        log_path=workdir / "parallel_v2" / "finalize.log",
        env=final_env,
    )
    logger.info(
        "stage timing stage=global_finalize date=- elapsed_s=%.3f status=ok",
        time.perf_counter() - started,
    )
    if not artifacts_ready(
        workdir,
        expected_dates=dates,
        train_end=effective_train_end,
        val_end=effective_val_end,
    ):
        msg = "global V2 finalize returned without audit-ready artifacts"
        raise RuntimeError(msg)


def main() -> None:
    """Parse CLI arguments for the formal parallel V2 data builder."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates-file", type=Path, default=DEFAULT_DATES)
    parser.add_argument("--symbols-sz-file", type=Path, default=DEFAULT_SZ)
    parser.add_argument("--symbols-sh-file", type=Path, default=DEFAULT_SH)
    parser.add_argument("--workdir", type=Path, default=Path("quant_fm/runs/v2_shared"))
    parser.add_argument("--groups", type=int, default=int(os.getenv("NGROUPS", "2")))
    parser.add_argument(
        "--clean-workers", type=int, default=int(os.getenv("CLEAN_WORKERS", "30"))
    )
    parser.add_argument(
        "--canon-workers", type=int, default=int(os.getenv("CANON_WORKERS", "8"))
    )
    parser.add_argument(
        "--tokenize-workers",
        type=int,
        default=int(os.getenv("TOKENIZE_WORKERS", "16")),
    )
    parser.add_argument("--train-end")
    parser.add_argument("--val-end")
    parser.add_argument("--n-bins", type=int, default=32)
    parser.add_argument("--v2-max-samples-per-field", type=int, default=5_000_000)
    parser.add_argument("--v2-seed", type=int, default=0)
    parser.add_argument(
        "--keep-events",
        action="store_true",
        help="tokenize 后保留 canonical events；默认删除以释放磁盘",
    )
    args = parser.parse_args()
    run(
        dates_file=args.dates_file,
        workdir=args.workdir,
        symbols_sz_file=args.symbols_sz_file,
        symbols_sh_file=args.symbols_sh_file,
        groups=args.groups,
        clean_workers=args.clean_workers,
        canon_workers=args.canon_workers,
        tokenize_workers=args.tokenize_workers,
        train_end=args.train_end,
        val_end=args.val_end,
        n_bins=args.n_bins,
        max_samples_per_field=args.v2_max_samples_per_field,
        seed=args.v2_seed,
        drop_events=not args.keep_events,
    )


if __name__ == "__main__":
    main()
