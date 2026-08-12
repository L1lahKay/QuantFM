"""
上传 quant_fm 产物到 MinIO model-cache（写 endpoint :9100）。

Also used by ``run_medium.py --upload-minio``.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import logging
import os
import re
import resource
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn

from quant_fm.data_coverage import coverage_set_sha256
from quant_fm.manifest.build_manifest import Manifest
from quant_fm.manifest.validation import (
    sha256_file,
    validate_manifest_shards,
    validate_manifest_vocab_contract,
)
from quant_fm.scripts.minio_config import (
    load_write_config,
    output_bucket,
    output_prefix,
)
from quant_fm.tokenizer.artifact_contract import (
    stable_vocab_sha256,
    token_contract_path,
)

logger = logging.getLogger(__name__)
POINTER_VERSION = 3
LEGACY_POINTER_VERSION = 1
LEGACY_STORAGE_POINTER_VERSION = 2
IMMUTABILITY_FENCE_VERSION = "lease_inotify_v1"
_GENERATION_RE = re.compile(r"^[0-9a-f]{64}$")
_MC_MISSING_OBJECT_MARKERS = (
    "nosuchkey",
    "no such key",
    "specified key does not exist",
    "object does not exist",
)
_LEASE_FD_HEADROOM = 32
_LEASE_SIGNAL_LOCK = threading.Lock()
_IN_ATTRIB = 0x00000004
_IN_CLOSE_WRITE = 0x00000008
_IN_MODIFY = 0x00000002
_IN_MOVED_FROM = 0x00000040
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_DELETE = 0x00000200
_IN_DELETE_SELF = 0x00000400
_IN_MOVE_SELF = 0x00000800
_IN_UNMOUNT = 0x00002000
_IN_Q_OVERFLOW = 0x00004000
_IN_IGNORED = 0x00008000
_IN_ONLYDIR = 0x01000000
_IN_CLOEXEC = os.O_CLOEXEC
_IN_NONBLOCK = os.O_NONBLOCK
_IN_EVENT_STRUCT_SIZE = 16
_INOTIFY_CHANGE_MASK = (
    _IN_MODIFY
    | _IN_CLOSE_WRITE
    | _IN_ATTRIB
    | _IN_CREATE
    | _IN_DELETE
    | _IN_MOVED_FROM
    | _IN_MOVED_TO
    | _IN_DELETE_SELF
    | _IN_MOVE_SELF
    | _IN_UNMOUNT
    | _IN_Q_OVERFLOW
)
_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.inotify_init1.argtypes = [ctypes.c_int]
_LIBC.inotify_init1.restype = ctypes.c_int
_LIBC.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
_LIBC.inotify_add_watch.restype = ctypes.c_int
_SUPPORTED_IMMUTABLE_FENCE_FILESYSTEMS = {
    "btrfs",
    "ext2",
    "ext3",
    "ext4",
    "overlay",
    "tmpfs",
    "xfs",
}


class _LeaseBreakError(RuntimeError):
    """Raised synchronously when another opener asks to break a read lease."""


@dataclass(frozen=True, slots=True)
class _LeasedFile:
    """One path and inode protected by an open-file-description read lease."""

    path: Path
    fd: int
    device: int
    inode: int


def _set_read_lease(fd: int) -> None:
    """Install the Linux read lease used by the immutable upload fence."""
    fcntl.fcntl(fd, fcntl.F_SETOWN, os.getpid())
    fcntl.fcntl(fd, fcntl.F_SETLEASE, fcntl.F_RDLCK)


def _regular_single_link_files(roots: list[Path]) -> list[Path]:
    """Resolve every lease target while rejecting links and special files."""
    files: list[Path] = []
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            msg = f"artifact lease root is not a regular directory: {root}"
            raise RuntimeError(msg)
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                msg = f"artifact lease target is a symlink: {path}"
                raise RuntimeError(msg)
            file_stat = path.stat()
            if stat.S_ISDIR(file_stat.st_mode):
                continue
            if not stat.S_ISREG(file_stat.st_mode):
                msg = f"artifact lease target is not a regular file: {path}"
                raise RuntimeError(msg)
            if file_stat.st_nlink != 1:
                msg = (
                    "artifact payload has an external hardlink and cannot be "
                    f"leased exclusively: {path} (st_nlink={file_stat.st_nlink})"
                )
                raise RuntimeError(msg)
            files.append(path)
    if not files:
        msg = "artifact claim has no regular files to lease"
        raise RuntimeError(msg)
    return files


def _preflight_read_lease_capacity(
    roots: list[Path],
    *,
    extra_files: int = 0,
) -> int:
    """Fail before claiming paths when RLIMIT_NOFILE cannot hold every lease."""
    file_count = len(_regular_single_link_files(roots)) + extra_files
    try:
        open_fds = len(list(Path("/proc/self/fd").iterdir()))
    except OSError as exc:
        msg = "cannot count open descriptors for artifact leases"
        raise RuntimeError(msg) from exc
    soft_limit, _hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    required = open_fds + file_count + _LEASE_FD_HEADROOM
    if soft_limit != resource.RLIM_INFINITY and required > soft_limit:
        msg = (
            "RLIMIT_NOFILE is too small for an immutable artifact lease fence: "
            f"open={open_fds}, payload_files={file_count}, "
            f"headroom={_LEASE_FD_HEADROOM}, soft_limit={soft_limit}"
        )
        raise RuntimeError(msg)
    return file_count


def _filesystem_type(
    path: Path, *, mountinfo_path: Path = Path("/proc/self/mountinfo")
) -> str:
    """Resolve the longest matching mount point and return its kernel fs type."""
    resolved = Path(path).resolve()
    try:
        lines = mountinfo_path.read_text(
            encoding="utf-8",
            errors="surrogateescape",
        ).splitlines()
    except OSError as exc:
        msg = "cannot inspect filesystem type for immutable artifact fencing"
        raise RuntimeError(msg) from exc
    best: tuple[int, str] | None = None
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator + 1 >= len(fields) or len(fields) < 6:
            continue
        mount_point = Path(fields[4].replace("\\040", " ").replace("\\011", "\t"))
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        candidate = (len(mount_point.parts), fields[separator + 1])
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        msg = f"cannot resolve filesystem mount for artifact root: {resolved}"
        raise RuntimeError(msg)
    return best[1]


def _preflight_supported_fence_filesystem(roots: list[Path]) -> None:
    """Allow only local filesystems with the lease/inotify semantics we require."""
    for root in roots:
        fs_type = _filesystem_type(root)
        if fs_type not in _SUPPORTED_IMMUTABLE_FENCE_FILESYSTEMS:
            msg = (
                "immutable upload fencing requires a supported local filesystem "
                f"(ext4/xfs/btrfs/overlay/tmpfs), got {fs_type!r} for {root}"
            )
            raise RuntimeError(msg)
        root_device = root.stat().st_dev
        for path in root.rglob("*"):
            if path.stat(follow_symlinks=False).st_dev != root_device:
                msg = (
                    "immutable upload fencing refuses an artifact tree that crosses "
                    f"filesystem devices: root={root}, path={path}"
                )
                raise RuntimeError(msg)


def _preflight_supported_fence_directory(path: Path) -> None:
    """Validate a future private staging parent without scanning its siblings."""
    fs_type = _filesystem_type(path)
    if fs_type not in _SUPPORTED_IMMUTABLE_FENCE_FILESYSTEMS:
        msg = (
            "immutable upload metadata staging requires a supported local filesystem "
            f"(ext4/xfs/btrfs/overlay/tmpfs), got {fs_type!r} for {path}"
        )
        raise RuntimeError(msg)


class _ClaimReadLeaseFence:
    """Hold kernel leases and mutation watches on one upload snapshot."""

    def __init__(self, roots: list[Path]) -> None:
        self._roots = [Path(root) for root in roots]
        self._leased: list[_LeasedFile] = []
        self._broken = False
        self._closing = False
        self._active = False
        self._acquired_once = False
        self._lock_held = False
        self._old_handler: Any = None
        self._old_mask: set[signal.Signals] | None = None
        self._inotify_fd: int | None = None
        self._watch_count = 0
        self._root_fds: dict[Path, int] = {}

    @property
    def fd_numbers(self) -> set[int]:
        """Descriptors intentionally visible in this process's proc table."""
        descriptors = {item.fd for item in self._leased}
        descriptors.update(self._root_fds.values())
        if self._inotify_fd is not None:
            descriptors.add(self._inotify_fd)
        return descriptors

    def root_fd_paths(self) -> dict[Path, str]:
        """Return stable proc paths for the leased claim directory descriptors."""
        paths: dict[Path, str] = {}
        for root, fd in self._root_fds.items():
            paths[root] = f"/proc/self/fd/{fd}"
        return paths

    @property
    def pass_fds(self) -> tuple[int, ...]:
        """Descriptors child upload processes must inherit explicitly."""
        return tuple(sorted(self._root_fds.values()))

    @property
    def broken(self) -> bool:
        """Whether a lease break or forbidden directory mutation was observed."""
        if not self._broken and self._active:
            try:
                self._drain_directory_events()
                for leased in self._leased:
                    lease_type = fcntl.fcntl(leased.fd, fcntl.F_GETLEASE)
                    descriptor_stat = os.fstat(leased.fd)
                    path_stat = leased.path.stat(follow_symlinks=False)
                    if (
                        lease_type != fcntl.F_RDLCK
                        or descriptor_stat.st_nlink != 1
                        or (descriptor_stat.st_dev, descriptor_stat.st_ino)
                        != (leased.device, leased.inode)
                        or (path_stat.st_dev, path_stat.st_ino)
                        != (leased.device, leased.inode)
                    ):
                        self._broken = True
                        break
            except (OSError, RuntimeError):
                self._broken = True
        return self._broken

    @property
    def acquired_once(self) -> bool:
        """Whether the complete lease/watch fence ever became active."""
        return self._acquired_once

    @property
    def partially_acquired(self) -> bool:
        """Whether at least one lease exists but the full fence never activated."""
        return bool(self._leased) and not self._acquired_once

    def fd_path_for(self, path: Path) -> str:
        """Resolve a file below one fenced root through its stable root fd."""
        resolved = Path(path).resolve()
        for root, fd in self._root_fds.items():
            try:
                relative = resolved.relative_to(root.resolve())
            except ValueError:
                continue
            base = f"/proc/self/fd/{fd}"
            return base if relative == Path() else f"{base}/{relative.as_posix()}"
        msg = f"upload source is outside every fenced root: {path}"
        raise RuntimeError(msg)

    def _start_directory_watches(self) -> None:
        """Watch every claim directory for path or byte mutations."""
        inotify_fd = _LIBC.inotify_init1(_IN_CLOEXEC | _IN_NONBLOCK)
        if inotify_fd < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        self._inotify_fd = inotify_fd
        for root in self._roots:
            directories = [root]
            directories.extend(
                path
                for path in sorted(root.rglob("*"))
                if path.is_dir() and not path.is_symlink()
            )
            for directory in directories:
                encoded = os.fsencode(directory)
                watch = _LIBC.inotify_add_watch(
                    inotify_fd,
                    encoded,
                    _INOTIFY_CHANGE_MASK | _IN_ONLYDIR,
                )
                if watch < 0:
                    error = ctypes.get_errno()
                    raise OSError(error, os.strerror(error), directory)
                self._watch_count += 1
        if self._watch_count == 0:
            msg = "artifact claim has no directories to watch"
            raise RuntimeError(msg)

    def _drain_directory_events(self) -> None:
        """Fail closed when any watched tree changed or the queue overflowed."""
        if self._inotify_fd is None:
            msg = "artifact directory mutation watcher is not active"
            raise RuntimeError(msg)
        observed_masks: list[int] = []
        while True:
            try:
                payload = os.read(self._inotify_fd, 1024 * 1024)
            except BlockingIOError:
                break
            except InterruptedError:
                continue
            if not payload:
                msg = "artifact directory mutation watcher closed unexpectedly"
                raise RuntimeError(msg)
            offset = 0
            while offset < len(payload):
                if len(payload) - offset < _IN_EVENT_STRUCT_SIZE:
                    msg = "artifact directory watcher returned a truncated event"
                    raise RuntimeError(msg)
                mask = int.from_bytes(payload[offset + 4 : offset + 8], "little")
                name_length = int.from_bytes(
                    payload[offset + 12 : offset + 16],
                    "little",
                )
                offset += _IN_EVENT_STRUCT_SIZE + name_length
                if offset > len(payload):
                    msg = "artifact directory watcher returned an invalid event"
                    raise RuntimeError(msg)
                if mask & (_INOTIFY_CHANGE_MASK | _IN_IGNORED):
                    observed_masks.append(mask)
        if observed_masks:
            self._broken = True
            masks = ",".join(f"0x{mask:08x}" for mask in observed_masks[:8])
            msg = (
                "artifact claim changed while immutable: "
                f"inotify masks={masks}; refusing remote commit/delete"
            )
            raise RuntimeError(msg)

    def _handle_break(self, _signum: int, _frame: Any) -> None:
        self._broken = True

    def acquire(self) -> None:
        """Acquire every lease atomically from the caller's safety perspective."""
        if self._active:
            msg = "artifact read lease fence is already active"
            raise RuntimeError(msg)
        if threading.current_thread() is not threading.main_thread():
            msg = "artifact read leases require the Python main thread"
            raise RuntimeError(msg)
        required_constants = ("F_SETLEASE", "F_GETLEASE", "F_SETOWN")
        if any(not hasattr(fcntl, name) for name in required_constants):
            msg = "kernel/Python does not support artifact read leases"
            raise RuntimeError(msg)
        if not _LEASE_SIGNAL_LOCK.acquire(blocking=False):
            msg = "another artifact read lease fence is already active"
            raise RuntimeError(msg)
        self._lock_held = True

        try:
            self._old_mask = signal.pthread_sigmask(
                signal.SIG_BLOCK,
                {signal.SIGIO},
            )
            if signal.SIGIO in self._old_mask:
                msg = "SIGIO is blocked; artifact lease breaks cannot fail closed"
                _fail_runtime(msg)
            self._old_handler = signal.getsignal(signal.SIGIO)
            signal.signal(signal.SIGIO, self._handle_break)
            try:
                self._start_directory_watches()
            except OSError as exc:
                msg = "kernel inotify unavailable for immutable artifact claims"
                raise RuntimeError(msg) from exc
            root_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
            if hasattr(os, "O_NOFOLLOW"):
                root_flags |= os.O_NOFOLLOW
            for root in self._roots:
                root_fd = os.open(root, root_flags)
                root_stat = os.fstat(root_fd)
                path_stat = root.stat(follow_symlinks=False)
                if not stat.S_ISDIR(root_stat.st_mode) or (
                    root_stat.st_dev,
                    root_stat.st_ino,
                ) != (path_stat.st_dev, path_stat.st_ino):
                    os.close(root_fd)
                    msg = "artifact claim root changed while its descriptor was opened"
                    _fail_runtime(msg)
                self._root_fds[root] = root_fd
            flags = os.O_RDONLY | os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            for path in _regular_single_link_files(self._roots):
                fd = os.open(path, flags)
                try:
                    descriptor_stat = os.fstat(fd)
                    path_stat = path.stat(follow_symlinks=False)
                    if (
                        not stat.S_ISREG(descriptor_stat.st_mode)
                        or descriptor_stat.st_nlink != 1
                        or (descriptor_stat.st_dev, descriptor_stat.st_ino)
                        != (path_stat.st_dev, path_stat.st_ino)
                    ):
                        msg = "artifact lease target changed while descriptors were opened"
                        _fail_runtime(msg)
                    try:
                        _set_read_lease(fd)
                    except OSError as exc:
                        if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EBUSY}:
                            self._broken = True
                            msg = (
                                f"a writer conflicts with artifact lease target: {path}"
                            )
                            raise RuntimeError(msg) from exc
                        if exc.errno in {
                            errno.EINVAL,
                            errno.ENOSYS,
                            errno.ENOTSUP,
                            errno.EOPNOTSUPP,
                        }:
                            msg = (
                                "kernel read lease unavailable for artifact file: "
                                f"{path}"
                            )
                            raise RuntimeError(msg) from exc
                        raise
                    self._leased.append(
                        _LeasedFile(
                            path=path,
                            fd=fd,
                            device=descriptor_stat.st_dev,
                            inode=descriptor_stat.st_ino,
                        )
                    )
                except BaseException:
                    if not any(item.fd == fd for item in self._leased):
                        os.close(fd)
                    raise
            self._active = True
            self._acquired_once = True
            signal.pthread_sigmask(signal.SIG_SETMASK, self._old_mask)
            self.assert_intact()
        except BaseException:
            if self._inotify_fd is not None:
                try:
                    self._drain_directory_events()
                except RuntimeError:
                    self._broken = True
            if signal.SIGIO in signal.sigpending():
                self._broken = True
            if (self._broken or self._leased) and self._old_mask is not None:
                # Preserve descriptors/watches/leases for the outer failure path,
                # which quarantines producer-visible roots before calling close().
                signal.pthread_sigmask(signal.SIG_SETMASK, self._old_mask)
                raise
            self.close()
            raise

    def assert_intact(self) -> None:
        """Check both lease state and the path-to-inode bindings."""
        if self._broken:
            msg = "artifact read lease was broken; refusing remote commit/delete"
            raise _LeaseBreakError(msg)
        if not self._active:
            msg = "artifact read lease fence is not active"
            raise RuntimeError(msg)
        self._drain_directory_events()
        for leased in self._leased:
            try:
                lease_type = fcntl.fcntl(
                    leased.fd,
                    fcntl.F_GETLEASE,
                )
                descriptor_stat = os.fstat(leased.fd)
                path_stat = leased.path.stat(follow_symlinks=False)
            except OSError as exc:
                msg = "artifact lease target disappeared while the fence was active"
                raise RuntimeError(msg) from exc
            if lease_type != fcntl.F_RDLCK:
                self._broken = True
                msg = "artifact read lease changed state; refusing remote commit/delete"
                raise _LeaseBreakError(msg)
            if (
                descriptor_stat.st_nlink != 1
                or not stat.S_ISREG(path_stat.st_mode)
                or (descriptor_stat.st_dev, descriptor_stat.st_ino)
                != (leased.device, leased.inode)
                or (path_stat.st_dev, path_stat.st_ino) != (leased.device, leased.inode)
            ):
                msg = "artifact lease path/inode changed; refusing remote commit/delete"
                raise RuntimeError(msg)

    def close(self) -> None:
        """Release all leases and restore the process signal state exactly once."""
        if (
            not self._lock_held
            and not self._leased
            and self._old_mask is None
            and self._inotify_fd is None
        ):
            return
        self._closing = True
        cleanup_error: OSError | None = None
        try:
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGIO})
            for leased in reversed(self._leased):
                try:
                    fcntl.fcntl(leased.fd, fcntl.F_SETLEASE, fcntl.F_UNLCK)
                except OSError as exc:
                    cleanup_error = cleanup_error or exc
                try:
                    os.close(leased.fd)
                except OSError as exc:
                    cleanup_error = cleanup_error or exc
            self._leased.clear()
            for root_fd in self._root_fds.values():
                try:
                    os.close(root_fd)
                except OSError as exc:
                    cleanup_error = cleanup_error or exc
            self._root_fds.clear()
            self._active = False
            if signal.SIGIO in signal.sigpending():
                self._broken = True
                signal.sigwait({signal.SIGIO})
            if self._old_handler is not None:
                signal.signal(signal.SIGIO, self._old_handler)
            if self._old_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, self._old_mask)
        finally:
            if self._inotify_fd is not None:
                try:
                    os.close(self._inotify_fd)
                except OSError as exc:
                    cleanup_error = cleanup_error or exc
                self._inotify_fd = None
                self._watch_count = 0
            self._old_mask = None
            self._old_handler = None
            self._closing = False
            if self._lock_held:
                self._lock_held = False
                _LEASE_SIGNAL_LOCK.release()
        if cleanup_error is not None:
            msg = "failed to release an artifact read lease"
            raise RuntimeError(msg) from cleanup_error

    def quarantine_roots(self, roots: list[Path]) -> tuple[list[Path], list[str]]:
        """Move dirty producer-visible roots aside before releasing their leases."""
        recovery: list[Path] = []
        errors: list[str] = []
        for root in roots:
            root = Path(root)
            root_fd = self._root_fds.get(root)
            if root_fd is None:
                errors.append(f"missing fenced root descriptor: {root}")
                continue
            expected = os.fstat(root_fd)
            quarantine = root.with_name(
                f".{root.name}.quantfm-recovery-{uuid.uuid4().hex}"
            )
            try:
                root.rename(quarantine)
                actual = quarantine.stat(follow_symlinks=False)
            except OSError as exc:
                errors.append(f"cannot quarantine {root}: {exc.strerror or exc}")
                continue
            recovery.append(quarantine)
            if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                errors.append(
                    f"quarantined path did not match fenced root inode: {quarantine}"
                )
        return recovery, errors


@dataclass(frozen=True, slots=True)
class _ValidatedGeneration:
    """A fully revalidated, content-addressed local artifact generation."""

    workdir: Path
    tokens: Path
    events: Path
    vocab: Path
    manifest: Path
    audit: Path | None
    coverage: Path | None
    core_generation_id: str
    generation_id: str
    storage_generation_id: str
    shard_count: int
    pointer: dict[str, Any]


def _vocab_artifact(workdir: Path) -> Path:
    """Resolve exactly one local vocab generation and reject mixed roots."""
    data_dir = Path(workdir) / "data"
    candidates = [
        path
        for path in (data_dir / "vocab_v2.json", data_dir / "vocab.json")
        if path.is_file()
    ]
    if len(candidates) != 1:
        msg = (
            f"expected exactly one vocab artifact under {data_dir}, "
            f"found {[path.name for path in candidates]}"
        )
        raise FileNotFoundError(msg)
    return candidates[0]


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


def _load_vocab(path: Path):
    """Load the concrete vocabulary type without trusting its filename alone."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("vocab_version") == "2.0":
        from quant_fm.tokenizer.vocab_v2 import VocabV2

        return VocabV2.load(path)
    from quant_fm.tokenizer.vocab import Vocab

    return Vocab.load(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON commit record atomically."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _content_inventory(
    root: Path,
    *,
    require_single_link: bool = False,
) -> list[dict[str, str]]:
    """Hash every regular file and optionally reject external hardlinks."""
    if not root.is_dir():
        return []
    inventory: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            msg = f"artifact generation contains a symlink: {path}"
            raise ValueError(msg)
        file_stat = path.stat()
        if stat.S_ISDIR(file_stat.st_mode):
            continue
        if not stat.S_ISREG(file_stat.st_mode):
            msg = f"artifact generation contains a non-regular file: {path}"
            raise ValueError(msg)
        if require_single_link and file_stat.st_nlink != 1:
            msg = (
                "artifact payload has an external hardlink and cannot be "
                f"exclusively claimed: {path} (st_nlink={file_stat.st_nlink})"
            )
            raise ValueError(msg)
        inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    return inventory


def _semantic_generation_payload(
    manifest: Manifest,
    *,
    vocab_sha256: str,
    coverage_inventory: list[dict[str, str]],
    event_inventory: list[dict[str, str]],
) -> dict[str, Any]:
    """Build a path-independent identity for one immutable generation."""
    shards = sorted(
        (
            {
                "market": shard.market,
                "symbol": shard.symbol,
                "date": shard.date,
                "rows": shard.rows,
                "sha256": shard.sha256,
                "split": shard.split,
                "data_contract_sha256": shard.data_contract_sha256,
            }
            for shard in manifest.shards
        ),
        key=lambda item: (
            str(item["market"]),
            str(item["symbol"]),
            str(item["date"]),
            str(item["split"]),
        ),
    )
    return {
        "schema_version": manifest.schema_version,
        "train_end": manifest.train_end,
        "val_end": manifest.val_end,
        "purge_days": manifest.purge_days,
        "embargo_days": manifest.embargo_days,
        "vocab_sha256": vocab_sha256,
        "event_ordering_version": manifest.event_ordering_version,
        "feature_transform_version": manifest.feature_transform_version,
        "shards": shards,
        "coverage": coverage_inventory,
        "events": event_inventory,
    }


def _payload_sha256(payload: dict[str, Any]) -> str:
    """Return the canonical identity of one generation payload."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _storage_generation_id(pointer: dict[str, Any]) -> str:
    """Derive the path-bound remote namespace from all persisted identities."""
    return _payload_sha256(
        {
            "core_generation_id": pointer["core_generation_id"],
            "generation_id": pointer["generation_id"],
            "manifest_sha256": pointer["manifest_sha256"],
            "vocab_sha256": pointer["vocab_sha256"],
            "audit_sha256": pointer.get("audit_sha256"),
            "coverage_sha256": pointer.get("coverage_sha256"),
            "immutability_fence_version": pointer.get("immutability_fence_version"),
        }
    )


def _validate_local_generation(
    workdir: Path,
    *,
    include_events: bool = False,
    run_live_audit: bool = True,
) -> _ValidatedGeneration:
    """Revalidate every live byte before it is uploaded or accepted locally."""
    root = Path(workdir).resolve()
    tokens = root / "tokens"
    events = root / "events"
    vocab_path = _vocab_artifact(root)
    manifest_path = root / "data" / "manifest.json"
    audit_path = root / "artifact_audit.json"
    coverage_path = root / "data" / "coverage"
    required = [tokens, vocab_path, manifest_path]
    missing = [path for path in required if not path.exists()]
    if missing:
        msg = f"missing: {', '.join(str(path) for path in missing)}; run data pipeline first"
        raise FileNotFoundError(msg)
    if not tokens.is_dir():
        msg = f"tokens artifact is not a directory: {tokens}"
        raise ValueError(msg)

    vocab = _load_vocab(vocab_path)
    manifest = Manifest.load(manifest_path)
    is_v2 = vocab_path.name == "vocab_v2.json"
    if is_v2:
        if not audit_path.is_file():
            msg = f"V2 upload requires a passed full-path artifact audit: {audit_path}"
            raise FileNotFoundError(msg)
        try:
            stored_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            msg = f"invalid V2 artifact audit {audit_path}: {exc}"
            raise ValueError(msg) from exc
        if stored_audit.get("contract_ready") is not True:
            msg = f"V2 artifact audit is not PASS: {audit_path}"
            raise RuntimeError(msg)
        if stored_audit.get("audit_version") != "2.0":
            msg = (
                f"V2 upload requires an identity-bound audit_version=2.0: {audit_path}"
            )
            raise RuntimeError(msg)
        if stored_audit.get("checked_all_paths") is not True:
            msg = f"V2 upload requires checked_all_paths=true: {audit_path}"
            raise RuntimeError(msg)
        if not coverage_path.is_dir():
            msg = (
                f"V2 upload requires exact universe coverage receipts: {coverage_path}"
            )
            raise FileNotFoundError(msg)
        coverage_inventory = _content_inventory(coverage_path)
        if not coverage_inventory or any(
            "/" in item["path"] or not item["path"].endswith(".json")
            for item in coverage_inventory
        ):
            msg = (
                "V2 coverage must be a non-empty flat directory containing only "
                f"JSON receipts: {coverage_path}"
            )
            raise ValueError(msg)
        manifest_file_sha256 = sha256_file(manifest_path)
        vocab_file_sha256 = sha256_file(vocab_path)
        coverage_sha256 = coverage_set_sha256(root)
        expected_audit_inputs = {
            "manifest_sha256": manifest_file_sha256,
            "vocab_file_sha256": vocab_file_sha256,
            "coverage_sha256": coverage_sha256,
        }
        mismatched_audit_inputs = {
            field: (stored_audit.get(field), expected)
            for field, expected in expected_audit_inputs.items()
            if stored_audit.get(field) != expected
        }
        audit_input_sha256 = hashlib.sha256(
            json.dumps(
                expected_audit_inputs,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if stored_audit.get("audit_input_sha256") != audit_input_sha256:
            mismatched_audit_inputs["audit_input_sha256"] = (
                stored_audit.get("audit_input_sha256"),
                audit_input_sha256,
            )
        if mismatched_audit_inputs:
            msg = (
                "V2 audit belongs to a different manifest/vocab/coverage generation: "
                f"{mismatched_audit_inputs}"
            )
            raise ValueError(msg)
        if run_live_audit:
            from quant_fm.scripts.audit_v2_artifacts import audit_v2_artifacts

            live_audit = audit_v2_artifacts(root, full_path_check=True)
            if live_audit.get("contract_ready") is not True:
                msg = f"live full-path V2 artifact audit failed: {audit_path}"
                raise RuntimeError(msg)
            if live_audit.get("checked_all_paths") is not True:
                msg = "live V2 artifact audit did not check every path"
                raise RuntimeError(msg)
            if any(
                live_audit.get(field) != expected
                for field, expected in {
                    **expected_audit_inputs,
                    "audit_input_sha256": audit_input_sha256,
                }.items()
            ):
                msg = "live V2 audit identity disagrees with local artifacts"
                raise RuntimeError(msg)
    else:
        audit_path = None
        coverage_path = None
        coverage_inventory = []

    expected_vocab_path = Path(manifest.vocab_path or "").resolve()
    if expected_vocab_path != vocab_path.resolve():
        msg = (
            "manifest points at a different vocabulary generation: "
            f"manifest={manifest.vocab_path!r}, upload={vocab_path}"
        )
        raise ValueError(msg)
    validate_manifest_vocab_contract(manifest, vocab, context="artifact upload")
    if not manifest.shards:
        msg = "artifact upload refuses an empty manifest"
        raise ValueError(msg)
    validate_manifest_shards(
        manifest,
        vocab,
        context="artifact upload",
        expected_tokens_root=tokens,
    )

    tokens_root = tokens.resolve()
    expected_files: set[Path] = set()
    logical_keys: set[tuple[str, str, str]] = set()
    for shard in manifest.shards:
        key = (shard.market, shard.symbol, shard.date)
        if key in logical_keys:
            msg = f"manifest contains a duplicate logical shard: {key}"
            raise ValueError(msg)
        logical_keys.add(key)
        shard_path = Path(shard.path).resolve()
        try:
            shard_path.relative_to(tokens_root)
        except ValueError as exc:
            msg = f"manifest token shard escapes the upload tokens root: {shard_path}"
            raise ValueError(msg) from exc
        expected_files.add(shard_path)
        sidecar = token_contract_path(shard_path)
        if sidecar.is_file():
            expected_files.add(sidecar.resolve())

    actual_files: set[Path] = set()
    for path in tokens.rglob("*"):
        if path.is_symlink():
            msg = f"tokens generation contains a symlink: {path}"
            raise ValueError(msg)
        file_stat = path.stat()
        if stat.S_ISDIR(file_stat.st_mode):
            continue
        if not stat.S_ISREG(file_stat.st_mode):
            msg = f"tokens generation contains a non-regular file: {path}"
            raise ValueError(msg)
        if file_stat.st_nlink != 1:
            msg = (
                "tokens payload has an external hardlink and cannot be "
                f"exclusively claimed: {path} (st_nlink={file_stat.st_nlink})"
            )
            raise ValueError(msg)
        actual_files.add(path.resolve())
    if actual_files != expected_files:
        extra = sorted(str(path) for path in actual_files - expected_files)
        missing_files = sorted(str(path) for path in expected_files - actual_files)
        msg = (
            "tokens directory does not exactly match the manifest generation: "
            f"extra={extra[:8]}, missing={missing_files[:8]}"
        )
        raise ValueError(msg)

    event_inventory = (
        _content_inventory(events, require_single_link=True) if include_events else []
    )
    core_payload = _semantic_generation_payload(
        manifest,
        vocab_sha256=stable_vocab_sha256(vocab),
        coverage_inventory=coverage_inventory,
        event_inventory=[],
    )
    core_generation_id = _payload_sha256(core_payload)
    generation_id = _payload_sha256({**core_payload, "events": event_inventory})
    pointer: dict[str, Any] = {
        "pointer_version": POINTER_VERSION,
        "immutability_fence_version": IMMUTABILITY_FENCE_VERSION,
        "core_generation_id": core_generation_id,
        "generation_id": generation_id,
        "schema_version": manifest.schema_version,
        "shard_count": len(manifest.shards),
        "event_file_count": len(event_inventory),
        "coverage_file_count": len(coverage_inventory),
        "manifest_sha256": sha256_file(manifest_path),
        "vocab_sha256": sha256_file(vocab_path),
        "audit_sha256": sha256_file(audit_path) if audit_path is not None else None,
        "coverage_sha256": coverage_set_sha256(root) if is_v2 else None,
    }
    storage_generation_id = _storage_generation_id(pointer)
    pointer["storage_generation_id"] = storage_generation_id
    return _ValidatedGeneration(
        workdir=root,
        tokens=tokens,
        events=events,
        vocab=vocab_path,
        manifest=manifest_path,
        audit=audit_path,
        coverage=coverage_path,
        core_generation_id=core_generation_id,
        generation_id=generation_id,
        storage_generation_id=storage_generation_id,
        shard_count=len(manifest.shards),
        pointer=pointer,
    )


def _checked_pointer(payload: Any, *, context: str) -> dict[str, Any]:
    """Validate an immutable-generation pointer read from local or remote JSON."""
    if not isinstance(payload, dict):
        msg = f"{context} generation pointer must be a JSON object"
        raise TypeError(msg)
    pointer_version = payload.get("pointer_version")
    if pointer_version not in {
        LEGACY_POINTER_VERSION,
        LEGACY_STORAGE_POINTER_VERSION,
        POINTER_VERSION,
    }:
        msg = f"{context} generation pointer has unsupported version"
        raise ValueError(msg)
    for field in ("core_generation_id", "generation_id"):
        value = payload.get(field)
        if not isinstance(value, str) or not _GENERATION_RE.fullmatch(value):
            msg = f"{context} generation pointer has an invalid {field}"
            raise ValueError(msg)
    storage_generation_id = payload.get("storage_generation_id")
    if storage_generation_id is not None and (
        not isinstance(storage_generation_id, str)
        or not _GENERATION_RE.fullmatch(storage_generation_id)
    ):
        msg = f"{context} generation pointer has an invalid storage_generation_id"
        raise ValueError(msg)
    shard_count = payload.get("shard_count")
    if (
        isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count < 1
    ):
        msg = f"{context} generation pointer has an invalid shard_count"
        raise ValueError(msg)
    for field in ("manifest_sha256", "vocab_sha256"):
        value = payload.get(field)
        if not isinstance(value, str) or not _GENERATION_RE.fullmatch(value):
            msg = f"{context} generation pointer has an invalid {field}"
            raise ValueError(msg)
    audit_hash = payload.get("audit_sha256")
    if audit_hash is not None and (
        not isinstance(audit_hash, str) or not _GENERATION_RE.fullmatch(audit_hash)
    ):
        msg = f"{context} generation pointer has an invalid audit_sha256"
        raise ValueError(msg)
    coverage_hash = payload.get("coverage_sha256")
    if coverage_hash is not None and (
        not isinstance(coverage_hash, str)
        or not _GENERATION_RE.fullmatch(coverage_hash)
    ):
        msg = f"{context} generation pointer has an invalid coverage_sha256"
        raise ValueError(msg)
    for field in ("event_file_count", "coverage_file_count"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            msg = f"{context} generation pointer has an invalid {field}"
            raise ValueError(msg)
    if payload.get("schema_version") == "cn_l2_v2":
        if audit_hash is None or coverage_hash is None:
            msg = f"{context} V2 generation pointer lacks audit/coverage identity"
            raise ValueError(msg)
        if payload["coverage_file_count"] < 1:
            msg = f"{context} V2 generation pointer has no coverage receipts"
            raise ValueError(msg)
    checksum_algorithm = payload.get("payload_transfer_checksum_algorithm")
    checksum_receipt = payload.get("payload_listing_receipt_sha256")
    if (checksum_algorithm is None) != (checksum_receipt is None):
        msg = f"{context} generation pointer has an incomplete checksum receipt"
        raise ValueError(msg)
    if checksum_algorithm is not None:
        if checksum_algorithm != "SHA256":
            msg = f"{context} generation pointer has an unsupported checksum algorithm"
            raise ValueError(msg)
        if not isinstance(checksum_receipt, str) or not _GENERATION_RE.fullmatch(
            checksum_receipt
        ):
            msg = f"{context} generation pointer has an invalid checksum receipt"
            raise ValueError(msg)
    if pointer_version == POINTER_VERSION:
        if storage_generation_id is None:
            msg = f"{context} generation pointer lacks storage_generation_id"
            raise ValueError(msg)
        if payload.get("immutability_fence_version") != IMMUTABILITY_FENCE_VERSION:
            msg = f"{context} generation pointer lacks the current immutability fence"
            raise ValueError(msg)
    return dict(payload)


def _read_remote_pointer(
    remote: str,
    *,
    filename: str = "current.json",
) -> dict[str, Any] | None:
    """Read a pointer, treating only an explicit missing object as legacy."""
    result = subprocess.run(
        ["mc", "cat", f"{remote}/{filename}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        diagnostic = f"{result.stdout}\n{result.stderr}".casefold()
        if any(marker in diagnostic for marker in _MC_MISSING_OBJECT_MARKERS):
            return None
        # Do not include mc output or argv: either can contain environment-specific
        # endpoint details, and alias setup is the only command that handles secrets.
        msg = f"failed to read remote {filename}; refusing legacy fallback"
        raise RuntimeError(msg)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        msg = f"remote {filename} is not valid JSON"
        raise ValueError(msg) from exc
    pointer = _checked_pointer(payload, context=f"remote {filename}")
    if filename == "current.json" and pointer.get("pointer_version") == POINTER_VERSION:
        if pointer.get("payload_listing_receipt_sha256") is None:
            msg = "remote current.json lacks its committed listing receipt"
            raise ValueError(msg)
        if pointer["storage_generation_id"] != _storage_generation_id(pointer):
            msg = "remote current.json has an inconsistent storage_generation_id"
            raise ValueError(msg)
    return pointer


def _generation_remote(remote: str, pointer: dict[str, Any]) -> str:
    """Resolve a content-addressed generation beneath a stable tag root."""
    storage_id = pointer.get("storage_generation_id") or pointer["generation_id"]
    return f"{remote}/generations/{storage_id}"


def _assert_compatible_commit(
    local: dict[str, Any],
    committed: dict[str, Any],
) -> None:
    """Reject reuse unless the remote commit has the same semantic identity."""
    identity_fields = (
        "pointer_version",
        "generation_id",
        "storage_generation_id",
        "core_generation_id",
        "schema_version",
        "shard_count",
        "event_file_count",
        "coverage_file_count",
        "manifest_sha256",
        "vocab_sha256",
        "audit_sha256",
        "coverage_sha256",
        "immutability_fence_version",
    )
    mismatches = {
        field: (committed.get(field), local.get(field))
        for field in identity_fields
        if committed.get(field) != local.get(field)
    }
    if mismatches:
        msg = f"remote immutable generation commit is incompatible: {mismatches}"
        raise RuntimeError(msg)


def _assert_unchanged_source(
    initial: dict[str, Any],
    revalidated: dict[str, Any],
) -> None:
    """Require the local source pointer to remain byte-for-byte identical."""
    if initial != revalidated:
        mismatches = {
            field: (initial.get(field), revalidated.get(field))
            for field in sorted(set(initial) | set(revalidated))
            if initial.get(field) != revalidated.get(field)
        }
        msg = f"local artifact generation changed during upload: {mismatches}"
        raise RuntimeError(msg)


def _path_is_within(path: str, roots: tuple[Path, ...]) -> bool:
    """Return whether one absolute proc path names an object below a claim."""
    if path.endswith(" (deleted)"):
        path = path[: -len(" (deleted)")]
    if not path.startswith("/"):
        return False
    candidate = Path(path)
    return any(candidate == root or candidate.is_relative_to(root) for root in roots)


def _proc_race_disappeared(exc: OSError) -> bool:
    """Whether a proc entry vanished because its process/fd exited concurrently."""
    return exc.errno in {errno.ENOENT, errno.ESRCH}


def _fail_runtime(message: str) -> NoReturn:
    """Raise a runtime safety failure from outside guarded cleanup blocks."""
    raise RuntimeError(message)


def _assert_no_same_uid_process_references(
    roots: list[Path],
    *,
    proc_root: Path = Path("/proc"),
    ignored_self_fds: set[int] | None = None,
) -> None:
    """
    Fail closed if a same-UID process can still mutate an upload claim.

    Renaming a tree blocks path-based writers, but an already-open descriptor or
    writable mapping survives the rename.  Scan fd, cwd, root and maps for every
    visible same-UID process.  Permission failures are safety failures rather than
    a reason to assume the claim is private.
    """
    resolved_roots = tuple(Path(root).resolve() for root in roots if root.exists())
    if not resolved_roots:
        return
    uid = os.getuid()
    self_pid = str(os.getpid())
    ignored_self_fds = ignored_self_fds or set()
    references: list[str] = []
    try:
        processes = list(proc_root.iterdir())
    except OSError as exc:
        msg = f"cannot inspect {proc_root} for open artifact handles"
        raise RuntimeError(msg) from exc

    for process in processes:
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != uid:
                continue
        except OSError as exc:
            if _proc_race_disappeared(exc):
                continue
            msg = "cannot identify a process while fencing artifact claims"
            raise RuntimeError(msg) from exc

        for name in ("cwd", "root"):
            try:
                target = str((process / name).readlink())
            except OSError as exc:
                if _proc_race_disappeared(exc):
                    continue
                msg = f"cannot inspect same-UID process {process.name} {name}"
                raise RuntimeError(msg) from exc
            if _path_is_within(target, resolved_roots):
                references.append(f"pid={process.name}:{name}")

        fd_root = process / "fd"
        try:
            descriptors = list(fd_root.iterdir())
        except OSError as exc:
            if _proc_race_disappeared(exc):
                continue
            msg = f"cannot inspect same-UID process {process.name} descriptors"
            raise RuntimeError(msg) from exc
        for descriptor in descriptors:
            if process.name == self_pid:
                try:
                    descriptor_number = int(descriptor.name)
                except ValueError:
                    descriptor_number = -1
                if descriptor_number in ignored_self_fds:
                    continue
            try:
                target = str(descriptor.readlink())
            except OSError as exc:
                if _proc_race_disappeared(exc):
                    continue
                msg = f"cannot inspect same-UID process {process.name} descriptor"
                raise RuntimeError(msg) from exc
            if _path_is_within(target, resolved_roots):
                references.append(f"pid={process.name}:fd={descriptor.name}")

        try:
            mappings = (process / "maps").read_text(
                encoding="utf-8",
                errors="surrogateescape",
            )
        except OSError as exc:
            if _proc_race_disappeared(exc):
                continue
            msg = f"cannot inspect same-UID process {process.name} mappings"
            raise RuntimeError(msg) from exc
        for line in mappings.splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and _path_is_within(fields[5], resolved_roots):
                references.append(f"pid={process.name}:map")

    if references:
        msg = (
            "artifact claim is still referenced by a same-UID process; "
            f"refusing upload/delete: {', '.join(sorted(set(references))[:8])}"
        )
        raise RuntimeError(msg)


def _stage_fenced_generation(
    initial: _ValidatedGeneration,
    metadata_root: Path,
) -> _ValidatedGeneration:
    """Copy metadata privately while payload trees stay producer-visible."""
    root = Path(metadata_root).resolve()
    data = root / "data"
    data.mkdir(parents=True)
    vocab = data / initial.vocab.name
    manifest = data / "manifest.json"
    shutil.copyfile(initial.vocab, vocab)
    shutil.copyfile(initial.manifest, manifest)
    coverage = None
    if initial.coverage is not None:
        coverage = data / "coverage"
        shutil.copytree(initial.coverage, coverage, symlinks=True)
    audit = None
    if initial.audit is not None:
        audit = root / "artifact_audit.json"
        shutil.copyfile(initial.audit, audit)
    return replace(
        initial,
        workdir=root,
        vocab=vocab,
        manifest=manifest,
        audit=audit,
        coverage=coverage,
    )


def _actual_local_upload_payload(
    generation: _ValidatedGeneration,
    *,
    include_events: bool,
) -> dict[str, str]:
    """Hash the exact private paths that an ensuing mc invocation will read."""
    actual = {
        f"tokens/{item['path']}": item["sha256"]
        for item in _content_inventory(
            generation.tokens,
            require_single_link=True,
        )
    }
    actual[f"data/{generation.vocab.name}"] = sha256_file(generation.vocab)
    actual["data/manifest.json"] = sha256_file(generation.manifest)
    if generation.audit is not None:
        actual["artifact_audit.json"] = sha256_file(generation.audit)
    if generation.coverage is not None:
        actual.update(
            {
                f"data/coverage/{item['path']}": item["sha256"]
                for item in _content_inventory(generation.coverage)
            }
        )
    if include_events and generation.events.is_dir():
        actual.update(
            {
                f"events/{item['path']}": item["sha256"]
                for item in _content_inventory(
                    generation.events,
                    require_single_link=True,
                )
            }
        )
    return actual


def _assert_claimed_generation_stable(
    generation: _ValidatedGeneration,
    expected: dict[str, str | None],
    *,
    include_events: bool,
    scan_roots: list[Path],
    lease_fence: _ClaimReadLeaseFence | None = None,
) -> None:
    """Rehash every claimed/private upload byte, then reject lingering handles."""
    if lease_fence is not None:
        lease_fence.assert_intact()
    frozen = {
        relative: content_hash
        for relative, content_hash in expected.items()
        if content_hash is not None
    }
    actual = _actual_local_upload_payload(
        generation,
        include_events=include_events,
    )
    if actual != frozen:
        missing = sorted(set(frozen) - set(actual))
        extra = sorted(set(actual) - set(frozen))
        changed = sorted(
            relative
            for relative in set(actual) & set(frozen)
            if actual[relative] != frozen[relative]
        )
        detail = (
            "parquet SHA-256 mismatch"
            if any(
                item.startswith("tokens/") and item.endswith(".parquet")
                for item in changed
            )
            else "artifact payload SHA-256 mismatch"
        )
        msg = (
            f"{detail} after atomic upload claim: missing={missing[:8]}, "
            f"extra={extra[:8]}, changed={changed[:8]}"
        )
        raise ValueError(msg)
    if lease_fence is not None:
        lease_fence.assert_intact()
    _assert_no_same_uid_process_references(
        scan_roots,
        ignored_self_fds=lease_fence.fd_numbers if lease_fence is not None else None,
    )
    if lease_fence is not None:
        lease_fence.assert_intact()


def _expected_remote_payload(
    generation: _ValidatedGeneration,
    pointer: dict[str, Any],
    *,
    include_commit: bool,
    include_events: bool,
) -> dict[str, str | None]:
    """Return every expected remote object and its exact content hash."""
    expected: dict[str, str | None] = {
        f"data/{generation.vocab.name}": pointer["vocab_sha256"],
        "data/manifest.json": pointer["manifest_sha256"],
    }
    if generation.audit is not None:
        expected["artifact_audit.json"] = pointer["audit_sha256"]
    if generation.coverage is not None:
        for item in _content_inventory(generation.coverage):
            expected[f"data/coverage/{item['path']}"] = item["sha256"]

    manifest = Manifest.load(generation.manifest)
    token_root = generation.tokens.resolve()
    for shard in manifest.shards:
        shard_path = Path(shard.path).resolve()
        relative = shard_path.relative_to(token_root).as_posix()
        expected[f"tokens/{relative}"] = shard.sha256
        sidecar = token_contract_path(shard_path)
        if sidecar.is_file():
            expected[f"tokens/{sidecar.relative_to(token_root).as_posix()}"] = (
                shard.data_contract_sha256 or sha256_file(sidecar)
            )
    if include_events:
        for item in _content_inventory(generation.events):
            expected[f"events/{item['path']}"] = item["sha256"]
    if include_commit:
        # Its parsed semantic fields are checked separately.  JSON whitespace is
        # intentionally not part of generation compatibility across producers.
        expected["generation.json"] = None
    return expected


def _remote_listing_inventory(remote: str) -> dict[str, tuple[int, str]]:
    """Fetch exact key, size, and ETag metadata with one recursive list call."""
    result = subprocess.run(
        ["mc", "ls", "--recursive", "--json", remote.rstrip("/") + "/"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        msg = "failed to read remote object inventory; refusing to publish pointer"
        raise RuntimeError(msg)
    inventory: dict[str, tuple[int, str]] = {}
    for raw_line in result.stdout.splitlines():
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            msg = "mc returned an invalid remote checksum inventory"
            raise RuntimeError(msg) from exc
        relative = item.get("key")
        size = item.get("size")
        etag_value = item.get("etag")
        etag = etag_value.strip().strip('"') if isinstance(etag_value, str) else ""
        if (
            item.get("status") != "success"
            or item.get("type") != "file"
            or not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in inventory
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not etag
        ):
            msg = "mc returned an invalid or duplicate remote object entry"
            raise RuntimeError(msg)
        inventory[relative] = (size, etag)
    return inventory


def _listing_inventory_sha256(inventory: dict[str, tuple[int, str]]) -> str:
    """Bind every remote key, byte size, and ETag in one compact commit field."""
    encoded = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _expected_remote_sizes(
    generation: _ValidatedGeneration,
    expected: dict[str, str | None],
) -> dict[str, int]:
    """Return frozen local byte sizes for every upload payload object."""
    roots = {
        "tokens": generation.tokens,
        "events": generation.events,
        "data/coverage": generation.coverage,
    }
    direct = {
        f"data/{generation.vocab.name}": generation.vocab,
        "data/manifest.json": generation.manifest,
        "artifact_audit.json": generation.audit,
    }
    sizes: dict[str, int] = {}
    for relative, content_hash in expected.items():
        if content_hash is None:
            continue
        path = direct.get(relative)
        if path is None:
            for prefix, root in roots.items():
                marker = prefix + "/"
                if relative.startswith(marker) and root is not None:
                    path = root / relative.removeprefix(marker)
                    break
        if path is None or not path.is_file():
            msg = f"cannot resolve frozen upload size for {relative}"
            raise RuntimeError(msg)
        sizes[relative] = path.stat().st_size
    return sizes


def _verify_remote_generation_payload(
    remote: str,
    expected: dict[str, str | None],
    *,
    expected_sizes: dict[str, int] | None = None,
) -> str:
    """
    Verify exact membership and return its listing receipt hash.

    ``mc cp --checksum SHA256`` makes the server validate every newly uploaded
    byte.  A single recursive list then binds each key, size, and non-empty ETag
    to the immutable commit without one HEAD or full network read per shard.
    """
    inventory = _remote_listing_inventory(remote)
    if set(inventory) != set(expected):
        missing = sorted(set(expected) - set(inventory))
        extra = sorted(set(inventory) - set(expected))
        msg = (
            "remote generation payload is incomplete or mixed: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
        raise RuntimeError(msg)
    if expected_sizes is not None:
        payload_paths = {
            relative
            for relative, content_hash in expected.items()
            if content_hash is not None
        }
        if set(expected_sizes) != payload_paths:
            msg = "frozen local size inventory does not match the upload payload"
            raise RuntimeError(msg)
        wrong_sizes = {
            relative: (expected_sizes[relative], inventory[relative][0])
            for relative in expected
            if inventory[relative][0] != expected_sizes[relative]
        }
        if wrong_sizes:
            msg = (
                "remote generation object size differs from the frozen upload: "
                f"{dict(sorted(wrong_sizes.items())[:8])}"
            )
            raise RuntimeError(msg)
    payload_inventory = {
        relative: inventory[relative]
        for relative, content_hash in expected.items()
        if content_hash is not None
    }
    return _listing_inventory_sha256(payload_inventory)


def _verify_committed_listing_receipt(
    remote: str,
    expected: dict[str, str | None],
    committed: dict[str, Any],
) -> None:
    """Fail closed unless a committed generation's listing receipt still matches."""
    algorithm = committed.get("payload_transfer_checksum_algorithm")
    expected_receipt = committed.get("payload_listing_receipt_sha256")
    if algorithm != "SHA256" or not isinstance(expected_receipt, str):
        msg = (
            "committed generation lacks a reusable listing receipt; "
            "refusing an unverified immutable-generation reuse"
        )
        raise RuntimeError(msg)
    actual_receipt = _verify_remote_generation_payload(remote, expected)
    if actual_receipt != expected_receipt:
        msg = "remote generation key/size/ETag receipt changed after commit"
        raise RuntimeError(msg)


def _verify_remote_commit_receipt(
    remote: str,
    committed: dict[str, Any],
) -> None:
    """Verify a committed listing receipt when no local inventory is available."""
    algorithm = committed.get("payload_transfer_checksum_algorithm")
    expected_receipt = committed.get("payload_listing_receipt_sha256")
    if algorithm is None and expected_receipt is None:
        return
    if algorithm != "SHA256" or not isinstance(expected_receipt, str):
        msg = "remote generation has an invalid listing receipt"
        raise RuntimeError(msg)
    generation_commit = _read_remote_pointer(remote, filename="generation.json")
    if generation_commit is None:
        msg = "remote generation is missing its immutable commit object"
        raise RuntimeError(msg)
    if generation_commit != committed:
        msg = "remote generation.json does not exactly match current.json"
        raise RuntimeError(msg)
    inventory = _remote_listing_inventory(remote)
    if inventory.pop("generation.json", None) is None:
        msg = "remote generation is missing its immutable commit object"
        raise RuntimeError(msg)
    if _listing_inventory_sha256(inventory) != expected_receipt:
        msg = "remote generation key/size/ETag receipt changed after commit"
        raise RuntimeError(msg)


def _upload_commands(
    generation: _ValidatedGeneration,
    generation_remote: str,
    *,
    include_events: bool,
    lease_fence: _ClaimReadLeaseFence | None = None,
) -> list[list[str]]:
    """Build the immutable payload copy commands for one generation."""
    source = (
        lease_fence.fd_path_for if lease_fence is not None else lambda path: str(path)
    )
    commands: list[list[str]] = [
        [
            "mc",
            "cp",
            "--recursive",
            "--checksum",
            "SHA256",
            source(generation.tokens) + "/",
            f"{generation_remote}/tokens/",
        ],
        [
            "mc",
            "cp",
            "--checksum",
            "SHA256",
            source(generation.vocab),
            f"{generation_remote}/data/{generation.vocab.name}",
        ],
        [
            "mc",
            "cp",
            "--checksum",
            "SHA256",
            source(generation.manifest),
            f"{generation_remote}/data/manifest.json",
        ],
    ]
    if generation.coverage is not None:
        commands.append(
            [
                "mc",
                "cp",
                "--recursive",
                "--checksum",
                "SHA256",
                source(generation.coverage) + "/",
                f"{generation_remote}/data/coverage/",
            ]
        )
    if generation.audit is not None:
        commands.append(
            [
                "mc",
                "cp",
                "--checksum",
                "SHA256",
                source(generation.audit),
                f"{generation_remote}/artifact_audit.json",
            ]
        )
    if include_events and generation.events.is_dir():
        commands.insert(
            0,
            [
                "mc",
                "cp",
                "--recursive",
                "--checksum",
                "SHA256",
                source(generation.events) + "/",
                f"{generation_remote}/events/",
            ],
        )
    return commands


def _claim_and_delete_local_payload(
    source_root: Path,
    expected: dict[str, str | None],
    *,
    include_events: bool,
) -> None:
    """Refuse the legacy automatic deletion path, which cannot be made atomic."""
    del source_root, expected, include_events
    msg = (
        "automatic local payload deletion is disabled: perform offline cleanup "
        "only after stopping every producer and independently verifying the upload"
    )
    raise RuntimeError(msg)


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
    source_root = Path(workdir).resolve()
    if delete_local:
        msg = (
            "automatic delete_local is disabled: upload claims cannot prove an "
            "atomic safe recursive deletion boundary; upload with delete_local=False "
            "and remove the verified local generation during an offline maintenance step"
        )
        raise RuntimeError(msg)
    bucket = output_bucket()
    prefix = output_prefix(tag)
    dest = f"{bucket}/{prefix}"
    if dry_run:
        generation = _validate_local_generation(
            source_root,
            include_events=include_events,
        )
        remote = f"fm_upload/{dest}"
        generation_remote = _generation_remote(remote, generation.pointer)
        commands = _upload_commands(
            generation,
            generation_remote,
            include_events=include_events,
        )
        for cmd in commands:
            logger.info("upload (dry-run): %s", " ".join(cmd))
        logger.info(
            "publish (dry-run): generation=%s -> %s/current.json",
            generation.generation_id,
            remote,
        )
    else:
        initial = _validate_local_generation(
            source_root,
            include_events=include_events,
        )
        frozen_expected = _expected_remote_payload(
            initial,
            initial.pointer,
            include_commit=False,
            include_events=include_events,
        )
        payload_targets = [initial.tokens]
        if include_events and initial.events.is_dir():
            payload_targets.append(initial.events)
        # Private staged metadata joins payload roots in the same fence before
        # any upload command is constructed.
        metadata_file_count = 2
        if initial.audit is not None:
            metadata_file_count += 1
        if initial.coverage is not None:
            metadata_file_count += len(_regular_single_link_files([initial.coverage]))
        _preflight_supported_fence_filesystem(payload_targets)
        _preflight_supported_fence_directory(source_root.parent)
        _preflight_read_lease_capacity(payload_targets, extra_files=metadata_file_count)
        alias = _ensure_mc_alias()
        remote = f"{alias}/{dest}"
        lease_fence: _ClaimReadLeaseFence | None = None
        with tempfile.TemporaryDirectory(
            prefix=".quantfm-upload-metadata-",
            dir=source_root.parent,
        ) as metadata_dir:
            try:
                # Existing handles are rejected before leases; a writer racing
                # this scan is rejected by F_SETLEASE or makes the watcher dirty.
                _assert_no_same_uid_process_references(payload_targets)
                generation = _stage_fenced_generation(
                    initial,
                    Path(metadata_dir),
                )
                scan_roots = [*payload_targets, Path(metadata_dir)]
                lease_fence = _ClaimReadLeaseFence(scan_roots)
                lease_fence.acquire()
                _assert_claimed_generation_stable(
                    generation,
                    frozen_expected,
                    include_events=include_events,
                    scan_roots=scan_roots,
                    lease_fence=lease_fence,
                )
                generation_remote = _generation_remote(remote, generation.pointer)
                frozen_sizes = _expected_remote_sizes(generation, frozen_expected)
                commands = _upload_commands(
                    generation,
                    generation_remote,
                    include_events=include_events,
                    lease_fence=lease_fence,
                )
                committed = _read_remote_pointer(
                    generation_remote,
                    filename="generation.json",
                )
                if committed is not None:
                    _assert_compatible_commit(generation.pointer, committed)
                with tempfile.TemporaryDirectory(
                    prefix="quantfm-upload-pointer-"
                ) as pointer_dir:
                    pointer_root = Path(pointer_dir)
                    generation_commit = pointer_root / "generation.json"
                    current_pointer = pointer_root / "current.json"
                    if committed is None:
                        for command in commands:
                            logger.info("upload: %s", " ".join(command))
                            subprocess.run(
                                command,
                                check=True,
                                pass_fds=lease_fence.pass_fds,
                            )
                        checksum_receipt = _verify_remote_generation_payload(
                            generation_remote,
                            frozen_expected,
                            expected_sizes=frozen_sizes,
                        )
                        if not _GENERATION_RE.fullmatch(checksum_receipt):
                            msg = "remote listing receipt is not a valid SHA-256 digest"
                            _fail_runtime(msg)
                        # This scan follows mc exit and therefore also detects any
                        # descriptor a writer retained through the transfer.
                        _assert_claimed_generation_stable(
                            generation,
                            frozen_expected,
                            include_events=include_events,
                            scan_roots=scan_roots,
                            lease_fence=lease_fence,
                        )
                        committed_pointer = {
                            **generation.pointer,
                            "payload_transfer_checksum_algorithm": "SHA256",
                            "payload_listing_receipt_sha256": checksum_receipt,
                        }
                        _atomic_json(generation_commit, committed_pointer)
                        subprocess.run(
                            [
                                "mc",
                                "cp",
                                str(generation_commit),
                                f"{generation_remote}/generation.json",
                            ],
                            check=True,
                        )
                        committed = _read_remote_pointer(
                            generation_remote,
                            filename="generation.json",
                        )
                        if committed is None:
                            msg = "generation commit was not readable after upload"
                            _fail_runtime(msg)
                        _assert_compatible_commit(generation.pointer, committed)
                        if committed != committed_pointer:
                            msg = "generation commit readback differs from uploaded commit"
                            _fail_runtime(msg)
                        _verify_committed_listing_receipt(
                            generation_remote,
                            {**frozen_expected, "generation.json": None},
                            committed,
                        )
                        lease_fence.assert_intact()
                    else:
                        logger.info(
                            "immutable generation %s already committed; verifying it",
                            generation.generation_id,
                        )
                        _verify_committed_listing_receipt(
                            generation_remote,
                            {**frozen_expected, "generation.json": None},
                            committed,
                        )
                        lease_fence.assert_intact()
                    # The snapshot was fully hashed before this branch; new payloads
                    # were fully rehashed after transfer, while reuse was checked
                    # against its bound remote receipt.  Read leases reject new
                    # writers and recursive inotify watches reject path mutation
                    # without another production-scale token reread.
                    lease_fence.assert_intact()
                    _assert_no_same_uid_process_references(
                        scan_roots,
                        ignored_self_fds=lease_fence.fd_numbers,
                    )
                    lease_fence.assert_intact()
                    _atomic_json(current_pointer, committed)
                    # The mutable pointer is published only after the immutable
                    # commit and exact key/size/ETag receipt have been verified.
                    subprocess.run(
                        [
                            "mc",
                            "cp",
                            str(current_pointer),
                            f"{remote}/current.json",
                        ],
                        check=True,
                    )
                    published = _read_remote_pointer(remote)
                    if published != committed:
                        msg = (
                            "stable current pointer readback disagrees with "
                            "generation commit"
                        )
                        _fail_runtime(msg)
                    lease_fence.assert_intact()
                    _assert_no_same_uid_process_references(
                        scan_roots,
                        ignored_self_fds=lease_fence.fd_numbers,
                    )
                    lease_fence.assert_intact()
            except Exception as exc:
                if lease_fence is not None:
                    dirty = lease_fence.broken or lease_fence.partially_acquired
                    if lease_fence.acquired_once and not dirty:
                        try:
                            lease_fence.assert_intact()
                        except (OSError, RuntimeError, ValueError):
                            dirty = True
                    if dirty:
                        recovery, quarantine_errors = lease_fence.quarantine_roots(
                            payload_targets
                        )
                        lease_fence.close()
                        paths = ", ".join(str(path) for path in recovery) or "none"
                        detail = "; ".join(quarantine_errors) or "none"
                        msg = (
                            f"{exc}; local artifact source became untrusted and was "
                            f"quarantined before lease release: paths={paths}, "
                            f"quarantine_errors={detail}"
                        )
                        raise RuntimeError(msg) from exc
                    lease_fence.close()
                raise

            if lease_fence is None:
                msg = "artifact upload fence was not acquired"
                _fail_runtime(msg)
            lease_fence.close()

    uri = remote_uri(tag)
    logger.info("uploaded generation %s → %s", generation.generation_id, uri)

    return uri


def verify_upload(tag: str) -> int:
    """Verify the committed remote generation; return its parquet shard count."""
    alias = _ensure_mc_alias()
    bucket = output_bucket()
    prefix = output_prefix(tag)
    remote = f"{alias}/{bucket}/{prefix}"
    pointer = _read_remote_pointer(remote)
    generation_remote = _generation_remote(remote, pointer) if pointer else remote
    if pointer is not None:
        _verify_remote_commit_receipt(generation_remote, pointer)
    result = subprocess.run(
        ["mc", "find", f"{generation_remote}/tokens", "--name", "*.parquet"],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    if pointer is not None and len(lines) != pointer["shard_count"]:
        msg = (
            f"remote generation {pointer['generation_id']} is incomplete: "
            f"pointer={pointer['shard_count']} parquet shards, remote={len(lines)}"
        )
        raise RuntimeError(msg)
    if pointer is not None:
        vocab_name = (
            "vocab_v2.json"
            if pointer.get("schema_version") == "cn_l2_v2"
            else "vocab.json"
        )
        required = [
            f"{generation_remote}/generation.json",
            f"{generation_remote}/data/manifest.json",
            f"{generation_remote}/data/{vocab_name}",
        ]
        if pointer.get("schema_version") == "cn_l2_v2":
            required.append(f"{generation_remote}/artifact_audit.json")
        for target in required:
            status = subprocess.run(
                ["mc", "stat", target],
                capture_output=True,
                text=True,
                check=False,
            )
            if status.returncode != 0:
                msg = f"remote generation is missing committed artifact: {target}"
                raise RuntimeError(msg)
        if pointer.get("schema_version") == "cn_l2_v2":
            coverage = subprocess.run(
                [
                    "mc",
                    "find",
                    f"{generation_remote}/data/coverage",
                    "--name",
                    "*.json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            receipts = [item for item in coverage.stdout.splitlines() if item.strip()]
            if (
                coverage.returncode != 0
                or len(receipts) != pointer["coverage_file_count"]
            ):
                msg = (
                    f"remote generation {pointer['generation_id']} has incomplete "
                    "coverage receipts: "
                    f"pointer={pointer['coverage_file_count']}, remote={len(receipts)}"
                )
                raise RuntimeError(msg)
    uri = remote_uri(tag)
    logger.info("remote parquet files under %s: %d", uri, len(lines))
    for path in lines[:5]:
        logger.info("  sample: %s", path)
    return len(lines)


def main() -> None:
    """Upload an experiment data snapshot or verify a remote tag."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, default=Path("quant_fm/runs/v2_shared"))
    parser.add_argument("--tag", default="v2_shared")
    parser.add_argument("--include-events", action="store_true")
    parser.add_argument(
        "--delete-local",
        action="store_true",
        help=("已禁用：自动递归删除无法形成原子安全边界；请停写并独立验收后离线清理"),
    )
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
