import errno
import json
import logging
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_fm.data_coverage import write_coverage_receipt
from quant_fm.manifest.build_manifest import Manifest, ShardEntry
from quant_fm.manifest.validation import sha256_file
from quant_fm.scripts.audit_v2_artifacts import audit_v2_artifacts
from quant_fm.scripts.download_from_minio import (
    _validate_pointer_files,
    download_workdir,
)
from quant_fm.scripts.upload_to_minio import upload_workdir
from quant_fm.tokenizer.artifact_contract import (
    stable_vocab_sha256,
    token_contract_path,
)
from tests.test_v2_artifact_audit import (
    _full_vocab,
    _semantic_frame,
    _write_contract,
)


def _uploadable_v2(tmp_path: Path) -> Path:
    root = tmp_path / "v2_shared"
    data_dir = root / "data"
    token_path = root / "tokens" / "SH" / "600000" / "2025-01-02.parquet"
    token_path.parent.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    vocab = _full_vocab()
    vocab_path = data_dir / "vocab_v2.json"
    vocab.save(vocab_path)
    frame = _semantic_frame(vocab)
    frame.write_parquet(token_path)
    _write_contract(token_path, vocab, metadata=None)
    Manifest(
        shards=[
            ShardEntry(
                market="SH",
                symbol="600000",
                date="2025-01-02",
                path=str(token_path.resolve()),
                rows=frame.height,
                sha256=sha256_file(token_path),
                split="train",
                data_contract_sha256=sha256_file(token_contract_path(token_path)),
            )
        ],
        vocab_path=str(vocab_path.resolve()),
        vocab_sha256=stable_vocab_sha256(vocab),
        schema_version=vocab.schema_version,
        event_ordering_version=vocab.event_ordering_version,
        feature_transform_version=vocab.feature_transform_version,
    ).save(data_dir / "manifest.json")

    clean_event = root / "clean" / "2025-01-02" / "SH" / "600000" / "events.parquet"
    clean_event.parent.mkdir(parents=True)
    clean_event.touch()
    write_coverage_receipt(
        workdir=root,
        clean_dir=root / "clean" / "2025-01-02",
        date="2025-01-02",
        symbols_sz=(),
        symbols_sh=("600000",),
    )
    (root / "validation_windows.json").write_text("{}\n", encoding="utf-8")
    audit = audit_v2_artifacts(root, full_path_check=True)
    assert audit["contract_ready"] is True
    (root / "artifact_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def test_upload_requires_full_path_audit(tmp_path: Path) -> None:
    root = _uploadable_v2(tmp_path)
    audit_path = root / "artifact_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["checked_all_paths"] = False
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(RuntimeError, match="checked_all_paths=true"):
        upload_workdir(root, tag="gate-test", dry_run=True)


def test_upload_revalidates_token_and_coverage_bytes(tmp_path: Path) -> None:
    token_root = _uploadable_v2(tmp_path / "token")
    token_path = next((token_root / "tokens").rglob("*.parquet"))
    with token_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match=r"Parquet magic|parquet SHA-256 mismatch"):
        upload_workdir(token_root, tag="token-tamper", dry_run=True)

    coverage_root = _uploadable_v2(tmp_path / "coverage")
    coverage_path = next((coverage_root / "data" / "coverage").glob("*.json"))
    coverage_path.write_text(
        coverage_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different manifest/vocab/coverage"):
        upload_workdir(coverage_root, tag="coverage-tamper", dry_run=True)


def test_upload_commits_generation_before_publishing_current_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    import quant_fm.scripts.upload_to_minio as uploader

    source_token = next((root / "tokens").rglob("*.parquet"))
    source_token_identity = (source_token.stat().st_dev, source_token.stat().st_ino)
    calls: list[list[str]] = []
    committed: dict[str, object] | None = None
    uploaded_token_identity: tuple[int, int] | None = None
    uploaded_token_source: Path | None = None
    full_payload_checks = 0
    original_stable = uploader._assert_claimed_generation_stable

    def count_full_payload_check(*args: object, **kwargs: object) -> None:
        nonlocal full_payload_checks
        original_stable(*args, **kwargs)
        full_payload_checks += 1

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal committed, uploaded_token_identity, uploaded_token_source
        calls.append(command)
        if command[:2] == ["mc", "cat"]:
            if committed is None:
                return subprocess.CompletedProcess(
                    command,
                    1,
                    stdout="",
                    stderr="The specified key does not exist",
                )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(committed),
                stderr="",
            )
        if command[:2] == ["mc", "cp"] and command[-1].endswith("/generation.json"):
            committed = json.loads(Path(command[-2]).read_text(encoding="utf-8"))
        if command[:3] == ["mc", "cp", "--recursive"] and command[-1].endswith(
            "/tokens/"
        ):
            uploaded_token_source = Path(command[-2])
            claimed_token = next(uploaded_token_source.rglob("*.parquet"))
            stat = claimed_token.stat()
            uploaded_token_identity = (stat.st_dev, stat.st_ino)
            assert source_token.exists()
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._ensure_mc_alias",
        lambda: "fm_test",
    )
    monkeypatch.setattr("quant_fm.scripts.upload_to_minio.subprocess.run", fake_run)
    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._verify_remote_generation_payload",
        lambda *_args, **_kwargs: "0" * 64,
    )
    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._verify_committed_listing_receipt",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        uploader,
        "_assert_claimed_generation_stable",
        count_full_payload_check,
    )

    upload_workdir(root, tag="immutable-test")

    destinations = [command[-1] for command in calls if command[:2] == ["mc", "cp"]]
    generation_targets = [
        target for target in destinations if "/generations/" in target
    ]
    assert generation_targets
    assert all(
        re.search(r"/generations/[0-9a-f]{64}/", target)
        for target in generation_targets
    )
    assert destinations[-2].endswith("/generation.json")
    assert destinations[-1].endswith("/current.json")
    assert any(target.endswith("/data/coverage/") for target in generation_targets)
    payload_copies = [
        command
        for command in calls
        if command[:2] == ["mc", "cp"]
        and "/generations/" in command[-1]
        and not command[-1].endswith("/generation.json")
    ]
    assert payload_copies
    assert all(
        "--checksum" in command and "SHA256" in command for command in payload_copies
    )
    assert uploaded_token_identity == source_token_identity
    assert uploaded_token_source is not None
    assert uploaded_token_source.as_posix().startswith("/proc/self/fd/")
    # One frozen preflight plus one post-transfer check; pointer publication must
    # not add production-scale token rereads.
    assert full_payload_checks == 2
    assert (root / "tokens").is_dir()
    assert not list(root.glob(".tokens.quantfm-upload-*"))


def test_upload_rejects_incompatible_existing_generation_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    from quant_fm.scripts.upload_to_minio import _validate_local_generation

    incompatible = _validate_local_generation(root, run_live_audit=False).pointer
    incompatible = {**incompatible, "coverage_sha256": "0" * 64}
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:2] == ["mc", "cat"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(incompatible),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._ensure_mc_alias",
        lambda: "fm_test",
    )
    monkeypatch.setattr("quant_fm.scripts.upload_to_minio.subprocess.run", fake_run)

    with pytest.raises(
        RuntimeError, match="immutable generation commit is incompatible"
    ):
        upload_workdir(root, tag="incompatible-generation")

    assert not any(command[:2] == ["mc", "cp"] for command in calls)


def test_download_pointer_rejects_coverage_generation_mismatch(tmp_path: Path) -> None:
    root = _uploadable_v2(tmp_path)
    from quant_fm.scripts.upload_to_minio import _validate_local_generation

    pointer = _validate_local_generation(root, run_live_audit=False).pointer
    receipt = next((root / "data" / "coverage").glob("*.json"))
    receipt.write_text(receipt.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="coverage generation mismatch"):
        _validate_pointer_files(
            root,
            pointer,
            vocab_name="vocab_v2.json",
            data_version="v2",
        )


def test_v2_download_plan_restores_coverage_receipts(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        download_workdir(tmp_path, tag="coverage-plan", dry_run=True)

    assert "/data/coverage/" in caplog.text


def test_download_rejects_nonempty_destination_before_contacting_minio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "old-generation.txt").write_text("old\n", encoding="utf-8")
    contacted = False

    def unexpected_alias() -> str:
        nonlocal contacted
        contacted = True
        return "fm_test"

    monkeypatch.setattr(
        "quant_fm.scripts.download_from_minio._ensure_mc_alias",
        unexpected_alias,
    )

    with pytest.raises(FileExistsError, match="prevent mixing artifact generations"):
        download_workdir(tmp_path, tag="new-generation")

    assert contacted is False


def test_upload_failure_does_not_delete_local_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    committed: dict[str, object] | None = None

    def fail_pointer_publish(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal committed
        if command[:2] == ["mc", "cat"]:
            if committed is None:
                return subprocess.CompletedProcess(
                    command,
                    1,
                    stdout="",
                    stderr="The specified key does not exist",
                )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(committed),
                stderr="",
            )
        if command[-1].endswith("/current.json"):
            raise subprocess.CalledProcessError(1, command)
        if command[:2] == ["mc", "cp"] and command[-1].endswith("/generation.json"):
            committed = json.loads(Path(command[-2]).read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._ensure_mc_alias",
        lambda: "fm_test",
    )
    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio.subprocess.run",
        fail_pointer_publish,
    )
    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._verify_remote_generation_payload",
        lambda *_args, **_kwargs: "0" * 64,
    )
    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._verify_committed_listing_receipt",
        lambda *_args: None,
    )

    with pytest.raises(subprocess.CalledProcessError):
        upload_workdir(root, tag="failed-publish", delete_local=False)

    assert (root / "tokens").is_dir()
    assert not list(root.glob(".tokens.quantfm-upload-*"))


def test_delete_local_is_rejected_before_alias_or_remote_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    contacted = False

    def contact() -> str:
        nonlocal contacted
        contacted = True
        return "fm_test"

    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._ensure_mc_alias",
        contact,
    )

    with pytest.raises(RuntimeError, match="automatic delete_local is disabled"):
        upload_workdir(root, tag="delete-disabled", delete_local=True)

    assert contacted is False
    assert (root / "tokens").is_dir()
    assert not list(root.glob(".tokens.quantfm-upload-*"))


@pytest.mark.parametrize("diagnostic", ["Access Denied", "dial tcp: i/o timeout"])
def test_remote_pointer_transport_or_auth_error_never_falls_back_to_legacy(
    monkeypatch: pytest.MonkeyPatch,
    diagnostic: str,
) -> None:
    from quant_fm.scripts.upload_to_minio import _read_remote_pointer

    def failed_cat(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr=diagnostic,
        )

    monkeypatch.setattr("quant_fm.scripts.upload_to_minio.subprocess.run", failed_cat)

    with pytest.raises(RuntimeError, match="refusing legacy fallback") as captured:
        _read_remote_pointer("fm_test/bucket/tag")

    assert diagnostic not in str(captured.value)


def test_remote_pointer_explicit_missing_object_allows_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_fm.scripts.upload_to_minio import _read_remote_pointer

    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio.subprocess.run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="The specified key does not exist",
        ),
    )

    assert _read_remote_pointer("fm_test/bucket/legacy") is None


def test_expected_payload_omits_events_when_not_requested(tmp_path: Path) -> None:
    root = _uploadable_v2(tmp_path)
    event = root / "events" / "SH" / "600000" / "events.parquet"
    event.parent.mkdir(parents=True)
    event.write_bytes(b"optional event bytes")
    from quant_fm.scripts.upload_to_minio import (
        _expected_remote_payload,
        _validate_local_generation,
    )

    generation = _validate_local_generation(root, include_events=False)
    expected = _expected_remote_payload(
        generation,
        generation.pointer,
        include_commit=False,
        include_events=False,
    )

    assert not any(path.startswith("events/") for path in expected)


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
@pytest.mark.parametrize("artifact_kind", ["parquet", "sidecar"])
def test_remote_payload_verifier_rejects_missing_or_corrupt_token_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    artifact_kind: str,
) -> None:
    root = _uploadable_v2(tmp_path)
    from quant_fm.scripts.upload_to_minio import (
        _expected_remote_payload,
        _expected_remote_sizes,
        _validate_local_generation,
        _verify_remote_generation_payload,
    )

    generation = _validate_local_generation(root, run_live_audit=False)
    expected = _expected_remote_payload(
        generation,
        generation.pointer,
        include_commit=False,
        include_events=False,
    )
    expected_sizes = _expected_remote_sizes(generation, expected)
    suffix = ".parquet" if artifact_kind == "parquet" else ".contract.json"
    token_object = next(
        path
        for path in expected
        if path.startswith("tokens/") and path.endswith(suffix)
    )
    inventory = {
        relative: (expected_sizes[relative], f"etag-{index}")
        for index, relative in enumerate(expected)
    }
    if failure == "missing":
        del inventory[token_object]
    remote = "fm_test/bucket/tag/generations/generation"
    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._remote_listing_inventory",
        lambda _remote: inventory,
    )
    if failure == "corrupt":
        size, etag = inventory[token_object]
        inventory[token_object] = (size + 1, etag)

    with pytest.raises(RuntimeError, match=r"incomplete or mixed|size differs"):
        _verify_remote_generation_payload(
            remote,
            expected,
            expected_sizes=expected_sizes,
        )


def test_committed_but_incomplete_generation_is_not_published_or_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    from quant_fm.scripts.upload_to_minio import _validate_local_generation

    committed = {
        **_validate_local_generation(root, run_live_audit=False).pointer,
        "payload_transfer_checksum_algorithm": "SHA256",
        "payload_listing_receipt_sha256": "0" * 64,
    }
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:2] == ["mc", "cat"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(committed),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._ensure_mc_alias",
        lambda: "fm_test",
    )
    monkeypatch.setattr("quant_fm.scripts.upload_to_minio.subprocess.run", fake_run)
    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._verify_committed_listing_receipt",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError("remote generation payload is incomplete or mixed")
        ),
    )

    with pytest.raises(RuntimeError, match="incomplete or mixed"):
        upload_workdir(root, tag="incomplete-commit", delete_local=False)

    assert not any(
        command[:2] == ["mc", "cp"] and command[-1].endswith("/current.json")
        for command in calls
    )
    assert (root / "tokens").is_dir()
    assert not list(root.glob(".tokens.quantfm-upload-*"))


def test_legacy_commit_without_listing_receipt_cannot_be_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    from quant_fm.scripts.upload_to_minio import _validate_local_generation

    legacy_commit = _validate_local_generation(root, run_live_audit=False).pointer
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:2] == ["mc", "cat"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(legacy_commit),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._ensure_mc_alias",
        lambda: "fm_test",
    )
    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio.subprocess.run",
        fake_run,
    )

    with pytest.raises(RuntimeError, match="lacks a reusable listing receipt"):
        upload_workdir(root, tag="legacy-commit", delete_local=False)

    assert not any(
        command[:2] == ["mc", "cp"] and command[-1].endswith("/current.json")
        for command in calls
    )
    assert (root / "tokens").is_dir()
    assert not list(root.glob(".tokens.quantfm-upload-*"))


def test_current_pointer_readback_mismatch_prevents_local_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    from quant_fm.scripts.upload_to_minio import _validate_local_generation

    committed = {
        **_validate_local_generation(root, run_live_audit=False).pointer,
        "payload_transfer_checksum_algorithm": "SHA256",
        "payload_listing_receipt_sha256": "0" * 64,
    }
    wrong_current = {**committed, "audit_sha256": "0" * 64}

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if command[:2] == ["mc", "cat"]:
            payload = (
                committed if command[-1].endswith("/generation.json") else wrong_current
            )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(payload),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._ensure_mc_alias",
        lambda: "fm_test",
    )
    monkeypatch.setattr("quant_fm.scripts.upload_to_minio.subprocess.run", fake_run)
    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._verify_committed_listing_receipt",
        lambda *_args: None,
    )

    with pytest.raises(ValueError, match="inconsistent storage_generation_id"):
        upload_workdir(root, tag="wrong-current", delete_local=False)

    assert (root / "tokens").is_dir()
    assert not list(root.glob(".tokens.quantfm-upload-*"))


def test_source_mutation_after_initial_validation_cannot_commit_or_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    token = next((root / "tokens").rglob("*.parquet"))
    import quant_fm.scripts.upload_to_minio as uploader

    original_validate = uploader._validate_local_generation
    validations = 0
    calls: list[list[str]] = []

    def mutate_after_first_validation(*args: object, **kwargs: object):
        nonlocal validations
        result = original_validate(*args, **kwargs)
        validations += 1
        if validations == 1:
            with token.open("ab") as stream:
                stream.write(b"changed-after-validation")
        return result

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:2] == ["mc", "cat"]:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="The specified key does not exist",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        uploader, "_validate_local_generation", mutate_after_first_validation
    )
    monkeypatch.setattr(uploader, "_ensure_mc_alias", lambda: "fm_test")
    monkeypatch.setattr(uploader.subprocess, "run", fake_run)
    monkeypatch.setattr(
        uploader,
        "_verify_remote_generation_payload",
        lambda *_args, **_kwargs: "0" * 64,
    )

    with pytest.raises(ValueError, match=r"Parquet magic|parquet SHA-256 mismatch"):
        upload_workdir(root, tag="source-mutated", delete_local=False)

    assert not any(
        command[:2] == ["mc", "cp"]
        and command[-1].endswith(("/generation.json", "/current.json"))
        for command in calls
    )
    assert (root / "tokens").is_dir()
    assert not list(root.glob(".tokens.quantfm-upload-*"))


def test_root_fd_upload_preserves_original_visible_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    token = next((root / "tokens").rglob("*.parquet"))
    original_bytes = token.read_bytes()
    committed: dict[str, object] | None = None
    uploaded_bytes: bytes | None = None

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal committed, uploaded_bytes
        if command[:2] == ["mc", "cat"]:
            if committed is None:
                return subprocess.CompletedProcess(
                    command,
                    1,
                    stdout="",
                    stderr="The specified key does not exist",
                )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(committed),
                stderr="",
            )
        if command[:3] == ["mc", "cp", "--recursive"] and command[-1].endswith(
            "/tokens/"
        ):
            claimed = next(Path(command[-2]).rglob("*.parquet"))
            uploaded_bytes = claimed.read_bytes()
            assert token.is_file()
        if command[:2] == ["mc", "cp"] and command[-1].endswith("/generation.json"):
            committed = json.loads(Path(command[-2]).read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._ensure_mc_alias",
        lambda: "fm_test",
    )
    monkeypatch.setattr("quant_fm.scripts.upload_to_minio.subprocess.run", fake_run)
    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._verify_remote_generation_payload",
        lambda *_args, **_kwargs: "0" * 64,
    )
    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._verify_committed_listing_receipt",
        lambda *_args: None,
    )

    upload_workdir(root, tag="root-fd-source", delete_local=False)

    assert uploaded_bytes == original_bytes
    assert token.read_bytes() == original_bytes
    assert not list(root.glob(".tokens.quantfm-recovery-*"))


def test_external_hardlink_is_rejected_before_transient_upload_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    token = next((root / "tokens").rglob("*.parquet"))
    external = tmp_path / "external-token.parquet"
    external.hardlink_to(token)
    calls: list[list[str]] = []

    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._ensure_mc_alias",
        lambda: "fm_test",
    )

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("quant_fm.scripts.upload_to_minio.subprocess.run", fake_run)

    with pytest.raises(ValueError, match="external hardlink"):
        upload_workdir(root, tag="external-hardlink", delete_local=False)

    # The external inode could otherwise be mutated and restored while mc reads
    # the renamed claim.  Refusal must happen before any remote payload copy.
    assert not any(command[:2] == ["mc", "cp"] for command in calls)
    assert token.is_file()
    assert external.stat().st_ino == token.stat().st_ino


def test_existing_generation_remains_locally_after_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    token = next((root / "tokens").rglob("*.parquet"))
    original_bytes = token.read_bytes()
    from quant_fm.scripts.upload_to_minio import _validate_local_generation

    committed = {
        **_validate_local_generation(root, run_live_audit=False).pointer,
        "payload_transfer_checksum_algorithm": "SHA256",
        "payload_listing_receipt_sha256": "0" * 64,
    }
    current_published = False

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal current_published
        if command[:2] == ["mc", "cat"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(committed),
                stderr="",
            )
        if command[:2] == ["mc", "cp"] and command[-1].endswith("/current.json"):
            current_published = True
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._ensure_mc_alias",
        lambda: "fm_test",
    )
    monkeypatch.setattr("quant_fm.scripts.upload_to_minio.subprocess.run", fake_run)
    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._verify_committed_listing_receipt",
        lambda *_args: None,
    )

    upload_workdir(root, tag="producer-visible", delete_local=False)

    assert current_published is True
    assert token.read_bytes() == original_bytes
    assert not list(root.glob(".tokens.quantfm-recovery-*"))


def test_legacy_delete_helper_is_fail_closed(
    tmp_path: Path,
) -> None:
    root = _uploadable_v2(tmp_path)
    from quant_fm.scripts.upload_to_minio import (
        _claim_and_delete_local_payload,
        _expected_remote_payload,
        _validate_local_generation,
    )

    generation = _validate_local_generation(root, run_live_audit=False)
    expected = _expected_remote_payload(
        generation,
        generation.pointer,
        include_commit=False,
        include_events=False,
    )
    token = next((root / "tokens").rglob("*.parquet"))
    original_bytes = token.read_bytes()

    with pytest.raises(RuntimeError, match="automatic local payload deletion"):
        _claim_and_delete_local_payload(root, expected, include_events=False)

    assert token.read_bytes() == original_bytes
    assert not list(root.glob(".tokens.quantfm-delete-*"))


def test_root_substitution_is_detected_and_live_path_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    import quant_fm.scripts.upload_to_minio as uploader

    committed = {
        **uploader._validate_local_generation(
            root,
            run_live_audit=False,
        ).pointer,
        "payload_transfer_checksum_algorithm": "SHA256",
        "payload_listing_receipt_sha256": "0" * 64,
    }
    original_assert = uploader._assert_claimed_generation_stable
    assertions = 0
    sentinel = root / "tokens" / "NEW_GENERATION"
    writer_moved = root / "writer-moved-tokens"

    def replace_after_claim_assert(*args: object, **kwargs: object) -> None:
        nonlocal assertions
        original_assert(*args, **kwargs)
        assertions += 1
        if assertions == 1:
            (root / "tokens").rename(writer_moved)
            (root / "tokens").mkdir()
            sentinel.write_text("survives\n", encoding="utf-8")

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if command[:2] == ["mc", "cat"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(committed),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(uploader, "_ensure_mc_alias", lambda: "fm_test")
    monkeypatch.setattr(uploader.subprocess, "run", fake_run)
    monkeypatch.setattr(
        uploader,
        "_verify_committed_listing_receipt",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        uploader, "_assert_claimed_generation_stable", replace_after_claim_assert
    )

    with pytest.raises(RuntimeError, match="became untrusted"):
        upload_workdir(root, tag="post-assert-replacement", delete_local=False)

    assert assertions == 1
    assert writer_moved.is_dir()
    assert not (root / "tokens").exists()
    recovery = list(root.glob(".tokens.quantfm-recovery-*"))
    assert len(recovery) == 1
    assert (recovery[0] / "NEW_GENERATION").read_text(encoding="utf-8") == "survives\n"


def test_delete_helper_has_no_mutating_inventory_bypass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    import quant_fm.scripts.upload_to_minio as uploader

    generation = uploader._validate_local_generation(root, run_live_audit=False)
    expected = uploader._expected_remote_payload(
        generation,
        generation.pointer,
        include_commit=False,
        include_events=False,
    )
    inventory_called = False

    def inventory(_path: Path) -> list[dict[str, str]]:
        nonlocal inventory_called
        inventory_called = True
        return []

    monkeypatch.setattr(uploader, "_content_inventory", inventory)

    with pytest.raises(RuntimeError, match="automatic local payload deletion"):
        uploader._claim_and_delete_local_payload(
            root,
            expected,
            include_events=False,
        )

    assert inventory_called is False
    assert next((root / "tokens").rglob("*.parquet")).is_file()
    assert not list(root.glob(".tokens.quantfm-delete-*"))


@pytest.mark.parametrize("destination_existed", [False, True])
def test_download_staging_failure_is_clean_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    destination_existed: bool,
) -> None:
    destination = tmp_path / "restored"
    if destination_existed:
        destination.mkdir()
    fail_first_copy = True

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal fail_first_copy
        assert command[:2] == ["mc", "cp"]
        target = Path(command[-1])
        if "--recursive" in command:
            target.mkdir(parents=True, exist_ok=True)
            (target / "downloaded.bin").write_bytes(b"payload")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("metadata\n", encoding="utf-8")
        if fail_first_copy:
            fail_first_copy = False
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "quant_fm.scripts.download_from_minio._ensure_mc_alias",
        lambda: "fm_test",
    )
    monkeypatch.setattr(
        "quant_fm.scripts.download_from_minio._read_remote_pointer",
        lambda _remote: None,
    )
    monkeypatch.setattr(
        "quant_fm.scripts.download_from_minio.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        "quant_fm.scripts.download_from_minio._rebase_downloaded_manifest",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "quant_fm.scripts.download_from_minio._validate_local_generation",
        lambda *_args, **_kwargs: SimpleNamespace(generation_id="legacy"),
    )

    with pytest.raises(subprocess.CalledProcessError):
        download_workdir(destination, tag="retry", data_version="v1")

    if destination_existed:
        assert destination.is_dir()
        assert not any(destination.iterdir())
    else:
        assert not destination.exists()

    download_workdir(destination, tag="retry", data_version="v1")

    assert (destination / "tokens" / "downloaded.bin").is_file()
    assert not list(tmp_path.glob(".restored.quantfm-download-*"))


def test_remote_listing_inventory_uses_one_recursive_list_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_fm.scripts.upload_to_minio import _remote_listing_inventory

    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        lines = [
            json.dumps(
                {
                    "status": "success",
                    "key": f"tokens/{index}.parquet",
                    "type": "file",
                    "size": index + 1,
                    "etag": f"etag-{index}",
                }
            )
            for index in range(3)
        ]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(lines),
            stderr="",
        )

    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio.subprocess.run",
        fake_run,
    )

    inventory = _remote_listing_inventory("fm_test/bucket/generation")

    assert len(inventory) == 3
    assert calls == [
        [
            "mc",
            "ls",
            "--recursive",
            "--json",
            "fm_test/bucket/generation/",
        ]
    ]


def test_committed_listing_receipt_rejects_etag_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_fm.scripts.upload_to_minio import (
        _listing_inventory_sha256,
        _verify_committed_listing_receipt,
    )

    payload = {"tokens/a.parquet": (10, "etag-before")}
    committed = {
        "payload_transfer_checksum_algorithm": "SHA256",
        "payload_listing_receipt_sha256": _listing_inventory_sha256(payload),
    }
    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio._remote_listing_inventory",
        lambda _remote: {
            "tokens/a.parquet": (10, "etag-after"),
            "generation.json": (100, "commit-etag"),
        },
    )

    with pytest.raises(RuntimeError, match="key/size/ETag receipt changed"):
        _verify_committed_listing_receipt(
            "fm_test/bucket/generation",
            {"tokens/a.parquet": "1" * 64, "generation.json": None},
            committed,
        )


def test_remote_listing_inventory_rejects_missing_etag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_fm.scripts.upload_to_minio import _remote_listing_inventory

    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio.subprocess.run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "status": "success",
                    "type": "file",
                    "key": "tokens/a.parquet",
                    "size": 10,
                    "etag": "",
                }
            ),
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="invalid or duplicate"):
        _remote_listing_inventory("fm_test/bucket/generation")


def test_successful_staged_download_rebases_manifest_to_published_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _uploadable_v2(tmp_path / "source")
    destination = tmp_path / "published"

    def fake_copy(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        remote_source = command[-2]
        target = Path(command[-1])
        if remote_source.endswith("/tokens/"):
            shutil.copytree(source / "tokens", target, dirs_exist_ok=True)
        elif remote_source.endswith("/data/coverage/"):
            shutil.copytree(
                source / "data" / "coverage",
                target,
                dirs_exist_ok=True,
            )
        elif remote_source.endswith("/data/vocab_v2.json"):
            shutil.copy2(source / "data" / "vocab_v2.json", target)
        elif remote_source.endswith("/data/manifest.json"):
            shutil.copy2(source / "data" / "manifest.json", target)
        elif remote_source.endswith("/artifact_audit.json"):
            shutil.copy2(source / "artifact_audit.json", target)
        else:
            raise AssertionError(remote_source)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "quant_fm.scripts.download_from_minio._ensure_mc_alias",
        lambda: "fm_test",
    )
    monkeypatch.setattr(
        "quant_fm.scripts.download_from_minio._read_remote_pointer",
        lambda _remote: None,
    )
    monkeypatch.setattr(
        "quant_fm.scripts.download_from_minio.subprocess.run",
        fake_copy,
    )

    download_workdir(destination, tag="staged-success")

    manifest = Manifest.load(destination / "data" / "manifest.json")
    assert Path(manifest.vocab_path).parent == (destination / "data").resolve()
    assert all(
        Path(shard.path).is_relative_to((destination / "tokens").resolve())
        for shard in manifest.shards
    )
    from quant_fm.scripts.upload_to_minio import _validate_local_generation

    _validate_local_generation(destination, run_live_audit=False)


def test_path_bound_storage_generation_id_isolates_rebased_workdirs(
    tmp_path: Path,
) -> None:
    from quant_fm.scripts.upload_to_minio import (
        _generation_remote,
        _validate_local_generation,
    )

    first = _validate_local_generation(
        _uploadable_v2(tmp_path / "first"),
        run_live_audit=False,
    )
    second = _validate_local_generation(
        _uploadable_v2(tmp_path / "second"),
        run_live_audit=False,
    )

    assert first.core_generation_id == second.core_generation_id
    assert first.generation_id == second.generation_id
    assert first.pointer["manifest_sha256"] != second.pointer["manifest_sha256"]
    assert first.storage_generation_id != second.storage_generation_id
    assert _generation_remote("alias/bucket/tag", first.pointer) != _generation_remote(
        "alias/bucket/tag", second.pointer
    )


def test_v3_storage_namespace_and_pointer_bind_immutability_fence(
    tmp_path: Path,
) -> None:
    from quant_fm.scripts.upload_to_minio import (
        IMMUTABILITY_FENCE_VERSION,
        _checked_pointer,
        _storage_generation_id,
        _validate_local_generation,
    )

    pointer = _validate_local_generation(
        _uploadable_v2(tmp_path),
        run_live_audit=False,
    ).pointer
    assert pointer["pointer_version"] == 3
    assert pointer["immutability_fence_version"] == IMMUTABILITY_FENCE_VERSION

    legacy_v2 = dict(pointer)
    legacy_v2["pointer_version"] = 2
    legacy_v2.pop("immutability_fence_version")
    legacy_v2["storage_generation_id"] = _storage_generation_id(legacy_v2)
    assert _checked_pointer(legacy_v2, context="legacy v2") == legacy_v2
    assert legacy_v2["storage_generation_id"] != pointer["storage_generation_id"]

    missing_fence = dict(pointer)
    missing_fence.pop("immutability_fence_version")
    with pytest.raises(ValueError, match="immutability fence"):
        _checked_pointer(missing_fence, context="v3")


def test_v2_remote_pointer_remains_download_read_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_fm.scripts.upload_to_minio import (
        _read_remote_pointer,
        _storage_generation_id,
        _validate_local_generation,
    )

    pointer = _validate_local_generation(
        _uploadable_v2(tmp_path),
        run_live_audit=False,
    ).pointer
    pointer["pointer_version"] = 2
    pointer.pop("immutability_fence_version")
    pointer["storage_generation_id"] = _storage_generation_id(pointer)
    pointer["payload_transfer_checksum_algorithm"] = "SHA256"
    pointer["payload_listing_receipt_sha256"] = "0" * 64
    monkeypatch.setattr(
        "quant_fm.scripts.upload_to_minio.subprocess.run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(pointer),
            stderr="",
        ),
    )

    assert _read_remote_pointer("fm_test/bucket/tag") == pointer


def test_v2_commit_cannot_be_reused_at_a_v3_storage_prefix(tmp_path: Path) -> None:
    from quant_fm.scripts.upload_to_minio import (
        _assert_compatible_commit,
        _validate_local_generation,
    )

    local = _validate_local_generation(
        _uploadable_v2(tmp_path),
        run_live_audit=False,
    ).pointer
    forged_v2 = dict(local)
    forged_v2["pointer_version"] = 2

    with pytest.raises(RuntimeError, match="pointer_version"):
        _assert_compatible_commit(local, forged_v2)


def test_writer_open_after_final_scan_blocks_publish_and_quarantines_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    import quant_fm.scripts.upload_to_minio as uploader

    committed = {
        **uploader._validate_local_generation(
            root,
            run_live_audit=False,
        ).pointer,
        "payload_transfer_checksum_algorithm": "SHA256",
        "payload_listing_receipt_sha256": "0" * 64,
    }
    original_scan = uploader._assert_no_same_uid_process_references
    scans = 0
    claim_token: list[Path] = []
    start_writer = threading.Event()
    writer_opening = threading.Event()
    writer_done = threading.Event()
    lease_break_observed = [False]
    writer_errors: list[BaseException] = []
    calls: list[list[str]] = []

    original_break_handler = uploader._ClaimReadLeaseFence._handle_break

    def observe_lease_break(
        self: object,
        signum: int,
        frame: object,
    ) -> None:
        original_break_handler(self, signum, frame)
        lease_break_observed[0] = True

    def writer() -> None:
        start_writer.wait()
        writer_opening.set()
        try:
            with claim_token[0].open("r+b") as stream:
                stream.write(b"BAD")
                stream.flush()
        except BaseException as exc:  # pragma: no cover - diagnostic only
            writer_errors.append(exc)
        finally:
            writer_done.set()

    thread = threading.Thread(target=writer)
    thread.start()

    def open_after_final_scan(*args: object, **kwargs: object) -> None:
        nonlocal scans
        original_scan(*args, **kwargs)
        scans += 1
        # Initial inherited-handle scan, prehash scan, then the final scan just
        # before current.json.  The lease must close this old TOCTOU window.
        if scans == 3:
            roots = args[0]
            claim_root = next(path for path in roots if Path(path).name == "tokens")
            claim_token.append(next(Path(claim_root).rglob("*.parquet")))
            start_writer.set()
            writer_opening.wait(timeout=2)
            deadline = time.monotonic() + 2
            while not lease_break_observed[0] and time.monotonic() < deadline:
                time.sleep(0.001)
            assert lease_break_observed[0]

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:2] == ["mc", "cat"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(committed),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(uploader, "_ensure_mc_alias", lambda: "fm_test")
    monkeypatch.setattr(
        uploader._ClaimReadLeaseFence,
        "_handle_break",
        observe_lease_break,
    )
    monkeypatch.setattr(uploader.subprocess, "run", fake_run)
    monkeypatch.setattr(
        uploader,
        "_verify_committed_listing_receipt",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        uploader,
        "_assert_no_same_uid_process_references",
        open_after_final_scan,
    )

    try:
        with pytest.raises(RuntimeError, match="became untrusted"):
            upload_workdir(root, tag="open-fd-delete", delete_local=False)
    finally:
        start_writer.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert not writer_errors
    assert scans == 3
    assert not any(
        command[:2] == ["mc", "cp"] and command[-1].endswith("/current.json")
        for command in calls
    )
    assert not (root / "tokens").exists()
    recovery = list(root.glob(".tokens.quantfm-recovery-*"))
    assert len(recovery) == 1
    assert next(recovery[0].rglob("*.parquet")).read_bytes().startswith(b"BAD")


def test_writer_open_during_transfer_blocks_commit_and_quarantines_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    import quant_fm.scripts.upload_to_minio as uploader

    claim_token: list[Path] = []
    start_writer = threading.Event()
    writer_opening = threading.Event()
    writer_done = threading.Event()
    writer_errors: list[BaseException] = []
    calls: list[list[str]] = []

    def writer() -> None:
        start_writer.wait()
        writer_opening.set()
        try:
            with claim_token[0].open("r+b"):
                pass
        except BaseException as exc:  # pragma: no cover - diagnostic only
            writer_errors.append(exc)
        finally:
            writer_done.set()

    thread = threading.Thread(target=writer)
    thread.start()

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:2] == ["mc", "cat"]:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="The specified key does not exist",
            )
        if command[:3] == ["mc", "cp", "--recursive"] and command[-1].endswith(
            "/tokens/"
        ):
            claim_token.append(next(Path(command[-2]).rglob("*.parquet")))
            start_writer.set()
            writer_opening.wait(timeout=2)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(uploader, "_ensure_mc_alias", lambda: "fm_test")
    monkeypatch.setattr(uploader.subprocess, "run", fake_run)
    monkeypatch.setattr(
        uploader,
        "_verify_remote_generation_payload",
        lambda *_args, **_kwargs: "0" * 64,
    )

    try:
        with pytest.raises(RuntimeError, match="became untrusted"):
            upload_workdir(root, tag="writer-during-transfer", delete_local=False)
    finally:
        start_writer.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert not writer_errors
    assert not any(
        command[:2] == ["mc", "cp"]
        and command[-1].endswith(("/generation.json", "/current.json"))
        for command in calls
    )
    assert not (root / "tokens").exists()
    recovery = list(root.glob(".tokens.quantfm-recovery-*"))
    assert len(recovery) == 1
    assert next(recovery[0].rglob("*.parquet")).is_file()


def test_rename_replace_restore_during_transfer_blocks_remote_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    import quant_fm.scripts.upload_to_minio as uploader

    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:2] == ["mc", "cat"]:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="The specified key does not exist",
            )
        if command[:3] == ["mc", "cp", "--recursive"] and command[-1].endswith(
            "/tokens/"
        ):
            token = next(Path(command[-2]).rglob("*.parquet"))
            original = token.with_name(f".{token.name}.writer-saved")
            original_bytes = token.read_bytes()
            token.rename(original)
            token.write_bytes(b"x" * len(original_bytes))
            token.unlink()
            original.rename(token)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(uploader, "_ensure_mc_alias", lambda: "fm_test")
    monkeypatch.setattr(uploader.subprocess, "run", fake_run)
    monkeypatch.setattr(
        uploader,
        "_verify_remote_generation_payload",
        lambda *_args, **_kwargs: "0" * 64,
    )

    with pytest.raises(RuntimeError, match="artifact claim changed"):
        upload_workdir(root, tag="rename-replace-restore", delete_local=False)

    assert not any(
        command[:2] == ["mc", "cp"]
        and command[-1].endswith(("/generation.json", "/current.json"))
        for command in calls
    )
    assert not (root / "tokens").exists()
    recovery = list(root.glob(".tokens.quantfm-recovery-*"))
    assert len(recovery) == 1
    assert next(recovery[0].rglob("*.parquet")).is_file()


def test_metadata_replace_restore_during_transfer_blocks_remote_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    import quant_fm.scripts.upload_to_minio as uploader

    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:2] == ["mc", "cat"]:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="The specified key does not exist",
            )
        if command[:3] == ["mc", "cp", "--recursive"] and command[-1].endswith(
            "/tokens/"
        ):
            metadata_root = next(
                path
                for path in root.parent.glob(".quantfm-upload-metadata-*")
                if path.is_dir()
            )
            manifest = metadata_root / "data" / "manifest.json"
            saved = manifest.with_name(".manifest.json.writer-saved")
            original_bytes = manifest.read_bytes()
            manifest.rename(saved)
            manifest.write_bytes(b"x" * len(original_bytes))
            manifest.unlink()
            saved.rename(manifest)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(uploader, "_ensure_mc_alias", lambda: "fm_test")
    monkeypatch.setattr(uploader.subprocess, "run", fake_run)
    monkeypatch.setattr(
        uploader,
        "_verify_remote_generation_payload",
        lambda *_args, **_kwargs: "0" * 64,
    )

    with pytest.raises(RuntimeError, match="artifact claim changed"):
        upload_workdir(root, tag="metadata-replace-restore", delete_local=False)

    assert not any(
        command[:2] == ["mc", "cp"]
        and command[-1].endswith(("/generation.json", "/current.json"))
        for command in calls
    )
    assert not (root / "tokens").exists()
    recovery = list(root.glob(".tokens.quantfm-recovery-*"))
    assert len(recovery) == 1
    assert next(recovery[0].rglob("*.parquet")).is_file()


def test_unsupported_read_lease_restores_claim_without_remote_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    import quant_fm.scripts.upload_to_minio as uploader

    calls: list[list[str]] = []
    old_handler = signal.getsignal(signal.SIGIO)

    def unsupported(_fd: int) -> None:
        raise OSError(errno.EOPNOTSUPP, "leases unsupported")

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(uploader, "_ensure_mc_alias", lambda: "fm_test")
    monkeypatch.setattr(uploader, "_set_read_lease", unsupported)
    monkeypatch.setattr(uploader.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="read lease unavailable"):
        upload_workdir(root, tag="unsupported-lease", delete_local=False)

    assert signal.getsignal(signal.SIGIO) is old_handler
    assert next((root / "tokens").rglob("*.parquet")).is_file()
    assert not list(root.glob(".tokens.quantfm-upload-*"))
    assert not any(command[:2] == ["mc", "cp"] for command in calls)


def test_partial_lease_conflict_quarantines_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    import quant_fm.scripts.upload_to_minio as uploader

    attempts = 0

    def conflict_after_one(fd: int) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise OSError(errno.EAGAIN, "writer conflict")
        uploader.fcntl.fcntl(fd, uploader.fcntl.F_SETOWN, uploader.os.getpid())
        uploader.fcntl.fcntl(fd, uploader.fcntl.F_SETLEASE, uploader.fcntl.F_RDLCK)

    monkeypatch.setattr(uploader, "_ensure_mc_alias", lambda: "fm_test")
    monkeypatch.setattr(uploader, "_set_read_lease", conflict_after_one)

    with pytest.raises(RuntimeError, match=r"writer conflicts.*recovery"):
        upload_workdir(root, tag="partial-lease-conflict", delete_local=False)

    assert attempts == 2
    assert not (root / "tokens").exists()
    recovery = list(root.glob(".tokens.quantfm-recovery-*"))
    assert len(recovery) == 1
    assert next(recovery[0].rglob("*.parquet")).is_file()


def test_partial_lease_late_break_quarantines_before_unlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    import quant_fm.scripts.upload_to_minio as uploader

    original_set_lease = uploader._set_read_lease
    original_sigpending = uploader.signal.sigpending
    leased_target: list[Path] = []
    attempts = 0
    pending_checks = 0
    start_writer = threading.Event()
    writer_entered = threading.Event()
    writer_errors: list[BaseException] = []

    def unsupported_after_one(fd: int) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise OSError(errno.EOPNOTSUPP, "later lease unsupported")
        original_set_lease(fd)
        leased_target.append(Path(f"/proc/self/fd/{fd}").resolve())

    def writer() -> None:
        start_writer.wait()
        writer_entered.set()
        try:
            with leased_target[0].open("r+b") as stream:
                stream.seek(0)
                stream.write(b"BAD")
                stream.flush()
        except BaseException as exc:  # pragma: no cover - diagnostic only
            writer_errors.append(exc)

    thread = threading.Thread(target=writer)
    thread.start()

    def late_pending_signal() -> set[signal.Signals]:
        nonlocal pending_checks
        # Snapshot first, then make a writer request the already-held lease.
        # This deterministically models the request arriving just after the
        # acquire exception's first pending-signal check.
        snapshot = original_sigpending()
        pending_checks += 1
        if pending_checks == 1:
            start_writer.set()
            assert writer_entered.wait(timeout=2)
            deadline = time.monotonic() + 2
            while signal.SIGIO not in original_sigpending():
                if time.monotonic() >= deadline:
                    msg = "writer did not request the partial read lease"
                    raise AssertionError(msg)
                time.sleep(0.001)
        return snapshot

    monkeypatch.setattr(uploader, "_ensure_mc_alias", lambda: "fm_test")
    monkeypatch.setattr(uploader, "_set_read_lease", unsupported_after_one)
    monkeypatch.setattr(uploader.signal, "sigpending", late_pending_signal)

    try:
        with pytest.raises(RuntimeError, match="became untrusted"):
            upload_workdir(root, tag="partial-lease-late-break", delete_local=False)
    finally:
        start_writer.set()
        thread.join(timeout=5)

    assert attempts == 2
    assert not thread.is_alive()
    assert not writer_errors
    assert leased_target[0].is_relative_to(root / "tokens")
    relative = leased_target[0].relative_to(root / "tokens")
    assert not (root / "tokens").exists()
    recovery = list(root.glob(".tokens.quantfm-recovery-*"))
    assert len(recovery) == 1
    assert (recovery[0] / relative).read_bytes().startswith(b"BAD")


def test_too_small_nofile_limit_fails_before_upload_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    import quant_fm.scripts.upload_to_minio as uploader

    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(uploader, "_ensure_mc_alias", lambda: "fm_test")
    monkeypatch.setattr(uploader.subprocess, "run", fake_run)
    monkeypatch.setattr(uploader.resource, "getrlimit", lambda _kind: (1, 1))

    with pytest.raises(RuntimeError, match="RLIMIT_NOFILE"):
        upload_workdir(root, tag="small-nofile", delete_local=False)

    assert (root / "tokens").is_dir()
    assert not list(root.glob(".tokens.quantfm-upload-*"))
    assert not any(command[:2] == ["mc", "cp"] for command in calls)


def test_remote_or_unknown_filesystem_is_rejected_before_upload_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    import quant_fm.scripts.upload_to_minio as uploader

    contacted = False

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal contacted
        contacted = True
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(uploader, "_ensure_mc_alias", lambda: "fm_test")
    monkeypatch.setattr(uploader.subprocess, "run", fake_run)
    monkeypatch.setattr(uploader, "_filesystem_type", lambda _path: "nfs4")

    with pytest.raises(RuntimeError, match="supported local filesystem"):
        upload_workdir(root, tag="remote-fs", delete_local=False)

    assert contacted is False
    assert (root / "tokens").is_dir()
    assert not list(root.glob(".tokens.quantfm-upload-*"))


def test_remote_metadata_staging_parent_is_rejected_before_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _uploadable_v2(tmp_path)
    import quant_fm.scripts.upload_to_minio as uploader

    contacted = False

    def alias() -> str:
        nonlocal contacted
        contacted = True
        return "fm_test"

    def fs_type(path: Path) -> str:
        return "nfs4" if Path(path) == root.parent else "ext4"

    monkeypatch.setattr(uploader, "_ensure_mc_alias", alias)
    monkeypatch.setattr(uploader, "_filesystem_type", fs_type)

    with pytest.raises(RuntimeError, match=r"metadata staging.*supported local"):
        upload_workdir(root, tag="remote-metadata-parent", delete_local=False)

    assert contacted is False
    assert (root / "tokens").is_dir()
    assert not list(root.glob(".tokens.quantfm-upload-*"))


def _reader_receipt_pointer() -> dict[str, object]:
    return {
        "storage_generation_id": "1" * 64,
        "payload_transfer_checksum_algorithm": "SHA256",
        "payload_listing_receipt_sha256": "2" * 64,
        "shard_count": 1,
        "coverage_file_count": 1,
    }


def test_remote_commit_receipt_requires_exact_generation_pointer_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_fm.scripts.upload_to_minio as uploader

    current = _reader_receipt_pointer()
    generation_commit = {**current, "unexpected_generation_field": True}
    listed = False

    def unexpected_listing(_remote: str) -> dict[str, tuple[int, str]]:
        nonlocal listed
        listed = True
        return {}

    monkeypatch.setattr(
        uploader,
        "_read_remote_pointer",
        lambda _remote, *, filename="current.json": generation_commit,
    )
    monkeypatch.setattr(uploader, "_remote_listing_inventory", unexpected_listing)

    with pytest.raises(RuntimeError, match=r"generation\.json.*current\.json"):
        uploader._verify_remote_commit_receipt(
            "fm_test/bucket/tag/generations/id",
            current,
        )

    assert listed is False


def test_download_rejects_generation_commit_that_differs_from_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_fm.scripts.download_from_minio as downloader
    import quant_fm.scripts.upload_to_minio as uploader

    current = _reader_receipt_pointer()
    generation_commit = {**current, "unexpected_generation_field": True}
    contacted_payload = False

    def unexpected_copy(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal contacted_payload
        contacted_payload = True
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(downloader, "_ensure_mc_alias", lambda: "fm_test")
    monkeypatch.setattr(downloader, "output_bucket", lambda: "bucket")
    monkeypatch.setattr(downloader, "output_prefix", lambda _tag: "tag")
    monkeypatch.setattr(downloader, "_read_remote_pointer", lambda _remote: current)
    monkeypatch.setattr(
        uploader,
        "_read_remote_pointer",
        lambda _remote, *, filename="current.json": generation_commit,
    )
    monkeypatch.setattr(downloader.subprocess, "run", unexpected_copy)

    with pytest.raises(RuntimeError, match=r"generation\.json.*current\.json"):
        download_workdir(tmp_path / "download", tag="mismatch")

    assert contacted_payload is False


def test_remote_ready_rejects_generation_commit_that_differs_from_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_fm.scripts.download_from_minio as downloader
    import quant_fm.scripts.upload_to_minio as uploader

    current = _reader_receipt_pointer()
    generation_commit = {**current, "unexpected_generation_field": True}

    monkeypatch.setattr(downloader, "_ensure_mc_alias", lambda: "fm_test")
    monkeypatch.setattr(downloader, "output_bucket", lambda: "bucket")
    monkeypatch.setattr(downloader, "output_prefix", lambda _tag: "tag")
    monkeypatch.setattr(downloader, "_read_remote_pointer", lambda _remote: current)
    monkeypatch.setattr(
        uploader,
        "_read_remote_pointer",
        lambda _remote, *, filename="current.json": generation_commit,
    )
    monkeypatch.setattr(
        downloader.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("payload checks must not run"),
    )

    assert downloader.remote_ready("mismatch") is False


def test_verify_upload_rejects_generation_commit_that_differs_from_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_fm.scripts.upload_to_minio as uploader

    current = _reader_receipt_pointer()
    generation_commit = {**current, "unexpected_generation_field": True}

    def pointer(_remote: str, *, filename: str = "current.json") -> dict[str, object]:
        return current if filename == "current.json" else generation_commit

    monkeypatch.setattr(uploader, "_ensure_mc_alias", lambda: "fm_test")
    monkeypatch.setattr(uploader, "output_bucket", lambda: "bucket")
    monkeypatch.setattr(uploader, "output_prefix", lambda _tag: "tag")
    monkeypatch.setattr(uploader, "_read_remote_pointer", pointer)
    monkeypatch.setattr(
        uploader.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("payload checks must not run"),
    )

    with pytest.raises(RuntimeError, match=r"generation\.json.*current\.json"):
        uploader.verify_upload("mismatch")


def test_legacy_flat_v2_remote_requires_nonempty_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_fm.scripts.download_from_minio as downloader

    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:2] == ["mc", "stat"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[:2] == ["mc", "find"]:
            stdout = "token.parquet\n" if command[-3].endswith("/tokens") else ""
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(downloader, "_ensure_mc_alias", lambda: "fm_test")
    monkeypatch.setattr(downloader, "output_bucket", lambda: "bucket")
    monkeypatch.setattr(downloader, "output_prefix", lambda _tag: "legacy")
    monkeypatch.setattr(downloader, "_read_remote_pointer", lambda _remote: None)
    monkeypatch.setattr(downloader.subprocess, "run", fake_run)

    assert downloader.remote_ready("legacy", data_version="v2") is False
    assert any(
        command[:2] == ["mc", "find"] and command[-3].endswith("/data/coverage")
        for command in calls
    )
