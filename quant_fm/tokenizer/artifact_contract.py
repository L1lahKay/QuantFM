"""Sidecar contract that binds token parquet data semantics to its vocabulary."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pylob.event_ordering import (
    LEGACY_LOCAL_TIME_V1,
    validate_event_ordering_version,
)

from quant_fm.tokenizer.transforms import (
    FEATURE_TRANSFORM_LEGACY_V1,
    reference_price_initialization,
    validate_feature_transform_version,
)

if TYPE_CHECKING:
    from typing import Any

TOKEN_ARTIFACT_CONTRACT_VERSION = "2.0"
_LEGACY_TOKEN_ARTIFACT_CONTRACT_VERSION = "1.0"


class TokenSemantics(Protocol):
    """Minimal Vocab/VocabV2 interface needed by the sidecar contract."""

    schema_version: str
    event_ordering_version: str
    feature_transform_version: str
    data_semantics_explicit: bool

    def to_json(self) -> str:
        """Return the stable serialized vocabulary payload."""


def stable_vocab_sha256(vocab: TokenSemantics) -> str:
    """
    Hash the exact stable serialization written by ``Vocab.save``.

    Both V1 and V2 vocabularies expose deterministic ``to_json`` methods.  Using
    that byte representation avoids path/whitespace-dependent identities while
    remaining equal to the bytes produced by the ordinary ``save`` methods.
    """
    return hashlib.sha256(vocab.to_json().encode("utf-8")).hexdigest()


def token_contract_path(token_path: Path) -> Path:
    """Return the sidecar path for a token parquet shard."""
    path = Path(token_path)
    return path.with_suffix(f"{path.suffix}.contract.json")


def token_contract_payload(
    vocab: TokenSemantics,
    *,
    storage_encoding: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the stable semantic payload stored next to every new token shard."""
    ordering = validate_event_ordering_version(vocab.event_ordering_version)
    transform = validate_feature_transform_version(vocab.feature_transform_version)
    payload: dict[str, object] = {
        "artifact_version": TOKEN_ARTIFACT_CONTRACT_VERSION,
        "schema_version": str(vocab.schema_version),
        "vocab_sha256": stable_vocab_sha256(vocab),
        "event_ordering_version": ordering,
        "feature_transform_version": transform,
        "reference_price_initialization": reference_price_initialization(transform),
    }
    if storage_encoding is not None:
        # The encoding owns its own version and SHA-256.  Keeping it in this
        # existing sidecar makes parquet bytes and their decode scale one
        # content-addressed manifest unit instead of adding a second sidecar.
        payload["storage_encoding"] = dict(storage_encoding)
    return payload


def write_token_contract(
    token_path: Path,
    vocab: TokenSemantics,
    *,
    storage_encoding: Mapping[str, object] | None = None,
) -> Path:
    """Atomically write one token shard's explicit data-semantics sidecar."""
    destination = token_contract_path(token_path)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(
            token_contract_payload(vocab, storage_encoding=storage_encoding),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def read_token_contract(token_path: Path) -> dict[str, Any]:
    """Read a sidecar; sidecar-less historical shards are identified as legacy."""
    path = token_contract_path(token_path)
    if not path.is_file():
        return {
            "artifact_version": "0",
            "schema_version": None,
            "event_ordering_version": LEGACY_LOCAL_TIME_V1,
            "feature_transform_version": FEATURE_TRANSFORM_LEGACY_V1,
            "reference_price_initialization": reference_price_initialization(
                FEATURE_TRANSFORM_LEGACY_V1
            ),
            "inferred_legacy": True,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"invalid token contract object: {path}"
        raise TypeError(msg)
    artifact_version = str(payload.get("artifact_version", ""))
    if artifact_version not in {
        _LEGACY_TOKEN_ARTIFACT_CONTRACT_VERSION,
        TOKEN_ARTIFACT_CONTRACT_VERSION,
    }:
        msg = f"unsupported token artifact contract version: {artifact_version!r}"
        raise ValueError(msg)
    ordering = validate_event_ordering_version(
        str(payload.get("event_ordering_version", ""))
    )
    transform = validate_feature_transform_version(
        str(payload.get("feature_transform_version", ""))
    )
    expected_initialization = reference_price_initialization(transform)
    if payload.get("reference_price_initialization") != expected_initialization:
        msg = (
            f"token contract reference-price policy disagrees with {transform}: {path}"
        )
        raise ValueError(msg)
    vocab_sha256 = payload.get("vocab_sha256")
    if vocab_sha256 is not None:
        vocab_sha256 = str(vocab_sha256)
        if len(vocab_sha256) != 64:
            msg = f"token contract vocab_sha256 must be a full SHA-256: {path}"
            raise ValueError(msg)
        try:
            bytes.fromhex(vocab_sha256)
        except ValueError as exc:
            msg = f"token contract vocab_sha256 must be hexadecimal: {path}"
            raise ValueError(msg) from exc
    storage = payload.get("storage_encoding")
    if storage is not None:
        if not isinstance(storage, Mapping):
            msg = f"token contract storage_encoding must be an object: {path}"
            raise TypeError(msg)
        # Local import avoids a module cycle: the storage codec itself uses this
        # module for the shared stable vocab identity.
        from quant_fm.tokenizer.storage_encoding_v2 import StorageEncodingMetadataV2

        metadata = StorageEncodingMetadataV2.from_dict(storage)
        storage_vocab_sha256 = metadata.vocab_sha256
        if vocab_sha256 is None or storage_vocab_sha256 != vocab_sha256:
            msg = (
                "token contract vocab identity disagrees with V2 storage metadata: "
                f"{path}"
            )
            raise ValueError(msg)
    return {
        **payload,
        "event_ordering_version": ordering,
        "feature_transform_version": transform,
        "inferred_legacy": False,
    }


def assert_token_contract_matches(token_path: Path, vocab: TokenSemantics) -> None:
    """Reject resume/manifest use when shard and vocabulary semantics differ."""
    actual = read_token_contract(token_path)
    if actual["inferred_legacy"] and vocab.data_semantics_explicit:
        msg = (
            f"token shard has no explicit data-semantics sidecar: {token_path}; "
            "use its legacy vocab or rebuild into a new output root"
        )
        raise ValueError(msg)
    expected = token_contract_payload(vocab)
    mismatches = {
        key: (actual.get(key), expected[key])
        for key in (
            "artifact_version",
            "schema_version",
            "vocab_sha256",
            "event_ordering_version",
            "feature_transform_version",
            "reference_price_initialization",
        )
        if not (
            actual["inferred_legacy"]
            and key in {"schema_version", "vocab_sha256", "artifact_version"}
        )
        and actual.get(key) != expected[key]
    }
    if mismatches:
        msg = f"token/vocab data-semantics mismatch for {token_path}: {mismatches}"
        raise ValueError(msg)
    storage = actual.get("storage_encoding")
    if storage is not None and getattr(vocab, "VOCAB_VERSION", None) == "2.0":
        from quant_fm.tokenizer.storage_encoding_v2 import (
            StorageEncodingMetadataV2,
            assert_storage_metadata_matches_vocab_v2,
        )

        assert_storage_metadata_matches_vocab_v2(
            StorageEncodingMetadataV2.from_dict(storage),
            vocab,  # type: ignore[arg-type]
        )


def token_contract_matches(token_path: Path, vocab: TokenSemantics) -> bool:
    """Return whether a token shard is safe to reuse with ``vocab``."""
    try:
        assert_token_contract_matches(token_path, vocab)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


__all__ = [
    "TOKEN_ARTIFACT_CONTRACT_VERSION",
    "assert_token_contract_matches",
    "read_token_contract",
    "stable_vocab_sha256",
    "token_contract_matches",
    "token_contract_path",
    "token_contract_payload",
    "write_token_contract",
]
