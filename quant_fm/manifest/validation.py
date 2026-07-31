"""Runtime validation for manifest, vocabulary, and token-shard provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import pyarrow.parquet as pq

from quant_fm.tokenizer.artifact_contract import (
    assert_token_contract_matches,
    stable_vocab_sha256,
    token_contract_path,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_fm.manifest.build_manifest import Manifest, ShardEntry


class ManifestVocab(Protocol):
    """Vocabulary surface needed for manifest/runtime validation."""

    schema_version: str
    event_ordering_version: str
    feature_transform_version: str
    data_semantics_explicit: bool

    def to_json(self) -> str:
        """Return the stable vocabulary serialization."""


def sha256_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    """Return a streaming SHA-256 for a materialized artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest_vocab_contract(
    manifest: Manifest,
    vocab: ManifestVocab,
    *,
    context: str,
) -> dict[str, object]:
    """
    Require manifest top-level semantics to agree with the loaded vocab.

    Historical vocabularies without explicit semantic fields remain legacy-only:
    a historical manifest may omit the corresponding fields, but it may never
    claim a different (especially causal) version.
    """
    mismatches: dict[str, tuple[object, object]] = {}
    if manifest.schema_version != vocab.schema_version:
        mismatches["schema_version"] = (
            manifest.schema_version,
            vocab.schema_version,
        )
    manifest_vocab_hash = manifest.vocab_sha256
    expected_vocab_hash = stable_vocab_sha256(vocab)
    if vocab.data_semantics_explicit:
        vocab_hash_compatible = manifest_vocab_hash == expected_vocab_hash
    else:
        vocab_hash_compatible = manifest_vocab_hash in {None, expected_vocab_hash}
    if not vocab_hash_compatible:
        mismatches["vocab_sha256"] = (manifest_vocab_hash, expected_vocab_hash)
    for field in ("event_ordering_version", "feature_transform_version"):
        manifest_value = getattr(manifest, field)
        vocab_value = getattr(vocab, field)
        if vocab.data_semantics_explicit:
            compatible = manifest_value == vocab_value
        else:
            compatible = manifest_value in {None, vocab_value}
        if not compatible:
            mismatches[field] = (manifest_value, vocab_value)
    if mismatches:
        msg = f"{context} manifest/vocab contract mismatch: {mismatches}"
        raise ValueError(msg)
    return {
        "schema_version": vocab.schema_version,
        "event_ordering_version": vocab.event_ordering_version,
        "feature_transform_version": vocab.feature_transform_version,
        "data_semantics_explicit": vocab.data_semantics_explicit,
    }


def validate_manifest_shards(
    manifest: Manifest,
    vocab: ManifestVocab,
    *,
    shards: Sequence[ShardEntry] | None = None,
    context: str,
) -> dict[str, int]:
    """
    Validate every selected shard against live bytes, sidecar, and vocab.

    Strict/causal manifests carry both parquet and sidecar hashes: the sidecar
    binds decoding semantics, while only the parquet hash binds the token bytes.
    Sidecar-less historical shards therefore remain legacy-only; legacy diagnostic
    manifests may omit the parquet hash only when another auditable identity exists.
    """
    selected = list(manifest.shards if shards is None else shards)
    unhashed_parquet = 0
    for shard in selected:
        path = Path(shard.path)
        if not path.is_file():
            msg = f"{context} token shard is missing: {path}"
            raise FileNotFoundError(msg)
        parquet = pq.ParquetFile(path)
        actual_rows = int(parquet.metadata.num_rows)
        if actual_rows != shard.rows:
            msg = (
                f"{context} token row-count mismatch for {path}: "
                f"manifest={shard.rows}, actual={actual_rows}"
            )
            raise ValueError(msg)
        if shard.sha256:
            actual_parquet_hash = sha256_file(path)
            if actual_parquet_hash != shard.sha256:
                msg = (
                    f"{context} token parquet SHA-256 mismatch for {path}: "
                    f"manifest={shard.sha256}, actual={actual_parquet_hash}"
                )
                raise ValueError(msg)
        else:
            if vocab.data_semantics_explicit:
                msg = (
                    f"{context} explicit token shard lacks a full manifest-recorded "
                    f"parquet SHA-256: {path}"
                )
                raise ValueError(msg)
            unhashed_parquet += 1

        assert_token_contract_matches(path, vocab)
        sidecar = token_contract_path(path)
        if shard.data_contract_sha256:
            if not sidecar.is_file():
                msg = f"{context} token sidecar is missing: {sidecar}"
                raise ValueError(msg)
            actual_contract_hash = sha256_file(sidecar)
            if actual_contract_hash != shard.data_contract_sha256:
                msg = (
                    f"{context} token sidecar SHA-256 mismatch for {path}: "
                    f"manifest={shard.data_contract_sha256}, "
                    f"actual={actual_contract_hash}"
                )
                raise ValueError(msg)
        elif vocab.data_semantics_explicit:
            msg = (
                f"{context} explicit token shard lacks a manifest-recorded sidecar "
                f"identity: {path}"
            )
            raise ValueError(msg)

        if not shard.sha256 and not shard.data_contract_sha256:
            msg = f"{context} token shard has no auditable source identity: {path}"
            raise ValueError(msg)

    return {
        "validated_shards": len(selected),
        "unhashed_parquet_shards": unhashed_parquet,
    }


__all__ = [
    "sha256_file",
    "validate_manifest_shards",
    "validate_manifest_vocab_contract",
]
