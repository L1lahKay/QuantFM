"""Strict frozen-FM representation gate for the production Top-K ranker."""

from __future__ import annotations

from typing import TYPE_CHECKING

from quant_fm.embedding.contract import validate_strict_causal_representation
from quant_fm.embedding.pooling_spec import DEFAULT_V2_MULTI_SCALE_OUTPUTS

if TYPE_CHECKING:
    from quant_fm.embedding.contract import EmbeddingContract

STRICT_POOLING_VERSION = "hierarchical_selected_v2"
STRICT_CONTEXT = 2048
STRICT_CHUNK_STRIDE = 512
STRICT_SCHEMA_VERSION = "cn_l2_v2"
STRICT_BOOK_STATE_TIMING = "post_event"


def validate_strict_topk_representation(
    contract: EmbeddingContract,
    *,
    context: str,
) -> dict[str, str | int | bool]:
    """Reject legacy/non-overlapping embeddings from strict Top-K workflows."""
    try:
        validate_strict_causal_representation(contract, require_overlap=True)
    except ValueError as exc:
        msg = (
            f"{context} is not a strict causal Top-K representation: {exc}; "
            "regenerate V2 embeddings or use the explicit legacy/diagnostic override"
        )
        raise ValueError(msg) from exc
    violations: list[str] = []
    if contract.schema_version != STRICT_SCHEMA_VERSION:
        violations.append(
            f"schema_version={contract.schema_version!r} must equal "
            f"{STRICT_SCHEMA_VERSION!r}"
        )
    if contract.book_state_timing != STRICT_BOOK_STATE_TIMING:
        violations.append(
            f"book_state_timing={contract.book_state_timing!r} must equal "
            f"{STRICT_BOOK_STATE_TIMING!r}"
        )
    if contract.context != STRICT_CONTEXT:
        violations.append(f"context={contract.context} must equal {STRICT_CONTEXT}")
    if contract.chunk_stride != STRICT_CHUNK_STRIDE:
        violations.append(
            f"chunk_stride={contract.chunk_stride} must equal {STRICT_CHUNK_STRIDE}"
        )
    if contract.pooling_version != STRICT_POOLING_VERSION:
        violations.append(f"pooling_version={contract.pooling_version!r}")
    if contract.pooling_components != DEFAULT_V2_MULTI_SCALE_OUTPUTS:
        violations.append(
            f"pooling_components={contract.pooling_components!r} must equal "
            f"{DEFAULT_V2_MULTI_SCALE_OUTPUTS!r}"
        )
    if contract.pooling_scalar_components:
        violations.append(
            f"pooling_scalar_components={contract.pooling_scalar_components!r}"
        )
    if violations:
        msg = (
            f"{context} is not a strict causal Top-K representation: "
            f"{', '.join(violations)}; regenerate V2 embeddings or use the explicit "
            "legacy/diagnostic override"
        )
        raise ValueError(msg)
    return {
        "verified": True,
        "format_version": contract.format_version,
        "schema_version": contract.schema_version,
        "book_state_timing": contract.book_state_timing,
        "event_ordering_version": contract.event_ordering_version,
        "feature_transform_version": contract.feature_transform_version,
        "encoder_semantics": contract.encoder_semantics,
        "context": contract.context,
        "chunk_stride": contract.chunk_stride,
        "pooling_version": contract.pooling_version,
    }
