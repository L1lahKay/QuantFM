from __future__ import annotations

import pytest

from quant_fm.downstream.representation import (
    STRICT_BOOK_STATE_TIMING,
    STRICT_CHUNK_STRIDE,
    STRICT_CONTEXT,
    STRICT_SCHEMA_VERSION,
    validate_strict_topk_representation,
)
from quant_fm.embedding.contract import (
    AFTER_CLOSE_AVAILABILITY,
    CAUSAL_CHUNKED_ENCODER,
    CAUSAL_OVERLAPPING_ENCODER,
    EMBEDDING_CONTRACT_VERSION,
    STOCK_DAY_GRANULARITY,
    STRICT_EVENT_ORDERING_VERSION,
    STRICT_FEATURE_TRANSFORM_VERSION,
    EmbeddingContract,
)
from quant_fm.embedding.pooling_spec import DEFAULT_V2_MULTI_SCALE_OUTPUTS


def _strict_contract() -> EmbeddingContract:
    width = len(DEFAULT_V2_MULTI_SCALE_OUTPUTS) * 2
    return EmbeddingContract(
        format_version=EMBEDDING_CONTRACT_VERSION,
        fm_checkpoint_sha256="a" * 64,
        vocab_sha256="b" * 64,
        schema_version=STRICT_SCHEMA_VERSION,
        book_state_timing=STRICT_BOOK_STATE_TIMING,
        pooling_version="hierarchical_selected_v2",
        granularity=STOCK_DAY_GRANULARITY,
        context=STRICT_CONTEXT,
        chunk_stride=STRICT_CHUNK_STRIDE,
        pooling="multi_scale",
        last_k=256,
        dtype="bf16",
        encoder_width=2,
        pooling_components=DEFAULT_V2_MULTI_SCALE_OUTPUTS,
        pooling_scalar_components=(),
        embedding_columns=tuple(f"emb_{index}" for index in range(width)),
        embedding_width=width,
        signal_availability=AFTER_CLOSE_AVAILABILITY,
        encoder_semantics=CAUSAL_OVERLAPPING_ENCODER,
        event_ordering_version=STRICT_EVENT_ORDERING_VERSION,
        feature_transform_version=STRICT_FEATURE_TRANSFORM_VERSION,
    )


def test_strict_topk_representation_accepts_causal_overlap_and_selected_pooling() -> (
    None
):
    result = validate_strict_topk_representation(
        _strict_contract(),
        context="training embeddings",
    )

    assert result["verified"]
    assert result["schema_version"] == STRICT_SCHEMA_VERSION
    assert result["book_state_timing"] == STRICT_BOOK_STATE_TIMING
    assert result["context"] == STRICT_CONTEXT
    assert result["chunk_stride"] == STRICT_CHUNK_STRIDE
    assert result["pooling_version"] == "hierarchical_selected_v2"


def test_strict_topk_representation_rejects_independent_chunks() -> None:
    contract = _strict_contract()
    legacy_chunking = EmbeddingContract.from_dict(
        {
            **contract.to_dict(),
            "chunk_stride": contract.context,
            "encoder_semantics": CAUSAL_CHUNKED_ENCODER,
        },
        require_vocab=True,
    )

    with pytest.raises(ValueError, match="strict causal Top-K"):
        validate_strict_topk_representation(
            legacy_chunking,
            context="training embeddings",
        )


@pytest.mark.parametrize(
    ("updates", "expected_message"),
    [
        ({"context": 4096}, "context=4096 must equal 2048"),
        ({"chunk_stride": 1024}, "chunk_stride=1024 must equal 512"),
        ({"schema_version": "cn_l2_v1"}, "schema_version='cn_l2_v1'"),
        ({"book_state_timing": "pre_event"}, "book_state_timing='pre_event'"),
    ],
)
def test_strict_topk_representation_locks_exact_context_and_stride(
    updates: dict[str, int | str],
    expected_message: str,
) -> None:
    contract = _strict_contract()
    changed = EmbeddingContract.from_dict(
        {**contract.to_dict(), **updates},
        require_vocab=True,
    )

    with pytest.raises(ValueError, match=expected_message):
        validate_strict_topk_representation(
            changed,
            context="training embeddings",
        )
