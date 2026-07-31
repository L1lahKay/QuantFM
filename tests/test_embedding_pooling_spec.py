from __future__ import annotations

import pytest

from quant_fm.embedding.pooling_spec import (
    DEFAULT_V2_MULTI_SCALE_OUTPUTS,
    LEGACY_MULTI_SCALE_POOLING_VERSION,
    MULTI_SCALE_POOLING_VERSION,
    MULTISCALE_VECTOR_NAMES,
    resolve_pooling_spec,
)


def test_selected_v2_width_is_derived_from_declared_outputs() -> None:
    spec = resolve_pooling_spec(
        "multi_scale",
        configured_version=MULTI_SCALE_POOLING_VERSION,
        configured_outputs=DEFAULT_V2_MULTI_SCALE_OUTPUTS,
    )

    assert spec.vector_components == DEFAULT_V2_MULTI_SCALE_OUTPUTS
    assert spec.scalar_components == ()
    assert spec.embedding_width(1024) == 4096


def test_legacy_multiscale_layout_remains_explicit() -> None:
    spec = resolve_pooling_spec(
        "multi_scale",
        configured_version=LEGACY_MULTI_SCALE_POOLING_VERSION,
    )

    assert spec.vector_components == MULTISCALE_VECTOR_NAMES
    assert spec.scalar_components == ("raw_event_count",)
    assert spec.embedding_width(1024) == 8193


def test_ambiguous_legacy_selected_outputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_pooling_spec(
            "multi_scale",
            configured_version=LEGACY_MULTI_SCALE_POOLING_VERSION,
            configured_outputs=DEFAULT_V2_MULTI_SCALE_OUTPUTS,
        )


def test_selected_v2_rejects_unknown_or_duplicate_outputs() -> None:
    with pytest.raises(ValueError, match="invalid multi-scale"):
        resolve_pooling_spec(
            "multi_scale",
            configured_version=MULTI_SCALE_POOLING_VERSION,
            configured_outputs=("mean_all", "not_real"),
        )
    with pytest.raises(ValueError, match="must be unique"):
        resolve_pooling_spec(
            "multi_scale",
            configured_version=MULTI_SCALE_POOLING_VERSION,
            configured_outputs=("mean_all", "mean_all"),
        )
