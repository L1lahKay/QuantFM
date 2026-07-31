"""
Pure, versioned stock-day pooling layout definitions.

This module intentionally does not import Torch.  Representation contracts and
artifact validation can therefore derive embedding widths without loading the
model runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

FLAT_POOLING_VERSION = "flat_v1"
LEGACY_MULTI_SCALE_POOLING_VERSION = "hierarchical_v1"
MULTI_SCALE_POOLING_VERSION = "hierarchical_selected_v2"

MULTISCALE_VECTOR_NAMES: tuple[str, ...] = (
    "mean_all",
    "last_256",
    "last_1024",
    "open_call",
    "continuous_am",
    "continuous_pm",
    "close_call",
    "close_30m",
)

# This is the four-route representation declared by the V2 configs.  The
# legacy implementation ignored that declaration and emitted all eight vectors
# plus a raw event-count scalar.
DEFAULT_V2_MULTI_SCALE_OUTPUTS: tuple[str, ...] = (
    "mean_all",
    "last_256",
    "continuous_pm",
    "close_30m",
)
LEGACY_MULTI_SCALE_SCALARS: tuple[str, ...] = ("raw_event_count",)

FLAT_POOLING_METHODS = frozenset({"mean", "last", "lastk_mean"})
POOLING_METHODS = frozenset((*FLAT_POOLING_METHODS, "multi_scale"))


@dataclass(frozen=True, slots=True)
class PoolingSpec:
    """Fully resolved pooling layout used to derive output width."""

    version: str
    method: str
    vector_components: tuple[str, ...]
    scalar_components: tuple[str, ...] = ()

    def validate(self) -> None:
        """Reject ambiguous, duplicate, or version-inconsistent layouts."""
        if self.method not in POOLING_METHODS:
            msg = f"unsupported embedding pooling: {self.method!r}"
            raise ValueError(msg)
        if len(set(self.vector_components)) != len(self.vector_components):
            msg = "pooling vector components must be unique"
            raise ValueError(msg)
        if len(set(self.scalar_components)) != len(self.scalar_components):
            msg = "pooling scalar components must be unique"
            raise ValueError(msg)

        if self.method in FLAT_POOLING_METHODS:
            if self.version != FLAT_POOLING_VERSION:
                msg = (
                    f"flat pooling requires version {FLAT_POOLING_VERSION!r}, "
                    f"got {self.version!r}"
                )
                raise ValueError(msg)
            if self.vector_components != (self.method,) or self.scalar_components:
                msg = "flat pooling must emit exactly its one vector and no scalars"
                raise ValueError(msg)
            return

        invalid = set(self.vector_components) - set(MULTISCALE_VECTOR_NAMES)
        if invalid or not self.vector_components:
            msg = f"invalid multi-scale pooling vector components: {sorted(invalid)}"
            raise ValueError(msg)
        if self.version == LEGACY_MULTI_SCALE_POOLING_VERSION:
            if self.vector_components != MULTISCALE_VECTOR_NAMES:
                msg = "legacy hierarchical_v1 must emit all eight vectors"
                raise ValueError(msg)
            if self.scalar_components != LEGACY_MULTI_SCALE_SCALARS:
                msg = "legacy hierarchical_v1 must emit raw_event_count"
                raise ValueError(msg)
        elif self.version == MULTI_SCALE_POOLING_VERSION:
            if self.scalar_components:
                msg = "hierarchical_selected_v2 does not emit unscaled scalar columns"
                raise ValueError(msg)
        else:
            msg = f"unsupported multi-scale pooling version: {self.version!r}"
            raise ValueError(msg)

    def embedding_width(self, encoder_width: int) -> int:
        """Return ``n_vectors*d_model + n_scalars`` after validation."""
        self.validate()
        if encoder_width < 1:
            msg = "encoder_width must be positive"
            raise ValueError(msg)
        return len(self.vector_components) * encoder_width + len(self.scalar_components)


def resolve_pooling_spec(
    method: str,
    *,
    configured_version: str | None = None,
    configured_outputs: tuple[str, ...] | list[str] = (),
) -> PoolingSpec:
    """
    Resolve an actual extraction method into an unambiguous layout.

    ``hierarchical_v1`` is retained only to interpret checkpoints that already
    froze the historical 8-vector + raw-count behavior.  New multi-scale
    checkpoints must use ``hierarchical_selected_v2`` and persist their ordered
    output list.
    """
    if method in FLAT_POOLING_METHODS:
        spec = PoolingSpec(FLAT_POOLING_VERSION, method, (method,))
        spec.validate()
        return spec
    if method != "multi_scale":
        msg = f"unsupported embedding pooling: {method!r}"
        raise ValueError(msg)

    version = configured_version or MULTI_SCALE_POOLING_VERSION
    outputs = tuple(str(value) for value in configured_outputs)
    if version == LEGACY_MULTI_SCALE_POOLING_VERSION:
        if outputs:
            msg = (
                "hierarchical_v1 checkpoint/config is ambiguous: it declares "
                "selected outputs although the legacy extractor emitted all eight "
                "vectors plus raw_event_count; use hierarchical_selected_v2 for "
                "new artifacts"
            )
            raise ValueError(msg)
        spec = PoolingSpec(
            version,
            method,
            MULTISCALE_VECTOR_NAMES,
            LEGACY_MULTI_SCALE_SCALARS,
        )
    elif version == MULTI_SCALE_POOLING_VERSION:
        if not outputs:
            msg = "hierarchical_selected_v2 requires an explicit ordered outputs list"
            raise ValueError(msg)
        spec = PoolingSpec(version, method, outputs)
    else:
        msg = f"unsupported multi-scale pooling version: {version!r}"
        raise ValueError(msg)
    spec.validate()
    return spec
