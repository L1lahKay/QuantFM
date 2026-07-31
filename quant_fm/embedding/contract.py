"""股日 embedding 在训练与生产评分之间的不可变表示契约。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from quant_fm.embedding.pooling_spec import PoolingSpec

EMBEDDING_CONTRACT_VERSION = "stock_day_embedding_v2"
AFTER_CLOSE_AVAILABILITY = "available_after_signal_date_close"
CAUSAL_CHUNKED_ENCODER = "causal_independent_chunks_v1"
CAUSAL_OVERLAPPING_ENCODER = "causal_overlap_unique_emit_v2"
STOCK_DAY_GRANULARITY = "stock_day"
STRICT_EVENT_ORDERING_VERSION = "exchange_time_sequence_v2"
STRICT_FEATURE_TRANSFORM_VERSION = "ew_vwap_causal_nan_v2"


def encoder_semantics_for(context: int, chunk_stride: int) -> str:
    """Return the exact cross-window context semantics for extraction."""
    if context < 1:
        msg = "embedding context must be positive"
        raise ValueError(msg)
    if not 1 <= chunk_stride <= context:
        msg = "embedding chunk_stride must be in [1, context]"
        raise ValueError(msg)
    return (
        CAUSAL_CHUNKED_ENCODER
        if chunk_stride == context
        else CAUSAL_OVERLAPPING_ENCODER
    )


def embedding_contract_path(embedding_path: Path) -> Path:
    """返回 embedding parquet 对应的稳定 sidecar 路径。"""
    path = Path(embedding_path)
    return path.with_name(f"{path.name}.contract.json")


def validate_strict_causal_representation(
    contract: EmbeddingContract,
    *,
    require_overlap: bool = True,
) -> None:
    """
    Reject representations built from legacy token or chunk semantics.

    The ordinary contract loader intentionally permits an auditable V2 sidecar
    that records legacy token transforms or independent chunks.  Production
    strict paths should additionally call this helper.
    """
    contract.validate(require_vocab=True)
    mismatches: dict[str, dict[str, object]] = {}
    expected_versions = {
        "event_ordering_version": STRICT_EVENT_ORDERING_VERSION,
        "feature_transform_version": STRICT_FEATURE_TRANSFORM_VERSION,
    }
    for field, expected in expected_versions.items():
        actual = getattr(contract, field)
        if actual != expected:
            mismatches[field] = {"expected": expected, "actual": actual}
    if require_overlap and contract.encoder_semantics != CAUSAL_OVERLAPPING_ENCODER:
        mismatches["encoder_semantics"] = {
            "expected": CAUSAL_OVERLAPPING_ENCODER,
            "actual": contract.encoder_semantics,
        }
    if mismatches:
        msg = f"embedding is not a strict causal representation: {mismatches}"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class EmbeddingContract:
    """描述不会随日期、split 或 shard 改变的 embedding 语义。"""

    format_version: str
    fm_checkpoint_sha256: str
    vocab_sha256: str | None
    schema_version: str
    book_state_timing: str
    pooling_version: str
    granularity: str
    context: int
    chunk_stride: int
    pooling: str
    last_k: int
    dtype: str
    encoder_width: int
    pooling_components: tuple[str, ...]
    pooling_scalar_components: tuple[str, ...]
    embedding_columns: tuple[str, ...]
    embedding_width: int
    signal_availability: str
    encoder_semantics: str
    event_ordering_version: str
    feature_transform_version: str

    def validate(self, *, require_vocab: bool = False) -> None:
        """拒绝不完整、含糊或内部矛盾的表示声明。"""
        if self.format_version != EMBEDDING_CONTRACT_VERSION:
            msg = f"unsupported embedding contract version: {self.format_version!r}"
            raise ValueError(msg)
        if not self.fm_checkpoint_sha256:
            msg = "embedding contract is missing fm_checkpoint_sha256"
            raise ValueError(msg)
        if require_vocab and not self.vocab_sha256:
            msg = "strict embedding contract is missing vocab_sha256"
            raise ValueError(msg)
        if not self.schema_version:
            msg = "embedding contract is missing schema_version"
            raise ValueError(msg)
        if not self.book_state_timing or not self.pooling_version:
            msg = "embedding contract is missing book/pooling version metadata"
            raise ValueError(msg)
        if self.granularity != STOCK_DAY_GRANULARITY:
            msg = f"unsupported embedding granularity: {self.granularity!r}"
            raise ValueError(msg)
        if self.context < 1 or self.last_k < 1 or self.encoder_width < 1:
            msg = "embedding context, last_k and encoder_width must be positive"
            raise ValueError(msg)
        expected_encoder_semantics = encoder_semantics_for(
            self.context,
            self.chunk_stride,
        )
        if self.encoder_semantics != expected_encoder_semantics:
            msg = (
                "embedding encoder_semantics does not match context/chunk_stride: "
                f"expected={expected_encoder_semantics!r}, "
                f"actual={self.encoder_semantics!r}"
            )
            raise ValueError(msg)
        pooling_spec = PoolingSpec(
            version=self.pooling_version,
            method=self.pooling,
            vector_components=self.pooling_components,
            scalar_components=self.pooling_scalar_components,
        )
        expected_width = pooling_spec.embedding_width(self.encoder_width)
        if self.embedding_width != expected_width:
            msg = (
                "embedding_width does not match the versioned pooling layout: "
                f"expected={expected_width}, actual={self.embedding_width}"
            )
            raise ValueError(msg)
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            msg = f"unsupported embedding dtype: {self.dtype!r}"
            raise ValueError(msg)
        if self.embedding_width != len(self.embedding_columns):
            msg = "embedding_width does not match embedding_columns"
            raise ValueError(msg)
        expected_columns = tuple(
            f"emb_{index}" for index in range(self.embedding_width)
        )
        if self.embedding_columns != expected_columns:
            msg = "embedding columns must be ordered contiguously from emb_0"
            raise ValueError(msg)
        if self.signal_availability != AFTER_CLOSE_AVAILABILITY:
            msg = (
                "unsupported embedding signal availability: "
                f"{self.signal_availability!r}"
            )
            raise ValueError(msg)
        if not self.event_ordering_version or not self.feature_transform_version:
            msg = "embedding contract is missing token data-semantics versions"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """返回稳定、JSON 可序列化的字典。"""
        payload = asdict(self)
        payload["pooling_components"] = list(self.pooling_components)
        payload["pooling_scalar_components"] = list(self.pooling_scalar_components)
        payload["embedding_columns"] = list(self.embedding_columns)
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        require_vocab: bool = False,
    ) -> EmbeddingContract:
        """从 sidecar/payload 重建并校验契约。"""
        required = {
            "format_version",
            "fm_checkpoint_sha256",
            "vocab_sha256",
            "schema_version",
            "book_state_timing",
            "pooling_version",
            "granularity",
            "context",
            "chunk_stride",
            "pooling",
            "last_k",
            "dtype",
            "encoder_width",
            "pooling_components",
            "pooling_scalar_components",
            "embedding_columns",
            "embedding_width",
            "signal_availability",
            "encoder_semantics",
            "event_ordering_version",
            "feature_transform_version",
        }
        version = payload.get("format_version")
        if version != EMBEDDING_CONTRACT_VERSION:
            msg = (
                f"unsupported embedding contract version: {version!r}; old sidecars "
                "cannot be safely upgraded because they do not prove stride, pooling "
                "layout, or token data semantics; regenerate embeddings with "
                "quant_fm.embedding.extract_hidden"
            )
            raise ValueError(msg)
        missing = sorted(required - set(payload))
        if missing:
            msg = f"embedding contract is missing fields: {missing}"
            raise ValueError(msg)
        contract = cls(
            format_version=str(payload["format_version"]),
            fm_checkpoint_sha256=str(payload["fm_checkpoint_sha256"]),
            vocab_sha256=(
                str(payload["vocab_sha256"])
                if payload["vocab_sha256"] is not None
                else None
            ),
            schema_version=str(payload["schema_version"]),
            book_state_timing=str(payload["book_state_timing"]),
            pooling_version=str(payload["pooling_version"]),
            granularity=str(payload["granularity"]),
            context=int(payload["context"]),
            chunk_stride=int(payload["chunk_stride"]),
            pooling=str(payload["pooling"]),
            last_k=int(payload["last_k"]),
            dtype=str(payload["dtype"]),
            encoder_width=int(payload["encoder_width"]),
            pooling_components=tuple(
                str(value) for value in payload["pooling_components"]
            ),
            pooling_scalar_components=tuple(
                str(value) for value in payload["pooling_scalar_components"]
            ),
            embedding_columns=tuple(
                str(value) for value in payload["embedding_columns"]
            ),
            embedding_width=int(payload["embedding_width"]),
            signal_availability=str(payload["signal_availability"]),
            encoder_semantics=str(payload["encoder_semantics"]),
            event_ordering_version=str(payload["event_ordering_version"]),
            feature_transform_version=str(payload["feature_transform_version"]),
        )
        contract.validate(require_vocab=require_vocab)
        return contract

    def fingerprint(self) -> str:
        """返回可进入 Ranker cache key 的完整表示指纹。"""
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


def write_embedding_contract(
    embedding_path: Path,
    contract: EmbeddingContract,
) -> Path:
    """原子写入 embedding sidecar。"""
    contract.validate()
    destination = embedding_contract_path(embedding_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(contract.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_embedding_contract(
    embedding_path: Path,
    *,
    required: bool = True,
    require_vocab: bool = False,
) -> EmbeddingContract | None:
    """读取 sidecar；严格路径默认拒绝旧 embedding。"""
    path = embedding_contract_path(embedding_path)
    if not path.is_file():
        if required:
            msg = (
                f"embedding representation contract is missing: {path}; "
                "a sidecar cannot be safely synthesized from column width; regenerate "
                "the embedding, or use an explicit legacy compatibility override"
            )
            raise ValueError(msg)
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"invalid embedding representation contract: {path}"
        raise ValueError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"embedding representation contract must be a JSON object: {path}"
        raise TypeError(msg)
    return EmbeddingContract.from_dict(payload, require_vocab=require_vocab)


def validate_embedding_columns(
    column_names: list[str] | tuple[str, ...],
    contract: EmbeddingContract,
    *,
    context: str,
) -> None:
    """交叉校验 parquet 的实际有序 embedding 列与 sidecar。"""
    actual = tuple(name for name in column_names if name.startswith("emb_"))
    if actual != contract.embedding_columns:
        msg = (
            f"{context} embedding columns do not match representation contract: "
            f"expected={list(contract.embedding_columns)}, actual={list(actual)}"
        )
        raise ValueError(msg)


def assert_embedding_contract_compatible(
    expected: EmbeddingContract,
    actual: EmbeddingContract,
    *,
    context: str,
) -> None:
    """要求训练和评分使用完全相同的表示语义。"""
    expected_payload = expected.to_dict()
    actual_payload = actual.to_dict()
    mismatches = {
        field: {"expected": expected_payload[field], "actual": actual_payload[field]}
        for field in expected_payload
        if expected_payload[field] != actual_payload[field]
    }
    if mismatches:
        msg = f"{context} embedding representation mismatch: {mismatches}"
        raise ValueError(msg)


def load_compatible_embedding_contracts(
    embedding_paths: list[Path],
    *,
    required: bool = True,
    require_vocab: bool = False,
    context: str,
) -> EmbeddingContract | None:
    """读取多个分片/split 契约并要求表示语义一致。"""
    if not embedding_paths:
        msg = f"{context} requires at least one embedding path"
        raise ValueError(msg)
    common: EmbeddingContract | None = None
    missing = False
    for path in embedding_paths:
        contract = load_embedding_contract(
            path,
            required=required,
            require_vocab=require_vocab,
        )
        if contract is None:
            missing = True
            continue
        validate_embedding_columns(
            list(_parquet_column_names(path)),
            contract,
            context=str(path),
        )
        if common is None:
            common = contract
        else:
            assert_embedding_contract_compatible(
                common,
                contract,
                context=context,
            )
    return None if missing else common


def propagate_embedding_contract(
    input_paths: list[Path],
    output_path: Path,
    *,
    context: str,
) -> EmbeddingContract:
    """校验一组合并输入并把共同契约原子传播到输出。"""
    contract = load_compatible_embedding_contracts(
        input_paths,
        required=True,
        context=context,
    )
    if contract is None:  # pragma: no cover - required=True invariant
        msg = f"{context} did not produce an embedding representation contract"
        raise RuntimeError(msg)
    validate_embedding_columns(
        list(_parquet_column_names(output_path)),
        contract,
        context=str(output_path),
    )
    write_embedding_contract(output_path, contract)
    return contract


def _parquet_column_names(path: Path) -> tuple[str, ...]:
    """延迟导入 Polars，避免仅处理 JSON 时加载 dataframe 依赖。"""
    import polars as pl

    return tuple(pl.read_parquet_schema(path).names())
