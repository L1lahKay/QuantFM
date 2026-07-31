"""V2 token payload 的无损窄 token 与对称 Q16 scalar 存储编码。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import polars as pl

from quant_fm.tokenizer.artifact_contract import stable_vocab_sha256

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quant_fm.tokenizer.vocab_v2 import VocabV2

STORAGE_ENCODING_VERSION_V2 = "token_uint_scalar_q16_v1"
STORAGE_ENCODING_SCHEME_V2 = "vocab_uint_and_symmetric_int16"
Q16_MAX = 32767

_TOKEN_DTYPES: dict[str, pl.DataType] = {
    "uint8": pl.UInt8,
    "uint16": pl.UInt16,
}


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    """返回用于契约哈希的稳定 JSON 字节。"""
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def vocab_sha256_v2(vocab: VocabV2) -> str:
    """哈希完整冻结 V2 vocab，而不是依赖文件路径或格式化方式。"""
    return stable_vocab_sha256(vocab)


def _validate_sha256(value: str, *, name: str) -> None:
    """拒绝不完整或非十六进制的 SHA-256。"""
    if len(value) != 64:
        msg = f"{name} must be a 64-character SHA-256"
        raise ValueError(msg)
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        msg = f"{name} must be hexadecimal"
        raise ValueError(msg) from exc


@dataclass(frozen=True, slots=True)
class TokenStorageFieldV2:
    """一个 token 列的冻结存储宽度与合法 id 空间。"""

    column: str
    storage_dtype: str
    vocab_size: int

    def __post_init__(self) -> None:
        """要求 dtype 是容纳完整 vocab 的最窄无符号整数。"""
        if not self.column.startswith("tok_"):
            msg = f"token storage column must start with 'tok_': {self.column!r}"
            raise ValueError(msg)
        if not 1 <= self.vocab_size <= 65536:
            msg = f"token vocab_size must be in [1, 65536]: {self.column!r}"
            raise ValueError(msg)
        expected = "uint8" if self.vocab_size <= 256 else "uint16"
        if self.storage_dtype != expected:
            msg = (
                f"token storage dtype for {self.column!r} must be {expected}, "
                f"got {self.storage_dtype}"
            )
            raise ValueError(msg)

    def to_dict(self) -> dict[str, str | int]:
        """返回稳定 JSON 字段描述。"""
        return {
            "column": self.column,
            "storage_dtype": self.storage_dtype,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TokenStorageFieldV2:
        """从 contract payload 恢复并校验 token 字段。"""
        return cls(
            column=str(payload["column"]),
            storage_dtype=str(payload["storage_dtype"]),
            vocab_size=int(payload["vocab_size"]),
        )


@dataclass(frozen=True, slots=True)
class ScalarStorageFieldV2:
    """一个冻结 normalizer clip 对应的对称 Int16 编码。"""

    column: str
    clip: float
    storage_dtype: str = "int16"
    decoded_dtype: str = "float32"
    quantization: str = "symmetric"
    quant_max: int = Q16_MAX

    def __post_init__(self) -> None:
        """拒绝不能被稳定反量化的 scalar 描述。"""
        if not self.column.startswith("val_"):
            msg = f"scalar storage column must start with 'val_': {self.column!r}"
            raise ValueError(msg)
        if not np.isfinite(self.clip) or self.clip <= 0:
            msg = f"scalar clip must be finite and positive: {self.column!r}"
            raise ValueError(msg)
        if self.storage_dtype != "int16" or self.decoded_dtype != "float32":
            msg = f"unsupported scalar dtype contract for {self.column!r}"
            raise ValueError(msg)
        if self.quantization != "symmetric" or self.quant_max != Q16_MAX:
            msg = f"unsupported scalar quantization contract for {self.column!r}"
            raise ValueError(msg)

    @property
    def scale(self) -> float:
        """返回一个整数步长对应的标准化标量宽度。"""
        return self.clip / self.quant_max

    def to_dict(self) -> dict[str, str | float | int]:
        """返回稳定 JSON 字段描述。"""
        return {
            "column": self.column,
            "storage_dtype": self.storage_dtype,
            "decoded_dtype": self.decoded_dtype,
            "quantization": self.quantization,
            "quant_max": self.quant_max,
            "clip": self.clip,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScalarStorageFieldV2:
        """从 contract payload 恢复并交叉验证派生 scale。"""
        field = cls(
            column=str(payload["column"]),
            clip=float(payload["clip"]),
            storage_dtype=str(payload["storage_dtype"]),
            decoded_dtype=str(payload["decoded_dtype"]),
            quantization=str(payload["quantization"]),
            quant_max=int(payload["quant_max"]),
        )
        declared_scale = float(payload["scale"])
        if not np.isfinite(declared_scale) or not np.isclose(
            declared_scale,
            field.scale,
            rtol=0.0,
            atol=np.finfo(np.float64).eps * max(1.0, abs(field.scale)),
        ):
            msg = f"declared Q16 scale disagrees with clip: {field.column!r}"
            raise ValueError(msg)
        return field


@dataclass(frozen=True, slots=True)
class StorageEncodingMetadataV2:
    """可嵌入现有 token contract 的版本化 Q16 存储元数据。"""

    schema_version: str
    vocab_sha256: str
    token_fields: tuple[TokenStorageFieldV2, ...]
    scalar_fields: tuple[ScalarStorageFieldV2, ...]
    format_version: str = STORAGE_ENCODING_VERSION_V2
    scheme: str = STORAGE_ENCODING_SCHEME_V2

    REQUIRED_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "format_version",
            "scheme",
            "schema_version",
            "vocab_sha256",
            "token_fields",
            "scalar_fields",
            "metadata_sha256",
        }
    )

    def __post_init__(self) -> None:
        """校验版本、hash 和字段唯一性。"""
        if self.format_version != STORAGE_ENCODING_VERSION_V2:
            msg = f"unsupported storage encoding version: {self.format_version!r}"
            raise ValueError(msg)
        if self.scheme != STORAGE_ENCODING_SCHEME_V2:
            msg = f"unsupported storage encoding scheme: {self.scheme!r}"
            raise ValueError(msg)
        if not self.schema_version:
            msg = "storage encoding schema_version must not be empty"
            raise ValueError(msg)
        _validate_sha256(self.vocab_sha256, name="vocab_sha256")
        columns = [field.column for field in (*self.token_fields, *self.scalar_fields)]
        if len(columns) != len(set(columns)):
            msg = "storage encoding contains duplicate columns"
            raise ValueError(msg)

    def _payload_without_hash(self) -> dict[str, Any]:
        """返回 metadata 自哈希覆盖的完整语义 payload。"""
        return {
            "format_version": self.format_version,
            "scheme": self.scheme,
            "schema_version": self.schema_version,
            "vocab_sha256": self.vocab_sha256,
            "token_fields": [field.to_dict() for field in self.token_fields],
            "scalar_fields": [field.to_dict() for field in self.scalar_fields],
        }

    @property
    def metadata_sha256(self) -> str:
        """返回覆盖版本、vocab、字段宽度与量化尺度的契约哈希。"""
        return hashlib.sha256(_canonical_json(self._payload_without_hash())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """生成可直接嵌入 token artifact contract 的字典。"""
        return {**self._payload_without_hash(), "metadata_sha256": self.metadata_sha256}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StorageEncodingMetadataV2:
        """从字典恢复 metadata，并拒绝缺字段、未知版本或哈希篡改。"""
        missing = sorted(cls.REQUIRED_KEYS - set(payload))
        if missing:
            msg = f"storage encoding metadata is missing fields: {missing}"
            raise ValueError(msg)
        unknown = sorted(set(payload) - cls.REQUIRED_KEYS)
        if unknown:
            msg = f"storage encoding metadata contains unknown fields: {unknown}"
            raise ValueError(msg)
        metadata = cls(
            format_version=str(payload["format_version"]),
            scheme=str(payload["scheme"]),
            schema_version=str(payload["schema_version"]),
            vocab_sha256=str(payload["vocab_sha256"]),
            token_fields=tuple(
                TokenStorageFieldV2.from_dict(field)
                for field in payload["token_fields"]
            ),
            scalar_fields=tuple(
                ScalarStorageFieldV2.from_dict(field)
                for field in payload["scalar_fields"]
            ),
        )
        declared_hash = str(payload["metadata_sha256"])
        _validate_sha256(declared_hash, name="metadata_sha256")
        if declared_hash != metadata.metadata_sha256:
            msg = "storage encoding metadata SHA-256 mismatch"
            raise ValueError(msg)
        return metadata


def build_storage_metadata_v2(vocab: VocabV2) -> StorageEncodingMetadataV2:
    """从冻结 vocab 构建跨 shard 稳定的最窄 token 与 Q16 字段契约。"""
    token_fields = tuple(
        TokenStorageFieldV2(
            column=column,
            storage_dtype="uint8" if size <= 256 else "uint16",
            vocab_size=size,
        )
        for column, size in vocab.token_field_sizes().items()
    )
    scalar_fields = tuple(
        ScalarStorageFieldV2(
            column=str(spec.value_column),
            clip=float(vocab.binned[spec.name].normalizer.clip),
        )
        for spec in vocab.field_specs
        if spec.value_column is not None
    )
    return StorageEncodingMetadataV2(
        schema_version=str(vocab.schema_version),
        vocab_sha256=vocab_sha256_v2(vocab),
        token_fields=token_fields,
        scalar_fields=scalar_fields,
    )


def assert_storage_metadata_matches_vocab_v2(
    metadata: StorageEncodingMetadataV2,
    vocab: VocabV2,
) -> None:
    """拒绝用其他 schema/vocab 的量化尺度解码当前 shard。"""
    expected = build_storage_metadata_v2(vocab)
    if metadata.to_dict() != expected.to_dict():
        msg = (
            "storage encoding metadata does not match V2 vocab: "
            f"expected={expected.metadata_sha256}, "
            f"actual={metadata.metadata_sha256}"
        )
        raise ValueError(msg)


def _validate_token_series(
    series: pl.Series,
    field: TokenStorageFieldV2,
    *,
    require_storage_dtype: bool,
) -> None:
    """在任何窄化转换前检查 token 类型、空值和完整 vocab 范围。"""
    if not series.dtype.is_integer():
        msg = f"token column must be integer: {field.column!r}"
        raise TypeError(msg)
    if series.null_count():
        msg = f"token column contains nulls: {field.column!r}"
        raise ValueError(msg)
    if require_storage_dtype and series.dtype != _TOKEN_DTYPES[field.storage_dtype]:
        msg = (
            f"encoded token dtype mismatch for {field.column!r}: "
            f"expected={field.storage_dtype}, actual={series.dtype}"
        )
        raise TypeError(msg)
    if series.len() == 0:
        return
    minimum = int(series.min())
    maximum = int(series.max())
    if minimum < 0 or maximum >= field.vocab_size:
        msg = (
            f"token ids out of vocab range for {field.column!r}: "
            f"observed=[{minimum}, {maximum}], expected=[0, {field.vocab_size - 1}]"
        )
        raise ValueError(msg)


def _quantize_scalar(series: pl.Series, field: ScalarStorageFieldV2) -> pl.Series:
    """将已标准化并 clip 的 Float scalar 映射到对称 Int16。"""
    if not series.dtype.is_float():
        msg = f"scalar column must be floating before Q16 encoding: {field.column!r}"
        raise TypeError(msg)
    if series.null_count():
        msg = f"scalar column contains nulls: {field.column!r}"
        raise ValueError(msg)
    values = series.to_numpy().astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        msg = f"scalar column contains non-finite values: {field.column!r}"
        raise ValueError(msg)
    tolerance = max(1e-6, field.clip * 1e-6)
    if values.size and np.max(np.abs(values)) > field.clip + tolerance:
        msg = (
            f"scalar column exceeds frozen normalizer clip for {field.column!r}: "
            f"clip={field.clip}"
        )
        raise ValueError(msg)
    clipped = np.clip(values, -field.clip, field.clip)
    encoded = np.rint(clipped * (field.quant_max / field.clip)).astype(np.int16)
    return pl.Series(field.column, encoded, dtype=pl.Int16)


def quantize_frame_v2(
    frame: pl.DataFrame,
    vocab: VocabV2,
    metadata: StorageEncodingMetadataV2 | None = None,
) -> tuple[pl.DataFrame, StorageEncodingMetadataV2]:
    """
    将完整 V2 token frame 编成窄 token 与 Q16 scalar。

    所有 vocab 声明字段必须存在。token 先检查完整 id 范围再窄化，避免无符号
    cast 环绕；scalar 必须是 tokenizer 输出的有限 Float 且位于冻结 clip 内。
    """
    contract = metadata or build_storage_metadata_v2(vocab)
    assert_storage_metadata_matches_vocab_v2(contract, vocab)
    required = {
        field.column for field in (*contract.token_fields, *contract.scalar_fields)
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        msg = f"V2 storage encoding frame is missing columns: {missing}"
        raise ValueError(msg)

    expressions: list[pl.Expr | pl.Series] = []
    for field in contract.token_fields:
        series = frame[field.column]
        _validate_token_series(series, field, require_storage_dtype=False)
        expressions.append(
            pl.col(field.column).cast(_TOKEN_DTYPES[field.storage_dtype])
        )
    for field in contract.scalar_fields:
        expressions.append(_quantize_scalar(frame[field.column], field))
    return frame.with_columns(expressions), contract


def _legacy_float32_frame(frame: pl.DataFrame, vocab: VocabV2 | None) -> pl.DataFrame:
    """兼容无 storage metadata 的历史 Float scalar，并拒绝裸量化整数。"""
    expected_token_fields = vocab.token_field_sizes() if vocab is not None else {}
    for column, vocab_size in expected_token_fields.items():
        if column not in frame.columns:
            continue
        field = TokenStorageFieldV2(
            column=column,
            storage_dtype="uint8" if vocab_size <= 256 else "uint16",
            vocab_size=vocab_size,
        )
        _validate_token_series(frame[column], field, require_storage_dtype=False)

    scalar_columns = [column for column in frame.columns if column.startswith("val_")]
    expressions: list[pl.Expr] = []
    for column in scalar_columns:
        dtype = frame.schema[column]
        if not dtype.is_float():
            msg = (
                f"integer scalar {column!r} has no Q16 storage metadata; refusing "
                "to guess a dequantization scale"
            )
            raise ValueError(msg)
        expressions.append(pl.col(column).cast(pl.Float32))
    return frame.with_columns(expressions) if expressions else frame


def dequantize_frame_v2(
    frame: pl.DataFrame,
    metadata: StorageEncodingMetadataV2 | None = None,
    *,
    vocab: VocabV2 | None = None,
) -> pl.DataFrame:
    """
    解码 Q16 scalar 为 Float32；无 metadata 时仅兼容历史 Float scalar。

    允许 DataLoader 只投影部分字段。存在于 frame 中的 encoded 字段必须严格
    匹配 metadata；缺少 metadata 的 Int16/其他整数 ``val_*`` 会被拒绝。
    """
    if metadata is None:
        return _legacy_float32_frame(frame, vocab)
    if vocab is not None:
        assert_storage_metadata_matches_vocab_v2(metadata, vocab)

    scalar_by_column = {field.column: field for field in metadata.scalar_fields}
    unexpected_scalars = sorted(
        column
        for column in frame.columns
        if column.startswith("val_") and column not in scalar_by_column
    )
    if unexpected_scalars:
        msg = f"scalar columns absent from Q16 storage metadata: {unexpected_scalars}"
        raise ValueError(msg)

    for field in metadata.token_fields:
        if field.column in frame.columns:
            _validate_token_series(
                frame[field.column],
                field,
                require_storage_dtype=True,
            )

    expressions: list[pl.Series] = []
    for column, field in scalar_by_column.items():
        if column not in frame.columns:
            continue
        series = frame[column]
        if series.dtype != pl.Int16:
            msg = (
                f"encoded scalar dtype mismatch for {column!r}: "
                f"expected=Int16, actual={series.dtype}"
            )
            raise TypeError(msg)
        if series.null_count():
            msg = f"encoded scalar column contains nulls: {column!r}"
            raise ValueError(msg)
        encoded = series.to_numpy()
        if encoded.size and np.max(np.abs(encoded.astype(np.int32))) > field.quant_max:
            msg = f"encoded scalar exceeds symmetric Q16 range: {column!r}"
            raise ValueError(msg)
        decoded = encoded.astype(np.float32) * np.float32(field.scale)
        expressions.append(pl.Series(column, decoded, dtype=pl.Float32))
    return frame.with_columns(expressions) if expressions else frame


def read_token_frame_v2(
    token_path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    vocab: VocabV2 | None = None,
) -> pl.DataFrame:
    """读取 token shard，并按同目录 sidecar 安全恢复模型语义 dtype。"""
    from quant_fm.tokenizer.artifact_contract import read_token_contract

    path = Path(token_path)
    contract = read_token_contract(path)
    storage_payload = contract.get("storage_encoding")
    if storage_payload is not None and not isinstance(storage_payload, Mapping):
        msg = f"token storage_encoding must be an object: {path}"
        raise TypeError(msg)
    metadata = (
        None
        if storage_payload is None
        else StorageEncodingMetadataV2.from_dict(storage_payload)
    )
    frame = pl.read_parquet(
        path,
        columns=None if columns is None else list(columns),
    )
    return dequantize_frame_v2(frame, metadata, vocab=vocab)


__all__ = [
    "Q16_MAX",
    "STORAGE_ENCODING_SCHEME_V2",
    "STORAGE_ENCODING_VERSION_V2",
    "ScalarStorageFieldV2",
    "StorageEncodingMetadataV2",
    "TokenStorageFieldV2",
    "assert_storage_metadata_matches_vocab_v2",
    "build_storage_metadata_v2",
    "dequantize_frame_v2",
    "quantize_frame_v2",
    "read_token_frame_v2",
    "vocab_sha256_v2",
]
