"""
用冻结 FM 对 token 分片编码并输出股日嵌入。

每个股日分片按 context 大小分块编码事件流；逐事件隐状态经
:mod:`pool_stock_day` 池化为每个 (date, symbol) 一向量。输出为整洁的嵌入 parquet，
便于下游与手工因子拼接。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch

from quant_fm.embedding.contract import (
    AFTER_CLOSE_AVAILABILITY,
    EMBEDDING_CONTRACT_VERSION,
    STOCK_DAY_GRANULARITY,
    EmbeddingContract,
    encoder_semantics_for,
    load_embedding_contract,
    validate_embedding_columns,
    write_embedding_contract,
)
from quant_fm.embedding.pool_stock_day import (
    MultiScaleStockDayPoolAccumulator,
    StockDayPoolAccumulator,
)
from quant_fm.embedding.pooling_spec import (
    DEFAULT_V2_MULTI_SCALE_OUTPUTS,
    MULTI_SCALE_POOLING_VERSION,
    resolve_pooling_spec,
)
from quant_fm.manifest.build_manifest import Manifest
from quant_fm.manifest.validation import (
    validate_manifest_shards,
    validate_manifest_vocab_contract,
)
from quant_fm.pretrain.data_contract import (
    load_checkpoint_contract,
    validate_checkpoint_data_contract,
)
from quant_fm.pretrain.train import _load_vocab, load_checkpoint, resolve_device
from quant_fm.tokenizer.artifact_contract import (
    stable_vocab_sha256,
    token_contract_path,
)
from quant_fm.tokenizer.storage_encoding_v2 import read_token_frame_v2
from quant_fm.tokenizer.transforms import int_time_to_ms

if TYPE_CHECKING:
    from quant_fm.embedding.pooling_spec import PoolingSpec
    from quant_fm.manifest.build_manifest import ShardEntry
    from quant_fm.pretrain.model import OrderFlowFM

logger = logging.getLogger(__name__)

DTYPE_MAP: dict[str, torch.dtype | None] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": None,
}

_CACHE_FORMAT_VERSION = 3


def _effective_stride(context: int, stride: int | None) -> int:
    """Resolve and validate extraction stride."""
    actual = context if stride is None else stride
    # Reuse the contract validator so CLI and sidecars share exact bounds.
    encoder_semantics_for(context, actual)
    return actual


def _causal_windows(
    n_events: int,
    *,
    context: int,
    stride: int,
) -> list[tuple[int, int, int]]:
    """
    Return ``(start, end, emit_start)`` windows in chronological order.

    With overlap, a later window includes up to ``context-stride`` historical
    events, while ``emit_start:end`` contains only events not emitted before.
    Consequently every event enters stock-day pooling exactly once.
    """
    _effective_stride(context, stride)
    if n_events < 0:
        msg = "n_events must be non-negative"
        raise ValueError(msg)
    if n_events == 0:
        return []
    if stride == context:
        return [
            (start, min(start + context, n_events), 0)
            for start in range(0, n_events, context)
        ]

    first_end = min(context, n_events)
    windows = [(0, first_end, 0)]
    emitted_until = first_end
    start = 0
    while emitted_until < n_events:
        start += stride
        end = min(start + context, n_events)
        emit_start = emitted_until - start
        windows.append((start, end, emit_start))
        emitted_until = end
    return windows


def _pooling_spec_for_model(model: OrderFlowFM, pooling: str) -> PoolingSpec:
    """Resolve the actual output layout from frozen checkpoint metadata."""
    if pooling != "multi_scale":
        return resolve_pooling_spec(pooling)
    configured_method = getattr(model.cfg, "pooling_method", "mean")
    configured_version = getattr(model.cfg, "pooling_version", "flat_v1")
    configured_outputs = tuple(getattr(model.cfg, "pooling_outputs", ()))
    if configured_method != "multi_scale" and not configured_version.startswith(
        "hierarchical_"
    ):
        # A flat V1 checkpoint may still be used in an explicit multi-scale
        # ablation.  Give that new artifact the corrected V2 layout.
        configured_version = MULTI_SCALE_POOLING_VERSION
        configured_outputs = DEFAULT_V2_MULTI_SCALE_OUTPUTS
    return resolve_pooling_spec(
        pooling,
        configured_version=configured_version,
        configured_outputs=configured_outputs,
    )


def _select_evenly_spaced_dates(
    shards: list[ShardEntry], max_dates: int
) -> list[ShardEntry]:
    """Keep complete cross-sections for up to ``max_dates`` representative dates."""
    if max_dates < 0:
        msg = "max_dates must be non-negative"
        raise ValueError(msg)
    dates = sorted({shard.date for shard in shards})
    if max_dates == 0 or len(dates) <= max_dates:
        return shards
    if max_dates == 1:
        selected = {dates[len(dates) // 2]}
    else:
        selected = {
            dates[index * (len(dates) - 1) // (max_dates - 1)]
            for index in range(max_dates)
        }
    return [shard for shard in shards if shard.date in selected]


def _sha256_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shards_fingerprint(shards: list[ShardEntry]) -> str:
    """Hash the exact selected token contents without rereading parquet files."""
    digest = hashlib.sha256()
    for shard in sorted(shards, key=lambda item: (item.date, item.market, item.symbol)):
        contract_id = shard.data_contract_sha256
        if not contract_id:
            sidecar = token_contract_path(Path(shard.path))
            if sidecar.is_file():
                contract_id = _sha256_file(sidecar)
        digest.update(
            (
                f"{shard.date}\0{shard.market}\0{shard.symbol}\0"
                f"{shard.rows}\0{shard.sha256}\0"
                f"{contract_id or ''}\n"
            ).encode()
        )
    return digest.hexdigest()


def _embedding_cache_spec(
    args: argparse.Namespace,
    shards: list[ShardEntry],
    *,
    checkpoint_id: str,
) -> dict[str, object]:
    return {
        "format_version": _CACHE_FORMAT_VERSION,
        "checkpoint_sha256": checkpoint_id,
        "shards_sha256": _shards_fingerprint(shards),
        "n_shards": len(shards),
        "split": args.split,
        "context": args.context,
        "stride": getattr(args, "stride", None),
        "pooling": args.pooling,
        "last_k": args.last_k,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "num_parts": args.num_parts,
        "part_index": args.part_index,
    }


def _cache_metadata_path(output: Path) -> Path:
    return output.with_name(f"{output.name}.meta.json")


def _embedding_cache_hit(
    output: Path, metadata_path: Path, expected: dict[str, object]
) -> bool:
    if not output.is_file() or not metadata_path.is_file():
        return False
    try:
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
        if actual != expected:
            return False
        keys = pl.read_parquet(output, columns=["date", "symbol", "market"])
        return keys.height == int(expected["n_shards"])
    except (OSError, ValueError, json.JSONDecodeError, pl.exceptions.PolarsError):
        return False


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _embedding_frame(
    metadata: list[dict[str, object]], vectors: list[np.ndarray] | np.ndarray
) -> pl.DataFrame:
    """Build the wide result column-wise, avoiding millions of Python floats."""
    if not metadata:
        return pl.DataFrame(
            schema={"date": pl.String, "symbol": pl.String, "market": pl.String}
        )
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(metadata):
        msg = f"invalid embedding matrix shape {matrix.shape} for {len(metadata)} rows"
        raise ValueError(msg)
    base = pl.DataFrame(metadata)
    values = pl.DataFrame(
        matrix,
        schema=[f"emb_{index}" for index in range(matrix.shape[1])],
        orient="row",
    )
    return base.hstack(values)


def _build_embedding_contract(
    args: argparse.Namespace,
    model: OrderFlowFM,
    embeddings: pl.DataFrame,
    *,
    checkpoint_id: str,
    vocab_path: Path | None,
    effective_dtype: str,
    chunk_stride: int,
    pooling_spec: PoolingSpec,
) -> EmbeddingContract:
    """从真实 FM 配置与实际输出列构造股日表示契约。"""
    embedding_columns = tuple(
        column for column in embeddings.columns if column.startswith("emb_")
    )
    provided_vocab_id = (
        stable_vocab_sha256(_load_vocab(vocab_path))
        if vocab_path is not None and vocab_path.is_file()
        else None
    )
    declared_vocab_id = model.cfg.vocab_sha256 or None
    if (
        declared_vocab_id is not None
        and provided_vocab_id is not None
        and declared_vocab_id != provided_vocab_id
    ):
        msg = "FM checkpoint vocab hash does not match extraction vocab"
        raise ValueError(msg)
    vocab_id = declared_vocab_id or provided_vocab_id
    expected_width = pooling_spec.embedding_width(model.cfg.d_model)
    if len(embedding_columns) != expected_width:
        msg = (
            "actual embedding width does not match resolved pooling profile: "
            f"expected={expected_width}, actual={len(embedding_columns)}, "
            f"profile={pooling_spec}"
        )
        raise ValueError(msg)
    contract = EmbeddingContract(
        format_version=EMBEDDING_CONTRACT_VERSION,
        fm_checkpoint_sha256=checkpoint_id,
        vocab_sha256=vocab_id,
        schema_version=model.cfg.schema_version,
        book_state_timing=model.cfg.book_state_timing,
        pooling_version=pooling_spec.version,
        granularity=STOCK_DAY_GRANULARITY,
        context=args.context,
        chunk_stride=chunk_stride,
        pooling=args.pooling,
        last_k=args.last_k,
        dtype=effective_dtype,
        encoder_width=model.cfg.d_model,
        pooling_components=pooling_spec.vector_components,
        pooling_scalar_components=pooling_spec.scalar_components,
        embedding_columns=embedding_columns,
        embedding_width=len(embedding_columns),
        signal_availability=AFTER_CLOSE_AVAILABILITY,
        encoder_semantics=encoder_semantics_for(args.context, chunk_stride),
        event_ordering_version=model.cfg.event_ordering_version,
        feature_transform_version=model.cfg.feature_transform_version,
    )
    contract.validate()
    return contract


def _embedding_contract_cache_hit(
    output: Path,
    args: argparse.Namespace,
    *,
    checkpoint_id: str,
    vocab_id: str | None,
    effective_dtype: str,
    chunk_stride: int | None,
) -> bool:
    """要求旧 parquet 的 sidecar 与本次表示请求完全一致。"""
    try:
        contract = load_embedding_contract(output, required=False)
        if contract is None:
            return False
        validate_embedding_columns(
            list(pl.read_parquet_schema(output).names()),
            contract,
            context=str(output),
        )
    except (OSError, ValueError, pl.exceptions.PolarsError):
        return False
    requested: dict[str, object] = {
        "fm_checkpoint_sha256": checkpoint_id,
        "vocab_sha256": vocab_id,
        "last_k": args.last_k,
        "dtype": effective_dtype,
    }
    if args.context is not None:
        requested["context"] = args.context
    if args.pooling is not None:
        requested["pooling"] = args.pooling
    if chunk_stride is not None:
        requested["chunk_stride"] = chunk_stride
    actual = contract.to_dict()
    return all(actual[field] == value for field, value in requested.items())


def _balanced_shard_partition(
    shards: list[ShardEntry], num_parts: int, part_index: int
) -> list[ShardEntry]:
    """Greedy row-balanced partition; file-count stride badly skews active stocks."""
    parts: list[list[ShardEntry]] = [[] for _ in range(num_parts)]
    loads = [0] * num_parts
    for shard in sorted(shards, key=lambda item: (-item.rows, item.path)):
        target = min(range(num_parts), key=lambda index: (loads[index], index))
        parts[target].append(shard)
        loads[target] += shard.rows
    logger.info("part token-row loads: %s", loads)
    return sorted(parts[part_index], key=lambda item: item.path)


def _autocast(device: torch.device, amp_dtype: torch.dtype | None):
    """与训练一致的自动混合精度上下文；``amp_dtype`` 为 None 时不启用。"""
    return torch.autocast(
        device_type=device.type,
        dtype=amp_dtype if amp_dtype is not None else torch.float32,
        enabled=amp_dtype is not None,
    )


@torch.no_grad()
def _embed_shard(
    model: OrderFlowFM,
    shard: ShardEntry,
    device: torch.device,
    *,
    context: int,
    stride: int | None = None,
    pooling: str,
    last_k: int = 256,
    amp_dtype: torch.dtype | None = None,
    pooling_spec: PoolingSpec | None = None,
) -> np.ndarray:
    """分块编码单个分片并返回池化嵌入向量。"""
    chunk_stride = _effective_stride(context, stride)
    resolved_pooling = pooling_spec or (
        resolve_pooling_spec(
            pooling,
            configured_version=MULTI_SCALE_POOLING_VERSION,
            configured_outputs=DEFAULT_V2_MULTI_SCALE_OUTPUTS,
        )
        if pooling == "multi_scale"
        else resolve_pooling_spec(pooling)
    )
    if resolved_pooling.method != pooling:
        msg = "pooling_spec method does not match requested pooling"
        raise ValueError(msg)
    token_fields = model.cfg.input_fields
    scalar_fields = (
        *model.cfg.scalar_fields,
        *model.cfg.standalone_scalar_fields,
    )
    input_columns = (*token_fields, *scalar_fields)
    read_columns = (
        (*input_columns, "int_time") if pooling == "multi_scale" else input_columns
    )
    df = read_token_frame_v2(shard.path, columns=read_columns)
    n = df.height
    fields = {field: df[field].to_numpy().astype(np.int64) for field in token_fields}
    fields.update(
        {field: df[field].to_numpy().astype(np.float32) for field in scalar_fields}
    )

    accumulator = (
        MultiScaleStockDayPoolAccumulator(model.cfg.d_model)
        if pooling == "multi_scale"
        else StockDayPoolAccumulator(model.cfg.d_model, method=pooling, last_k=last_k)
    )
    elapsed = (
        int_time_to_ms(df["int_time"].to_numpy()) if pooling == "multi_scale" else None
    )
    for start, end, emit_start in _causal_windows(
        n,
        context=context,
        stride=chunk_stride,
    ):
        length = end - start
        batch = {
            field: torch.from_numpy(fields[field][start:end].copy())
            .unsqueeze(0)
            .to(device)
            for field in input_columns
        }
        batch["attention_mask"] = torch.ones(
            (1, length), dtype=torch.bool, device=device
        )
        batch["_all_tokens_valid"] = True
        with _autocast(device, amp_dtype):
            hidden = model.encode(batch)[0]  # [L, d]
        if isinstance(accumulator, MultiScaleStockDayPoolAccumulator):
            if elapsed is None:  # pragma: no cover - construction invariant
                msg = "missing intraday time array"
                raise RuntimeError(msg)
            time_tensor = torch.from_numpy(elapsed[start + emit_start : end].copy()).to(
                device
            )
            accumulator.update(
                hidden[emit_start:],
                batch["attention_mask"][0, emit_start:],
                time_tensor,
            )
        else:
            accumulator.update(
                hidden[emit_start:],
                batch["attention_mask"][0, emit_start:],
            )

    value = (
        accumulator.concatenate(
            vector_components=resolved_pooling.vector_components,
            scalar_components=resolved_pooling.scalar_components,
        )
        if isinstance(accumulator, MultiScaleStockDayPoolAccumulator)
        else accumulator.value()
    )
    return value.numpy().astype(np.float32)


def extract_stock_day_embeddings(
    model: OrderFlowFM,
    shards: list[ShardEntry],
    device: torch.device,
    *,
    context: int = 2048,
    stride: int | None = None,
    pooling: str = "mean",
    last_k: int = 256,
    amp_dtype: torch.dtype | None = None,
    pooling_spec: PoolingSpec | None = None,
) -> pl.DataFrame:
    """生成 ``(date, symbol, market, emb_*)`` 嵌入数据帧。"""
    metadata: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    for shard in shards:
        vec = _embed_shard(
            model,
            shard,
            device,
            context=context,
            stride=stride,
            pooling=pooling,
            last_k=last_k,
            amp_dtype=amp_dtype,
            pooling_spec=pooling_spec,
        )
        metadata.append(
            {"date": shard.date, "symbol": shard.symbol, "market": shard.market}
        )
        vectors.append(vec)
        logger.info("embedded %s %s", shard.date, shard.symbol)
    return _embedding_frame(metadata, vectors)


@torch.no_grad()
def extract_stock_day_embeddings_batched(
    model: OrderFlowFM,
    shards: list[ShardEntry],
    device: torch.device,
    *,
    context: int = 2048,
    stride: int | None = None,
    pooling: str = "mean",
    last_k: int = 256,
    batch_size: int = 16,
    log_every: int = 200,
    amp_dtype: torch.dtype | None = None,
    pooling_spec: PoolingSpec | None = None,
) -> pl.DataFrame:
    """
    批处理版：把多个（股日）分片的 context 分块打包进同一 forward，显著提高 GPU 吞吐。

    每个窗口最多含 ``context`` 个事件；``stride<context`` 时携带历史前缀，
    但仅把新增后缀送入池化，保证每个事件严格计入一次。
    """
    chunk_stride = _effective_stride(context, stride)
    resolved_pooling = pooling_spec or (
        resolve_pooling_spec(
            pooling,
            configured_version=MULTI_SCALE_POOLING_VERSION,
            configured_outputs=DEFAULT_V2_MULTI_SCALE_OUTPUTS,
        )
        if pooling == "multi_scale"
        else resolve_pooling_spec(pooling)
    )
    if resolved_pooling.method != pooling:
        msg = "pooling_spec method does not match requested pooling"
        raise ValueError(msg)
    d = model.cfg.d_model
    token_fields = model.cfg.input_fields
    scalar_fields = (
        *model.cfg.scalar_fields,
        *model.cfg.standalone_scalar_fields,
    )
    input_columns = (*token_fields, *scalar_fields)
    read_columns = (
        (*input_columns, "int_time") if pooling == "multi_scale" else input_columns
    )

    # 预扫描出所有 (shard_idx, 起止) 块，块内数据延迟读取以省内存。
    meta: list[dict] = [
        {"date": s.date, "symbol": s.symbol, "market": s.market} for s in shards
    ]
    # Mean is the production default.  Keep its partial sums on the accelerator
    # and transfer once at the end; the former per-chunk ``.cpu()`` synchronized
    # the GPU thousands of times.  Other pooling modes retain their specialised
    # CPU accumulators because their state is small and order-sensitive.
    mean_sums = (
        torch.zeros((len(shards), d), dtype=torch.float32, device=device)
        if pooling == "mean"
        else None
    )
    mean_counts = (
        torch.zeros(len(shards), dtype=torch.int64, device=device)
        if pooling == "mean"
        else None
    )
    accumulators = (
        None
        if pooling == "mean"
        else (
            [MultiScaleStockDayPoolAccumulator(d) for _ in shards]
            if pooling == "multi_scale"
            else [
                StockDayPoolAccumulator(d, method=pooling, last_k=last_k)
                for _ in shards
            ]
        )
    )

    # Exact-length buckets keep default mean-pooling batches padding-free.
    # Mixing one tail chunk with full context chunks forces every Transformer
    # layer off the fused causal SDPA path.  Order-sensitive pooling modes keep
    # the original FIFO queue so a later tail can never overtake an earlier
    # full chunk from the same shard.
    WindowBatchItem = tuple[int, dict[str, np.ndarray], int, int]
    pending_by_length: dict[int, list[WindowBatchItem]] = {}
    pending_ordered: list[WindowBatchItem] = []

    def flush(batch_chunks: list[WindowBatchItem]) -> None:
        if not batch_chunks:
            return
        max_len = max(length for _, _, length, _ in batch_chunks)
        bsz = len(batch_chunks)
        fields_t: dict[str, torch.Tensor] = {
            field: torch.zeros((bsz, max_len), dtype=torch.long)
            for field in token_fields
        }
        fields_t.update(
            {
                field: torch.zeros((bsz, max_len), dtype=torch.float32)
                for field in scalar_fields
            }
        )
        mask = torch.zeros((bsz, max_len), dtype=torch.bool)
        for bi, (_, fields, length, _) in enumerate(batch_chunks):
            for field in input_columns:
                fields_t[field][bi, :length] = torch.from_numpy(fields[field])
            mask[bi, :length] = True
        batch = {field: fields_t[field].to(device) for field in input_columns}
        batch["attention_mask"] = mask.to(device)
        batch["_all_tokens_valid"] = all(
            length == max_len for _, _, length, _ in batch_chunks
        )
        with _autocast(device, amp_dtype):
            hidden = model.encode(batch)  # [B, Lmax, d]
        if mean_sums is not None and mean_counts is not None:
            chunk_sums = torch.stack(
                [
                    hidden[bi, emit_start:length].float().sum(dim=0)
                    for bi, (_, _, length, emit_start) in enumerate(batch_chunks)
                ]
            )
            # A batch often contains many chunks from the same active stock.
            # Consolidate duplicate destinations before index_add_ so CUDA does
            # not need competing atomics for the same accumulator row.
            positions_by_shard: dict[int, list[int]] = {}
            for position, (sidx, _, _, _) in enumerate(batch_chunks):
                positions_by_shard.setdefault(sidx, []).append(position)
            unique_indices = list(positions_by_shard)
            aggregated_sums = torch.stack(
                [
                    chunk_sums[positions].sum(dim=0)
                    for positions in positions_by_shard.values()
                ]
            )
            aggregated_lengths = torch.tensor(
                [
                    sum(
                        batch_chunks[position][2] - batch_chunks[position][3]
                        for position in positions
                    )
                    for positions in positions_by_shard.values()
                ],
                dtype=torch.int64,
                device=device,
            )
            shard_indices = torch.tensor(
                unique_indices, dtype=torch.int64, device=device
            )
            mean_sums.index_add_(0, shard_indices, aggregated_sums)
            mean_counts.index_add_(0, shard_indices, aggregated_lengths)
            return
        if accumulators is None:  # pragma: no cover - construction invariant
            msg = "missing pooling accumulators"
            raise RuntimeError(msg)
        for bi, (sidx, fields, length, emit_start) in enumerate(batch_chunks):
            accumulator = accumulators[sidx]
            # Keep all three streams aligned to the real chunk length.  In a
            # mixed-length batch ``hidden`` and ``attention_mask`` are padded
            # to ``max_len`` while the intraday timestamps are not.
            device_mask = batch["attention_mask"][bi, emit_start:length]
            chunk_hidden = hidden[bi, emit_start:length]
            if isinstance(accumulator, MultiScaleStockDayPoolAccumulator):
                times = torch.from_numpy(fields["__time_of_day_ms"][emit_start:]).to(
                    device
                )
                accumulator.update(chunk_hidden, device_mask, times)
            else:
                accumulator.update(chunk_hidden, device_mask)

    for sidx, shard in enumerate(shards):
        df = read_token_frame_v2(shard.path, columns=read_columns)
        n = df.height
        cols = {field: df[field].to_numpy().astype(np.int64) for field in token_fields}
        cols.update(
            {field: df[field].to_numpy().astype(np.float32) for field in scalar_fields}
        )
        if pooling == "multi_scale":
            cols["__time_of_day_ms"] = int_time_to_ms(df["int_time"].to_numpy())
        for start, end, emit_start in _causal_windows(
            n,
            context=context,
            stride=chunk_stride,
        ):
            chunk = {
                field: cols[field][start:end].copy()
                for field in (
                    (*input_columns, "__time_of_day_ms")
                    if pooling == "multi_scale"
                    else input_columns
                )
            }
            length = end - start
            pending = (
                pending_by_length.setdefault(length, [])
                if pooling == "mean"
                else pending_ordered
            )
            pending.append((sidx, chunk, length, emit_start))
            if len(pending) >= batch_size:
                flush(pending)
                pending.clear()
        if (sidx + 1) % log_every == 0:
            logger.info("embedded %d/%d shards", sidx + 1, len(shards))
    if pooling == "mean":
        for length in sorted(pending_by_length, reverse=True):
            flush(pending_by_length[length])
    else:
        flush(pending_ordered)

    mean_vectors = None
    if mean_sums is not None and mean_counts is not None:
        mean_vectors = (
            (mean_sums / mean_counts.clamp_min(1).to(torch.float32).unsqueeze(1))
            .cpu()
            .numpy()
        )

    vectors: list[np.ndarray] = []
    for sidx in range(len(meta)):
        if mean_vectors is not None:
            vec = mean_vectors[sidx]
        else:
            if accumulators is None:  # pragma: no cover - construction invariant
                msg = "missing pooling accumulators"
                raise RuntimeError(msg)
            accumulator = accumulators[sidx]
            vec_tensor = (
                accumulator.concatenate(
                    vector_components=resolved_pooling.vector_components,
                    scalar_components=resolved_pooling.scalar_components,
                )
                if isinstance(accumulator, MultiScaleStockDayPoolAccumulator)
                else accumulator.value()
            )
            vec = vec_tensor.numpy().astype(np.float32)
        vectors.append(vec)
    logger.info("embedded %d/%d shards (done)", len(shards), len(shards))
    return _embedding_frame(meta, vectors)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--vocab",
        type=Path,
        help="v2 checkpoint 必填，用于校验 schema/字段顺序/vocab hash",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--context",
        type=int,
        default=None,
        help=(
            "编码窗口长度；默认使用 checkpoint 固化的 context_horizon，"
            "旧 checkpoint 回退为 2048"
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "embedding 窗口步长；小于 context 时使用因果重叠窗口且每个事件只池化一次。"
            "默认读取 checkpoint pooling.stride，旧 checkpoint 等于 context"
        ),
    )
    parser.add_argument(
        "--pooling",
        default=None,
        choices=["mean", "last", "lastk_mean", "multi_scale"],
        help="默认使用 checkpoint 固化的 pooling.method（旧 checkpoint 为 mean）",
    )
    parser.add_argument(
        "--last-k",
        type=int,
        default=256,
        help="lastk_mean 跨整个股日保留的最后事件数",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="推理精度；默认 bf16（与训练一致，约 2× 提速且省显存）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=">1 时启用批处理编码（多股日打包一次 forward），大幅提速",
    )
    parser.add_argument(
        "--num-parts",
        type=int,
        default=1,
        help="将该 split 的分片按 stride 均分为 N 份（多卡并行时用）",
    )
    parser.add_argument(
        "--part-index",
        type=int,
        default=0,
        help="本进程处理第 part-index 份（0..num-parts-1）",
    )
    parser.add_argument(
        "--max-dates",
        type=int,
        default=0,
        help="每个 split 最多等距选 N 个完整横截面日期；0 表示全量",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="checkpoint、分片和推理参数均相同时复用已完成的输出",
    )
    parser.add_argument(
        "--checkpoint-id",
        default=None,
        help=(
            "可选的 checkpoint SHA-256 期望值；只作一致性断言，worker 仍会"
            "重新哈希 live checkpoint"
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    """CLI 入口。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype = DTYPE_MAP[args.dtype] if device.type == "cuda" else None
    effective_dtype = args.dtype if amp_dtype is not None else "fp32"
    manifest_path = Path(args.manifest)
    manifest = Manifest.load(manifest_path)
    vocab_path = args.vocab
    if vocab_path is None and manifest.vocab_path:
        vocab_path = Path(manifest.vocab_path)
    if vocab_path is None:
        msg = "embedding extraction requires a vocab to verify token provenance"
        raise ValueError(msg)
    vocab = _load_vocab(vocab_path)
    validate_manifest_vocab_contract(
        manifest,
        vocab,
        context="embedding extraction",
    )
    shards = manifest.split(args.split)
    if args.max_dates < 0:
        msg = "--max-dates must be non-negative"
        raise ValueError(msg)
    if args.stride is not None and args.context is not None:
        _effective_stride(args.context, args.stride)
    before_dates = len({shard.date for shard in shards})
    shards = _select_evenly_spaced_dates(shards, args.max_dates)
    after_dates = len({shard.date for shard in shards})
    if after_dates < before_dates:
        logger.info(
            "representative-date subset: %d/%d dates, %d shards",
            after_dates,
            before_dates,
            len(shards),
        )
    if args.num_parts > 1:
        shards = _balanced_shard_partition(shards, args.num_parts, args.part_index)
        logger.info(
            "part %d/%d: %d shards, %d token rows (dtype=%s)",
            args.part_index,
            args.num_parts,
            len(shards),
            sum(shard.rows for shard in shards),
            args.dtype,
        )
    validate_manifest_shards(
        manifest,
        vocab,
        shards=shards,
        context="embedding extraction",
        expected_tokens_root=(
            manifest_path.parent.parent / "tokens"
            if manifest_path.parent.name == "data"
            else manifest_path.parent / "tokens"
        ),
    )
    checkpoint_contract = load_checkpoint_contract(
        args.checkpoint,
        expected_checkpoint_sha256=args.checkpoint_id,
        required=vocab.data_semantics_explicit,
    )
    checkpoint_id = (
        str(checkpoint_contract["checkpoint_sha256"])
        if checkpoint_contract is not None
        else _sha256_file(args.checkpoint)
    )
    validate_checkpoint_data_contract(
        checkpoint_contract,
        checkpoint_payload=None,
        vocab=vocab,
    )
    cache_spec = _embedding_cache_spec(
        args,
        shards,
        checkpoint_id=checkpoint_id,
    )
    metadata_path = _cache_metadata_path(args.out)
    vocab_id = (
        stable_vocab_sha256(vocab)
        if vocab_path is not None and vocab_path.is_file()
        else None
    )
    if (
        args.resume
        and _embedding_cache_hit(args.out, metadata_path, cache_spec)
        and _embedding_contract_cache_hit(
            args.out,
            args,
            checkpoint_id=checkpoint_id,
            vocab_id=vocab_id,
            effective_dtype=effective_dtype,
            chunk_stride=args.stride,
        )
    ):
        logger.info("cache hit: %s (%d shards)", args.out, len(shards))
        return

    # Loading the 230M model happens only after full provenance and cache checks.
    model = load_checkpoint(
        args.checkpoint,
        device,
        vocab_path=vocab_path,
        checkpoint_sha256=checkpoint_id,
    )
    configured_context = int(getattr(model.cfg, "context_horizon", 0))
    args.context = (
        args.context if args.context is not None else configured_context or 2048
    )
    configured_stride = int(getattr(model.cfg, "pooling_stride", 0))
    requested_stride = (
        args.stride if args.stride is not None else configured_stride or args.context
    )
    chunk_stride = _effective_stride(
        args.context,
        requested_stride,
    )
    args.pooling = args.pooling or getattr(model.cfg, "pooling_method", "mean")
    pooling_spec = _pooling_spec_for_model(model, args.pooling)
    logger.info(
        "embedding representation: pooling=%s/%s components=%s context=%d stride=%d",
        pooling_spec.version,
        pooling_spec.method,
        pooling_spec.vector_components,
        args.context,
        chunk_stride,
    )
    if args.batch_size > 1:
        embeddings = extract_stock_day_embeddings_batched(
            model,
            shards,
            device,
            context=args.context,
            stride=chunk_stride,
            pooling=args.pooling,
            last_k=args.last_k,
            batch_size=args.batch_size,
            amp_dtype=amp_dtype,
            pooling_spec=pooling_spec,
        )
    else:
        embeddings = extract_stock_day_embeddings(
            model,
            shards,
            device,
            context=args.context,
            stride=chunk_stride,
            pooling=args.pooling,
            last_k=args.last_k,
            amp_dtype=amp_dtype,
            pooling_spec=pooling_spec,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp")
    embeddings.write_parquet(temporary)
    temporary.replace(args.out)
    embedding_contract = _build_embedding_contract(
        args,
        model,
        embeddings,
        checkpoint_id=checkpoint_id,
        vocab_path=vocab_path,
        effective_dtype=effective_dtype,
        chunk_stride=chunk_stride,
        pooling_spec=pooling_spec,
    )
    write_embedding_contract(args.out, embedding_contract)
    _atomic_write_json(metadata_path, cache_spec)
    logger.info("wrote %d embeddings to %s", embeddings.height, args.out)


if __name__ == "__main__":
    main()
