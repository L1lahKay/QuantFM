#!/usr/bin/env bash
# 多卡并行抽取某个 split 的股日 embedding，然后合并为单个 parquet。
#
# 用法：
#   SPLIT=train bash quant_fm/scripts/extract_embeddings_parallel.sh
#   NPROC=8 BATCH=16 SPLIT=val bash ...
#
# 每张 GPU 处理 split 分片的 1/NPROC（stride 均分），各写一个 part 文件；
# 全部完成后合并为 {EMB_DIR}/{SPLIT}.parquet。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORKDIR="${WORKDIR:-quant_fm/runs/medium_300m}"
CKPT="${CKPT:-$WORKDIR/run/best.pt}"
MANIFEST="${MANIFEST:-$WORKDIR/data/manifest.json}"
EMB_DIR="${EMB_DIR:-$WORKDIR/embeddings}"
SPLIT="${SPLIT:-train}"
NPROC="${NPROC:-8}"
BATCH="${BATCH:-16}"
# Empty means "use the immutable values stored in the checkpoint".  Setting
# these environment variables remains an explicit research/legacy override;
# extract_hidden records the resolved values in the representation sidecar.
CONTEXT="${CONTEXT-}"
POOLING="${POOLING-}"
STRIDE="${STRIDE-}"
DTYPE="${DTYPE:-bf16}"
MAX_DATES="${MAX_DATES:-0}"
RESUME="${RESUME:-1}"
MIN_FREE_MEM_GB="${MIN_FREE_MEM_GB:-32}"
LOW_MEM_NPROC="${LOW_MEM_NPROC:-4}"
SCORE_LOCK="${SCORE_LOCK:-$WORKDIR/delivery_oos/.score.lock}"

if ! [[ "$MAX_DATES" =~ ^[0-9]+$ ]]; then
  echo "ERROR: MAX_DATES must be a non-negative integer, got $MAX_DATES" >&2
  exit 2
fi
if [[ -n "$CONTEXT" && ! "$CONTEXT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CONTEXT must be a positive integer when set, got $CONTEXT" >&2
  exit 2
fi
if [[ -n "$STRIDE" && ! "$STRIDE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: STRIDE must be a positive integer when set, got $STRIDE" >&2
  exit 2
fi
if [[ -n "$CONTEXT" && -n "$STRIDE" ]] && (( STRIDE > CONTEXT )); then
  echo "ERROR: STRIDE=$STRIDE cannot exceed CONTEXT=$CONTEXT" >&2
  exit 2
fi

# GPU 0 上的 Ranker 打分与多卡 FM 推理互斥，避免两套模型争显存。
if [[ -f "$SCORE_LOCK" ]]; then
  score_pid="$(cat "$SCORE_LOCK" 2>/dev/null || true)"
  while [[ "$score_pid" =~ ^[0-9]+$ ]] && kill -0 "$score_pid" 2>/dev/null; do
    echo "WAIT score pid=$score_pid 正在使用 GPU 0；15s 后重试 embedding"
    sleep 15
  done
  rm -f "$SCORE_LOCK"
fi

# 两个全日订单簿清洗进程会占用 400+ GiB 主存。低内存时仍强启 8 个模型进程
# 曾导致全部 embedding worker 被 OOM killer 杀死；自动降卡不改变结果，只降低吞吐。
mem_free_kb="$(awk '/^MemFree:/ {print $2}' /proc/meminfo)"
min_free_kb=$((MIN_FREE_MEM_GB * 1024 * 1024))
if (( mem_free_kb < min_free_kb && NPROC > LOW_MEM_NPROC )); then
  echo "WARN host MemFree=$((mem_free_kb / 1024 / 1024))GiB < ${MIN_FREE_MEM_GB}GiB; NPROC $NPROC → $LOW_MEM_NPROC 防止 OOM"
  NPROC="$LOW_MEM_NPROC"
fi

PARTS_DIR="$EMB_DIR/parts"
mkdir -p "$PARTS_DIR"

# Freeze the expected checkpoint identity in the parent.  Each worker still
# re-hashes the live bytes before trusting the sidecar; this value is an assertion,
# not a provenance-validation bypass.
CHECKPOINT_ID="$(sha256sum "$CKPT")"
CHECKPOINT_ID="${CHECKPOINT_ID%% *}"
EXTRA_ARGS=(--checkpoint-id "$CHECKPOINT_ID")
if [[ -n "$CONTEXT" ]]; then
  EXTRA_ARGS+=(--context "$CONTEXT")
fi
if [[ -n "$POOLING" ]]; then
  EXTRA_ARGS+=(--pooling "$POOLING")
fi
if [[ -n "$STRIDE" ]]; then
  EXTRA_ARGS+=(--stride "$STRIDE")
fi
if [[ "$RESUME" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
fi
if (( MAX_DATES > 0 )); then
  EXTRA_ARGS+=(--max-dates "$MAX_DATES")
fi

echo "======== $(date -Is) parallel-extract split=$SPLIT nproc=$NPROC batch=$BATCH dtype=$DTYPE max_dates=$MAX_DATES resume=$RESUME context=${CONTEXT:-checkpoint} pooling=${POOLING:-checkpoint} stride=${STRIDE:-checkpoint} ========"

pids=()
for ((g=0; g<NPROC; g++)); do
  out="$PARTS_DIR/${SPLIT}.part${g}of${NPROC}.parquet"
  log="$PARTS_DIR/${SPLIT}.part${g}of${NPROC}.log"
  CUDA_VISIBLE_DEVICES="$g" uv run python -m quant_fm.embedding.extract_hidden \
    --checkpoint "$CKPT" \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --out "$out" \
    --dtype "$DTYPE" \
    --batch-size "$BATCH" \
    --num-parts "$NPROC" \
    --part-index "$g" \
    --device cuda:0 \
    "${EXTRA_ARGS[@]}" \
    > "$log" 2>&1 &
  pids+=("$!")
  echo "  launched part $g on GPU $g (pid ${pids[-1]}) → $out"
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "ERROR: part $i (pid ${pids[$i]}) failed; see $PARTS_DIR/${SPLIT}.part${i}of${NPROC}.log" >&2
    fail=1
  fi
done
if [[ "$fail" != "0" ]]; then
  echo "ERROR: 至少一个 part 失败，未合并" >&2
  exit 1
fi

echo "==> 合并 parts → $EMB_DIR/${SPLIT}.parquet"
uv run python - "$PARTS_DIR" "$SPLIT" "$NPROC" "$EMB_DIR" <<'PY'
import sys
import json
from pathlib import Path
import polars as pl
from quant_fm.embedding.contract import (
    assert_embedding_contract_compatible,
    load_compatible_embedding_contracts,
    load_embedding_contract,
    propagate_embedding_contract,
    validate_embedding_columns,
)

parts_dir, split, nproc, emb_dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
paths = [Path(parts_dir) / f"{split}.part{g}of{nproc}.parquet" for g in range(nproc)]
metadata_paths = [path.with_name(f"{path.name}.meta.json") for path in paths]
if not all(path.is_file() for path in paths + metadata_paths):
    missing = [str(path) for path in paths + metadata_paths if not path.is_file()]
    raise SystemExit(f"missing embedding part artifacts: {missing}")
part_specs = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths]
part_contract = load_compatible_embedding_contracts(
    paths,
    required=True,
    context=f"embedding parts for split={split}",
)
if part_contract is None:
    raise SystemExit(f"missing embedding representation contract for split={split}")
merged_spec = {"format_version": 1, "split": split, "parts": part_specs}
out = Path(emb_dir) / f"{split}.parquet"
out_metadata = out.with_name(f"{out.name}.meta.json")
if out.is_file() and out_metadata.is_file():
    try:
        existing = json.loads(out_metadata.read_text(encoding="utf-8"))
        rows = pl.read_parquet(out, columns=["date"]).height
        expected_rows = sum(int(spec["n_shards"]) for spec in part_specs)
        if existing == merged_spec and rows == expected_rows:
            output_contract = load_embedding_contract(out, required=True)
            assert_embedding_contract_compatible(
                part_contract,
                output_contract,
                context=f"cached merged embedding split={split}",
            )
            validate_embedding_columns(
                pl.read_parquet_schema(out).names(),
                output_contract,
                context=str(out),
            )
            print(f"merged cache hit: {out} rows={rows}")
            raise SystemExit(0)
    except (OSError, ValueError, json.JSONDecodeError, pl.exceptions.PolarsError):
        pass
frames = [pl.read_parquet(p) for p in paths if p.exists()]
if not frames:
    raise SystemExit(f"no part files for split={split}")
merged = pl.concat(frames, how="vertical_relaxed").sort(["date", "symbol"])
tmp = out.with_name(f".{out.name}.tmp")
merged.write_parquet(tmp)
tmp.replace(out)
propagate_embedding_contract(
    paths,
    out,
    context=f"embedding parts for split={split}",
)
tmp_metadata = out_metadata.with_name(f".{out_metadata.name}.tmp")
tmp_metadata.write_text(
    json.dumps(merged_spec, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
tmp_metadata.replace(out_metadata)
print(f"merged {sum(f.height for f in frames)} rows → {out}")
PY

echo "======== $(date -Is) DONE parallel-extract split=$SPLIT ========"
