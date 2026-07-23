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
CONTEXT="${CONTEXT:-2048}"
POOLING="${POOLING:-mean}"
DTYPE="${DTYPE:-bf16}"
MIN_FREE_MEM_GB="${MIN_FREE_MEM_GB:-32}"
LOW_MEM_NPROC="${LOW_MEM_NPROC:-4}"
SCORE_LOCK="${SCORE_LOCK:-$WORKDIR/delivery_oos/.score.lock}"

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

echo "======== $(date -Is) parallel-extract split=$SPLIT nproc=$NPROC batch=$BATCH dtype=$DTYPE ========"

pids=()
for ((g=0; g<NPROC; g++)); do
  out="$PARTS_DIR/${SPLIT}.part${g}of${NPROC}.parquet"
  log="$PARTS_DIR/${SPLIT}.part${g}of${NPROC}.log"
  CUDA_VISIBLE_DEVICES="$g" uv run python -m quant_fm.embedding.extract_hidden \
    --checkpoint "$CKPT" \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --out "$out" \
    --context "$CONTEXT" \
    --pooling "$POOLING" \
    --dtype "$DTYPE" \
    --batch-size "$BATCH" \
    --num-parts "$NPROC" \
    --part-index "$g" \
    --device cuda:0 \
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
from pathlib import Path
import polars as pl

parts_dir, split, nproc, emb_dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
paths = [Path(parts_dir) / f"{split}.part{g}of{nproc}.parquet" for g in range(nproc)]
frames = [pl.read_parquet(p) for p in paths if p.exists()]
if not frames:
    raise SystemExit(f"no part files for split={split}")
merged = pl.concat(frames, how="vertical_relaxed").sort(["date", "symbol"])
out = Path(emb_dir) / f"{split}.parquet"
merged.write_parquet(out)
print(f"merged {sum(f.height for f in frames)} rows → {out}")
PY

echo "======== $(date -Is) DONE parallel-extract split=$SPLIT ========"
