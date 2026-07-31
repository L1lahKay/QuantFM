#!/usr/bin/env bash
# 2026 跨期 OOS 下游编排（严格晚于 FM 预训练期）。
#
# 流程：
#   1) 轮询等待 oos2026 的 manifest.json 就绪（tokenize 完成）
#   2) 用 medium_300m 的 best.pt 多卡抽 2026 各 split embedding，合并为 all.parquet
#   3) 用 2025 数据冻结 Ranker、对 2026 无标签打分 → scores.parquet + manifest
#
# 用法：
#   nohup bash quant_fm/scripts/run_oos2026_downstream.sh > quant_fm/runs/oos2026/oos_downstream.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TRAIN_WORKDIR="${TRAIN_WORKDIR:-quant_fm/runs/medium_300m}"
OOS_WORKDIR="${OOS_WORKDIR:-quant_fm/runs/oos2026}"
CKPT="${CKPT:-$TRAIN_WORKDIR/run/best.pt}"
MANIFEST="$OOS_WORKDIR/data/manifest.json"
EMB_DIR="$OOS_WORKDIR/embeddings"
NPROC="${NPROC:-8}"
BATCH="${BATCH:-16}"
DTYPE="${DTYPE:-bf16}"
CONTEXT="${CONTEXT-}"
POOLING="${POOLING-}"
STRIDE="${STRIDE-}"
TRAIN_PANEL="${TRAIN_PANEL:-$TRAIN_WORKDIR/panel/daily_panel.parquet}"
TRAIN_CALENDAR="${TRAIN_CALENDAR:-}"
TRAIN_UNIVERSE="${TRAIN_UNIVERSE:-}"
OOS_UNIVERSE="${OOS_UNIVERSE:-}"
TRAIN_EMB="${TRAIN_EMB:-$TRAIN_WORKDIR/embeddings/all.parquet}"
SIGNAL_ARTIFACT="${SIGNAL_ARTIFACT:-$TRAIN_WORKDIR/signal_artifact}"

[[ -n "$TRAIN_CALENDAR" && -f "$TRAIN_CALENDAR" ]] || {
  echo "ERROR: 必须设置严格训练标签使用的 TRAIN_CALENDAR" >&2
  exit 2
}
[[ -n "$TRAIN_UNIVERSE" && -f "$TRAIN_UNIVERSE" ]] || {
  echo "ERROR: 必须设置逐日 PIT TRAIN_UNIVERSE" >&2
  exit 2
}
[[ -n "$OOS_UNIVERSE" && -f "$OOS_UNIVERSE" ]] || {
  echo "ERROR: 必须设置逐日 PIT OOS_UNIVERSE；embedding 不能代替股票池" >&2
  exit 2
}

echo "======== $(date -Is) 等待 oos2026 manifest 就绪 ========"
while [[ ! -f "$MANIFEST" ]]; do sleep 60; done
echo "manifest ready: $MANIFEST"

for SPLIT in train val test; do
  echo "-------- 抽 embedding split=$SPLIT --------"
  WORKDIR="$OOS_WORKDIR" CKPT="$CKPT" MANIFEST="$MANIFEST" EMB_DIR="$EMB_DIR" \
    SPLIT="$SPLIT" NPROC="$NPROC" BATCH="$BATCH" DTYPE="$DTYPE" \
    CONTEXT="$CONTEXT" POOLING="$POOLING" STRIDE="$STRIDE" \
    bash quant_fm/scripts/extract_embeddings_parallel.sh
done

echo "-------- 合并 2026 all.parquet --------"
uv run python - "$EMB_DIR" <<'PY'
from pathlib import Path
import polars as pl
from quant_fm.embedding.contract import propagate_embedding_contract
emb = Path(__import__("sys").argv[1])
paths = [emb / f"{s}.parquet" for s in ("train","val","test") if (emb / f"{s}.parquet").exists()]
frames = [pl.read_parquet(path) for path in paths]
merged = pl.concat(frames, how="vertical_relaxed").sort(["date","symbol"])
out = emb / "all.parquet"
merged.write_parquet(out)
propagate_embedding_contract(paths, out, context="2026 embedding splits")
print(f"all.parquet rows={merged.height} days={merged['date'].n_unique()}")
PY

if [[ ! -f "$SIGNAL_ARTIFACT/ranker.pt" ]]; then
  echo "-------- 离线冻结 Ranker（仅历史期读取标签）--------"
  uv run python -m quant_fm.signal.train \
    --embeddings "$TRAIN_EMB" \
    --panel "$TRAIN_PANEL" \
    --calendar "$TRAIN_CALENDAR" \
    --universe "$TRAIN_UNIVERSE" \
    --out-dir "$SIGNAL_ARTIFACT" \
    --device cuda:0
fi

echo "-------- 跨期 OOS 无标签打分 --------"
uv run python -m quant_fm.signal.generate \
  --embeddings "$EMB_DIR/all.parquet" \
  --ranker "$SIGNAL_ARTIFACT/ranker.pt" \
  --ranker-metadata "$SIGNAL_ARTIFACT/ranker_metadata.json" \
  --fm-checkpoint "$CKPT" \
  --vocab "$TRAIN_WORKDIR/data/vocab.json" \
  --universe "$OOS_UNIVERSE" \
  --out-dir "$OOS_WORKDIR/delivery_oos" \
  --device cuda:0

echo "======== $(date -Is) OOS 交付完成 → $OOS_WORKDIR/delivery_oos ========"
