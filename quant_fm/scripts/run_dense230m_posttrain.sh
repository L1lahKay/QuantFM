#!/usr/bin/env bash
# Dense230M-V1-compat 训练完成后的串行验收：
#   预训练 val → 旧 300M 同窗 val → 非劣 gate → test
#   → train/val/test embedding → RankIC/CPCV/成本后组合回测。
#
# 本脚本不会等待训练，也不会抢占运行中的 GPU；完成标志或进程条件不满足时直接退出。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORKDIR="${WORKDIR:-quant_fm/runs/dense_230m_v1}"
CONFIG="${CONFIG:-$WORKDIR/config.yaml}"
RUN_DIR="$WORKDIR/run"
CKPT="${CKPT:-$RUN_DIR/best.pt}"
BASELINE_CKPT="${BASELINE_CKPT:-quant_fm/runs/medium_300m/run/best.pt}"
MANIFEST="${MANIFEST:-quant_fm/runs/cont60/data/manifest.json}"
PANEL="${PANEL:-quant_fm/runs/cont60/panel/daily_panel_cont60.parquet}"
MAX_DATES="${MAX_DATES:-0}"
if ! [[ "$MAX_DATES" =~ ^[0-9]+$ ]]; then
  echo "ERROR: MAX_DATES must be a non-negative integer, got $MAX_DATES" >&2
  exit 2
fi
if (( MAX_DATES > 0 )); then
  EMB_DIR="${EMB_DIR:-$WORKDIR/embeddings_quick_${MAX_DATES}d}"
else
  EMB_DIR="${EMB_DIR:-$WORKDIR/embeddings}"
fi
NPROC="${NPROC:-8}"
BATCH="${BATCH:-8}"
DTYPE="${DTYPE:-bf16}"
EPOCHS="${EPOCHS:-30}"
SEED="${SEED:-42}"

for required in \
  "$CONFIG" "$RUN_DIR/final.pt" "$RUN_DIR/final_resume.pt" \
  "$CKPT" "$BASELINE_CKPT" "$MANIFEST" "$PANEL"; do
  if [[ ! -f "$required" ]]; then
    echo "BLOCKED: required artifact is missing: $required" >&2
    exit 2
  fi
done

if pgrep -f "quant_fm.pretrain.train.*dense_230m_v1/config.yaml" >/dev/null; then
  echo "BLOCKED: Dense230M training process is still alive" >&2
  exit 2
fi

echo "======== $(date -Is) Dense230M post-train acceptance ========"

# Gate 1/2：同一个 validation plan 上比较；非劣失败时命令返回非零，不会继续 test。
.venv/bin/python -m quant_fm.scripts.posttrain_evaluation \
  --config "$CONFIG" \
  --baseline-checkpoint "$BASELINE_CKPT" \
  --noninferiority-tolerance 0.01 \
  --device cuda \
  --execute

# Gate 3 输入：三个 split 使用同一 checkpoint、manifest、pooling 和 dtype。
for split in train val test; do
  echo "==> extract embeddings split=$split"
  WORKDIR="$WORKDIR" CKPT="$CKPT" MANIFEST="$MANIFEST" EMB_DIR="$EMB_DIR" \
    SPLIT="$split" NPROC="$NPROC" BATCH="$BATCH" DTYPE="$DTYPE" \
    MAX_DATES="$MAX_DATES" \
    bash quant_fm/scripts/extract_embeddings_parallel.sh
done

echo "==> RankIC / CPCV / cost-aware portfolio evaluation"
.venv/bin/python -m quant_fm.downstream.run_judge \
  --workdir "$WORKDIR" \
  --checkpoint "$CKPT" \
  --panel "$PANEL" \
  --emb-dir "$EMB_DIR" \
  --epochs "$EPOCHS" \
  --device cuda:0 \
  --seed "$SEED"

echo "======== $(date -Is) Dense230M post-train acceptance complete ========"
echo "pretrain: $RUN_DIR/pretrain_acceptance.json"
echo "downstream: $WORKDIR/downstream/latest.json"
