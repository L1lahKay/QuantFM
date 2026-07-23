#!/usr/bin/env bash
# 连续 60 交易日下游一条龙（过夜自动接力）：
#   等待 cont60 tokens/manifest 就绪 → 抽 train/test embedding
#   → 历史 train 标签离线冻结 Ranker → 对 test 无标签生成 score。
#
# 依赖：run_medium(--reuse-vocab) 正在/已生成 quant_fm/runs/cont60/data/manifest.json
#       预训练权重 quant_fm/runs/medium_300m/run/best.pt
#       连续 panel   quant_fm/runs/cont60/panel/daily_panel_cont60.parquet
#
# 用法：
#   nohup bash quant_fm/scripts/run_cont60_downstream.sh > quant_fm/runs/cont60/downstream.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f "$HOME/.minio_fm_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.minio_fm_env.sh"
fi

WORKDIR="quant_fm/runs/cont60"
PRETRAIN="quant_fm/runs/medium_300m"
CKPT="${CKPT:-$PRETRAIN/run/best.pt}"
MANIFEST="$WORKDIR/data/manifest.json"
EMB_DIR="$WORKDIR/embeddings"
PANEL="${PANEL:-$WORKDIR/panel/daily_panel_cont60.parquet}"
NPROC="${NPROC:-8}"
BATCH="${BATCH:-16}"
DTYPE="${DTYPE:-bf16}"
EPOCHS="${EPOCHS:-30}"
SEED="${SEED:-42}"

echo "======== $(date -Is) cont60 下游一条龙 ========"
echo "  ckpt=$CKPT  manifest=$MANIFEST  panel=$PANEL"

# 1) 等待 tokens/manifest 就绪
echo "==> 等待 $MANIFEST ..."
while [[ ! -f "$MANIFEST" ]]; do
  sleep 60
done
echo "==> manifest 就绪：$(date -Is)"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: 预训练权重不存在: $CKPT" >&2
  exit 1
fi
if [[ ! -f "$PANEL" ]]; then
  echo "ERROR: 连续 panel 不存在: $PANEL" >&2
  exit 1
fi

# 2) 多卡抽 embedding（训练 Ranker 用 train，生产出分用 test）
for split in train test; do
  echo "==> 抽 embedding split=$split"
  WORKDIR="$WORKDIR" CKPT="$CKPT" MANIFEST="$MANIFEST" EMB_DIR="$EMB_DIR" \
    SPLIT="$split" NPROC="$NPROC" BATCH="$BATCH" DTYPE="$DTYPE" \
    bash "$ROOT/quant_fm/scripts/extract_embeddings_parallel.sh"
done

# 3) 训练期标签只在此离线步骤使用，冻结后生产打分不再读取 panel
echo "==> train frozen signal ranker"
uv run python -m quant_fm.signal.train \
  --embeddings "$EMB_DIR/train.parquet" \
  --panel "$PANEL" \
  --out-dir "$WORKDIR/signal_artifact" \
  --epochs "$EPOCHS" \
  --device cuda:0 \
  --seed "$SEED"

# 4) 唯一生产终点：test 日期 score + manifest
echo "==> generate label-free score signal"
uv run python -m quant_fm.signal.generate \
  --embeddings "$EMB_DIR/test.parquet" \
  --ranker "$WORKDIR/signal_artifact/ranker.pt" \
  --ranker-metadata "$WORKDIR/signal_artifact/ranker_metadata.json" \
  --out-dir "$WORKDIR/delivery" \
  --device cuda:0 \
  --fm-checkpoint "$CKPT" \
  --vocab "$WORKDIR/data/vocab.json"

echo "======== $(date -Is) DONE cont60 下游一条龙 ========"
echo "  交付: $WORKDIR/delivery/scores.parquet"
