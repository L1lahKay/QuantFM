#!/usr/bin/env bash
# 302M 全市场下游验收（多卡加速版）：
#   train/val/test 各用 NPROC 张卡并行抽 embedding → 合并 → panel（若缺）→ run_judge
#
# 用法：
#   bash quant_fm/scripts/run_judge_300m_fast.sh
#   NPROC=8 BATCH=16 bash ...
#   MAX_DATES=8 EPOCHS=10 bash ...  # 快速候选筛选（每 split 8 个完整日期）
#   SKIP_EMB=1 bash ...             # 显式复用已合并 embeddings
#
# 依赖：训练已产出 best.pt；本地 tokens/manifest 就绪。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORKDIR="${WORKDIR:-quant_fm/runs/medium_300m}"
DATA_WORKDIR="${DATA_WORKDIR:-$WORKDIR}"
CKPT="${CKPT:-$WORKDIR/run/best.pt}"
MANIFEST="${MANIFEST:-$DATA_WORKDIR/data/manifest.json}"
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
if [[ -z "${DEV_ONLY+x}" ]]; then
  if [[ "$MAX_DATES" == "0" ]]; then
    DEV_ONLY=0
  else
    DEV_ONLY=1
  fi
fi
if [[ "$DEV_ONLY" != "0" && "$DEV_ONLY" != "1" ]]; then
  echo "ERROR: DEV_ONLY must be 0 or 1, got $DEV_ONLY" >&2
  exit 2
fi
if [[ "$DEV_ONLY" == "1" ]]; then
  JUDGE_WORKDIR="${JUDGE_WORKDIR:-$WORKDIR/quick_eval_${MAX_DATES}d}"
  SPLITS=(train val)
  JUDGE_EXTRA_ARGS=(--dev-only)
else
  JUDGE_WORKDIR="${JUDGE_WORKDIR:-$WORKDIR}"
  SPLITS=(train val test)
  JUDGE_EXTRA_ARGS=()
fi
PANEL="${PANEL:-$DATA_WORKDIR/panel/daily_panel.parquet}"
DATES="${DATES:-quant_fm/data/medium_300m_22_dates.txt}"
NPROC="${NPROC:-8}"
BATCH="${BATCH:-16}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-cuda:0}"
SKIP_EMB="${SKIP_EMB:-0}"
SKIP_PANEL="${SKIP_PANEL:-0}"
EPOCHS="${EPOCHS:-30}"
TOP_K="${TOP_K:-300}"

if [[ ! -f "$CKPT" ]]; then echo "ERROR: 缺 checkpoint $CKPT" >&2; exit 1; fi
if [[ ! -f "$MANIFEST" ]]; then echo "ERROR: 缺 manifest $MANIFEST" >&2; exit 1; fi

echo "======== $(date -Is) judge-300m-fast (NPROC=$NPROC BATCH=$BATCH DTYPE=$DTYPE MAX_DATES=$MAX_DATES DEV_ONLY=$DEV_ONLY) ========"

if [[ "$SKIP_EMB" != "1" ]]; then
  for split in "${SPLITS[@]}"; do
    resume=1
    [[ "${FORCE_EMB:-0}" == "1" ]] && resume=0
    WORKDIR="$WORKDIR" CKPT="$CKPT" MANIFEST="$MANIFEST" EMB_DIR="$EMB_DIR" \
      SPLIT="$split" NPROC="$NPROC" BATCH="$BATCH" DTYPE="$DTYPE" \
      MAX_DATES="$MAX_DATES" RESUME="$resume" \
      bash "$ROOT/quant_fm/scripts/extract_embeddings_parallel.sh"
  done
  if [[ "${BUILD_ALL:-0}" == "1" || ! -f "$PANEL" ]]; then
    uv run python - "$EMB_DIR" "${SPLITS[*]}" <<'PY'
import sys
from pathlib import Path
import polars as pl
from quant_fm.embedding.contract import propagate_embedding_contract
emb = Path(sys.argv[1])
splits = sys.argv[2].split()
paths = [emb / f"{split}.parquet" for split in splits]
parts = [pl.read_parquet(path) for path in paths]
allp = pl.concat(parts, how="vertical_relaxed")
out = emb / "all.parquet"
tmp = emb / ".all.parquet.tmp"
allp.write_parquet(tmp)
tmp.replace(out)
propagate_embedding_contract(paths, out, context="judge embedding splits")
print(f"wrote {out} rows={allp.height}")
PY
  else
    echo "==> panel 已存在，跳过非必要的 all.parquet 全量重写"
  fi
else
  echo "==> SKIP_EMB=1"
fi

if [[ "$SKIP_PANEL" != "1" && ! -f "$PANEL" ]]; then
  echo "==> build daily panel"
  mkdir -p "$(dirname "$PANEL")"
  if [[ -f "$EMB_DIR/all.parquet" ]]; then
    uv run python -m quant_fm.downstream.build_panel_from_minio \
      --from-embeddings "$EMB_DIR/all.parquet" --out "$PANEL"
  else
    uv run python -m quant_fm.downstream.build_panel_from_minio \
      --dates-file "$DATES" --out "$PANEL"
  fi
fi

if [[ ! -f "$PANEL" ]]; then echo "ERROR: 缺 panel $PANEL" >&2; exit 1; fi

echo "==> run_judge"
uv run python -m quant_fm.downstream.run_judge \
  --workdir "$JUDGE_WORKDIR" \
  --checkpoint "$CKPT" \
  --panel "$PANEL" \
  --emb-dir "$EMB_DIR" \
  --epochs "$EPOCHS" \
  --device "$DEVICE" \
  --top-k "$TOP_K" \
  "${JUDGE_EXTRA_ARGS[@]}"

echo "======== $(date -Is) DONE judge-300m-fast ========"
echo "  report: $JUDGE_WORKDIR/downstream/latest.json"
