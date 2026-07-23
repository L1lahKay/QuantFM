#!/usr/bin/env bash
# 302M 全市场：抽 embedding → 建日频 panel → 跑下游 judge
#
# 用法：
#   bash quant_fm/scripts/run_judge_300m.sh
#   CKPT=quant_fm/runs/medium_300m/run/final.pt bash ...
#   SKIP_PANEL=1 bash ...          # 复用已有 panel
#   SKIP_EMB=1 bash ...            # 复用已有 embeddings
#   DEVICE=cuda:0 bash ...
#
# 依赖：训练产出 best.pt（或指定 CKPT），本地 tokens/manifest 就绪。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORKDIR="${WORKDIR:-quant_fm/runs/medium_300m}"
CKPT="${CKPT:-$WORKDIR/run/best.pt}"
MANIFEST="$WORKDIR/data/manifest.json"
DATES="${DATES:-quant_fm/data/medium_300m_22_dates.txt}"
EMB_DIR="$WORKDIR/embeddings"
PANEL="$WORKDIR/panel/daily_panel.parquet"
DEVICE="${DEVICE:-cuda:0}"
SKIP_EMB="${SKIP_EMB:-0}"
SKIP_PANEL="${SKIP_PANEL:-0}"
EPOCHS="${EPOCHS:-30}"
TOP_K="${TOP_K:-50}"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint 不存在: $CKPT" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: manifest 不存在: $MANIFEST" >&2
  exit 1
fi

echo "======== $(date -Is) judge-300m ========"
echo "workdir: $WORKDIR"
echo "ckpt:    $CKPT"
echo "device:  $DEVICE"

mkdir -p "$EMB_DIR" "$(dirname "$PANEL")"

if [[ "$SKIP_EMB" != "1" ]]; then
  for split in train val test; do
    out="$EMB_DIR/${split}.parquet"
    if [[ -f "$out" && "${FORCE_EMB:-0}" != "1" ]]; then
      echo "==> skip embedding $split (exists: $out)"
      continue
    fi
    echo "==> extract embedding: $split"
    uv run python -m quant_fm.embedding.extract_hidden \
      --checkpoint "$CKPT" \
      --manifest "$MANIFEST" \
      --split "$split" \
      --out "$out" \
      --context 2048 \
      --pooling mean \
      --device "$DEVICE"
  done
  # 合并一份 all.parquet，便于 panel --from-embeddings
  uv run python - <<PY
import polars as pl
from pathlib import Path
emb = Path("$EMB_DIR")
parts = [pl.read_parquet(emb / f"{s}.parquet") for s in ("train", "val", "test")]
all_df = pl.concat(parts, how="vertical_relaxed")
all_df.write_parquet(emb / "all.parquet")
print(f"wrote {emb/'all.parquet'} rows={all_df.height}")
PY
else
  echo "==> SKIP_EMB=1"
fi

if [[ "$SKIP_PANEL" != "1" ]]; then
  echo "==> build daily panel from MinIO snapshots"
  # 优先用 embeddings 推断 date/symbol；否则用日期文件
  if [[ -f "$EMB_DIR/all.parquet" ]]; then
    uv run python -m quant_fm.downstream.build_panel_from_minio \
      --from-embeddings "$EMB_DIR/all.parquet" \
      --out "$PANEL"
  else
    uv run python -m quant_fm.downstream.build_panel_from_minio \
      --dates-file "$DATES" \
      --out "$PANEL"
  fi
else
  echo "==> SKIP_PANEL=1"
fi

if [[ ! -f "$PANEL" ]]; then
  echo "ERROR: panel 不存在: $PANEL" >&2
  exit 1
fi

echo "==> run_judge"
uv run python -m quant_fm.downstream.run_judge \
  --workdir "$WORKDIR" \
  --checkpoint "$CKPT" \
  --panel "$PANEL" \
  --emb-dir "$EMB_DIR" \
  --epochs "$EPOCHS" \
  --device "$DEVICE" \
  --top-k "$TOP_K"

echo "======== $(date -Is) DONE judge-300m ========"
echo "  report: $WORKDIR/downstream/latest.json"
