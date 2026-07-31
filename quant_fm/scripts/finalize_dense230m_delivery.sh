#!/usr/bin/env bash
# 等待 Dense230M 真实信号就绪，执行 Ranker 输入审计、信号门禁和冻结打包。
# 该脚本只读生产输入；报告和最终回测包写入独立目录，默认绝不覆盖。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
WORKDIR="${WORKDIR:-quant_fm/runs/oos2026_dense230}"
TRAIN_EMBEDDINGS="${TRAIN_EMBEDDINGS:-quant_fm/runs/dense_230m_v1/embeddings/all.parquet}"
TRAIN_PANEL="${TRAIN_PANEL:-quant_fm/runs/cont60/panel/daily_panel_cont60.parquet}"
OOS_EMBEDDINGS="${OOS_EMBEDDINGS:-$WORKDIR/embeddings/all.parquet}"
SOURCE_DIR="${SOURCE_DIR:-$WORKDIR/delivery_oos}"
DATES_FILE="${DATES_FILE:-quant_fm/data/oos2026_dates.txt}"
ACCEPTANCE_DIR="${ACCEPTANCE_DIR:-$WORKDIR/acceptance_v1}"
PACKAGE_DIR="${PACKAGE_DIR:-$WORKDIR/backtest_delivery_v1}"
ARCHIVE="${ARCHIVE:-$WORKDIR/backtest_delivery_v1.tar.gz}"
BASELINE_SCORES="${BASELINE_SCORES:-}"
EVAL_PANEL="${EVAL_PANEL:-}"
WAIT="${WAIT:-1}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MIN_NAMES_PER_DAY="${MIN_NAMES_PER_DAY:-20}"
TOP_K="${TOP_K:-50}"

log() { echo "[$(date -Is)] $*"; }

required_inputs=(
  "$TRAIN_EMBEDDINGS"
  "$TRAIN_PANEL"
  "$OOS_EMBEDDINGS"
  "$SOURCE_DIR/scores.parquet"
  "$SOURCE_DIR/signal_manifest.json"
  "$DATES_FILE"
)

missing_inputs() {
  local path missing=0
  for path in "${required_inputs[@]}"; do
    if [[ ! -f "$path" ]]; then
      echo "$path"
      missing=1
    fi
  done
  return "$missing"
}

if [[ -f "$PACKAGE_DIR/delivery_manifest.json" && -f "$ARCHIVE" ]]; then
  log "交付包已存在且看起来完整；不覆盖：$PACKAGE_DIR"
  exit 0
fi

while mapfile -t pending < <(missing_inputs); ((${#pending[@]} > 0)); do
  if [[ "$WAIT" != "1" ]]; then
    log "输入尚未就绪且 WAIT=$WAIT：${pending[*]}"
    exit 3
  fi
  log "等待真实信号输入（缺 ${#pending[@]} 项）：${pending[*]}"
  sleep "$POLL_SECONDS"
done

ranker_report_dir="$ACCEPTANCE_DIR/ranker_inputs"
signal_report_dir="$ACCEPTANCE_DIR/signal_quality"
mkdir -p "$ranker_report_dir" "$signal_report_dir"

log "审计 Ranker 历史/OOS 输入与严格时间隔离…"
"$PYTHON" -m quant_fm.scripts.audit_ranker_inputs \
  --train-embeddings "$TRAIN_EMBEDDINGS" \
  --train-panel "$TRAIN_PANEL" \
  --oos-embeddings "$OOS_EMBEDDINGS" \
  --min-train-coverage 0.95 \
  --out-dir "$ranker_report_dir"

quality_args=(
  -m quant_fm.scripts.signal_quality_gate
  --scores "$SOURCE_DIR/scores.parquet"
  --manifest "$SOURCE_DIR/signal_manifest.json"
  --expected-dates "$DATES_FILE"
  --min-names-per-day "$MIN_NAMES_PER_DAY"
  --top-k "$TOP_K"
  --out-dir "$signal_report_dir"
)
if [[ -n "$EVAL_PANEL" ]]; then
  [[ -f "$EVAL_PANEL" ]] || { log "EVAL_PANEL 不存在：$EVAL_PANEL"; exit 4; }
  quality_args+=(--panel "$EVAL_PANEL")
fi
if [[ -n "$BASELINE_SCORES" ]]; then
  [[ -f "$BASELINE_SCORES" ]] || { log "BASELINE_SCORES 不存在：$BASELINE_SCORES"; exit 4; }
  quality_args+=(--baseline-scores "$BASELINE_SCORES")
fi

log "执行信号 schema、覆盖率、分布、换手与可选基线门禁…"
"$PYTHON" "${quality_args[@]}"

log "冻结回测交付目录并生成完整 SHA-256 与 tar.gz…"
"$PYTHON" -m quant_fm.scripts.package_signal_delivery \
  --source-dir "$SOURCE_DIR" \
  --out-dir "$PACKAGE_DIR" \
  --archive "$ARCHIVE" \
  --report "$ranker_report_dir/ranker_input_audit.json" \
  --report "$ranker_report_dir/ranker_input_audit.md" \
  --report "$signal_report_dir/signal_quality.json" \
  --report "$signal_report_dir/signal_quality.md"

log "Dense230M 回测交付完成：$PACKAGE_DIR"
log "压缩包：$ARCHIVE"
