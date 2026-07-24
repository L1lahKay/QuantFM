#!/usr/bin/env bash
# 对冻结 2026 score 构建严格执行面板并运行 research-only 评估。
# STRICT_TRADABLE=vwap_t1_vwap_t2 需要日历比最后一个信号日至少多两个交易日。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OOS_WORKDIR="${OOS_WORKDIR:-quant_fm/runs/oos2026}"
SCORES="${SCORES:-$OOS_WORKDIR/delivery_oos/scores.parquet}"
CALENDAR="${CALENDAR:-}"
RETURN_SPEC="${RETURN_SPEC:-vwap_t1_vwap_t2}"
RESEARCH_DIR="${RESEARCH_DIR:-$OOS_WORKDIR/research}"
PANEL="${PANEL:-$RESEARCH_DIR/execution_panel.parquet}"

if [[ -z "$CALENDAR" ]]; then
  echo "ERROR: 必须显式设置 CALENDAR，且日历要覆盖最后一个 signal 的 T+2。" >&2
  echo "       quant_fm/data/oos2026_dates.txt 只有 60 个信号日 + 1 个末日，不足以运行 vwap_t1_vwap_t2。" >&2
  exit 2
fi

for path in "$SCORES" "$CALENDAR"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: 缺少输入 $path" >&2
    exit 1
  fi
done

mkdir -p "$RESEARCH_DIR"

echo "==> build strict execution panel ($RETURN_SPEC)"
uv run python -m quant_fm.downstream.build_panel_from_minio \
  --from-embeddings "$SCORES" \
  --calendar-file "$CALENDAR" \
  --return-spec "$RETURN_SPEC" \
  --out "$PANEL"

echo "==> evaluate frozen OOS scores"
uv run python -m quant_fm.downstream.run_score_evaluation \
  --scores "$SCORES" \
  --panel "$PANEL" \
  --out-dir "$RESEARCH_DIR/evaluation"

echo "==> report: $RESEARCH_DIR/evaluation/metrics.json"
