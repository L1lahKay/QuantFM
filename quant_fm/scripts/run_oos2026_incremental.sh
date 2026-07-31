#!/usr/bin/env bash
# 2026 OOS「增量出 score」编排：不等 61 天全 tokenize 完，
# 对**已完成整天**滚动抽 embedding → 跨期打分，先交付第一版真·OOS 信号，
# 后续天数补齐后自动覆盖更新。
#
# 关键设计：
#   * 以 pipeline 日志里 "day done (tokenized): <date>" 为「整天完成」的唯一真相，
#     避免抽到半写入的当天分片。
#   * embedding 只抽**新天**（增量），累积进 oos_all.parquet，绝不重抽已抽过的天。
#   * 历史期只训练一次冻结 Ranker，随后复用 signal.generate 做 2026 无标签打分；
#     最新尚无未来收益的日期也能立即产生 score。
#   * 数据生产进程结束且无新天可抽 → 做最后一遍打分后退出。
#
# 用法：
#   nohup bash quant_fm/scripts/run_oos2026_incremental.sh \
#     > quant_fm/runs/oos2026/incremental.log 2>&1 &
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ---- 路径与参数（均可用环境变量覆盖）----
TRAIN_WORKDIR="${TRAIN_WORKDIR:-quant_fm/runs/medium_300m}"
OOS_WORKDIR="${OOS_WORKDIR:-quant_fm/runs/oos2026}"
CKPT="${CKPT:-$TRAIN_WORKDIR/run/best.pt}"
VOCAB="${VOCAB:-$TRAIN_WORKDIR/data/vocab.json}"
TRAIN_EMB_DIR="${TRAIN_EMB_DIR:-$TRAIN_WORKDIR/embeddings}"
TRAIN_EMB="${TRAIN_EMB:-$TRAIN_EMB_DIR/all.parquet}"
TRAIN_PANEL="${TRAIN_PANEL:-$TRAIN_WORKDIR/panel/daily_panel.parquet}"
TRAIN_CALENDAR="${TRAIN_CALENDAR:-}"
TRAIN_UNIVERSE="${TRAIN_UNIVERSE:-}"
OOS_UNIVERSE="${OOS_UNIVERSE:-}"
SIGNAL_ARTIFACT="${SIGNAL_ARTIFACT:-$TRAIN_WORKDIR/signal_artifact}"
TOKENS_DIR="${TOKENS_DIR:-$OOS_WORKDIR/tokens}"
PIPELINE_LOG="${PIPELINE_LOG:-$OOS_WORKDIR/pipeline2.log}"
DATES_FILE="${DATES_FILE:-quant_fm/data/oos2026_dates.txt}"

INCR_DIR="$OOS_WORKDIR/embeddings/incr"
EMBEDDED_DATES="$INCR_DIR/embedded_dates.txt"   # 已抽过 embedding 的天（累积）
OOS_ALL="$INCR_DIR/oos_all.parquet"             # 累积 2026 embedding（全部已抽天）
DELIVERY="${DELIVERY:-$OOS_WORKDIR/delivery_oos}"

NPROC="${NPROC:-8}"
BATCH="${BATCH:-16}"
DTYPE="${DTYPE:-bf16}"
CONTEXT="${CONTEXT-}"
POOLING="${POOLING-}"
STRIDE="${STRIDE-}"
MIN_DAYS="${MIN_DAYS:-1}"           # 至少一个完整信号日即可开始打分
INTERVAL="${INTERVAL:-600}"          # 巡检间隔（秒）
TOTAL_DAYS="$(grep -cve '^[[:space:]]*$' "$DATES_FILE" 2>/dev/null || echo 61)"

mkdir -p "$INCR_DIR"
touch "$EMBEDDED_DATES"

log() { echo "[$(date -Is)] $*"; }

# 已「整天完成 tokenize」的日期。权威来源 = run_medium 写的 data/.done/<date> 标记
# （内容为 "tokenized"），与串行/并行编排方式无关，且不受日志截断影响。
DONE_MARKERS="$OOS_WORKDIR/data/.done"
completed_dates() {
  [[ -d "$DONE_MARKERS" ]] || return 0
  for f in "$DONE_MARKERS"/2026-*; do
    [[ -f "$f" ]] || continue
    # 只认已 tokenize 的天（canonicalized-only 的旧标记不算）。
    if grep -q tokenized "$f" 2>/dev/null; then
      basename "$f"
    fi
  done | sort -u
}

pipeline_alive() {
  # 串行 run_medium 或并行驱动，任一在跑都算数据生产未结束。
  pgrep -f 'quant_fm.scripts.run_medium' >/dev/null 2>&1 \
    || pgrep -f 'run_medium_parallel_days' >/dev/null 2>&1
}

# 对累积 embedding 跑一遍跨期打分（覆盖更新 DELIVERY）。
score_now() {
  local n_days
  n_days="$(wc -l < "$EMBEDDED_DATES" | tr -d ' ')"
  if [[ "$n_days" -lt "$MIN_DAYS" ]]; then
    log "已抽 $n_days 天 < MIN_DAYS=$MIN_DAYS，暂不打分（继续累积）。"
    return 0
  fi
  log "跨期无标签打分：冻结 Ranker → 2026($n_days 天) score …"
  uv run python -m quant_fm.signal.generate \
    --embeddings "$OOS_ALL" \
    --ranker "$SIGNAL_ARTIFACT/ranker.pt" \
    --ranker-metadata "$SIGNAL_ARTIFACT/ranker_metadata.json" \
    --fm-checkpoint "$CKPT" \
    --vocab "$VOCAB" \
    --universe "$OOS_UNIVERSE" \
    --out-dir "$DELIVERY" \
    --device cuda:0 \
    && log "✅ 交付更新 → $DELIVERY（覆盖 $n_days 天）" \
    || log "⚠️ 打分失败（见上）；下轮重试。"
}

log "======== 增量编排启动 (total=$TOTAL_DAYS, min_days=$MIN_DAYS, interval=${INTERVAL}s) ========"
log "训练侧: emb=$TRAIN_EMB_DIR panel=$TRAIN_PANEL ckpt=$CKPT"
log "OOS 侧: tokens=$TOKENS_DIR（生产打分不读取未来收益 panel）"

for f in "$CKPT" "$VOCAB" "$TRAIN_EMB" "$TRAIN_PANEL" "$TRAIN_CALENDAR" "$TRAIN_UNIVERSE" "$OOS_UNIVERSE"; do
  [[ -e "$f" ]] || { log "❌ 缺少必要输入: $f；退出。"; exit 1; }
done

if [[ ! -f "$SIGNAL_ARTIFACT/ranker.pt" ]]; then
  log "离线训练并冻结 Ranker（此后增量出分不再读取标签）…"
  uv run python -m quant_fm.signal.train \
    --embeddings "$TRAIN_EMB" \
    --panel "$TRAIN_PANEL" \
    --calendar "$TRAIN_CALENDAR" \
    --universe "$TRAIN_UNIVERSE" \
    --out-dir "$SIGNAL_ARTIFACT" \
    --device cuda:0 || exit 1
fi

while true; do
  # 1) 计算「已完成 - 已抽」= 本轮新天
  comp_tmp="$INCR_DIR/.completed.txt"
  completed_dates > "$comp_tmp"
  n_comp="$(wc -l < "$comp_tmp" | tr -d ' ')"
  new_tmp="$INCR_DIR/.new_dates.txt"
  sort -u "$EMBEDDED_DATES" -o "$EMBEDDED_DATES"
  comm -23 "$comp_tmp" "$EMBEDDED_DATES" > "$new_tmp"
  n_new="$(wc -l < "$new_tmp" | tr -d ' ')"
  n_emb="$(wc -l < "$EMBEDDED_DATES" | tr -d ' ')"
  log "巡检: 已完成 $n_comp/$TOTAL_DAYS 天, 已抽 $n_emb 天, 本轮新增 $n_new 天"

  # 2) 有新天 → 抽 embedding（增量）
  if [[ "$n_new" -gt 0 ]]; then
    cycle="$INCR_DIR/_cycle"
    rm -rf "$cycle"; mkdir -p "$cycle"
    log "生成临时 manifest（白名单=本轮新天）…"
    uv run python -m quant_fm.scripts.make_adhoc_manifest \
      --tokens-dir "$TOKENS_DIR" \
      --out "$cycle/manifest.json" \
      --split test \
      --include-dates-file "$new_tmp" \
      --vocab "$VOCAB"

    log "多卡抽 embedding（$n_new 新天）…"
    if WORKDIR="$OOS_WORKDIR" CKPT="$CKPT" MANIFEST="$cycle/manifest.json" \
       EMB_DIR="$cycle" SPLIT=test NPROC="$NPROC" BATCH="$BATCH" DTYPE="$DTYPE" \
       CONTEXT="$CONTEXT" POOLING="$POOLING" STRIDE="$STRIDE" \
       bash quant_fm/scripts/extract_embeddings_parallel.sh; then
      # 3) 累积进 oos_all.parquet（去重保最新）+ 记账
      uv run python - "$cycle/test.parquet" "$OOS_ALL" <<'PY'
import sys
from pathlib import Path
import polars as pl
from quant_fm.embedding.contract import (
    load_compatible_embedding_contracts,
    validate_embedding_columns,
    write_embedding_contract,
)
new_p, all_p = Path(sys.argv[1]), Path(sys.argv[2])
new = pl.read_parquet(new_p)
contract_sources = [new_p]
if all_p.exists():
    old = pl.read_parquet(all_p)
    contract_sources.insert(0, all_p)
    merged = pl.concat([old, new], how="vertical_relaxed")
    merged = merged.unique(subset=["date", "symbol"], keep="last")
else:
    merged = new
merged = merged.sort(["date", "symbol"])
contract = load_compatible_embedding_contracts(
    contract_sources,
    required=True,
    context="incremental OOS embeddings",
)
if contract is None:
    raise SystemExit("missing incremental OOS embedding representation contract")
merged.write_parquet(all_p)
validate_embedding_columns(merged.columns, contract, context=str(all_p))
write_embedding_contract(all_p, contract)
print(f"oos_all rows={merged.height} days={merged['date'].n_unique()}")
PY
      cat "$new_tmp" >> "$EMBEDDED_DATES"
      sort -u "$EMBEDDED_DATES" -o "$EMBEDDED_DATES"
      rm -rf "$cycle"
    else
      log "⚠️ embedding 抽取失败；本轮不记账，下轮重试。"
    fi
  fi

  # 4) 打分（累计天数够就覆盖更新交付）
  score_now
  n_emb="$(wc -l < "$EMBEDDED_DATES" | tr -d ' ')"

  # 5) 退出条件：数据生产结束且无新天可抽 → 已追平，做最后一遍后收工
  if ! pipeline_alive && [[ "$n_new" -eq 0 ]]; then
    log "数据生产进程已结束且无新天可抽（已抽 $n_emb 天）；增量编排收工。"
    exit 0
  fi

  sleep "$INTERVAL"
done
