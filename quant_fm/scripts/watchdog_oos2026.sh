#!/usr/bin/env bash
# oos2026 流水线看门狗：进程死亡 / 长时间无进展 → 自动杀掉重启（--resume 续跑）。
# 全部 61 天 tokenize 完成后自动退出（交给 run_oos2026_downstream.sh 接力）。
#
# 用法：
#   nohup bash quant_fm/scripts/watchdog_oos2026.sh > quant_fm/runs/oos2026/watchdog.log 2>&1 &

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORKDIR="quant_fm/runs/oos2026"
DATES_FILE="quant_fm/data/oos2026_dates.txt"
VOCAB="quant_fm/runs/medium_300m/data/vocab.json"
SZ_FILE="${SZ_FILE:-quant_fm/data/oos2026_liquid_sz.txt}"
SH_FILE="${SH_FILE:-quant_fm/data/oos2026_liquid_sh.txt}"
PIPELINE_LOG="$WORKDIR/pipeline2.log"
ENV_FILE="$HOME/.minio_fm_env.sh"

CHECK_EVERY="${CHECK_EVERY:-120}"      # 每 2 分钟巡检一次
# 卡死阈值放到 2 小时：清洗单日的订单簿重建阶段本就静默 30-40 分钟，
# 阈值太小会误杀健康进程→反复重下→恶性循环（这正是上次空转 12h 的原因）。
# 原则：宁可多等，绝不误杀；只有进程真正“死亡”才立即重启（resume 续跑）。
STALL_LIMIT="${STALL_LIMIT:-7200}"
TOTAL_DAYS="$(grep -cve '^[[:space:]]*$' "$DATES_FILE")"

log() { echo "[$(date -Is)] $*"; }

pipeline_pid() {
  pgrep -f "python3 -m quant_fm.scripts.run_medium.*oos2026" | head -1
}

done_days() {
  local dir="$WORKDIR/data/.done" n=0
  [[ -d "$dir" ]] || { echo 0; return; }
  while IFS= read -r date; do
    [[ -f "$dir/$date" ]] && grep -q '^tokenized' "$dir/$date" 2>/dev/null && n=$((n+1))
  done < <(grep -ve '^[[:space:]]*$' "$DATES_FILE")
  echo "$n"
}

manifest_ready() {
  python3 - "$WORKDIR/data/manifest.json" "$DATES_FILE" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

manifest_path, dates_path = map(Path, sys.argv[1:3])
payload = json.loads(manifest_path.read_text())
shards = payload.get("shards", [])
expected = {line.strip() for line in dates_path.read_text().splitlines() if line.strip()}
actual = {item.get("date") for item in shards}
raise SystemExit(0 if shards and expected <= actual else 1)
PY
}

start_pipeline() {
  [[ -f "$ENV_FILE" ]] && source "$ENV_FILE"
  log "启动 run_medium（resume）…"
  TOKENIZE_WORKERS="${TOKENIZE_WORKERS:-16}" CLEAN_WORKERS="${CLEAN_WORKERS:-32}" \
  nohup uv run python -m quant_fm.scripts.run_medium \
    --dates-file "$DATES_FILE" \
    --workdir "$WORKDIR" \
    --reuse-vocab "$VOCAB" \
    --symbols-sz-file "$SZ_FILE" \
    --symbols-sh-file "$SH_FILE" \
    --fast-clean \
    --drop-clean --drop-events --resume \
    >> "$PIPELINE_LOG" 2>&1 &
  log "已启动 pid=$!"
}

kill_pipeline() {
  log "杀掉卡死的 run_medium 全家…"
  pkill -9 -f "run_medium.*oos2026" 2>/dev/null || true
  sleep 3
}

log "======== 看门狗启动 (total_days=$TOTAL_DAYS, check=${CHECK_EVERY}s, stall=${STALL_LIMIT}s) ========"

restarts=0
while true; do
  done_now="$(done_days)"
  # marker 与 manifest 内容都完整才算完成；空/旧 manifest 不再误报成功。
  if [[ "$done_now" -ge "$TOTAL_DAYS" ]] && manifest_ready; then
    log "✅ tokenize 全部完成 (done=$done_now/$TOTAL_DAYS, manifest=valid)；看门狗退出。"
    exit 0
  fi

  pid="$(pipeline_pid)"
  if [[ -z "$pid" ]]; then
    log "⚠️ 未发现 run_medium 进程 (done=$done_now/$TOTAL_DAYS)；重启。"
    start_pipeline; restarts=$((restarts+1)); sleep "$CHECK_EVERY"; continue
  fi

  # 卡死检测：日志文件多久没更新
  if [[ -f "$PIPELINE_LOG" ]]; then
    age=$(( $(date +%s) - $(stat -c %Y "$PIPELINE_LOG") ))
    if [[ "$age" -gt "$STALL_LIMIT" ]]; then
      log "⚠️ 日志 ${age}s 无更新 > ${STALL_LIMIT}s，判定卡死 (done=$done_now/$TOTAL_DAYS)；杀掉重启。"
      kill_pipeline; start_pipeline; restarts=$((restarts+1)); sleep "$CHECK_EVERY"; continue
    fi
  fi

  log "ok pid=$pid done=$done_now/$TOTAL_DAYS 日志${age:-?}s前更新 (累计重启 $restarts)"
  sleep "$CHECK_EVERY"
done
