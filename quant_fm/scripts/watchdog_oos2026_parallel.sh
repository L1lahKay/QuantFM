#!/usr/bin/env bash
# oos2026 跨日并行流水线看门狗：监督 run_medium_parallel_days.sh，
# 崩溃/中断 → --resume 自动重启（fast-clean 本地缓存 + 标的级跳过，重启秒级续跑）；
# 全部 tokenize 完成（manifest 出现或达到 TOTAL_DAYS）→ 退出。
#
# 用法：
#   nohup bash quant_fm/scripts/watchdog_oos2026_parallel.sh \
#     > quant_fm/runs/oos2026/watchdog_parallel.log 2>&1 &
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORKDIR="${WORKDIR:-quant_fm/runs/oos2026}"
DATES_FILE="${DATES_FILE:-quant_fm/data/oos2026_dates.txt}"
NGROUPS="${NGROUPS:-2}"
CLEAN_WORKERS="${CLEAN_WORKERS:-26}"
TOKENIZE_WORKERS="${TOKENIZE_WORKERS:-8}"
CHECK_EVERY="${CHECK_EVERY:-120}"
STALL_LIMIT="${STALL_LIMIT:-7200}"     # 2h：清洗静默期本就长，宁等勿误杀
DRIVER_LOG="$WORKDIR/parallel/driver.log"
ENV_FILE="$HOME/.minio_fm_env.sh"
TOTAL_DAYS="$(grep -cve '^[[:space:]]*$' "$DATES_FILE")"

log() { echo "[$(date -Is)] $*"; }

driver_alive() { pgrep -f 'run_medium_parallel_days' >/dev/null 2>&1; }

done_days() {
  # 权威来源 = data/.done/<date> 标记（内容含 tokenized），不受组日志截断影响。
  local dir="$WORKDIR/data/.done" n=0
  [[ -d "$dir" ]] || { echo 0; return; }
  for f in "$dir"/2026-*; do
    [[ -f "$f" ]] && grep -q tokenized "$f" 2>/dev/null && n=$((n+1))
  done
  echo "$n"
}

start_driver() {
  [[ -f "$ENV_FILE" ]] && source "$ENV_FILE"
  log "启动并行驱动（NGROUPS=$NGROUPS CLEAN_WORKERS=$CLEAN_WORKERS，resume）…"
  NGROUPS="$NGROUPS" CLEAN_WORKERS="$CLEAN_WORKERS" TOKENIZE_WORKERS="$TOKENIZE_WORKERS" \
    nohup bash quant_fm/scripts/run_medium_parallel_days.sh >> "$DRIVER_LOG" 2>&1 &
  log "已启动 pid=$!"
}

log "======== 并行看门狗启动 (total=$TOTAL_DAYS, ngroups=$NGROUPS, stall=${STALL_LIMIT}s) ========"
mkdir -p "$WORKDIR/parallel"

restarts=0
while true; do
  dn="$(done_days)"
  if [[ -f "$WORKDIR/data/manifest.json" ]] || [[ "$dn" -ge "$TOTAL_DAYS" ]]; then
    log "✅ 完成 (done=$dn/$TOTAL_DAYS, manifest=$([[ -f $WORKDIR/data/manifest.json ]] && echo yes || echo no))；退出。"
    exit 0
  fi

  if ! driver_alive; then
    log "⚠️ 驱动不在运行 (done=$dn/$TOTAL_DAYS)；重启。"
    start_driver; restarts=$((restarts+1)); sleep "$CHECK_EVERY"; continue
  fi

  # 卡死检测：任一组日志的最新 mtime 超过阈值未更新。
  newest=0
  for f in "$WORKDIR"/parallel/group*.log; do
    [[ -f "$f" ]] || continue
    m=$(stat -c %Y "$f"); (( m > newest )) && newest=$m
  done
  if (( newest > 0 )); then
    age=$(( $(date +%s) - newest ))
    if (( age > STALL_LIMIT )); then
      log "⚠️ 组日志 ${age}s 无更新 > ${STALL_LIMIT}s，判定卡死；杀掉重启。"
      pkill -9 -f 'run_medium_parallel_days' 2>/dev/null || true
      pkill -9 -f 'quant_fm.scripts.run_medium' 2>/dev/null || true
      sleep 3; start_driver; restarts=$((restarts+1)); sleep "$CHECK_EVERY"; continue
    fi
  fi

  log "ok done=$dn/$TOTAL_DAYS 组日志${age:-?}s前更新 (累计重启 $restarts)"
  sleep "$CHECK_EVERY"
done
