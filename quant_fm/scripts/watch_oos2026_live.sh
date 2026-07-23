#!/usr/bin/env bash
# oos2026 实时进度看板（在你自己的终端里跑，每 REFRESH 秒刷新一屏）。
#
# 用法（在交互终端里）：
#   bash quant_fm/scripts/watch_oos2026_live.sh
#   REFRESH=3 bash quant_fm/scripts/watch_oos2026_live.sh   # 自定义刷新间隔

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORKDIR="quant_fm/runs/oos2026"
DATES_FILE="quant_fm/data/oos2026_dates.txt"
PIPELINE_LOG="$WORKDIR/pipeline2.log"
WATCHDOG_LOG="$WORKDIR/watchdog.log"
OOS_LOG="$WORKDIR/oos_downstream.log"
REFRESH="${REFRESH:-5}"
TOTAL_DAYS="$(grep -cve '^[[:space:]]*$' "$DATES_FILE" 2>/dev/null || echo 61)"

alive() { pgrep -f "$1" >/dev/null 2>&1 && echo "🟢 运行中" || echo "🔴 未运行"; }

while true; do
  now="$(date '+%Y-%m-%d %H:%M:%S')"
  wd="$(alive 'watchdog_oos2026')"
  pl="$(alive 'python3 -m quant_fm.scripts.run_medium.*oos2026')"
  orch="$(alive 'run_oos2026_downstream')"

  done_days="$(grep -ac 'day done (tokenized)' "$PIPELINE_LOG" 2>/dev/null || true)"
  done_days="${done_days:-0}"
  tokens="$(find "$WORKDIR/tokens" -name '*.parquet' 2>/dev/null | wc -l)"
  last="$(tail -1 "$PIPELINE_LOG" 2>/dev/null | grep -aoE 'clean progress (SZ|SH) [0-9]+/[0-9]+|canonicalize progress [0-9-]+ [0-9]+/[0-9]+|canonicalized [0-9]+|dropped clean/[0-9-]+|tokenize_dir day=[0-9-]+ shards=[0-9]+|day done \(tokenized\): [0-9-]+' | tail -1)"
  disk="$(df -h . | awk 'NR==2{print $4" 空闲 / "$2}')"
  log_age="?"
  [[ -f "$PIPELINE_LOG" ]] && log_age="$(( $(date +%s) - $(stat -c %Y "$PIPELINE_LOG") ))s前"

  # 进度条
  pct=$(( done_days * 100 / TOTAL_DAYS ))
  filled=$(( done_days * 40 / TOTAL_DAYS ))
  bar="$(printf '%*s' "$filled" '' | tr ' ' '#')$(printf '%*s' $((40-filled)) '' | tr ' ' '-')"

  clear
  cat <<EOF
================ oos2026 实时看板  $now ================

  看门狗   : $wd     流水线 : $pl     下游编排 : $orch

  tokenize 进度 : [$bar] $done_days/$TOTAL_DAYS 天 ($pct%)
  tokens 分片   : $tokens
  当前活动      : ${last:-（等待中/日志无匹配）}   [日志 $log_age 更新]
  磁盘          : $disk

  --- 看门狗最近 3 条 ---
$(tail -3 "$WATCHDOG_LOG" 2>/dev/null | sed 's/^/  /')
EOF
  if [[ -f "$OOS_LOG" ]]; then
    echo "  --- 下游编排最近 2 条 ---"
    tail -2 "$OOS_LOG" 2>/dev/null | sed 's/^/  /'
  fi
  echo
  echo "  (Ctrl-C 退出看板，不影响后台任务；每 ${REFRESH}s 刷新)"
  sleep "$REFRESH"
done
