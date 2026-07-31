#!/usr/bin/env bash
# Dense230M 下游全链路实时终端看板。
#
# 用法：
#   bash quant_fm/scripts/watch_dense230m_downstream.sh
#   REFRESH=10 bash quant_fm/scripts/watch_dense230m_downstream.sh
#   ONCE=1 NO_CLEAR=1 bash quant_fm/scripts/watch_dense230m_downstream.sh

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORKDIR="${WORKDIR:-quant_fm/runs/oos2026_dense230}"
DATES_FILE="${DATES_FILE:-quant_fm/data/oos2026_dates.txt}"
HIST_EMB_DIR="${HIST_EMB_DIR:-quant_fm/runs/dense_230m_v1/embeddings}"
HIST_TOTAL_SHARDS="${HIST_TOTAL_SHARDS:-310636}"
NAMESPACE="${NAMESPACE:-khalil}"
HIST_JOB="${HIST_JOB:-dense230m-historical-embeddings-20260728}"
RELEASE_JOB="${RELEASE_JOB:-dense230m-oos-release-v2-20260728}"
OOS_JOB="${OOS_JOB:-dense230m-oos-signal-20260728}"
REFRESH="${REFRESH:-5}"
ONCE="${ONCE:-0}"
NO_CLEAR="${NO_CLEAR:-0}"
SKIP_K8S="${SKIP_K8S:-0}"

TOTAL_DAYS="$(grep -cve '^[[:space:]]*$' "$DATES_FILE" 2>/dev/null || echo 61)"
DONE_DIR="$WORKDIR/data/.done"
MANIFEST="$WORKDIR/data/manifest.json"
DELIVERY_DIR="$WORKDIR/delivery_oos"

bar() {
  local done="$1" total="$2" width="${3:-32}" filled pct
  if (( total <= 0 )); then
    printf '[%-*s] n/a' "$width" ''
    return
  fi
  (( done > total )) && done="$total"
  filled=$((done * width / total))
  pct=$((done * 100 / total))
  printf '[%s%s] %d/%d (%d%%)' \
    "$(printf '%*s' "$filled" '' | tr ' ' '#')" \
    "$(printf '%*s' $((width - filled)) '' | tr ' ' '-')" \
    "$done" "$total" "$pct"
}

done_days() {
  local count=0 date
  while IFS= read -r date; do
    [[ -f "$DONE_DIR/$date" ]] && \
      grep -q '^tokenized' "$DONE_DIR/$date" 2>/dev/null && count=$((count + 1))
  done < <(grep -ve '^[[:space:]]*$' "$DATES_FILE")
  echo "$count"
}

group_summary() {
  local log="$1" date latest age=0
  if [[ ! -f "$log" ]]; then
    echo "尚无日志"
    return
  fi
  date="$(rg -N 'clean\(fast\) [0-9]{4}-[0-9]{2}-[0-9]{2}' "$log" 2>/dev/null \
    | tail -1 | sed -n 's/.*clean(fast) \([0-9-]*\).*/\1/p')"
  latest="$(rg -N 'clean progress|retry failed symbols|fused tokenize progress|day done \(tokenized\)|raw 缓存已落盘|MinIO 瞬时读取失败' "$log" 2>/dev/null | tail -1)"
  age=$(( $(date +%s) - $(stat -c %Y "$log") ))
  printf '%s  %s  [日志%ds前]\n' "${date:-等待日期}" "${latest:-等待新进度}" "$age"
}

split_progress() {
  local directory="$1" split="$2" file line done=0 total=0 value
  shopt -s nullglob
  for file in "$directory"/parts/"$split".part*of*.log; do
    line="$(rg -N 'embedded [0-9]+/[0-9]+ shards' "$file" 2>/dev/null | tail -1)"
    [[ -n "$line" ]] || continue
    value="$(sed -n 's/.*embedded \([0-9][0-9]*\)\/\([0-9][0-9]*\) shards.*/\1 \2/p' <<< "$line")"
    [[ -n "$value" ]] || continue
    done=$((done + ${value%% *}))
    total=$((total + ${value##* }))
  done
  shopt -u nullglob
  echo "$done $total"
}

all_embedding_done() {
  local directory="$1" split values done total aggregate=0
  for split in train val test; do
    values="$(split_progress "$directory" "$split")"
    done="${values%% *}"
    total="${values##* }"
    (( total > 0 )) && aggregate=$((aggregate + done))
  done
  echo "$aggregate"
}

current_split() {
  local directory="$1"
  if compgen -G "$directory/parts/test.part*of*.log" >/dev/null; then
    echo test
  elif compgen -G "$directory/parts/val.part*of*.log" >/dev/null; then
    echo val
  else
    echo train
  fi
}

job_phase() {
  local table="$1" job="$2" phase
  phase="$(awk -v name="$job" '$1 == name {print $2 " (" $3 ")"}' <<< "$table")"
  echo "${phase:-Unknown}"
}

artifact_state() {
  local path="$1"
  if [[ -f "$path" ]]; then
    printf 'READY (%s)' "$(du -h "$path" 2>/dev/null | awk '{print $1}')"
  else
    printf '等待中'
  fi
}

while true; do
  now="$(date -Is)"
  done_count="$(done_days)"
  pipeline_state="未运行"
  pgrep -f 'quant_fm.scripts.run_medium.*oos2026_dense230' >/dev/null 2>&1 && \
    pipeline_state="运行中"

  hist_split="$(current_split "$HIST_EMB_DIR")"
  hist_split_values="$(split_progress "$HIST_EMB_DIR" "$hist_split")"
  hist_split_done="${hist_split_values%% *}"
  hist_split_total="${hist_split_values##* }"
  hist_done="$(all_embedding_done "$HIST_EMB_DIR")"

  oos_split_values="$(split_progress "$WORKDIR/embeddings" test)"
  oos_emb_done="${oos_split_values%% *}"
  oos_emb_total="${oos_split_values##* }"

  if [[ "$SKIP_K8S" == "1" ]]; then
    jobs_table=""
    hist_phase="跳过查询"
    release_phase="跳过查询"
    oos_phase="跳过查询"
  else
    jobs_table="$(kubectl --request-timeout=4s -n "$NAMESPACE" get jobs \
      "$HIST_JOB" "$RELEASE_JOB" "$OOS_JOB" --no-headers 2>/dev/null || true)"
    hist_phase="$(job_phase "$jobs_table" "$HIST_JOB")"
    release_phase="$(job_phase "$jobs_table" "$RELEASE_JOB")"
    oos_phase="$(job_phase "$jobs_table" "$OOS_JOB")"
  fi

  manifest_bytes=0
  [[ -f "$MANIFEST" ]] && manifest_bytes="$(stat -c %s "$MANIFEST")"
  fatal_count="$(rg -c '502 Bad Gateway|Traceback|FATAL' \
    "$WORKDIR"/parallel/group*.log "$WORKDIR/parallel/driver.log" 2>/dev/null \
    | awk -F: '{sum += $NF} END {print sum + 0}')"
  timeout_count="$(rg -c 'ERROR failed symbol=.*TimeoutError' \
    "$WORKDIR"/parallel/group*.log 2>/dev/null \
    | awk -F: '{sum += $NF} END {print sum + 0}')"
  disk_free="$(df -h "$WORKDIR" | awk 'NR == 2 {print $4}')"
  gpu_summary="$(nvidia-smi --query-gpu=utilization.gpu \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk '{sum += $1; count += 1} END {if (count) printf "avg %.0f%% / %d GPUs", sum/count, count; else print "n/a"}')"

  if [[ "$NO_CLEAR" != "1" && -t 1 ]]; then
    printf '\033[H\033[2J'
  fi
  cat <<EOF
================ Dense230M 下游信号实时看板 ================
时间: $now    刷新: ${REFRESH}s    磁盘剩余: $disk_free

1) OOS 数据准备（MinIO -> clean -> tokens -> manifest）
   $(bar "$done_count" "$TOTAL_DAYS")    进程: $pipeline_state
   G0: $(group_summary "$WORKDIR/parallel/group0.log")
   G1: $(group_summary "$WORKDIR/parallel/group1.log")
   manifest: ${manifest_bytes} bytes    致命错误: $fatal_count    单股超时待重试: $timeout_count

2) 历史 embedding（Ranker 训练输入，8 x RTX 5090）
   K8s: $hist_phase    GPU: $gpu_summary
   当前 $hist_split: $(bar "$hist_split_done" "$hist_split_total")
   全历史: $(bar "$hist_done" "$HIST_TOTAL_SHARDS")

3) 自动放行门禁
   release-v2: $release_phase
   条件: historical=Complete + OOS days=61/61 + manifest>1000 bytes

4) OOS embedding -> Ranker -> 信号交付
   K8s OOS Job: $oos_phase
   OOS embedding: $(bar "$oos_emb_done" "$oos_emb_total")
   Ranker checkpoint: $(artifact_state "$DELIVERY_DIR/ranker_checkpoint.pt")
   scores.parquet:    $(artifact_state "$DELIVERY_DIR/scores.parquet")
   signal manifest:   $(artifact_state "$DELIVERY_DIR/signal_manifest.json")

Ctrl-C 只退出看板，不影响后台流水线或 K8s Job。
EOF

  [[ "$ONCE" == "1" ]] && break
  sleep "$REFRESH"
done
