#!/usr/bin/env bash
# 跨日并行清洗驱动（P0-2）：把日期切成 G 份，并发跑 G 个 run_medium（--resume 续跑），
# 各组跳过收尾 manifest，最后由本脚本统一建一次。
#
# 为什么无需改 run_medium：clean/<date>、raw_cache/<date>、data/.done/<date>、
# tokens/<market>/<symbol>/<date>.parquet 全部按 <date> 分文件，不同日期天然不冲突。
#
# 核数预算：每组 clean 用 CLEAN_WORKERS 个进程，NGROUPS*CLEAN_WORKERS 应 ≤ nproc。
#   64 核建议：NGROUPS=2, CLEAN_WORKERS=30（2*30=60，留 4 核给 OS/IO）。
#
# ⚠️ 用 NGROUPS 而非 GROUPS：GROUPS 是 bash 保留只读变量（当前用户的 GID 数组），
#    赋值会被忽略、${GROUPS:-2} 会取到某个 GID（几百），从而启动几百个进程（fork 炸弹）。
#
# 用法：
#   NGROUPS=2 CLEAN_WORKERS=30 TOKENIZE_WORKERS=8 \
#   nohup bash quant_fm/scripts/run_medium_parallel_days.sh \
#     > quant_fm/runs/oos2026/parallel.log 2>&1 &
#
# 基准测试模式（只跑指定日期、跳过收尾 manifest）：额外传 BENCH=1 DATES_FILE=<子集>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORKDIR="${WORKDIR:-quant_fm/runs/oos2026}"
DATES_FILE="${DATES_FILE:-quant_fm/data/oos2026_dates.txt}"
VOCAB="${VOCAB:-quant_fm/runs/medium_300m/data/vocab.json}"
SZ_FILE="${SZ_FILE:-quant_fm/data/oos2026_liquid_sz.txt}"
SH_FILE="${SH_FILE:-quant_fm/data/oos2026_liquid_sh.txt}"
NGROUPS="${NGROUPS:-2}"
CLEAN_WORKERS="${CLEAN_WORKERS:-26}"
TOKENIZE_WORKERS="${TOKENIZE_WORKERS:-8}"

# 防呆：NGROUPS 必须是 1..16 的整数，否则可能误启动海量进程。
if ! [[ "$NGROUPS" =~ ^[0-9]+$ ]] || (( NGROUPS < 1 || NGROUPS > 16 )); then
  echo "FATAL: NGROUPS=$NGROUPS 非法（需 1..16 的整数）" >&2
  exit 2
fi
TRAIN_END="${TRAIN_END:-0000-00-00}"   # 全部日期 > val_end → 归 test（OOS 设计）
VAL_END="${VAL_END:-0000-00-00}"
BENCH="${BENCH:-0}"

SPLIT_DIR="$WORKDIR/parallel"
mkdir -p "$SPLIT_DIR"
ENV_FILE="$HOME/.minio_fm_env.sh"
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"

log() { echo "[$(date -Is)] $*"; }

# 1) 把日期交错切成 NGROUPS 份（round-robin，负载更均衡）。
mapfile -t ALL_DATES < <(grep -ve '^[[:space:]]*$' "$DATES_FILE")
n_dates="${#ALL_DATES[@]}"
for ((g=0; g<NGROUPS; g++)); do : > "$SPLIT_DIR/dates.g${g}.txt"; done
for ((i=0; i<n_dates; i++)); do
  g=$(( i % NGROUPS ))
  echo "${ALL_DATES[$i]}" >> "$SPLIT_DIR/dates.g${g}.txt"
done
log "======== 跨日并行启动 NGROUPS=$NGROUPS CLEAN_WORKERS=$CLEAN_WORKERS (总核约 $((NGROUPS*CLEAN_WORKERS))) 日期=$n_dates ========"

# 2) 并发启动 NGROUPS 个 run_medium。
pids=(); groups_ok=1
for ((g=0; g<NGROUPS; g++)); do
  chunk="$SPLIT_DIR/dates.g${g}.txt"
  glog="$SPLIT_DIR/group${g}.log"
  cnt="$(wc -l < "$chunk" | tr -d ' ')"
  log "  组 $g: $cnt 天 → $glog"
  CLEAN_WORKERS="$CLEAN_WORKERS" TOKENIZE_WORKERS="$TOKENIZE_WORKERS" \
  nohup uv run python -m quant_fm.scripts.run_medium \
    --dates-file "$chunk" \
    --workdir "$WORKDIR" \
    --reuse-vocab "$VOCAB" \
    --symbols-sz-file "$SZ_FILE" \
    --symbols-sh-file "$SH_FILE" \
    --train-end "$TRAIN_END" --val-end "$VAL_END" \
    --fast-clean --skip-manifest \
    --drop-clean --drop-events --resume \
    > "$glog" 2>&1 &
  pids+=("$!")
done

# 3) 等全部组结束。
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    log "⚠️ 组 $i (pid ${pids[$i]}) 非零退出；见 $SPLIT_DIR/group${i}.log"
    groups_ok=0
  fi
done
log "所有并行组结束 (ok=$groups_ok)"

# 4) 统一构建一次 manifest（BENCH 模式跳过）。
if [[ "$BENCH" == "1" ]]; then
  log "BENCH 模式：跳过收尾 manifest。"
else
  if [[ -f "$WORKDIR/data/.prune_embedded_tokens" ]]; then
    log "流式 OOS 模式：已消费 tokens 按日释放，跳过不完整的全量 manifest。"
  else
    log "统一构建 manifest（train_end=$TRAIN_END val_end=$VAL_END）…"
    uv run python - "$WORKDIR" "$VOCAB" "$TRAIN_END" "$VAL_END" <<'PY'
import sys
from pathlib import Path
from quant_fm.manifest.build_manifest import build_manifest
workdir, vocab, train_end, val_end = sys.argv[1:5]
m = build_manifest(
    Path(workdir) / "tokens",
    train_end=train_end, val_end=val_end,
    markets=("SZ", "SH"), vocab_path=vocab,
)
out = Path(workdir) / "data" / "manifest.json"
out.parent.mkdir(parents=True, exist_ok=True)
m.save(out)
print(f"manifest → {out}: {len(m.shards)} shards")
PY
  fi
fi
log "======== 跨日并行完成 ========"
