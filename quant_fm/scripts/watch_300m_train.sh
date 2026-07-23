#!/usr/bin/env bash
# 监控 302M 训练进度；训完后可选自动跑下游 judge。
#
# 用法：
#   bash quant_fm/scripts/watch_300m_train.sh
#   AUTO_JUDGE=1 bash quant_fm/scripts/watch_300m_train.sh   # 训完自动 judge
#   INTERVAL=60 bash ...                                      # 轮询间隔秒

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORKDIR="${WORKDIR:-quant_fm/runs/medium_300m}"
LOG="${LOG:-$WORKDIR/train_nohup.out}"
RUN_DIR="$WORKDIR/run"
INTERVAL="${INTERVAL:-120}"
AUTO_JUDGE="${AUTO_JUDGE:-0}"
MAX_STEPS="${MAX_STEPS:-80000}"

echo "======== watch 300m train ========"
echo "log: $LOG"
echo "interval: ${INTERVAL}s  AUTO_JUDGE=$AUTO_JUDGE"

last_step=-1
while true; do
  ts="$(date -Is)"
  if pgrep -f "quant_fm.pretrain.train --config quant_fm/pretrain/config_medium_300m" >/dev/null; then
    running=1
  else
    running=0
  fi

  step_line="$(rg -N "INFO step [0-9]+ " "$LOG" 2>/dev/null | tail -1 || true)"
  val_line="$(rg -N "val_loss|new best" "$LOG" 2>/dev/null | tail -1 || true)"
  step=0
  if [[ -n "$step_line" ]]; then
    step="$(echo "$step_line" | sed -n 's/.*step \([0-9][0-9]*\).*/\1/p')"
  fi

  gpu="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || echo '?')"
  ckpts="$(ls -1 "$RUN_DIR"/*.pt 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ' || true)"

  echo "[$ts] running=$running step=${step:-?} gpu0=$gpu ckpts=[$ckpts]"
  if [[ -n "$step_line" ]]; then echo "  $step_line"; fi
  if [[ -n "$val_line" ]]; then echo "  $val_line"; fi

  # 训练进程结束
  if [[ "$running" -eq 0 ]]; then
    if rg -q "DONE 300M|saved checkpoint.*final" "$LOG" 2>/dev/null \
      || [[ -f "$RUN_DIR/final.pt" ]]; then
      echo "==> 训练已完成"
      if [[ "$AUTO_JUDGE" == "1" ]]; then
        echo "==> AUTO_JUDGE=1 → 启动 run_judge_300m.sh"
        bash "$ROOT/quant_fm/scripts/run_judge_300m.sh"
      fi
      exit 0
    fi
    # 进程没了但也没 final：可能崩溃
    if [[ "${step:-0}" -gt 0 ]]; then
      echo "WARN: 训练进程已退出且无 final.pt（last step=$step）。可用 --resume auto 续训。"
      exit 1
    fi
  fi

  # 达到 max_steps 也提示
  if [[ -n "${step:-}" && "$step" -ge "$MAX_STEPS" ]]; then
    echo "==> 已达 max_steps=$MAX_STEPS，等待进程收尾..."
  fi

  if [[ "$step" != "$last_step" && -n "$step" ]]; then
    last_step="$step"
  fi
  sleep "$INTERVAL"
done
