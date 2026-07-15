#!/usr/bin/env bash
# 启动 TensorBoard（medium 训练日志目录）。
#
# 用法：
#   bash quant_fm/scripts/start_tensorboard_medium.sh
#   TB_PORT=6007 bash quant_fm/scripts/start_tensorboard_medium.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TB_PORT="${TB_PORT:-6006}"
LOGDIR="${TB_LOGDIR:-$ROOT/quant_fm/runs/medium/run/tb}"
RUN_DIR="$(dirname "$LOGDIR")"
PIDFILE="${TB_PIDFILE:-$RUN_DIR/tensorboard.pid}"
TB_LOG="${TB_LOG:-$RUN_DIR/tensorboard.log}"

mkdir -p "$LOGDIR" "$(dirname "$PIDFILE")"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "TensorBoard already running (pid $(cat "$PIDFILE"))"
  echo "  http://127.0.0.1:${TB_PORT}"
  exit 0
fi

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

nohup tensorboard \
  --logdir "$LOGDIR" \
  --port "$TB_PORT" \
  --bind_all \
  --load_fast=false \
  > "$TB_LOG" 2>&1 &

echo $! > "$PIDFILE"
sleep 2

if curl -sf "http://127.0.0.1:${TB_PORT}" >/dev/null 2>&1; then
  echo "TensorBoard started (pid $(cat "$PIDFILE"))"
  echo "  logdir: $LOGDIR"
  echo "  local:  http://127.0.0.1:${TB_PORT}"
  echo "  remote: use Cursor Ports or ssh -L ${TB_PORT}:127.0.0.1:${TB_PORT}"
else
  echo "WARN: TensorBoard may still be starting; check $TB_LOG"
fi
