#!/usr/bin/env bash
# 8 卡 medium 预训练（torchrun + FSDP），训练前自动启动 TensorBoard。
#
# 用法：
#   bash quant_fm/scripts/train_medium_8gpu.sh
#   CONFIG=quant_fm/pretrain/config_medium_try_8gpu.yaml bash ...
#   MEDIUM_WORKDIR=quant_fm/runs/medium_try bash ...

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

CONFIG="${CONFIG:-quant_fm/pretrain/config_medium_8gpu.yaml}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"

if [[ -n "${MEDIUM_WORKDIR:-}" ]]; then
  WORKDIR="$ROOT/$MEDIUM_WORKDIR"
else
  WORKDIR="$(python - <<PY
import pathlib, yaml
cfg = yaml.safe_load(pathlib.Path("$CONFIG").read_text())
print(pathlib.Path(cfg["runtime"]["out_dir"]).resolve().parent)
PY
)"
fi

MANIFEST="$WORKDIR/data/manifest.json"
VOCAB="$WORKDIR/data/vocab.json"

if [[ ! -f "$MANIFEST" || ! -f "$VOCAB" ]]; then
  echo "ERROR: medium 数据未就绪: $WORKDIR" >&2
  echo "  缺少 manifest 或 vocab；请先 run_medium_pipeline.sh 或 make medium" >&2
  exit 1
fi

python - <<'PY' || { echo "ERROR: CUDA 不可用" >&2; exit 1; }
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() == False"
print(f"torch {torch.__version__}, GPUs: {torch.cuda.device_count()}")
PY

export TB_LOGDIR="$WORKDIR/run/tb"
bash "$ROOT/quant_fm/scripts/start_tensorboard_medium.sh"

echo "==> config=$CONFIG  nproc=$NPROC  port=$MASTER_PORT"
echo "==> workdir=$WORKDIR"
echo "==> TensorBoard: http://127.0.0.1:${TB_PORT:-6006}"
echo "==> checkpoints: $WORKDIR/run/"

exec torchrun \
  --standalone \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  -m quant_fm.pretrain.train \
  --config "$CONFIG"
