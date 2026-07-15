#!/usr/bin/env bash
# 8 卡 pilot 预训练启动脚本（torchrun + FSDP）。
#
# 用法：
#   bash quant_fm/scripts/train_pilot_8gpu.sh
#   NPROC=4 bash quant_fm/scripts/train_pilot_8gpu.sh   # 临时只用 4 卡
#
# 前提：
#   1. source .venv/bin/activate 且已安装 CUDA 版 torch
#   2. 已运行 make pilot，存在 quant_fm/runs/pilot/data/{manifest,vocab}.json

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

CONFIG="${CONFIG:-quant_fm/pretrain/config_pilot_8gpu.yaml}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"

MANIFEST="$ROOT/quant_fm/runs/pilot/data/manifest.json"
VOCAB="$ROOT/quant_fm/runs/pilot/data/vocab.json"

if [[ ! -f "$MANIFEST" || ! -f "$VOCAB" ]]; then
  echo "ERROR: pilot 数据未就绪，请先运行 make pilot" >&2
  echo "  缺少: $MANIFEST 或 $VOCAB" >&2
  exit 1
fi

python - <<'PY' || { echo "ERROR: CUDA 不可用，请先安装 cu128 版 torch" >&2; exit 1; }
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() == False"
print(f"torch {torch.__version__}, GPUs: {torch.cuda.device_count()}")
PY

echo "==> config=$CONFIG  nproc=$NPROC  port=$MASTER_PORT"
echo "==> 日志与 checkpoint: quant_fm/runs/pilot/run/"

exec torchrun \
  --standalone \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  -m quant_fm.pretrain.train \
  --config "$CONFIG"
