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
if [[ -f "$WORKDIR/data/vocab_v2.json" ]]; then
  VOCAB="$WORKDIR/data/vocab_v2.json"
else
  VOCAB="$WORKDIR/data/vocab.json"
fi

if [[ ! -f "$MANIFEST" || ! -f "$VOCAB" ]]; then
  echo "ERROR: medium 数据未就绪: $WORKDIR" >&2
  echo "  缺少 manifest 或 vocab；请先 run_medium_pipeline.sh 或 make medium" >&2
  exit 1
fi

# MEDIUM_WORKDIR is an explicit artifact-root override.  Materialize an exact
# config snapshot so V2 try/smoke/full can share the reviewed model configs
# without accidentally reading the hard-coded v2_shared data root.
EFFECTIVE_CONFIG="$CONFIG"
if [[ -n "${MEDIUM_WORKDIR:-}" ]]; then
  EFFECTIVE_CONFIG="$WORKDIR/data/train_config.generated.yaml"
  python - "$CONFIG" "$WORKDIR" "$VOCAB" "$EFFECTIVE_CONFIG" <<'PY'
import pathlib
import sys

import yaml

source, workdir, vocab, destination = map(pathlib.Path, sys.argv[1:])
cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
cfg["data"]["manifest"] = str(workdir / "data" / "manifest.json")
cfg["data"]["vocab"] = str(vocab)
if "validation_plan" in cfg["data"]:
    cfg["data"]["validation_plan"] = str(workdir / "validation_windows.json")
cfg["runtime"]["out_dir"] = str(workdir / "run")
destination.write_text(
    yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)
PY
fi

python - <<'PY' || { echo "ERROR: CUDA 不可用" >&2; exit 1; }
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() == False"
print(f"torch {torch.__version__}, GPUs: {torch.cuda.device_count()}")
PY

RUN_DIR="$WORKDIR/run"
shopt -s dotglob nullglob
existing_run_entries=("$RUN_DIR"/*)
if (( ${#existing_run_entries[@]} > 0 )); then
  echo "ERROR: refusing fresh training in non-empty run directory: $RUN_DIR" >&2
  echo "  Use the train CLI with --resume auto, or choose a new workdir." >&2
  exit 1
fi

export TB_LOGDIR="$RUN_DIR/tb"

echo "==> config=$EFFECTIVE_CONFIG  nproc=$NPROC  port=$MASTER_PORT"
echo "==> workdir=$WORKDIR"
echo "==> TensorBoard: http://127.0.0.1:${TB_PORT:-6006}"
echo "==> checkpoints: $WORKDIR/run/"

# train.py writes config.snapshot.yaml only after all resume/provenance checks pass.
# Start telemetry after that point so it cannot make a fresh run directory non-empty
# before the training preflight has claimed it.
launcher_pid=$BASHPID
(
  while kill -0 "$launcher_pid" 2>/dev/null; do
    if [[ -f "$RUN_DIR/config.snapshot.yaml" ]]; then
      bash "$ROOT/quant_fm/scripts/start_tensorboard_medium.sh"
      exit 0
    fi
    sleep 1
  done
) &

exec torchrun \
  --standalone \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  -m quant_fm.pretrain.train \
  --config "$EFFECTIVE_CONFIG"
