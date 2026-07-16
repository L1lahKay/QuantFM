#!/usr/bin/env bash
# 300M 模型：按 Chinchilla 估数据量 → MinIO 读 L2 → tokens → 8 卡训练
#
# 规模依据（见脚本顶部注释 / 运行时打印）：
#   N ≈ 302.3M  →  D_train ≈ 20N ≈ 6.05B events
#   70/15/15 切分 → 总事件 ≈ 8.6B
#   全市场 ~5104 标的 × 实测 ~7.8 万 events/股日 → 约 22 个交易日
#
# 用法：
#   bash quant_fm/scripts/run_minio_300m_pipeline.sh
#   SKIP_TRAIN=1 bash ...          # 只做数据
#   SKIP_DATA=1 bash ...           # 本地已有 tokens，直接训练
#   SKIP_UPLOAD=1 bash ...         # 不上传 model-cache
#   CLEAN_WORKERS=16 bash ...      # 并行洗股进程数（默认 min(32, CPU/2)）
#
# 默认启用断点续跑：按日期记录 clean/canonicalize 完成状态，并跳过已生成
# 的 events/tokens；数据完成后训练会从 runtime.resume 指定的 checkpoint 恢复。
#
# 凭据：~/.minio_fm_env.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f "$HOME/.minio_fm_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.minio_fm_env.sh"
fi

WORKDIR="$ROOT/quant_fm/runs/medium_300m"
TAG="medium_300m"
DATES="$ROOT/quant_fm/data/medium_300m_22_dates.txt"
CONFIG="${CONFIG:-quant_fm/pretrain/config_medium_300m_8gpu.yaml}"
LOG="${LOG:-$WORKDIR/pipeline.log}"
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_UPLOAD="${SKIP_UPLOAD:-1}"
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-29521}"
FIT_SAMPLE_DAYS="${FIT_SAMPLE_DAYS:-5}"

mkdir -p "$WORKDIR"
exec > >(tee -a "$LOG") 2>&1

echo "======== $(date -Is) 300M MinIO pipeline ========"
echo "目标参数量: ~302.3M (1024×18)"
echo "Chinchilla: D_train≈6.05B → 总事件≈8.6B (70/15/15)"
echo "日期文件: $DATES ($(grep -c . "$DATES") 天) × 沪深全市场"
echo "workdir: $WORKDIR"
echo "config:  $CONFIG"
echo "read:    zeus-cn-quote @ ${MINIO_READ_ENDPOINT:-192.168.2.11:9000}"
echo "write:   model-cache @ ${MINIO_WRITE_ENDPOINT:-192.168.2.11:9100} (SKIP_UPLOAD=$SKIP_UPLOAD)"

uv run python -m quant_fm.scripts.check_minio
uv run python -m quant_fm.scripts.run_medium \
  --estimate-only \
  --dates-file "$DATES" \
  --workdir "$WORKDIR"

local_ready() {
  [[ -f "$WORKDIR/data/manifest.json" && -f "$WORKDIR/data/vocab.json" && -d "$WORKDIR/tokens" ]]
}

if [[ "$SKIP_DATA" == "1" ]]; then
  if ! local_ready; then
    echo "ERROR: SKIP_DATA=1 但本地数据未就绪: $WORKDIR" >&2
    exit 1
  fi
  echo "==> SKIP_DATA=1，复用本地 tokens"
else
  echo "==> MinIO raw → clean → events → tokens → manifest"
  UPLOAD_FLAGS=()
  if [[ "$SKIP_UPLOAD" != "1" ]]; then
    UPLOAD_FLAGS=(--upload-minio --upload-tag "$TAG")
  fi
  uv run python -m quant_fm.scripts.run_medium \
    --dates-file "$DATES" \
    --workdir "$WORKDIR" \
    --drop-clean \
    --drop-events \
    --resume \
    --fit-sample-days "$FIT_SAMPLE_DAYS" \
    "${UPLOAD_FLAGS[@]}"
fi

if ! local_ready; then
  echo "ERROR: tokens/manifest 未就绪: $WORKDIR" >&2
  exit 1
fi

# 打印真实事件量与 Chinchilla 对照
uv run python - <<'PY'
import json
from pathlib import Path
m = json.loads(Path("quant_fm/runs/medium_300m/data/manifest.json").read_text())
rows = {"train": 0, "val": 0, "test": 0}
dates = {"train": set(), "val": set(), "test": set()}
for s in m["shards"]:
    rows[s["split"]] += s["rows"]
    dates[s["split"]].add(s["date"])
total = sum(rows.values())
print("==> 真实事件量:")
for sp in ("train", "val", "test"):
    print(f"  {sp}: {len(dates[sp])} days, {rows[sp]:,} ({rows[sp]/1e9:.3f}B)")
print(f"  total: {total:,} ({total/1e9:.3f}B)")
print(f"  Chinchilla N from train: {rows['train']/20/1e6:.1f}M")
print(f"  configured model: 302.3M")
PY

if [[ "$SKIP_TRAIN" == "1" ]]; then
  echo "==> SKIP_TRAIN=1 → 仅数据完成"
  exit 0
fi

echo "==> 8-GPU train (302.3M)"
export TB_LOGDIR="$WORKDIR/run/tb"
bash "$ROOT/quant_fm/scripts/start_tensorboard_medium.sh" || true

uv run python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  -m quant_fm.pretrain.train \
  --config "$CONFIG" \
  --resume auto

echo "======== $(date -Is) DONE 300M ========"
echo "  ckpt: $WORKDIR/run/"
