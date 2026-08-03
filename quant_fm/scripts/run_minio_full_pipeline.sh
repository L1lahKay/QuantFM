#!/usr/bin/env bash
# 完整 V2 流水线：MinIO L2 → 真实盘口 V2 tokens → 审计/上传 → 8 卡训练
#
#   【读】:9000 / zeus-cn-quote
#      → run_medium（clean → events → vocab → tokens → manifest）
#   【写】:9100 / model-cache / ${MINIO_OUTPUT_PREFIX:-fm-pretrain/$USER}/{tag}/
#      → train_medium_8gpu（用本地 tokens；同时已备份到 MinIO）
#
# 用法：
#   bash quant_fm/scripts/run_minio_full_pipeline.sh              # 试跑 5日×30股
#   MODE=full bash quant_fm/scripts/run_minio_full_pipeline.sh    # 60日×全市场（≈总量1/10）
#   MODE=smoke bash ...                                          # 60日×50股/市场
#
# 环境变量：
#   MODE=try|smoke|full          数据规模（默认 try）
#   SKIP_DATA=1                  跳过清洗（本地已有或从 MinIO 拉）
#   SKIP_TRAIN=1                 只做数据+上传，不训练
#   SKIP_UPLOAD=1                不上传（仅本地）
#   FORCE_DOWNLOAD=1             忽略本地，强制从 model-cache 下载
#   DELETE_LOCAL_AFTER_TRAIN=1   训练结束后删本地 tokens（MinIO 已有备份）
#   NPROC / CONFIG / TB_PORT     传给训练脚本
#
# 凭据：~/.minio_fm_env.sh（读写密钥可不同，见 minio_env.example.sh）

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

if [[ -f "$HOME/.minio_fm_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/.minio_fm_env.sh"
fi

MODE="${MODE:-try}"
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
DELETE_LOCAL_AFTER_TRAIN="${DELETE_LOCAL_AFTER_TRAIN:-0}"
LOG="${LOG:-$ROOT/quant_fm/runs/minio_full_pipeline.log}"

case "$MODE" in
  try)
    WORKDIR="$ROOT/quant_fm/runs/v2_try"
    TAG="v2_try"
    CONFIG="${CONFIG:-quant_fm/pretrain/config_v2_25m.yaml}"
    DATES="$ROOT/quant_fm/data/medium_try_5_dates.txt"
    EXTRA=(--dates-file "$DATES" --max-symbols-per-market 30)
    ;;
  smoke)
    WORKDIR="$ROOT/quant_fm/runs/v2_smoke"
    TAG="v2_smoke"
    CONFIG="${CONFIG:-quant_fm/pretrain/config_v2_25m.yaml}"
    EXTRA=(--max-symbols-per-market 50)
    ;;
  full)
    # 「全量」= 文档约定的 medium：60 个均匀交易日 × 沪深全市场（≈ 总数据 1/10）
    WORKDIR="$ROOT/quant_fm/runs/v2_shared"
    TAG="v2_shared"
    CONFIG="${CONFIG:-quant_fm/pretrain/config_v2_230m.yaml}"
    EXTRA=(--resume --v2-full-audit)
    ;;
  *)
    echo "MODE must be try|smoke|full" >&2
    exit 1
    ;;
esac

mkdir -p "$(dirname "$LOG")" "$WORKDIR"
exec > >(tee -a "$LOG") 2>&1

echo "======== $(date -Is) MinIO FULL pipeline MODE=$MODE ========"
echo "read:   zeus-cn-quote @ ${MINIO_READ_ENDPOINT:-192.168.2.11:9000}"
OUT_PREFIX="${MINIO_OUTPUT_PREFIX:-fm-pretrain/${USER:-user}}"
echo "write:  model-cache @ ${MINIO_WRITE_ENDPOINT:-192.168.2.11:9100}/$OUT_PREFIX/$TAG/"
echo "workdir:$WORKDIR"
echo "config: $CONFIG"
echo "flags:  SKIP_DATA=$SKIP_DATA SKIP_TRAIN=$SKIP_TRAIN SKIP_UPLOAD=$SKIP_UPLOAD FORCE_DOWNLOAD=$FORCE_DOWNLOAD"

python -m quant_fm.scripts.check_minio

local_ready() {
  [[ -f "$WORKDIR/data/manifest.json" && -f "$WORKDIR/data/vocab_v2.json" && -f "$WORKDIR/artifact_audit.json" && -d "$WORKDIR/tokens" ]]
}

# ── 1) 数据：读 MinIO → tokens（或从 model-cache 恢复）──
if [[ "$FORCE_DOWNLOAD" == "1" ]] || { [[ "$SKIP_DATA" == "1" ]] && ! local_ready; }; then
  echo "==> download tokens from model-cache (tag=$TAG)"
  python -m quant_fm.scripts.download_from_minio --workdir "$WORKDIR" --tag "$TAG" --data-version v2
elif [[ "$SKIP_DATA" == "1" ]]; then
  echo "==> SKIP_DATA=1 and local ready → $WORKDIR"
else
  echo "==> run_medium: MinIO raw → tokens"
  UPLOAD_FLAGS=()
  if [[ "$SKIP_UPLOAD" != "1" ]]; then
    # 上传到 MinIO，但保留本地 tokens 供训练（勿加 --delete-local-after-upload）
    UPLOAD_FLAGS=(--upload-minio --upload-tag "$TAG")
  fi
  python -m quant_fm.scripts.run_medium \
    --workdir "$WORKDIR" \
    --data-version v2 \
    --drop-clean \
    --drop-events \
    "${UPLOAD_FLAGS[@]}" \
    "${EXTRA[@]}"

  if [[ "$SKIP_UPLOAD" != "1" ]]; then
    echo "==> verify / ensure upload to model-cache"
    python -m quant_fm.scripts.upload_to_minio --tag "$TAG" --verify-only
  fi
fi

if ! local_ready; then
  echo "ERROR: tokens/manifest 未就绪: $WORKDIR" >&2
  echo "  可试: FORCE_DOWNLOAD=1 MODE=$MODE bash quant_fm/scripts/run_minio_full_pipeline.sh" >&2
  exit 1
fi

echo "==> data ready: $WORKDIR/data/manifest.json"
python -m quant_fm.scripts.audit_v2_artifacts \
  --root "$WORKDIR" \
  --sample-shards 12 \
  --out "$WORKDIR/artifact_audit.json"

# ── 2) 训练（读本地 tokens；MinIO 已有副本）──
if [[ "$SKIP_TRAIN" == "1" ]]; then
  echo "==> SKIP_TRAIN=1 → 仅数据+上传完成"
  echo "======== $(date -Is) DONE (no train) → s3://model-cache/.../$TAG/ ========"
  exit 0
fi

echo "==> 8-GPU train config=$CONFIG"
export CONFIG
export MEDIUM_WORKDIR="${WORKDIR#"$ROOT/"}"
bash "$ROOT/quant_fm/scripts/train_medium_8gpu.sh"

if [[ "$DELETE_LOCAL_AFTER_TRAIN" == "1" ]]; then
  echo "==> delete local tokens/ (kept on MinIO)"
  rm -rf "$WORKDIR/tokens"
fi

echo "======== $(date -Is) DONE FULL pipeline MODE=$MODE ========"
echo "  MinIO:  s3://model-cache/$OUT_PREFIX/$TAG/"
echo "  ckpt:   $WORKDIR/run/"
