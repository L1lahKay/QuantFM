#!/usr/bin/env bash
# MinIO V2 数据流水线（无训练）：真实盘口回放 → V2 tokens → 审计 → MinIO
# 上传后保留本地产物；自动递归删除已禁用。离线停写并独立验收后再显式清理。
# 若还要训练，请用：
#   make minio-full-pipeline / run_minio_full_pipeline.sh
#
# 用法：
#   bash quant_fm/scripts/run_minio_data_pipeline.sh              # 5日×30股试跑
#   MODE=full bash quant_fm/scripts/run_minio_data_pipeline.sh    # 60日×全市场
#   MODE=smoke bash quant_fm/scripts/run_minio_data_pipeline.sh    # 60日×50股
#
# 凭据：~/.minio_fm_env.sh 或 mc alias myminio

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
LOG="${LOG:-$ROOT/quant_fm/runs/minio_pipeline.log}"
PARALLEL_V2=0

case "$MODE" in
  try)
    WORKDIR="$ROOT/quant_fm/runs/v2_try"
    TAG="v2_try"
    DATES="$ROOT/quant_fm/data/medium_try_5_dates.txt"
    EXTRA=(--dates-file "$DATES" --max-symbols-per-market 30)
    ;;
  smoke)
    WORKDIR="$ROOT/quant_fm/runs/v2_smoke"
    TAG="v2_smoke"
    EXTRA=(--max-symbols-per-market 50)
    ;;
  full)
    WORKDIR="$ROOT/quant_fm/runs/v2_shared"
    TAG="v2_shared"
    DATES="$ROOT/quant_fm/data/medium_60_dates.txt"
    PARALLEL_V2=1
    EXTRA=()
    ;;
  *)
    echo "MODE must be try|smoke|full" >&2
    exit 1
    ;;
esac

mkdir -p "$(dirname "$LOG")" "$WORKDIR"

exec > >(tee -a "$LOG") 2>&1

echo "======== $(date -Is) minio data pipeline MODE=$MODE ========"
OUT_PREFIX="${MINIO_OUTPUT_PREFIX:-fm-pretrain/${USER:-user}}"
echo "read:  zeus-cn-quote @ 192.168.2.11:9000"
echo "write: model-cache @ 192.168.2.11:9100/$OUT_PREFIX/$TAG/"
echo "workdir: $WORKDIR"
echo "schema: cn_l2_v2 (real post-event book state, Q16 scalar storage)"

python -m quant_fm.scripts.check_minio

if [[ "$PARALLEL_V2" == "1" ]]; then
  python -m quant_fm.scripts.run_v2_parallel_data \
    --workdir "$WORKDIR" \
    --dates-file "$DATES" \
    --groups "${NGROUPS:-2}" \
    --clean-workers "${CLEAN_WORKERS:-30}" \
    --canon-workers "${CANON_WORKERS:-8}" \
    --tokenize-workers "${TOKENIZE_WORKERS:-16}"
  python -m quant_fm.scripts.upload_to_minio \
    --workdir "$WORKDIR" --tag "$TAG"
else
  python -m quant_fm.scripts.run_medium \
    --workdir "$WORKDIR" \
    --data-version v2 \
    --fast-clean \
    --drop-clean \
    --drop-events \
    --v2-full-audit \
    --upload-minio \
    --upload-tag "$TAG" \
    "${EXTRA[@]}"
fi

python -m quant_fm.scripts.upload_to_minio --tag "$TAG" --verify-only

echo "======== $(date -Is) DONE → s3://model-cache/$OUT_PREFIX/$TAG/ ========"
