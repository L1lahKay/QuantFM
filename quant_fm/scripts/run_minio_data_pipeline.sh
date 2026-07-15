#!/usr/bin/env bash
# MinIO 数据流水线（无训练）：读 :9000/zeus-cn-quote → 本地处理 → 写 :9100/model-cache
# 上传后默认删除本地 tokens。若还要训练，请用：
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

case "$MODE" in
  try)
    WORKDIR="$ROOT/quant_fm/runs/medium_try"
    TAG="medium_try"
    DATES="$ROOT/quant_fm/data/medium_try_5_dates.txt"
    EXTRA=(--dates-file "$DATES" --max-symbols-per-market 30)
    ;;
  smoke)
    WORKDIR="$ROOT/quant_fm/runs/medium_smoke"
    TAG="medium_smoke"
    EXTRA=(--max-symbols-per-market 50)
    ;;
  full)
    WORKDIR="$ROOT/quant_fm/runs/medium"
    TAG="medium"
    EXTRA=(--resume)
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

python -m quant_fm.scripts.check_minio

python -m quant_fm.scripts.run_medium \
  --workdir "$WORKDIR" \
  --drop-clean \
  --drop-events \
  --upload-minio \
  --upload-tag "$TAG" \
  --delete-local-after-upload \
  "${EXTRA[@]}"

python -m quant_fm.scripts.upload_to_minio --tag "$TAG" --verify 2>/dev/null || \
  python3 <<PY
from quant_fm.scripts.upload_to_minio import _ensure_mc_alias, remote_uri, output_bucket, output_prefix
import subprocess
alias = _ensure_mc_alias()
remote = f"{alias}/{output_bucket()}/{output_prefix('$TAG')}"
r = subprocess.run(["mc", "find", remote, "--name", "*.parquet"], capture_output=True, text=True, check=True)
n = len([l for l in r.stdout.splitlines() if l.strip()])
print(f"verify: {remote_uri('$TAG')} parquet count={n}")
PY

echo "======== $(date -Is) DONE → s3://model-cache/$OUT_PREFIX/$TAG/ ========"
