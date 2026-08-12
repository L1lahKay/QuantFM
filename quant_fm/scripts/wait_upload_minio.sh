#!/usr/bin/env bash
# 等待本地 manifest 就绪后上传到 model-cache 并验证。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKDIR="${1:-$ROOT/quant_fm/runs/medium_try}"
TAG="${2:-medium_try}"
source "$ROOT/.venv/bin/activate" 2>/dev/null || true
[[ -f "$HOME/.minio_fm_env.sh" ]] && source "$HOME/.minio_fm_env.sh"
echo "waiting for $WORKDIR/data/manifest.json ..."
while [[ ! -f "$WORKDIR/data/manifest.json" ]]; do sleep 30; done
echo "uploading..."
python -m quant_fm.scripts.upload_to_minio \
  --workdir "$WORKDIR" --tag "$TAG" --verify
