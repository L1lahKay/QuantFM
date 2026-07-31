#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
BENCH_DIR="${SCRIPT_DIR}/volcano/bench"
RAW_DIR="${REPO_ROOT}/docs/assets/gpu-scheduler-evaluation/raw/volcano"
KUBECTL=(kubectl --kubeconfig /etc/rancher/k3s/k3s.yaml --context default)
NAMESPACE=gpu-dev
TIMELINE="${RAW_DIR}/cuda-smoke-timeline.txt"
TRANSCRIPT="${RAW_DIR}/cuda-smoke-transcript.txt"

: >"${TIMELINE}"
: >"${TRANSCRIPT}"
exec > >(tee -a "${TRANSCRIPT}") 2>&1

record() {
  local label=$1
  printf '%s utc=%s monotonic_ns=%s\n' \
    "${label}" "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" \
    "$(python3 -c 'import time; print(time.monotonic_ns())')" | tee -a "${TIMELINE}"
}

cleanup() {
  set +e
  "${KUBECTL[@]}" delete job -n "${NAMESPACE}" khalil-volcano-smoke --ignore-not-found=true --wait=true --timeout=120s
  "${KUBECTL[@]}" delete podgroup -n "${NAMESPACE}" khalil-volcano-smoke-pg --ignore-not-found=true --wait=true
  "${KUBECTL[@]}" delete queue khalil-volcano-smoke --ignore-not-found=true --wait=true
  set -e
}
trap cleanup EXIT

for object in \
  job.batch/khalil-volcano-smoke \
  podgroup.scheduling.volcano.sh/khalil-volcano-smoke-pg; do
  if "${KUBECTL[@]}" get -n "${NAMESPACE}" "${object}" >/dev/null 2>&1; then
    echo "ERROR ${object} already exists" >&2
    exit 2
  fi
done
if "${KUBECTL[@]}" get queue khalil-volcano-smoke >/dev/null 2>&1; then
  echo "ERROR queue khalil-volcano-smoke already exists" >&2
  exit 2
fi

"${KUBECTL[@]}" get resourcequota -n "${NAMESPACE}" -o json >"${RAW_DIR}/cuda-smoke-resourcequota-before.json"
GPU_USED=$(python3 -c '
import json,sys
items=json.load(open(sys.argv[1]))["items"]
q=next(x for x in items if x["metadata"]["name"]=="gpu-quota")
print(q["status"]["used"].get("requests.nvidia.com/gpu", "0"))
' "${RAW_DIR}/cuda-smoke-resourcequota-before.json")
if [[ ${GPU_USED} -ne 0 ]]; then
  echo "ERROR gpu-dev GPU quota is in use" >&2
  exit 2
fi

record CUDA_SMOKE_QUEUE_CREATE_BEGIN
"${KUBECTL[@]}" apply --dry-run=server -f "${BENCH_DIR}/00-smoke-queue.yaml"
"${KUBECTL[@]}" apply -f "${BENCH_DIR}/00-smoke-queue.yaml"
record CUDA_SMOKE_QUEUE_CREATE_RETURN
"${KUBECTL[@]}" apply --dry-run=server -f "${BENCH_DIR}/01-smoke.yaml" >"${RAW_DIR}/cuda-smoke-server-dry-run.txt"
record CUDA_SMOKE_JOB_CREATE_BEGIN
"${KUBECTL[@]}" apply -f "${BENCH_DIR}/01-smoke.yaml"
record CUDA_SMOKE_JOB_CREATE_RETURN
"${KUBECTL[@]}" wait -n "${NAMESPACE}" --for=condition=complete job/khalil-volcano-smoke --timeout=120s
record CUDA_SMOKE_COMPLETE_OBSERVED

"${KUBECTL[@]}" get job -n "${NAMESPACE}" khalil-volcano-smoke -o json >"${RAW_DIR}/cuda-smoke-job.json"
"${KUBECTL[@]}" get pods -n "${NAMESPACE}" -l evaluation.quantfm/scenario=smoke -o json >"${RAW_DIR}/cuda-smoke-pods.json"
"${KUBECTL[@]}" get podgroup -n "${NAMESPACE}" khalil-volcano-smoke-pg -o json >"${RAW_DIR}/cuda-smoke-podgroup.json"
"${KUBECTL[@]}" get queue khalil-volcano-smoke -o json >"${RAW_DIR}/cuda-smoke-queue.json"
"${KUBECTL[@]}" get events -n "${NAMESPACE}" -o json >"${RAW_DIR}/cuda-smoke-events.json"
"${KUBECTL[@]}" logs -n "${NAMESPACE}" -l evaluation.quantfm/scenario=smoke --prefix=true >"${RAW_DIR}/cuda-smoke-container-output.txt"

grep -q 'VISIBLE_GPU_COUNT=1' "${RAW_DIR}/cuda-smoke-container-output.txt"
grep -q 'TORCH_CUDA_AVAILABLE=true' "${RAW_DIR}/cuda-smoke-container-output.txt"
grep -q 'CUDA_TENSOR_RESULT=42' "${RAW_DIR}/cuda-smoke-container-output.txt"
echo "CUDA_SMOKE_ASSERTIONS visible_gpu=1 torch_cuda=true tensor_result=42"

cleanup
"${KUBECTL[@]}" get resourcequota -n "${NAMESPACE}" -o json >"${RAW_DIR}/cuda-smoke-resourcequota-after.json"
record CUDA_SMOKE_CLEANUP_COMPLETE
trap - EXIT
