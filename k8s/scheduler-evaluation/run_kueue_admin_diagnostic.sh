#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="$ROOT/k8s/scheduler-evaluation/gpu-dev/kueue-admin-diagnostic.yaml"
RESULT_DIR="${RESULT_DIR:-$ROOT/docs/assets/gpu-scheduler-evaluation/raw/kueue-admin}"
KUBECONFIG_PATH=/etc/rancher/k3s/k3s.yaml
KUBE_CONTEXT=default
NAMESPACE=gpu-dev
JOB=khalil-kueue-admin-diag

if [[ "${RUN_KUEUE_ADMIN_DIAGNOSTIC:-}" != "1" ]]; then
  echo "Refusing to submit: set RUN_KUEUE_ADMIN_DIAGNOSTIC=1 explicitly" >&2
  exit 2
fi

mkdir -p "$RESULT_DIR"
TRANSCRIPT="$RESULT_DIR/transcript.txt"
TIMELINE="$RESULT_DIR/timeline.txt"
: > "$TIMELINE"
exec > >(tee "$TRANSCRIPT") 2>&1

kctl=(kubectl --kubeconfig "$KUBECONFIG_PATH" --context "$KUBE_CONTEXT" --namespace "$NAMESPACE")

stamp() {
  local label=$1 utc monotonic
  utc=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
  monotonic=$(python3 -c 'import time; print(time.monotonic_ns())')
  printf '%s utc=%s monotonic_ns=%s\n' "$label" "$utc" "$monotonic" | tee -a "$TIMELINE"
}

cleanup() {
  "${kctl[@]}" delete job "$JOB" --ignore-not-found --wait=true || true
}
trap cleanup EXIT

echo "ADMIN_KUBECONFIG_PATH=$KUBECONFIG_PATH"
echo "ADMIN_CONTEXT=$KUBE_CONTEXT"
echo "NAMESPACE=$NAMESPACE"
stamp DIAGNOSTIC_BEGIN

if "${kctl[@]}" get job "$JOB" >/dev/null 2>&1; then
  echo "Refusing to overwrite existing Job $JOB" >&2
  exit 3
fi

"${kctl[@]}" get localqueues.kueue.x-k8s.io -o json > "$RESULT_DIR/localqueues-before.json"
localqueue_count=$("${kctl[@]}" get localqueues.kueue.x-k8s.io -o jsonpath='{.items[*].metadata.name}' | wc -w)
echo "LOCALQUEUE_COUNT=$localqueue_count"

"${kctl[@]}" apply --server-side --dry-run=server -f "$MANIFEST" > "$RESULT_DIR/server-dry-run.txt"
stamp JOB_CREATE_BEGIN
"${kctl[@]}" create -f "$MANIFEST"
stamp JOB_CREATE_RETURN
job_uid=$("${kctl[@]}" get job "$JOB" -o jsonpath='{.metadata.uid}')

workload_found=false
for _ in $(seq 1 40); do
  if "${kctl[@]}" get workloads.kueue.x-k8s.io -o json \
      | python3 -c '
import json,sys
uid=sys.argv[1]
items=json.load(sys.stdin)["items"]
raise SystemExit(0 if any(any(ref.get("uid")==uid for ref in x["metadata"].get("ownerReferences",[])) for x in items) else 1)
' "$job_uid"; then
    workload_found=true
    break
  fi
  sleep 0.25
done
stamp WORKLOAD_OBSERVED
if [[ "$workload_found" != "true" ]]; then
  echo "Kueue Workload was not created" >&2
  exit 4
fi

sleep 3
"${kctl[@]}" get job "$JOB" -o json > "$RESULT_DIR/job.json"
"${kctl[@]}" get workloads.kueue.x-k8s.io -o json > "$RESULT_DIR/workloads.json"
"${kctl[@]}" get pods -l batch.kubernetes.io/job-name="$JOB" -o json > "$RESULT_DIR/pods.json"
"${kctl[@]}" get events -o json > "$RESULT_DIR/events.json"
"${kctl[@]}" get resourcequota gpu-quota -o json > "$RESULT_DIR/resourcequota.json"

pod_count=$("${kctl[@]}" get pods -l batch.kubernetes.io/job-name="$JOB" -o jsonpath='{.items[*].metadata.name}' | wc -w)
suspended=$("${kctl[@]}" get job "$JOB" -o jsonpath='{.spec.suspend}')
echo "OBSERVED job_uid=$job_uid suspend=$suspended pods=$pod_count"
echo "RESULT localqueue_count=$localqueue_count workload_created=true suspended=$suspended pods=$pod_count"

stamp CLEANUP_BEGIN
trap - EXIT
cleanup
stamp CLEANUP_END
"${kctl[@]}" get job "$JOB" > "$RESULT_DIR/cleanup-job-absent.txt" 2>&1 || true
"${kctl[@]}" get workloads.kueue.x-k8s.io -o json > "$RESULT_DIR/workloads-after-cleanup.json"
stamp DIAGNOSTIC_COMPLETE
