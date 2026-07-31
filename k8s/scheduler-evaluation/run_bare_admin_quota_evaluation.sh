#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST_DIR="$ROOT/k8s/scheduler-evaluation/bare/admin-safe"
RESULT_DIR="${RESULT_DIR:-$ROOT/docs/assets/gpu-scheduler-evaluation/raw/bare-current}"
KUBECONFIG_PATH=/etc/rancher/k3s/k3s.yaml
KUBE_CONTEXT=default
NAMESPACE=gpu-dev
HOLDER=khalil-bare-quota-holder
WAITER=khalil-bare-quota-waiter

if [[ "${RUN_BARE_ADMIN_EVALUATION:-}" != "1" ]]; then
  echo "Refusing to submit: set RUN_BARE_ADMIN_EVALUATION=1 explicitly" >&2
  exit 2
fi

mkdir -p "$RESULT_DIR"
TRANSCRIPT="$RESULT_DIR/transcript.txt"
TIMELINE="$RESULT_DIR/timeline.txt"
: > "$TIMELINE"
exec > >(tee "$TRANSCRIPT") 2>&1

kctl=(
  kubectl
  --kubeconfig "$KUBECONFIG_PATH"
  --context "$KUBE_CONTEXT"
  --namespace "$NAMESPACE"
)

stamp() {
  local label=$1
  local utc monotonic
  utc=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
  monotonic=$(python3 -c 'import time; print(time.monotonic_ns())')
  printf '%s utc=%s monotonic_ns=%s\n' "$label" "$utc" "$monotonic" | tee -a "$TIMELINE"
}

cleanup() {
  stamp CLEANUP_BEGIN
  "${kctl[@]}" delete job "$HOLDER" "$WAITER" --ignore-not-found --wait=true || true
  stamp CLEANUP_END
}
trap cleanup EXIT

echo "ADMIN_KUBECONFIG_PATH=$KUBECONFIG_PATH"
echo "ADMIN_CONTEXT=$KUBE_CONTEXT"
echo "NAMESPACE=$NAMESPACE"
stamp EVALUATION_BEGIN

for name in "$HOLDER" "$WAITER"; do
  if "${kctl[@]}" get job "$name" >/dev/null 2>&1; then
    echo "Refusing to overwrite existing Job $name" >&2
    exit 3
  fi
done

"${kctl[@]}" get resourcequota gpu-quota -o json > "$RESULT_DIR/quota-before.json"
"${kctl[@]}" get pods -o json > "$RESULT_DIR/pods-before.json"
running_gpu=$("${kctl[@]}" get pods -o json | python3 -c '
import json, sys
pods=json.load(sys.stdin)["items"]
print(sum(int(c.get("resources",{}).get("requests",{}).get("nvidia.com/gpu",0)) for p in pods if p.get("status",{}).get("phase")=="Running" for c in p["spec"]["containers"]))
')
quota_hard=$("${kctl[@]}" get resourcequota gpu-quota -o jsonpath='{.status.hard.requests\.nvidia\.com/gpu}')
quota_used=$("${kctl[@]}" get resourcequota gpu-quota -o jsonpath='{.status.used.requests\.nvidia\.com/gpu}')
echo "PREFLIGHT_RUNNING_GPU_REQUEST_TOTAL=$running_gpu"
echo "GPU_DEV_QUOTA_HARD=$quota_hard USED=$quota_used"
if [[ "$running_gpu" != "0" || "$quota_used" != "0" ]]; then
  echo "Refusing to consume shared GPU quota while another GPU workload is active" >&2
  exit 4
fi

for manifest in 01-quota-holder.yaml 02-quota-waiter.yaml; do
  echo "SERVER_DRY_RUN $manifest"
  "${kctl[@]}" apply --server-side --dry-run=server -f "$MANIFEST_DIR/$manifest"
done
stamp SERVER_DRY_RUN_COMPLETE

stamp HOLDER_CREATE_BEGIN
"${kctl[@]}" create -f "$MANIFEST_DIR/01-quota-holder.yaml"
stamp HOLDER_CREATE_RETURN
"${kctl[@]}" wait --for=condition=Ready pod -l 'experiment=bare-current-quota,role=holder' --timeout=60s
stamp HOLDER_READY_OBSERVED

holder_pod=$("${kctl[@]}" get pod -l 'experiment=bare-current-quota,role=holder' -o jsonpath='{.items[0].metadata.name}')
holder_uid=$("${kctl[@]}" get pod "$holder_pod" -o jsonpath='{.metadata.uid}')
used_while_holder=$("${kctl[@]}" get resourcequota gpu-quota -o jsonpath='{.status.used.requests\.nvidia\.com/gpu}')
echo "HOLDER_POD=$holder_pod UID=$holder_uid QUOTA_USED_GPU=$used_while_holder"
if [[ "$used_while_holder" != "4" ]]; then
  echo "Expected holder to consume all 4 quota GPUs, got $used_while_holder" >&2
  exit 5
fi

stamp WAITER_CREATE_BEGIN
"${kctl[@]}" create -f "$MANIFEST_DIR/02-quota-waiter.yaml"
stamp WAITER_CREATE_RETURN
waiter_uid=$("${kctl[@]}" get job "$WAITER" -o jsonpath='{.metadata.uid}')

quota_rejection_seen=false
for _ in $(seq 1 30); do
  if "${kctl[@]}" get events --field-selector "involvedObject.uid=$waiter_uid" -o jsonpath='{range .items[*]}{.reason}{" "}{.message}{"\n"}{end}' \
      | grep -q 'FailedCreate.*exceeded quota: gpu-quota'; then
    quota_rejection_seen=true
    break
  fi
  sleep 0.5
done
stamp QUOTA_REJECTION_OBSERVED
if [[ "$quota_rejection_seen" != "true" ]]; then
  echo "Did not observe expected ResourceQuota FailedCreate" >&2
  exit 6
fi

"${kctl[@]}" get jobs "$HOLDER" "$WAITER" -o json > "$RESULT_DIR/during-jobs.json"
"${kctl[@]}" get pods -l experiment=bare-current-quota -o json > "$RESULT_DIR/during-pods.json"
"${kctl[@]}" get events -o json > "$RESULT_DIR/during-events.json"
"${kctl[@]}" get resourcequota gpu-quota -o json > "$RESULT_DIR/during-quota.json"
waiter_pods_during=$("${kctl[@]}" get pods -l 'experiment=bare-current-quota,role=waiter' -o jsonpath='{.items[*].metadata.name}')
echo "DURING waiter_pods=${waiter_pods_during:-NONE} quota_rejection=true"
if [[ -n "$waiter_pods_during" ]]; then
  echo "Waiter Pod unexpectedly existed while quota was full" >&2
  exit 7
fi

"${kctl[@]}" wait --for=condition=complete "job/$HOLDER" --timeout=120s
stamp HOLDER_COMPLETE_OBSERVED
"${kctl[@]}" wait --for=condition=complete "job/$WAITER" --timeout=180s
stamp WAITER_COMPLETE_OBSERVED

"${kctl[@]}" get jobs "$HOLDER" "$WAITER" -o json > "$RESULT_DIR/final-jobs.json"
"${kctl[@]}" get pods -l experiment=bare-current-quota -o json > "$RESULT_DIR/final-pods.json"
"${kctl[@]}" get events -o json > "$RESULT_DIR/final-events.json"
"${kctl[@]}" get resourcequota gpu-quota -o json > "$RESULT_DIR/final-quota.json"
"${kctl[@]}" logs "job/$HOLDER" > "$RESULT_DIR/holder-output.txt"
"${kctl[@]}" logs "job/$WAITER" > "$RESULT_DIR/waiter-output.txt"

waiter_pod=$("${kctl[@]}" get pod -l 'experiment=bare-current-quota,role=waiter' -o jsonpath='{.items[0].metadata.name}')
waiter_scheduler=$("${kctl[@]}" get pod "$waiter_pod" -o jsonpath='{.spec.schedulerName}')
echo "FINAL WAITER_POD=$waiter_pod SCHEDULER=$waiter_scheduler"
echo "RESULT quota_full_failedcreate=true release_then_run=true waiter_complete=true scheduler=default-scheduler"

trap - EXIT
cleanup
"${kctl[@]}" get job "$HOLDER" "$WAITER" > "$RESULT_DIR/cleanup-jobs-absent.txt" 2>&1 || true
"${kctl[@]}" get resourcequota gpu-quota -o json > "$RESULT_DIR/quota-after-cleanup.json"
stamp EVALUATION_COMPLETE
