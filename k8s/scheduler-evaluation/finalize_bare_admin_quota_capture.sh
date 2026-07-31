#!/usr/bin/env bash

# Resume evidence capture if the interactive runner is detached while its
# Kubernetes Jobs continue.  This script only addresses the two exact test
# names created by run_bare_admin_quota_evaluation.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_DIR="${RESULT_DIR:-$ROOT/docs/assets/gpu-scheduler-evaluation/raw/bare-current}"
TIMELINE="$RESULT_DIR/timeline.txt"
KUBECONFIG_PATH=/etc/rancher/k3s/k3s.yaml
KUBE_CONTEXT=default
NAMESPACE=gpu-dev
HOLDER=khalil-bare-quota-holder
WAITER=khalil-bare-quota-waiter

if [[ "${FINALIZE_BARE_ADMIN_EVALUATION:-}" != "1" ]]; then
  echo "Refusing to finalize: set FINALIZE_BARE_ADMIN_EVALUATION=1 explicitly" >&2
  exit 2
fi

kctl=(kubectl --kubeconfig "$KUBECONFIG_PATH" --context "$KUBE_CONTEXT" --namespace "$NAMESPACE")

stamp() {
  local label=$1 utc monotonic
  utc=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
  monotonic=$(python3 -c 'import time; print(time.monotonic_ns())')
  printf '%s utc=%s monotonic_ns=%s\n' "$label" "$utc" "$monotonic" | tee -a "$TIMELINE"
}

"${kctl[@]}" get job "$HOLDER" "$WAITER" >/dev/null
"${kctl[@]}" wait --for=condition=complete "job/$HOLDER" --timeout=30s
"${kctl[@]}" wait --for=condition=complete "job/$WAITER" --timeout=90s
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

stamp CLEANUP_BEGIN
"${kctl[@]}" delete job "$HOLDER" "$WAITER" --wait=true
stamp CLEANUP_END
"${kctl[@]}" get job "$HOLDER" "$WAITER" > "$RESULT_DIR/cleanup-jobs-absent.txt" 2>&1 || true
"${kctl[@]}" get resourcequota gpu-quota -o json > "$RESULT_DIR/quota-after-cleanup.json"
stamp EVALUATION_COMPLETE
