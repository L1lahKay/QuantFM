#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="$ROOT/k8s/scheduler-evaluation/gpu-dev/kueue-probe.yaml"
RESULT_DIR="${RESULT_DIR:-$ROOT/docs/assets/gpu-scheduler-evaluation/raw/current}"
KUBECONFIG_PATH=/etc/rancher/k3s/k3s.yaml
KUBE_CONTEXT=default
NAMESPACE=gpu-dev
POLL_ATTEMPTS="${PROBE_POLL_ATTEMPTS:-10}"

if [[ "${RUN_GPU_SCHEDULER_PROBE:-}" != "1" ]]; then
  echo "Refusing to submit: set RUN_GPU_SCHEDULER_PROBE=1 explicitly" >&2
  exit 2
fi

mkdir -p "$RESULT_DIR"
LOG="$RESULT_DIR/kueue-probe.txt"
exec > >(tee "$LOG") 2>&1

kctl=(
  kubectl
  "--kubeconfig=$KUBECONFIG_PATH"
  "--context=$KUBE_CONTEXT"
  --namespace "$NAMESPACE"
)

job_name=""
cleanup() {
  if [[ -n "$job_name" ]]; then
    "${kctl[@]}" delete job "$job_name" --ignore-not-found --wait=true
  fi
}
trap cleanup EXIT

stamp() {
  date -u +%Y-%m-%dT%H:%M:%S.%3NZ
}

echo "PROBE_CLIENT_START $(stamp)"
echo "PROBE_CONFIG poll_attempts=$POLL_ATTEMPTS queue_candidate=gpu-dev explicit_suspend=true"
"${kctl[@]}" get jobs,pods -o wide
echo "JOB_SUBMIT_CLIENT $(stamp)"
created=$("${kctl[@]}" create -f "$MANIFEST" -o name)
job_name=${created#job.batch/}
echo "JOB_CREATED_NAME $job_name"

for attempt in $(seq 0 $((POLL_ATTEMPTS - 1))); do
  now=$(stamp)
  job_state=$("${kctl[@]}" get job "$job_name" \
    -o 'jsonpath={.metadata.creationTimestamp}{" suspend="}{.spec.suspend}{" active="}{.status.active}{" succeeded="}{.status.succeeded}{"\n"}')
  pod_state=$("${kctl[@]}" get pods \
    -l "batch.kubernetes.io/job-name=$job_name" \
    -o 'jsonpath={range .items[*]}{.metadata.name}{" phase="}{.status.phase}{" started="}{.status.containerStatuses[0].state.terminated.startedAt}{.status.containerStatuses[0].state.running.startedAt}{" finished="}{.status.containerStatuses[0].state.terminated.finishedAt}{"\n"}{end}')
  echo "POLL attempt=$attempt client_at=$now job=[$job_state] pod=[$pod_state]"
  if "${kctl[@]}" get job "$job_name" \
    -o 'jsonpath={.status.conditions[?(@.type=="Complete")].status}' \
    | grep -qx True; then
    break
  fi
  sleep 1
done

"${kctl[@]}" get job "$job_name" -o json > "$RESULT_DIR/kueue-probe-job.json"
"${kctl[@]}" get pods \
  -l "batch.kubernetes.io/job-name=$job_name" \
  -o json > "$RESULT_DIR/kueue-probe-pods.json"
"${kctl[@]}" get events \
  --field-selector "involvedObject.name=$job_name" \
  -o json > "$RESULT_DIR/kueue-probe-events.json"
"${kctl[@]}" get events \
  --field-selector "involvedObject.name=$job_name" \
  --sort-by=.metadata.creationTimestamp || true
pod_resource=$("${kctl[@]}" get pods \
  -l "batch.kubernetes.io/job-name=$job_name" \
  -o name | head -n 1)
pod_name=${pod_resource#pod/}
if [[ -n "$pod_name" ]]; then
  "${kctl[@]}" logs "$pod_name" --pod-running-timeout=5s || true
else
  echo "NO_POD_LOGS Job remained gated before Pod creation"
fi

complete=$("${kctl[@]}" get job "$job_name" \
  -o 'jsonpath={.status.conditions[?(@.type=="Complete")].status}')
suspended=$("${kctl[@]}" get job "$job_name" -o 'jsonpath={.spec.suspend}')
if [[ "$complete" == "True" ]]; then
  echo "RESULT COMPLETED"
elif [[ "$suspended" == "true" && -z "$pod_name" ]]; then
  echo "RESULT SUSPENDED_NO_POD"
else
  echo "RESULT INCONCLUSIVE suspend=$suspended pod=$pod_name"
fi
echo "PROBE_CLIENT_END $(stamp)"
