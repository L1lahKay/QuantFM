#!/usr/bin/env bash

set -euo pipefail

echo "ARCHIVED ONLY: this 2026-07-30 privileged benchmark creates objects that are forbidden by the current gpu-dev-khalil rules." >&2
echo "Use run_gpu_dev_kueue_probe.sh for the current restricted-account probe." >&2
exit 2

# The original orchestration is retained below only to explain the captured
# raw evidence. It must not be executed in the current GPU environment.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST_DIR="$ROOT/k8s/scheduler-evaluation/bare"
RESULT_DIR="${RESULT_DIR:-$ROOT/docs/assets/gpu-scheduler-evaluation/raw}"
NS=khalil-scheduler-bench
QUOTA_NS=khalil-quota-bench

mkdir -p "$RESULT_DIR"
LOG="$RESULT_DIR/bare-k8s-benchmark.txt"
exec > >(tee "$LOG") 2>&1

stamp() {
  date -u +%Y-%m-%dT%H:%M:%S.%3NZ
}

echo "BENCHMARK_START $(stamp)"
kubectl apply -f "$MANIFEST_DIR/namespace.yaml"

echo "EXPERIMENT_CONTENTION_START $(stamp)"
kubectl -n "$NS" delete job bare-holder-5gpu bare-waiter-5gpu --ignore-not-found
kubectl apply -f "$MANIFEST_DIR/contention-holder.yaml"
echo "HOLDER_SUBMITTED $(stamp)"
kubectl -n "$NS" wait \
  --for=condition=Ready pod -l role=holder --timeout=45s
echo "HOLDER_READY $(stamp)"
kubectl apply -f "$MANIFEST_DIR/contention-waiter.yaml"
echo "WAITER_SUBMITTED $(stamp)"
sleep 3
echo "CONTENTTION_PENDING_SNAPSHOT $(stamp)"
kubectl -n "$NS" get jobs,pods -o wide
waiter_pod="$(kubectl -n "$NS" get pod -l role=waiter -o jsonpath='{.items[0].metadata.name}')"
kubectl -n "$NS" get events \
  --field-selector "involvedObject.name=$waiter_pod" \
  --sort-by=.lastTimestamp
kubectl -n "$NS" wait \
  --for=condition=complete job/bare-holder-5gpu --timeout=90s
kubectl -n "$NS" wait \
  --for=condition=complete job/bare-waiter-5gpu --timeout=90s
echo "CONTENTION_COMPLETE $(stamp)"
kubectl -n "$NS" get jobs,pods -o wide
kubectl -n "$NS" logs job/bare-holder-5gpu
kubectl -n "$NS" logs job/bare-waiter-5gpu
kubectl -n "$NS" get jobs -l experiment=bare-contention -o json \
  > "$RESULT_DIR/contention-jobs.json"
kubectl -n "$NS" get pods -l experiment=bare-contention -o json \
  > "$RESULT_DIR/contention-pods.json"

echo "EXPERIMENT_QUOTA_START $(stamp)"
kubectl apply -f "$MANIFEST_DIR/quota.yaml"
kubectl -n "$QUOTA_NS" delete job bare-over-quota-2gpu --ignore-not-found
kubectl apply -f "$MANIFEST_DIR/quota-over-limit-job.yaml"
echo "OVER_QUOTA_JOB_SUBMITTED $(stamp)"
sleep 4
kubectl -n "$QUOTA_NS" get resourcequota,jobs,pods -o wide
kubectl -n "$QUOTA_NS" describe job bare-over-quota-2gpu
kubectl -n "$QUOTA_NS" get job bare-over-quota-2gpu -o json \
  > "$RESULT_DIR/quota-job.json"
kubectl -n "$QUOTA_NS" get resourcequota bare-gpu-quota -o json \
  > "$RESULT_DIR/quota.json"
echo "EXPERIMENT_QUOTA_CAPTURED $(stamp)"

echo "EXPERIMENT_PREEMPTION_START $(stamp)"
kubectl apply -f "$MANIFEST_DIR/priorityclasses.yaml"
kubectl -n "$NS" delete pod bare-preempt-low-5gpu bare-preempt-high-5gpu \
  --ignore-not-found --wait=true
kubectl apply -f "$MANIFEST_DIR/preemption-low.yaml"
echo "LOW_SUBMITTED $(stamp)"
kubectl -n "$NS" wait \
  --for=condition=Ready pod/bare-preempt-low-5gpu --timeout=45s
echo "LOW_READY $(stamp)"
kubectl apply -f "$MANIFEST_DIR/preemption-high.yaml"
echo "HIGH_SUBMITTED $(stamp)"
sleep 3
echo "PREEMPTION_SNAPSHOT $(stamp)"
kubectl -n "$NS" get pods \
  -l experiment=bare-preemption \
  -o custom-columns='NAME:.metadata.name,PRIORITY:.spec.priority,PHASE:.status.phase,NODE:.spec.nodeName,NOMINATED:.status.nominatedNodeName'
kubectl -n "$NS" get events \
  --field-selector involvedObject.name=bare-preempt-low-5gpu \
  --sort-by=.lastTimestamp
kubectl -n "$NS" get events \
  --field-selector involvedObject.name=bare-preempt-high-5gpu \
  --sort-by=.lastTimestamp
kubectl -n "$NS" wait \
  --for=jsonpath='{.status.phase}'=Succeeded \
  pod/bare-preempt-high-5gpu --timeout=60s
echo "PREEMPTION_COMPLETE $(stamp)"
kubectl -n "$NS" get pods \
  -l experiment=bare-preemption \
  -o custom-columns='NAME:.metadata.name,PRIORITY:.spec.priority,PHASE:.status.phase,NODE:.spec.nodeName,REASON:.status.reason'
kubectl -n "$NS" logs bare-preempt-high-5gpu
kubectl -n "$NS" get pods -l experiment=bare-preemption -o json \
  > "$RESULT_DIR/preemption-pods.json"

echo "BENCHMARK_END $(stamp)"
