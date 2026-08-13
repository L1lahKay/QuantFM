#!/usr/bin/env bash

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULT_DIR="${RESULT_DIR:-$ROOT/docs/assets/gpu-scheduler-evaluation/raw/current}"
KUBECONFIG_PATH=/etc/rancher/k3s/k3s.yaml
KUBE_CONTEXT=default
NAMESPACE=gpu-dev

mkdir -p "$RESULT_DIR"
OUTPUT="$RESULT_DIR/cluster-access.txt"
exec > >(tee "$OUTPUT") 2>&1

kctl=(
  kubectl
  "--kubeconfig=$KUBECONFIG_PATH"
  "--context=$KUBE_CONTEXT"
)

can_i() {
  local verb=$1
  local resource=$2
  local scope=${3:-namespace}
  local answer
  if [[ "$scope" == "cluster" ]]; then
    answer=$("${kctl[@]}" auth can-i "$verb" "$resource" 2>/dev/null)
  else
    answer=$("${kctl[@]}" auth can-i "$verb" "$resource" -n "$NAMESPACE" 2>/dev/null)
  fi
  printf 'CAN_I %-7s %-45s %s\n' "$verb" "$resource" "$answer"
}

echo "CAPTURED_AT $(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
"${kctl[@]}" version

echo "KUEUE_API_RESOURCES"
"${kctl[@]}" api-resources --api-group=kueue.x-k8s.io
echo "VOLCANO_API_RESOURCES"
volcano_resources=$("${kctl[@]}" api-resources \
  --api-group=scheduling.volcano.sh --no-headers 2>&1)
if [[ -n "$volcano_resources" ]]; then
  printf '%s\n' "$volcano_resources"
else
  echo "NONE"
fi

echo "RESTRICTED_ACCOUNT_PERMISSIONS"
can_i create jobs.batch
can_i create pods
can_i get resourcequotas
can_i get localqueues.kueue.x-k8s.io
can_i get workloads.kueue.x-k8s.io
can_i list clusterqueues.kueue.x-k8s.io cluster
can_i create persistentvolumeclaims
can_i create namespaces cluster
can_i create priorityclasses.scheduling.k8s.io cluster
can_i get nodes cluster

echo "VISIBLE_JOBS_AND_PODS"
"${kctl[@]}" -n "$NAMESPACE" get jobs,pods -o wide

echo "DIRECT_QUOTA_READ_EXPECTED_FORBIDDEN"
"${kctl[@]}" -n "$NAMESPACE" get resourcequota 2>&1 || true
echo "DIRECT_KUEUE_READ_EXPECTED_FORBIDDEN"
"${kctl[@]}" -n "$NAMESPACE" get localqueues.kueue.x-k8s.io 2>&1 || true
"${kctl[@]}" -n "$NAMESPACE" get workloads.kueue.x-k8s.io 2>&1 || true
