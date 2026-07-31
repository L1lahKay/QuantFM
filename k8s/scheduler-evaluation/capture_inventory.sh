#!/usr/bin/env bash

set -euo pipefail

echo "ARCHIVED ONLY: this inventory used historical administrator visibility." >&2
echo "Use capture_gpu_dev_state.sh with the required restricted kubeconfig." >&2
exit 2

# The original commands are retained below only to explain the 2026-07-30
# snapshot. They must not be executed in the current GPU environment.

date -u +%Y-%m-%dT%H:%M:%SZ
kubectl get node gpu-dev-01 -L accelerator \
  -o custom-columns='NAME:.metadata.name,READY:.status.conditions[?(@.type=="Ready")].status,GPU:.status.allocatable.nvidia\.com/gpu,ACCELERATOR:.metadata.labels.accelerator,K8S:.status.nodeInfo.kubeletVersion'

if ! kubectl get crd | rg -i 'kueue|volcano|podgroup|clusterqueue'; then
  echo "No Kueue or Volcano CRDs installed"
fi

kubectl api-resources --api-group=scheduling.k8s.io
kubectl get resourcequota -A
kubectl get storageclass
kubectl get pvc -A -o wide
df -hT /home/khalil/DataCleaning7.3/QuantFM /data/k3s/storage
du -sh /home/khalil/DataCleaning7.3/QuantFM/quant_fm/runs
