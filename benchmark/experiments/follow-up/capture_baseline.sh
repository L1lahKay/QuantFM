#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
KUBECONFIG_PATH=/etc/rancher/k3s/k3s.yaml
KUBE_CONTEXT=default
NAMESPACE=gpu-dev
EVIDENCE_ROOT="${REPO_ROOT}/benchmark/results/follow-up-baseline"

usage() {
  echo "Usage: capture_baseline.sh [plan|capture] [--execute]" >&2
}

ACTION="${1:-plan}"
EXECUTE=false
if [[ $# -gt 0 ]]; then
  shift
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute) EXECUTE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unsupported argument: $1" >&2; usage; exit 2 ;;
  esac
done

case "${ACTION}" in
  plan)
    [[ "${EXECUTE}" == false ]] || { echo "ERROR: --execute is invalid with plan" >&2; exit 2; }
    cat <<EOF
mutation=none
kubeconfig=${KUBECONFIG_PATH}
context=${KUBE_CONTEXT}
namespace=${NAMESPACE}
captures=node/gpu/numa topology,schedulers,quota,monitoring,storage
privacy=no Secret,kubeconfig contents,PID,user,or command fields
next=capture_baseline.sh capture --execute
EOF
    exit 0
    ;;
  capture)
    [[ "${EXECUTE}" == true ]] || { echo "ERROR: capture requires --execute" >&2; exit 2; }
    ;;
  *) usage; exit 2 ;;
esac

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EVIDENCE_DIR="${EVIDENCE_ROOT}/${RUN_STAMP}"
[[ ! -e "${EVIDENCE_DIR}" ]] || { echo "ERROR: evidence directory exists" >&2; exit 2; }
mkdir -p -- "${EVIDENCE_DIR}"
KUBECTL=(kubectl --kubeconfig "${KUBECONFIG_PATH}" --context "${KUBE_CONTEXT}")

printf 'observed_at_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${EVIDENCE_DIR}/metadata.tsv"
printf 'kube_context\t%s\nnamespace\t%s\n' "${KUBE_CONTEXT}" "${NAMESPACE}" >>"${EVIDENCE_DIR}/metadata.tsv"

{
  printf 'scope\tverb\tresource\tallowed\n'
  for check in \
    "namespace|get|resourcequotas" \
    "namespace|list|pods" \
    "namespace|create|jobs.batch" \
    "namespace|get|jobs.batch" \
    "namespace|delete|jobs.batch" \
    "namespace|get|localqueues.kueue.x-k8s.io" \
    "namespace|create|workloads.kueue.x-k8s.io" \
    "namespace|get|podgroups.scheduling.volcano.sh" \
    "namespace|create|podgroups.scheduling.volcano.sh" \
    "cluster|get|nodes" \
    "cluster|get|resourceflavors.kueue.x-k8s.io" \
    "cluster|get|clusterqueues.kueue.x-k8s.io" \
    "cluster|get|queues.scheduling.volcano.sh"; do
    IFS='|' read -r scope verb resource <<<"${check}"
    if [[ "${scope}" == "namespace" ]]; then
      allowed="$("${KUBECTL[@]}" auth can-i "${verb}" "${resource}" --namespace "${NAMESPACE}")"
    else
      allowed="$("${KUBECTL[@]}" auth can-i "${verb}" "${resource}" --all-namespaces)"
    fi
    printf '%s\t%s\t%s\t%s\n' "${scope}" "${verb}" "${resource}" "${allowed}"
  done
} >"${EVIDENCE_DIR}/auth-can-i.tsv"

"${KUBECTL[@]}" version -o json >"${EVIDENCE_DIR}/kubernetes-version.json"
"${KUBECTL[@]}" get nodes \
  -o 'custom-columns=NAME:.metadata.name,READY:.status.conditions[?(@.type=="Ready")].status,UNSCHEDULABLE:.spec.unschedulable,GPU:.status.allocatable.nvidia\.com/gpu,CPU:.status.allocatable.cpu,MEMORY:.status.allocatable.memory,ACCELERATOR:.metadata.labels.accelerator,HOSTNAME:.metadata.labels.kubernetes\.io/hostname,ZONE:.metadata.labels.topology\.kubernetes\.io/zone,REGION:.metadata.labels.topology\.kubernetes\.io/region' \
  >"${EVIDENCE_DIR}/nodes.tsv"
"${KUBECTL[@]}" get nodes \
  -o 'custom-columns=NAME:.metadata.name,BLOCK:.metadata.labels.quantfm\.io/topology-block,RACK:.metadata.labels.quantfm\.io/topology-rack,HOSTNAME:.metadata.labels.kubernetes\.io/hostname' \
  >"${EVIDENCE_DIR}/tas-node-labels.tsv"
"${KUBECTL[@]}" -n "${NAMESPACE}" get resourcequota -o json \
  >"${EVIDENCE_DIR}/resourcequotas.json"
"${KUBECTL[@]}" get pods --all-namespaces --field-selector=status.phase=Running \
  -o 'custom-columns=NAMESPACE:.metadata.namespace,NAME:.metadata.name,GPU:.spec.containers[*].resources.requests.nvidia\.com/gpu,INIT_GPU:.spec.initContainers[*].resources.requests.nvidia\.com/gpu' \
  >"${EVIDENCE_DIR}/running-gpu-requests.tsv"

"${KUBECTL[@]}" get topology.kueue.x-k8s.io,resourceflavor.kueue.x-k8s.io,clusterqueue.kueue.x-k8s.io \
  -o json >"${EVIDENCE_DIR}/kueue-cluster-objects.json"
"${KUBECTL[@]}" -n "${NAMESPACE}" get localqueue.kueue.x-k8s.io,workload.kueue.x-k8s.io \
  -o json >"${EVIDENCE_DIR}/kueue-gpu-dev-objects.json"
"${KUBECTL[@]}" -n kueue-system get deployment kueue-controller-manager -o json \
  >"${EVIDENCE_DIR}/kueue-controller.json"
"${KUBECTL[@]}" -n kueue-system get configmap kueue-manager-config -o json \
  >"${EVIDENCE_DIR}/kueue-manager-config.json"

"${KUBECTL[@]}" get queue.scheduling.volcano.sh -o json \
  >"${EVIDENCE_DIR}/volcano-queues.json"
"${KUBECTL[@]}" -n "${NAMESPACE}" get podgroup.scheduling.volcano.sh -o json \
  >"${EVIDENCE_DIR}/volcano-podgroups.json"
"${KUBECTL[@]}" -n volcano-system get deployment volcano-scheduler -o json \
  >"${EVIDENCE_DIR}/volcano-scheduler.json"
"${KUBECTL[@]}" -n volcano-system get configmap volcano-scheduler-configmap -o json \
  >"${EVIDENCE_DIR}/volcano-scheduler-config.json"

"${KUBECTL[@]}" get storageclass -o json >"${EVIDENCE_DIR}/storageclasses.json"
"${KUBECTL[@]}" -n "${NAMESPACE}" get persistentvolumeclaim -o json \
  >"${EVIDENCE_DIR}/persistentvolumeclaims.json"
"${KUBECTL[@]}" get persistentvolume -o json >"${EVIDENCE_DIR}/persistentvolumes.json"

"${KUBECTL[@]}" get deployment,daemonset --all-namespaces \
  -o 'custom-columns=KIND:.kind,NAMESPACE:.metadata.namespace,NAME:.metadata.name,READY:.status.numberReady,AVAILABLE:.status.availableReplicas' \
  >"${EVIDENCE_DIR}/monitoring-and-device-components.tsv"
for crd in \
  clusterpolicies.nvidia.com \
  servicemonitors.monitoring.coreos.com \
  podmonitors.monitoring.coreos.com \
  prometheuses.monitoring.coreos.com; do
  if "${KUBECTL[@]}" get customresourcedefinition "${crd}" -o name \
      >>"${EVIDENCE_DIR}/monitoring-crds.txt" 2>/dev/null; then
    :
  else
    printf 'ABSENT %s\n' "${crd}" >>"${EVIDENCE_DIR}/monitoring-crds.txt"
  fi
done

nvidia-smi -L >"${EVIDENCE_DIR}/nvidia-smi-list.txt"
nvidia-smi topo -m >"${EVIDENCE_DIR}/nvidia-smi-topology.txt"
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version,compute_cap \
  --format=csv,noheader,nounits >"${EVIDENCE_DIR}/nvidia-gpus.csv"
nvidia-smi --query-compute-apps=gpu_uuid,used_memory \
  --format=csv,noheader,nounits >"${EVIDENCE_DIR}/host-gpu-compute-processes.csv"
lscpu --json >"${EVIDENCE_DIR}/lscpu.json"
if command -v numactl >/dev/null; then
  numactl --hardware >"${EVIDENCE_DIR}/numactl-hardware.txt"
else
  printf 'numactl unavailable\n' >"${EVIDENCE_DIR}/numactl-hardware.txt"
fi

lsblk --json \
  --output NAME,KNAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS,MODEL,SERIAL,WWN \
  >"${EVIDENCE_DIR}/block-devices.json"
findmnt --json --target / \
  --output TARGET,SOURCE,FSTYPE,OPTIONS >"${EVIDENCE_DIR}/root-mount.json"
findmnt --json --target /data \
  --output TARGET,SOURCE,FSTYPE,OPTIONS >"${EVIDENCE_DIR}/data-mount.json"

quantfm_pvc_name=quantfm-data
quantfm_pvc_uid="$("${KUBECTL[@]}" -n "${NAMESPACE}" get pvc "${quantfm_pvc_name}" -o jsonpath='{.metadata.uid}')"
quantfm_pv_name="$("${KUBECTL[@]}" -n "${NAMESPACE}" get pvc "${quantfm_pvc_name}" -o jsonpath='{.spec.volumeName}')"
quantfm_pv_uid="$("${KUBECTL[@]}" get pv "${quantfm_pv_name}" -o jsonpath='{.metadata.uid}')"
quantfm_local_path="$("${KUBECTL[@]}" get pv "${quantfm_pv_name}" -o jsonpath='{.spec.local.path}')"
quantfm_mount_target=""
quantfm_mount_source=""
quantfm_mount_fstype=""
while read -r mount_target mount_source mount_fstype; do
  if [[ "${mount_target}" == "/" || \
        "${quantfm_local_path}" == "${mount_target}" || \
        "${quantfm_local_path}" == "${mount_target}/"* ]]; then
    if (( ${#mount_target} > ${#quantfm_mount_target} )); then
      quantfm_mount_target="${mount_target}"
      quantfm_mount_source="${mount_source}"
      quantfm_mount_fstype="${mount_fstype}"
    fi
  fi
done < <(findmnt --raw --noheadings --output TARGET,SOURCE,FSTYPE)
[[ -n "${quantfm_mount_target}" ]] || {
  echo "ERROR: unable to resolve the filesystem backing ${NAMESPACE}/${quantfm_pvc_name}" >&2
  exit 2
}
{
  printf 'pvc_namespace\tpvc_name\tpvc_uid\tpv_name\tpv_uid\tlocal_path\tmount_target\tmount_source\tmount_fstype\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${NAMESPACE}" "${quantfm_pvc_name}" "${quantfm_pvc_uid}" \
    "${quantfm_pv_name}" "${quantfm_pv_uid}" "${quantfm_local_path}" \
    "${quantfm_mount_target}" "${quantfm_mount_source}" "${quantfm_mount_fstype}"
} >"${EVIDENCE_DIR}/quantfm-pvc-backing.tsv"

{
  printf 'interface\toperstate\tmtu\tspeed_mbps\n'
  for interface_path in /sys/class/net/*; do
    interface="${interface_path##*/}"
    operstate="$(cat "${interface_path}/operstate" 2>/dev/null || printf 'unknown')"
    mtu="$(cat "${interface_path}/mtu" 2>/dev/null || printf 'unknown')"
    speed="$(cat "${interface_path}/speed" 2>/dev/null || printf 'unknown')"
    printf '%s\t%s\t%s\t%s\n' "${interface}" "${operstate}" "${mtu}" "${speed}"
  done
} >"${EVIDENCE_DIR}/network-links.tsv"
if command -v rdma >/dev/null; then
  rdma link show >"${EVIDENCE_DIR}/rdma-links.txt" 2>&1 || true
else
  printf 'rdma command unavailable\n' >"${EVIDENCE_DIR}/rdma-links.txt"
fi

(
  cd -- "${EVIDENCE_DIR}"
  sha256sum -- * >SHA256SUMS
)
echo "follow_up_baseline=${EVIDENCE_DIR}"
