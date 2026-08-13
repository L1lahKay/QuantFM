#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
KUBECONFIG_PATH=/etc/rancher/k3s/k3s.yaml
KUBE_CONTEXT=default
NAMESPACE=gpu-dev
FIELD_MANAGER=quantfm-scheduler-benchmark
OWNER=khalil
PURPOSE=scheduler-benchmark-runtime

RESOURCE_FLAVOR=khalil-kueue-nvidia
CLUSTER_QUEUE=khalil-kueue-eval
LOCAL_QUEUE=khalil-kueue-eval
VOLCANO_QUEUE=khalil-volcano-smoke
GPU_QUOTA=gpu-quota
EVIDENCE_ROOT="${REPO_ROOT}/benchmark/results/scheduler-setup"

MANIFESTS=(
  "${SCRIPT_DIR}/00-kueue-resourceflavor.yaml"
  "${SCRIPT_DIR}/01-kueue-clusterqueue.yaml"
  "${SCRIPT_DIR}/02-kueue-localqueue.yaml"
  "${SCRIPT_DIR}/03-volcano-queue.yaml"
)

usage() {
  cat <<'EOF'
Usage: manage.sh [plan|apply|cleanup] [--execute] [--evidence-dir PATH]

plan is non-mutating and is the default. apply and cleanup require --execute.
The Kubernetes identity is fixed to /etc/rancher/k3s/k3s.yaml, context default.
Evidence must remain below benchmark/results/scheduler-setup.
EOF
}

ACTION=plan
EXECUTE=false
EVIDENCE_DIR=""
if [[ $# -gt 0 && "$1" != --* ]]; then
  ACTION="$1"
  shift
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      EXECUTE=true
      shift
      ;;
    --evidence-dir)
      [[ $# -ge 2 ]] || { echo "ERROR: --evidence-dir requires a path" >&2; exit 2; }
      EVIDENCE_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unsupported argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${ACTION}" in
  plan|apply|cleanup) ;;
  *) echo "ERROR: action must be plan, apply, or cleanup" >&2; exit 2 ;;
esac
if [[ "${ACTION}" != plan && "${EXECUTE}" != true ]]; then
  echo "ERROR: ${ACTION} requires --execute" >&2
  exit 2
fi
if [[ "${ACTION}" == plan && "${EXECUTE}" == true ]]; then
  echo "ERROR: --execute is not valid with plan" >&2
  exit 2
fi

if [[ -z "${EVIDENCE_DIR}" ]]; then
  RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  EVIDENCE_DIR="${EVIDENCE_ROOT}/${ACTION}-${RUN_STAMP}"
fi
EVIDENCE_DIR="$(realpath -m -- "${EVIDENCE_DIR}")"
case "${EVIDENCE_DIR}" in
  "${EVIDENCE_ROOT}"/*) ;;
  *)
    echo "ERROR: evidence directory must be below ${EVIDENCE_ROOT}" >&2
    exit 2
    ;;
esac
if [[ -e "${EVIDENCE_DIR}" ]]; then
  echo "ERROR: refusing to overwrite evidence directory: ${EVIDENCE_DIR}" >&2
  exit 2
fi
mkdir -p -- "${EVIDENCE_DIR}"

KUBECTL=(kubectl --kubeconfig "${KUBECONFIG_PATH}" --context "${KUBE_CONTEXT}")

record() {
  printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"${EVIDENCE_DIR}/timeline.txt"
}

run_auth_check() {
  local verb="$1"
  local resource="$2"
  local scope="$3"
  local answer
  if [[ "${scope}" == namespace ]]; then
    answer="$("${KUBECTL[@]}" -n "${NAMESPACE}" auth can-i "${verb}" "${resource}")"
  else
    answer="$("${KUBECTL[@]}" auth can-i "${verb}" "${resource}")"
  fi
  printf '%s\t%s\t%s\t%s\n' "${verb}" "${resource}" "${scope}" "${answer}" \
    >>"${EVIDENCE_DIR}/auth-can-i.tsv"
  [[ "${answer}" == yes ]] || {
    echo "ERROR: denied: ${verb} ${resource} (${scope})" >&2
    exit 1
  }
}

permission_preflight() {
  : >"${EVIDENCE_DIR}/auth-can-i.tsv"
  local verb
  for verb in get create patch delete; do
    run_auth_check "${verb}" resourceflavors.kueue.x-k8s.io cluster
    run_auth_check "${verb}" clusterqueues.kueue.x-k8s.io cluster
    run_auth_check "${verb}" localqueues.kueue.x-k8s.io namespace
    run_auth_check "${verb}" queues.scheduling.volcano.sh cluster
  done
  run_auth_check list jobs.batch namespace
  run_auth_check list pods namespace
  run_auth_check list workloads.kueue.x-k8s.io namespace
  run_auth_check list podgroups.scheduling.volcano.sh namespace
  run_auth_check list podgroups.scheduling.volcano.sh cluster
  run_auth_check get resourcequotas namespace
}

capture_preflight() {
  "${KUBECTL[@]}" -n "${NAMESPACE}" get resourcequota -o json \
    >"${EVIDENCE_DIR}/resourcequota-before.json"
  "${KUBECTL[@]}" -n "${NAMESPACE}" get jobs.batch,pods,workloads.kueue.x-k8s.io,podgroups.scheduling.volcano.sh \
    -o json >"${EVIDENCE_DIR}/evaluation-objects-before.json"
  "${KUBECTL[@]}" get resourceflavors.kueue.x-k8s.io,clusterqueues.kueue.x-k8s.io,queues.scheduling.volcano.sh \
    -o json >"${EVIDENCE_DIR}/cluster-scheduler-objects-before.json"
  "${KUBECTL[@]}" -n "${NAMESPACE}" get localqueues.kueue.x-k8s.io \
    -o json >"${EVIDENCE_DIR}/localqueues-before.json"
  "${KUBECTL[@]}" get pods --all-namespaces --field-selector=status.phase=Running \
    -o 'custom-columns=NAMESPACE:.metadata.namespace,NAME:.metadata.name,GPU:.spec.containers[*].resources.requests.nvidia\.com/gpu,INIT_GPU:.spec.initContainers[*].resources.requests.nvidia\.com/gpu' \
    >"${EVIDENCE_DIR}/running-pod-gpu-requests.txt"
}

HOST_PROCESS_QUERY_SUCCESS=false
HOST_PROCESS_COUNT=unknown
capture_host_gpu_processes() {
  local suffix="$1"
  local process_file="${EVIDENCE_DIR}/host-gpu-processes-${suffix}.csv"
  local meta_file="${EVIDENCE_DIR}/host-gpu-processes-${suffix}.meta.tsv"
  local temporary_file="${EVIDENCE_DIR}/.host-gpu-processes-${suffix}.tmp"
  local returncode
  set +e
  nvidia-smi --query-compute-apps=gpu_uuid,used_memory \
    --format=csv,noheader,nounits >"${temporary_file}" 2>/dev/null
  returncode=$?
  set -e
  printf 'gpu_uuid,used_memory_mib\n' >"${process_file}"
  if [[ ${returncode} -eq 0 ]]; then
    awk 'NF {print}' "${temporary_file}" >>"${process_file}"
    HOST_PROCESS_COUNT="$(awk 'NR > 1 && NF {count++} END {print count+0}' "${process_file}")"
    HOST_PROCESS_QUERY_SUCCESS=true
  else
    HOST_PROCESS_COUNT=unknown
    HOST_PROCESS_QUERY_SUCCESS=false
  fi
  rm -f -- "${temporary_file}"
  {
    printf 'observed_at_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'query_success\t%s\n' "${HOST_PROCESS_QUERY_SUCCESS}"
    printf 'returncode\t%s\n' "${returncode}"
    printf 'process_count\t%s\n' "${HOST_PROCESS_COUNT}"
    printf 'fields_recorded\tgpu_uuid,used_memory_mib\n'
  } >"${meta_file}"
  record "HOST_GPU_PROCESS_QUERY suffix=${suffix} success=${HOST_PROCESS_QUERY_SUCCESS} count=${HOST_PROCESS_COUNT}"
}

require_host_gpus_idle() {
  if [[ "${HOST_PROCESS_QUERY_SUCCESS}" != true ]]; then
    echo "ERROR: host GPU compute-process state could not be verified; refusing setup apply" >&2
    exit 1
  fi
  if [[ "${HOST_PROCESS_COUNT}" != 0 ]]; then
    echo "ERROR: host GPU compute processes are active; refusing setup apply" >&2
    exit 1
  fi
}

capture_quota_hard() {
  local destination="$1"
  "${KUBECTL[@]}" -n "${NAMESPACE}" get resourcequota "${GPU_QUOTA}" \
    -o go-template='requests.gpu={{ index .status.hard "requests.nvidia.com/gpu" }}{{ "\n" }}limits.gpu={{ index .status.hard "limits.nvidia.com/gpu" }}{{ "\n" }}requests.cpu={{ index .status.hard "requests.cpu" }}{{ "\n" }}limits.cpu={{ index .status.hard "limits.cpu" }}{{ "\n" }}requests.memory={{ index .status.hard "requests.memory" }}{{ "\n" }}limits.memory={{ index .status.hard "limits.memory" }}{{ "\n" }}' \
    >"${destination}"
}

verify_quota_and_idle_gpu_requests() {
  local requests_hard limits_hard requests_used limits_used
  requests_hard="$("${KUBECTL[@]}" -n "${NAMESPACE}" get resourcequota "${GPU_QUOTA}" \
    -o go-template='{{ index .status.hard "requests.nvidia.com/gpu" }}')"
  limits_hard="$("${KUBECTL[@]}" -n "${NAMESPACE}" get resourcequota "${GPU_QUOTA}" \
    -o go-template='{{ index .status.hard "limits.nvidia.com/gpu" }}')"
  requests_used="$("${KUBECTL[@]}" -n "${NAMESPACE}" get resourcequota "${GPU_QUOTA}" \
    -o go-template='{{ index .status.used "requests.nvidia.com/gpu" }}')"
  limits_used="$("${KUBECTL[@]}" -n "${NAMESPACE}" get resourcequota "${GPU_QUOTA}" \
    -o go-template='{{ index .status.used "limits.nvidia.com/gpu" }}')"
  if [[ -z "${requests_used}" || "${requests_used}" == "<no value>" ]]; then
    requests_used=0
  fi
  if [[ -z "${limits_used}" || "${limits_used}" == "<no value>" ]]; then
    limits_used=0
  fi
  if [[ "${requests_hard}" != 4 || "${limits_hard}" != 4 || \
        "${requests_used}" != 0 || "${limits_used}" != 0 ]]; then
    echo "ERROR: gpu-dev GPU ResourceQuota must remain hard requests=4, limits=4, used requests=0, limits=0" >&2
    exit 1
  fi
  if awk 'NR > 1 && (($3 != "<none>" && $3 != "") || ($4 != "<none>" && $4 != "")) {found=1} END {exit found ? 0 : 1}' \
      "${EVIDENCE_DIR}/running-pod-gpu-requests.txt"; then
    echo "ERROR: unrelated Running GPU requests exist; refusing scheduler setup mutation" >&2
    exit 1
  fi
  capture_quota_hard "${EVIDENCE_DIR}/resourcequota-hard-before.txt"
}

server_dry_run() {
  : >"${EVIDENCE_DIR}/server-dry-run.yaml"
  local manifest
  for manifest in "${MANIFESTS[@]}"; do
    "${KUBECTL[@]}" apply --server-side --dry-run=server \
      --field-manager="${FIELD_MANAGER}" -f "${manifest}" -o yaml \
      >>"${EVIDENCE_DIR}/server-dry-run.yaml"
  done
}

GUARD_STATE=unknown
guard_owned_object() {
  local resource="$1"
  local name="$2"
  local -a scoped=()
  if [[ "${resource}" == localqueues.kueue.x-k8s.io ]]; then
    scoped=(-n "${NAMESPACE}")
  fi
  local response_file="${EVIDENCE_DIR}/guard-${resource%%.*}-${name}.txt"
  if "${KUBECTL[@]}" "${scoped[@]}" get "${resource}" "${name}" -o name \
      >"${response_file}" 2>&1; then
    local owner purpose
    owner="$("${KUBECTL[@]}" "${scoped[@]}" get "${resource}" "${name}" \
      -o go-template='{{ index .metadata.labels "evaluation.quantfm/owner" }}')"
    purpose="$("${KUBECTL[@]}" "${scoped[@]}" get "${resource}" "${name}" \
      -o go-template='{{ index .metadata.labels "evaluation.quantfm/purpose" }}')"
    if [[ "${owner}" != "${OWNER}" || "${purpose}" != "${PURPOSE}" ]]; then
      echo "ERROR: refusing to reconcile unowned ${resource}/${name}" >&2
      exit 1
    fi
    GUARD_STATE=owned
    record "OWNED_OBJECT_RECONCILE_ALLOWED ${resource}/${name}"
  else
    if ! rg -qi 'not[ ]*found|notfound' "${response_file}"; then
      echo "ERROR: cannot determine presence of ${resource}/${name}" >&2
      exit 1
    fi
    GUARD_STATE=absent
    record "EXACT_OBJECT_ABSENT ${resource}/${name}"
  fi
}

guard_all_exact_objects() {
  guard_owned_object resourceflavors.kueue.x-k8s.io "${RESOURCE_FLAVOR}"
  guard_owned_object clusterqueues.kueue.x-k8s.io "${CLUSTER_QUEUE}"
  guard_owned_object localqueues.kueue.x-k8s.io "${LOCAL_QUEUE}"
  guard_owned_object queues.scheduling.volcano.sh "${VOLCANO_QUEUE}"
}

apply_exact_objects() {
  local manifest
  for manifest in "${MANIFESTS[@]}"; do
    "${KUBECTL[@]}" apply --server-side --field-manager="${FIELD_MANAGER}" \
      -f "${manifest}" -o json >>"${EVIDENCE_DIR}/apply-responses.jsonl"
  done
  "${KUBECTL[@]}" wait --for=condition=Active \
    clusterqueue.kueue.x-k8s.io/"${CLUSTER_QUEUE}" --timeout=120s
  "${KUBECTL[@]}" -n "${NAMESPACE}" wait --for=condition=Active \
    localqueue.kueue.x-k8s.io/"${LOCAL_QUEUE}" --timeout=120s
  "${KUBECTL[@]}" wait --for=jsonpath='{.status.state}'=Open \
    queue.scheduling.volcano.sh/"${VOLCANO_QUEUE}" --timeout=120s
  "${KUBECTL[@]}" get resourceflavor.kueue.x-k8s.io/"${RESOURCE_FLAVOR}" \
    clusterqueue.kueue.x-k8s.io/"${CLUSTER_QUEUE}" -o json \
    >"${EVIDENCE_DIR}/kueue-cluster-objects-after.json"
  "${KUBECTL[@]}" -n "${NAMESPACE}" get localqueue.kueue.x-k8s.io/"${LOCAL_QUEUE}" \
    -o json >"${EVIDENCE_DIR}/kueue-localqueue-after.json"
  "${KUBECTL[@]}" get queue.scheduling.volcano.sh/"${VOLCANO_QUEUE}" \
    -o json >"${EVIDENCE_DIR}/volcano-queue-after.json"
  capture_quota_hard "${EVIDENCE_DIR}/resourcequota-hard-after.txt"
  cmp "${EVIDENCE_DIR}/resourcequota-hard-before.txt" \
    "${EVIDENCE_DIR}/resourcequota-hard-after.txt"
}

assert_no_benchmark_objects() {
  local resource output
  : >"${EVIDENCE_DIR}/cleanup-benchmark-object-check.txt"
  for resource in jobs.batch workloads.kueue.x-k8s.io podgroups.scheduling.volcano.sh pods; do
    output="$("${KUBECTL[@]}" -n "${NAMESPACE}" get "${resource}" \
      -l app.kubernetes.io/name=khalil-scheduler-benchmark -o name)"
    if [[ -n "${output}" ]]; then
      printf '%s\n' "${output}" >>"${EVIDENCE_DIR}/cleanup-benchmark-object-check.txt"
      echo "ERROR: benchmark ${resource} still exist; refusing queue cleanup" >&2
      exit 1
    fi
    output="$("${KUBECTL[@]}" -n "${NAMESPACE}" get "${resource}" -o name \
      | awk -F/ '$2 ~ /^khalil-bm-/ {print}')"
    if [[ -n "${output}" ]]; then
      printf '%s\n' "${output}" >>"${EVIDENCE_DIR}/cleanup-benchmark-object-check.txt"
      echo "ERROR: khalil-bm prefixed ${resource} still exist; refusing queue cleanup" >&2
      exit 1
    fi
  done
  record NO_BENCHMARK_OBJECTS_CONFIRMED
}

assert_dedicated_queues_unused() {
  local output
  : >"${EVIDENCE_DIR}/cleanup-queue-reference-check.txt"

  output="$("${KUBECTL[@]}" -n "${NAMESPACE}" get workloads.kueue.x-k8s.io \
    -o 'custom-columns=NAME:.metadata.name,LOCAL_QUEUE:.spec.queueName,CLUSTER_QUEUE:.status.admission.clusterQueue' \
    --no-headers \
    | awk -v local_queue="${LOCAL_QUEUE}" -v cluster_queue="${CLUSTER_QUEUE}" \
        '$2 == local_queue || $3 == cluster_queue {print}')"
  if [[ -n "${output}" ]]; then
    printf 'Kueue Workload references:\n%s\n' "${output}" \
      >>"${EVIDENCE_DIR}/cleanup-queue-reference-check.txt"
    echo "ERROR: Workloads still reference the dedicated Kueue queues" >&2
    exit 1
  fi

  output="$("${KUBECTL[@]}" get podgroups.scheduling.volcano.sh --all-namespaces \
    -o 'custom-columns=NAMESPACE:.metadata.namespace,NAME:.metadata.name,QUEUE:.spec.queue' \
    --no-headers \
    | awk -v queue="${VOLCANO_QUEUE}" '$3 == queue {print}')"
  if [[ -n "${output}" ]]; then
    printf 'Volcano PodGroup references:\n%s\n' "${output}" \
      >>"${EVIDENCE_DIR}/cleanup-queue-reference-check.txt"
    echo "ERROR: PodGroups still reference the dedicated Volcano queue" >&2
    exit 1
  fi

  record DEDICATED_QUEUES_UNUSED_CONFIRMED
}

delete_owned_exact() {
  local resource="$1"
  local name="$2"
  local -a scoped=()
  if [[ "${resource}" == localqueues.kueue.x-k8s.io ]]; then
    scoped=(-n "${NAMESPACE}")
  fi
  guard_owned_object "${resource}" "${name}"
  if [[ "${GUARD_STATE}" == absent ]]; then
    record "EXACT_OBJECT_ALREADY_ABSENT ${resource}/${name}"
    return
  fi
  local uid
  uid="$("${KUBECTL[@]}" "${scoped[@]}" get "${resource}" "${name}" \
    -o jsonpath='{.metadata.uid}')"
  printf '%s\t%s\t%s\n' "${resource}" "${name}" "${uid}" \
    >>"${EVIDENCE_DIR}/cleanup-target-uids.tsv"
  "${KUBECTL[@]}" "${scoped[@]}" delete "${resource}" "${name}" \
    --wait=true --timeout=120s
  if "${KUBECTL[@]}" "${scoped[@]}" get "${resource}" "${name}" >/dev/null 2>&1; then
    echo "ERROR: ${resource}/${name} remains after cleanup" >&2
    exit 1
  fi
  record "EXACT_OBJECT_ABSENT_AFTER_CLEANUP ${resource}/${name} uid=${uid}"
}

cleanup_exact_objects() {
  : >"${EVIDENCE_DIR}/cleanup-target-uids.tsv"
  assert_no_benchmark_objects
  assert_dedicated_queues_unused
  delete_owned_exact localqueues.kueue.x-k8s.io "${LOCAL_QUEUE}"
  delete_owned_exact clusterqueues.kueue.x-k8s.io "${CLUSTER_QUEUE}"
  delete_owned_exact resourceflavors.kueue.x-k8s.io "${RESOURCE_FLAVOR}"
  delete_owned_exact queues.scheduling.volcano.sh "${VOLCANO_QUEUE}"
  assert_no_benchmark_objects
  capture_quota_hard "${EVIDENCE_DIR}/resourcequota-hard-after.txt"
  cmp "${EVIDENCE_DIR}/resourcequota-hard-before.txt" \
    "${EVIDENCE_DIR}/resourcequota-hard-after.txt"
}

record "BEGIN action=${ACTION} context=${KUBE_CONTEXT} namespace=${NAMESPACE}"
permission_preflight
capture_preflight
capture_host_gpu_processes initial
verify_quota_and_idle_gpu_requests
if [[ "${ACTION}" == cleanup ]]; then
  guard_all_exact_objects
  cleanup_exact_objects
else
  server_dry_run
  guard_all_exact_objects
  if [[ "${ACTION}" == apply ]]; then
    assert_no_benchmark_objects
    assert_dedicated_queues_unused
    # Repeat immediately before the first mutation; do not rely on an older
    # inventory or the initial observation made earlier in this invocation.
    capture_host_gpu_processes pre-apply
    require_host_gpus_idle
    apply_exact_objects
  fi
fi
record "COMPLETE action=${ACTION}"
echo "scheduler_setup_action=${ACTION}"
echo "evidence_directory=${EVIDENCE_DIR}"
