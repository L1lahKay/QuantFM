#!/usr/bin/env bash
set -Eeuo pipefail

# This is an administrator-only, destructive-to-test-resources evaluation.
# It is intentionally pinned to the one-time kubeconfig exception recorded in
# AGENTS.md. The kubeconfig is read in place and is never copied or printed.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
BENCH_DIR="${SCRIPT_DIR}/volcano/bench"
RAW_DIR="${REPO_ROOT}/docs/assets/gpu-scheduler-evaluation/raw/volcano"
NAMESPACE="gpu-dev"
KUBECTL=(kubectl --kubeconfig /etc/rancher/k3s/k3s.yaml --context default)
LABEL_SELECTOR="evaluation.quantfm/owner=khalil,evaluation.quantfm/suite=volcano-v1-15-1"
TRANSCRIPT="${RAW_DIR}/evaluation-transcript.txt"
TIMELINE="${RAW_DIR}/evaluation-timeline.txt"
ORIGINAL_CONFIG="${RAW_DIR}/scheduler-config-before.json"
ORIGINAL_PATCH="${RAW_DIR}/scheduler-config-restore-patch.json"
SCHEDULER_CHANGED=0

QUEUE_NAMES=(
  khalil-volcano-smoke
  khalil-volcano-quota
  khalil-volcano-gang
  khalil-volcano-preempt
)
PRIORITY_NAMES=(
  khalil-volcano-low
  khalil-volcano-high
)
JOB_NAMES=(
  khalil-volcano-smoke
  khalil-volcano-quota-holder
  khalil-volcano-quota-waiter
  khalil-volcano-gang
  khalil-volcano-preempt-victim
  khalil-volcano-preempt-high
)
PODGROUP_NAMES=(
  khalil-volcano-smoke-pg
  khalil-volcano-quota-holder-pg
  khalil-volcano-quota-waiter-pg
  khalil-volcano-gang-pg
  khalil-volcano-preempt-victim-pg
  khalil-volcano-preempt-high-pg
)

mkdir -p "${RAW_DIR}"
: >"${TRANSCRIPT}"
: >"${TIMELINE}"
exec > >(tee -a "${TRANSCRIPT}") 2>&1

record() {
  local label=$1
  local utc_now monotonic_now
  utc_now=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
  monotonic_now=$(python3 -c 'import time; print(time.monotonic_ns())')
  printf '%s utc=%s monotonic_ns=%s\n' "${label}" "${utc_now}" "${monotonic_now}" | tee -a "${TIMELINE}"
}

write_config_patch() {
  local config_file=$1
  local patch_file=$2
  python3 -c '
import json
import pathlib
import sys
conf = pathlib.Path(sys.argv[1]).read_text()
pathlib.Path(sys.argv[2]).write_text(json.dumps({"data": {"volcano-scheduler.conf": conf}}, indent=2) + "\n")
' "${config_file}" "${patch_file}"
}

write_restore_patch() {
  python3 -c '
import json
import pathlib
import sys
source = json.loads(pathlib.Path(sys.argv[1]).read_text())
pathlib.Path(sys.argv[2]).write_text(json.dumps({"data": source["data"]}, indent=2) + "\n")
' "${ORIGINAL_CONFIG}" "${ORIGINAL_PATCH}"
}

running_gpu_total() {
  local snapshot=$1
  python3 -c '
import json
import sys
data = json.load(open(sys.argv[1]))
total = 0
for pod in data.get("items", []):
    if pod.get("status", {}).get("phase") != "Running":
        continue
    for container in pod.get("spec", {}).get("containers", []):
        value = container.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0)
        total += int(str(value))
print(total)
' "${snapshot}"
}

capture_state() {
  local tag=$1
  record "CAPTURE_${tag}"
  "${KUBECTL[@]}" get jobs -n "${NAMESPACE}" -o json >"${RAW_DIR}/${tag}-jobs.json" 2>/dev/null || true
  "${KUBECTL[@]}" get pods -n "${NAMESPACE}" -o json >"${RAW_DIR}/${tag}-pods.json" 2>/dev/null || true
  "${KUBECTL[@]}" get podgroups.scheduling.volcano.sh -n "${NAMESPACE}" -o json >"${RAW_DIR}/${tag}-podgroups.json" 2>/dev/null || true
  "${KUBECTL[@]}" get queues.scheduling.volcano.sh -l "${LABEL_SELECTOR}" -o json >"${RAW_DIR}/${tag}-queues.json" 2>/dev/null || true
  "${KUBECTL[@]}" get resourcequota -n "${NAMESPACE}" -o json >"${RAW_DIR}/${tag}-resourcequota.json" 2>/dev/null || true
  "${KUBECTL[@]}" get events -n "${NAMESPACE}" -o json >"${RAW_DIR}/${tag}-events.json" 2>/dev/null || true
  "${KUBECTL[@]}" get pods -A -o json >"${RAW_DIR}/${tag}-all-pods.json" 2>/dev/null || true
  "${KUBECTL[@]}" get jobs,pods,podgroups.scheduling.volcano.sh -n "${NAMESPACE}" -o wide 2>/dev/null || true
  "${KUBECTL[@]}" get queues.scheduling.volcano.sh -l "${LABEL_SELECTOR}" \
    -o custom-columns='NAME:.metadata.name,STATE:.status.state,WEIGHT:.spec.weight,CAP_GPU:.spec.capability.nvidia\.com/gpu,DESERVED_GPU:.spec.deserved.nvidia\.com/gpu,ALLOC_GPU:.status.allocated.nvidia\.com/gpu,INQUEUE:.status.inqueue,RUNNING:.status.running' 2>/dev/null || true
  "${KUBECTL[@]}" get events -n "${NAMESPACE}" --sort-by=.metadata.creationTimestamp 2>/dev/null | tail -60 || true
  "${KUBECTL[@]}" logs -n "${NAMESPACE}" -l "${LABEL_SELECTOR}" --all-containers=true --prefix=true \
    >"${RAW_DIR}/${tag}-container-output.txt" 2>&1 || true
}

capture_scheduler() {
  local tag=$1
  "${KUBECTL[@]}" get configmap -n volcano-system volcano-scheduler-configmap -o yaml \
    >"${RAW_DIR}/${tag}-scheduler-config.yaml"
  "${KUBECTL[@]}" get deployment,pod -n volcano-system -l app=volcano-scheduler -o wide \
    >"${RAW_DIR}/${tag}-scheduler-health.txt"
  "${KUBECTL[@]}" logs -n volcano-system deployment/volcano-scheduler --since=10m \
    >"${RAW_DIR}/${tag}-scheduler-output.txt" 2>&1 || true
}

set_scheduler_config() {
  local config_file=$1
  local tag=$2
  local patch_file="${RAW_DIR}/${tag}-scheduler-patch.json"
  write_config_patch "${config_file}" "${patch_file}"
  record "${tag}_CONFIG_PATCH_BEGIN"
  "${KUBECTL[@]}" patch configmap -n volcano-system volcano-scheduler-configmap \
    --type=merge --patch-file "${patch_file}"
  SCHEDULER_CHANGED=1
  "${KUBECTL[@]}" rollout restart deployment/volcano-scheduler -n volcano-system
  "${KUBECTL[@]}" rollout status deployment/volcano-scheduler -n volcano-system --timeout=120s
  record "${tag}_CONFIG_READY"
  capture_scheduler "${tag}"
}

restore_scheduler() {
  if [[ ${SCHEDULER_CHANGED} -ne 1 ]]; then
    return
  fi
  record "SCHEDULER_RESTORE_BEGIN"
  write_restore_patch
  "${KUBECTL[@]}" patch configmap -n volcano-system volcano-scheduler-configmap \
    --type=merge --patch-file "${ORIGINAL_PATCH}"
  "${KUBECTL[@]}" rollout restart deployment/volcano-scheduler -n volcano-system
  "${KUBECTL[@]}" rollout status deployment/volcano-scheduler -n volcano-system --timeout=120s
  SCHEDULER_CHANGED=0
  record "SCHEDULER_RESTORE_READY"
  capture_scheduler "restored"
}

cleanup_exact() {
  set +e
  record "CLEANUP_BEGIN"
  restore_scheduler
  "${KUBECTL[@]}" delete job -n "${NAMESPACE}" "${JOB_NAMES[@]}" --ignore-not-found=true --wait=true --timeout=120s
  "${KUBECTL[@]}" delete podgroup -n "${NAMESPACE}" "${PODGROUP_NAMES[@]}" --ignore-not-found=true --wait=true
  "${KUBECTL[@]}" delete priorityclass "${PRIORITY_NAMES[@]}" --ignore-not-found=true --wait=true
  "${KUBECTL[@]}" delete queue.scheduling.volcano.sh "${QUEUE_NAMES[@]}" --ignore-not-found=true --wait=true
  record "CLEANUP_END"
  set -e
}

trap cleanup_exact EXIT

record "EVALUATION_BEGIN"
echo "ADMIN_KUBECONFIG_PATH=/etc/rancher/k3s/k3s.yaml"
echo "ADMIN_CONTEXT=$("${KUBECTL[@]}" config current-context)"
echo "NAMESPACE=${NAMESPACE}"

if [[ $("${KUBECTL[@]}" config current-context) != "default" ]]; then
  echo "ERROR unexpected administrator context" >&2
  exit 2
fi

for check in \
  'create queues.scheduling.volcano.sh' \
  'create podgroups.scheduling.volcano.sh' \
  'create priorityclasses.scheduling.k8s.io' \
  'create jobs.batch' \
  'patch configmaps' \
  'patch deployments.apps'; do
  verb=${check%% *}
  resource=${check#* }
  printf 'CAN_I %s %s: ' "${verb}" "${resource}"
  "${KUBECTL[@]}" auth can-i "${verb}" "${resource}" --all-namespaces
done

if ! "${KUBECTL[@]}" get namespace "${NAMESPACE}" >/dev/null 2>&1; then
  echo "ERROR required namespace ${NAMESPACE} does not exist" >&2
  exit 2
fi
for queue_name in "${QUEUE_NAMES[@]}"; do
  if "${KUBECTL[@]}" get queue.scheduling.volcano.sh "${queue_name}" >/dev/null 2>&1; then
    echo "ERROR queue ${queue_name} already exists; refusing to reuse it" >&2
    exit 2
  fi
done
for job_name in "${JOB_NAMES[@]}"; do
  if "${KUBECTL[@]}" get job -n "${NAMESPACE}" "${job_name}" >/dev/null 2>&1; then
    echo "ERROR job ${NAMESPACE}/${job_name} already exists; refusing to reuse it" >&2
    exit 2
  fi
done
for podgroup_name in "${PODGROUP_NAMES[@]}"; do
  if "${KUBECTL[@]}" get podgroup -n "${NAMESPACE}" "${podgroup_name}" >/dev/null 2>&1; then
    echo "ERROR podgroup ${NAMESPACE}/${podgroup_name} already exists; refusing to reuse it" >&2
    exit 2
  fi
done
for priority_name in "${PRIORITY_NAMES[@]}"; do
  if "${KUBECTL[@]}" get priorityclass "${priority_name}" >/dev/null 2>&1; then
    echo "ERROR priorityclass ${priority_name} already exists; refusing to reuse it" >&2
    exit 2
  fi
done

"${KUBECTL[@]}" get pods -A -o json >"${RAW_DIR}/evaluation-preflight-all-pods.json"
PREFLIGHT_GPU_TOTAL=$(running_gpu_total "${RAW_DIR}/evaluation-preflight-all-pods.json")
echo "PREFLIGHT_RUNNING_GPU_REQUEST_TOTAL=${PREFLIGHT_GPU_TOTAL}"
if [[ ${PREFLIGHT_GPU_TOTAL} -ne 0 ]]; then
  echo "ERROR running GPU workloads detected; refusing to run a shared-quota preemption test" >&2
  exit 2
fi

"${KUBECTL[@]}" get resourcequota -n "${NAMESPACE}" -o json >"${RAW_DIR}/evaluation-resourcequota-before.json"
read -r QUOTA_GPU_HARD QUOTA_GPU_USED < <(python3 -c '
import json,sys
items=json.load(open(sys.argv[1])).get("items",[])
quota=next(item for item in items if item["metadata"]["name"]=="gpu-quota")
print(quota["status"]["hard"]["requests.nvidia.com/gpu"], quota["status"]["used"].get("requests.nvidia.com/gpu", "0"))
' "${RAW_DIR}/evaluation-resourcequota-before.json")
echo "GPU_DEV_QUOTA_HARD=${QUOTA_GPU_HARD} USED=${QUOTA_GPU_USED}"
if [[ ${QUOTA_GPU_HARD} -ne 4 || ${QUOTA_GPU_USED} -ne 0 ]]; then
  echo "ERROR gpu-dev must have all four shared quota GPUs free before this test" >&2
  exit 2
fi

"${KUBECTL[@]}" get node gpu-dev-01 -o json >"${RAW_DIR}/evaluation-node.json"
NODE_GPU=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"]["allocatable"]["nvidia.com/gpu"])' "${RAW_DIR}/evaluation-node.json")
echo "NODE_GPU_ALLOCATABLE=${NODE_GPU}"
if [[ ${NODE_GPU} -ne 8 ]]; then
  echo "ERROR this pinned test matrix expects exactly 8 allocatable GPUs" >&2
  exit 2
fi

"${KUBECTL[@]}" get configmap -n volcano-system volcano-scheduler-configmap -o json >"${ORIGINAL_CONFIG}"
"${KUBECTL[@]}" get deployments -n volcano-system -o json >"${RAW_DIR}/installation-deployments.json"
"${KUBECTL[@]}" get customresourcedefinitions -o json >"${RAW_DIR}/installation-crds.json"
"${KUBECTL[@]}" get mutatingwebhookconfigurations,validatingwebhookconfigurations -o json >"${RAW_DIR}/installation-webhooks.json"
"${KUBECTL[@]}" get endpointslices -n volcano-system -o json >"${RAW_DIR}/installation-endpointslices.json"
"${KUBECTL[@]}" get deployment -n kueue-system kueue-controller-manager -o json >"${RAW_DIR}/installation-kueue-health.json"
sha256sum "${SCRIPT_DIR}/volcano/upstream/volcano-development-v1.15.1.yaml"
capture_scheduler "installed-default"

record "EMPTY_QUEUE_PREREQUISITES_BEGIN"
"${KUBECTL[@]}" apply --dry-run=server -f "${BENCH_DIR}/00-queues.yaml"
"${KUBECTL[@]}" apply -f "${BENCH_DIR}/00-queues.yaml"
"${KUBECTL[@]}" apply --dry-run=server -f "${BENCH_DIR}/06-priorityclasses.yaml"
"${KUBECTL[@]}" apply -f "${BENCH_DIR}/06-priorityclasses.yaml"
record "EMPTY_QUEUE_PREREQUISITES_READY"

: >"${RAW_DIR}/evaluation-server-dry-run.txt"
for manifest in "${BENCH_DIR}"/[0-9][0-9]-*.yaml; do
  [[ -f ${manifest} ]] || continue
  printf 'DRY_RUN %s\n' "$(basename "${manifest}")" | tee -a "${RAW_DIR}/evaluation-server-dry-run.txt"
  "${KUBECTL[@]}" apply --dry-run=server -f "${manifest}" | tee -a "${RAW_DIR}/evaluation-server-dry-run.txt"
done
record "SERVER_DRY_RUN_COMPLETE"

echo "SCENARIO_SMOKE_BEGIN"
record "SMOKE_CREATE_BEGIN"
"${KUBECTL[@]}" apply -f "${BENCH_DIR}/01-smoke.yaml"
record "SMOKE_CREATE_RETURN"
"${KUBECTL[@]}" wait -n "${NAMESPACE}" --for=condition=complete job/khalil-volcano-smoke --timeout=120s
record "SMOKE_COMPLETE_OBSERVED"
capture_state "smoke-final"
SMOKE_VISIBLE=$(grep -c 'VISIBLE_GPU_COUNT=1' "${RAW_DIR}/smoke-final-container-output.txt" || true)
SMOKE_SCHEDULER=$(python3 -c '
import json,sys
pods=json.load(open(sys.argv[1])).get("items",[])
matches=[p for p in pods if p.get("metadata",{}).get("labels",{}).get("evaluation.quantfm/scenario")=="smoke"]
print(matches[0]["spec"].get("schedulerName", "") if matches else "")
' "${RAW_DIR}/smoke-final-pods.json")
echo "SMOKE_VISIBLE_GPU_ASSERTIONS=${SMOKE_VISIBLE}"
echo "SMOKE_POD_SCHEDULER=${SMOKE_SCHEDULER}"
if [[ ${SMOKE_VISIBLE} -lt 1 || ${SMOKE_SCHEDULER} != "volcano" ]]; then
  echo "ERROR Volcano GPU smoke assertion failed" >&2
  exit 3
fi
"${KUBECTL[@]}" delete job -n "${NAMESPACE}" khalil-volcano-smoke --wait=true
"${KUBECTL[@]}" delete podgroup -n "${NAMESPACE}" khalil-volcano-smoke-pg --ignore-not-found=true
"${KUBECTL[@]}" delete queue khalil-volcano-smoke --ignore-not-found=true
echo "SCENARIO_SMOKE_END"

echo "SCENARIO_QUEUE_CAPABILITY_BEGIN"
"${KUBECTL[@]}" apply -f "${BENCH_DIR}/02-queue-base.yaml"
sleep 2
record "QUEUE_HOLDER_CREATE_BEGIN"
"${KUBECTL[@]}" apply -f "${BENCH_DIR}/03-queue-holder.yaml"
record "QUEUE_HOLDER_CREATE_RETURN"
"${KUBECTL[@]}" wait -n "${NAMESPACE}" --for=condition=Ready pod -l evaluation.quantfm/scenario=queue-holder --timeout=120s
record "QUEUE_HOLDER_READY_OBSERVED"
record "QUEUE_WAITER_CREATE_BEGIN"
"${KUBECTL[@]}" apply -f "${BENCH_DIR}/04-queue-waiter.yaml"
record "QUEUE_WAITER_CREATE_RETURN"
sleep 5
capture_state "queue-during"
QUEUE_WAITER_BOUND=$(python3 -c '
import json,sys
pods=json.load(open(sys.argv[1])).get("items",[])
matches=[p for p in pods if p.get("metadata",{}).get("labels",{}).get("evaluation.quantfm/scenario")=="queue-waiter"]
print(sum(1 for p in matches if p.get("spec",{}).get("nodeName")))
' "${RAW_DIR}/queue-during-pods.json")
QUEUE_DURING_GPU_TOTAL=$(running_gpu_total "${RAW_DIR}/queue-during-all-pods.json")
QUEUE_DURING_PHYSICAL_FREE=$((NODE_GPU - QUEUE_DURING_GPU_TOTAL))
echo "QUEUE_DURING_WAITER_BOUND=${QUEUE_WAITER_BOUND}"
echo "QUEUE_DURING_RUNNING_GPU_REQUEST_TOTAL=${QUEUE_DURING_GPU_TOTAL}"
echo "QUEUE_DURING_PHYSICAL_FREE_GPU=${QUEUE_DURING_PHYSICAL_FREE}"
if [[ ${QUEUE_WAITER_BOUND} -ne 0 || ${QUEUE_DURING_PHYSICAL_FREE} -lt 1 ]]; then
  echo "ERROR Queue capability isolation assertion failed" >&2
  exit 3
fi
"${KUBECTL[@]}" wait -n "${NAMESPACE}" --for=condition=complete job/khalil-volcano-quota-holder --timeout=120s
record "QUEUE_HOLDER_COMPLETE_OBSERVED"
"${KUBECTL[@]}" wait -n "${NAMESPACE}" --for=condition=complete job/khalil-volcano-quota-waiter --timeout=120s
record "QUEUE_WAITER_COMPLETE_OBSERVED"
capture_state "queue-final"
"${KUBECTL[@]}" delete job -n "${NAMESPACE}" khalil-volcano-quota-holder khalil-volcano-quota-waiter --wait=true
"${KUBECTL[@]}" delete podgroup -n "${NAMESPACE}" khalil-volcano-quota-holder-pg khalil-volcano-quota-waiter-pg --ignore-not-found=true
"${KUBECTL[@]}" delete queue khalil-volcano-quota --ignore-not-found=true
echo "SCENARIO_QUEUE_CAPABILITY_END"

echo "SCENARIO_GANG_BEGIN"
record "GANG_CREATE_BEGIN"
"${KUBECTL[@]}" apply -f "${BENCH_DIR}/05-gang.yaml"
record "GANG_CREATE_RETURN"
sleep 5
capture_state "gang-below-threshold"
GANG_BOUND_BELOW=$(python3 -c '
import json,sys
pods=json.load(open(sys.argv[1])).get("items",[])
matches=[p for p in pods if p.get("metadata",{}).get("labels",{}).get("evaluation.quantfm/scenario")=="gang"]
print(sum(1 for p in matches if p.get("spec",{}).get("nodeName")))
' "${RAW_DIR}/gang-below-threshold-pods.json")
echo "GANG_BOUND_BELOW_THRESHOLD=${GANG_BOUND_BELOW}"
if [[ ${GANG_BOUND_BELOW} -ne 0 ]]; then
  echo "ERROR Gang minMember assertion failed below threshold" >&2
  exit 3
fi
record "GANG_CAPABILITY_PATCH_BEGIN"
"${KUBECTL[@]}" patch queue khalil-volcano-gang --type=merge \
  -p '{"spec":{"capability":{"cpu":"2","memory":"4Gi","nvidia.com/gpu":"2"}}}'
record "GANG_CAPABILITY_PATCH_RETURN"
"${KUBECTL[@]}" wait -n "${NAMESPACE}" --for=condition=complete job/khalil-volcano-gang --timeout=120s
record "GANG_COMPLETE_OBSERVED"
capture_state "gang-final"
"${KUBECTL[@]}" delete job -n "${NAMESPACE}" khalil-volcano-gang --wait=true
"${KUBECTL[@]}" delete podgroup -n "${NAMESPACE}" khalil-volcano-gang-pg --ignore-not-found=true
"${KUBECTL[@]}" delete queue khalil-volcano-gang --ignore-not-found=true
echo "SCENARIO_GANG_END"

echo "SCENARIO_PREEMPT_BEGIN"
"${KUBECTL[@]}" apply -f "${BENCH_DIR}/06-priorityclasses.yaml"
set_scheduler_config "${BENCH_DIR}/scheduler-preempt.conf" "preempt"
"${KUBECTL[@]}" apply -f "${BENCH_DIR}/07-preempt-base.yaml"
sleep 2
record "PREEMPT_VICTIMS_CREATE_BEGIN"
"${KUBECTL[@]}" apply -f "${BENCH_DIR}/08-preempt-victim.yaml"
record "PREEMPT_VICTIMS_CREATE_RETURN"
"${KUBECTL[@]}" wait -n "${NAMESPACE}" --for=condition=Ready pod -l evaluation.quantfm/scenario=preempt-victim --timeout=120s
record "PREEMPT_VICTIMS_READY_OBSERVED"
capture_state "preempt-before"
record "PREEMPT_HIGH_CREATE_BEGIN"
"${KUBECTL[@]}" apply -f "${BENCH_DIR}/10-preempt-high.yaml"
record "PREEMPT_HIGH_CREATE_RETURN"
PREEMPT_EFFECTIVE=false
if "${KUBECTL[@]}" wait -n "${NAMESPACE}" --for=condition=Ready pod -l evaluation.quantfm/scenario=preempt-high --timeout=60s; then
  record "PREEMPT_HIGH_READY_OBSERVED"
  PREEMPT_EFFECTIVE=true
  "${KUBECTL[@]}" wait -n "${NAMESPACE}" --for=condition=complete job/khalil-volcano-preempt-high --timeout=120s
  record "PREEMPT_HIGH_COMPLETE_OBSERVED"
else
  record "PREEMPT_HIGH_READY_TIMEOUT"
fi
capture_state "preempt-after"
capture_scheduler "preempt-after"
echo "PREEMPT_EFFECTIVE=${PREEMPT_EFFECTIVE}"
"${KUBECTL[@]}" delete job -n "${NAMESPACE}" khalil-volcano-preempt-victim khalil-volcano-preempt-high --ignore-not-found=true --wait=true
"${KUBECTL[@]}" delete podgroup -n "${NAMESPACE}" khalil-volcano-preempt-victim-pg khalil-volcano-preempt-high-pg --ignore-not-found=true
"${KUBECTL[@]}" delete queue khalil-volcano-preempt --ignore-not-found=true
echo "SCENARIO_PREEMPT_END"

: >"${RAW_DIR}/cleanup-jobs-podgroups-absent.txt"
: >"${RAW_DIR}/cleanup-queues-absent.txt"
: >"${RAW_DIR}/cleanup-priorityclasses-absent.txt"
cleanup_exact
for job_name in "${JOB_NAMES[@]}"; do
  "${KUBECTL[@]}" get job -n "${NAMESPACE}" "${job_name}" >>"${RAW_DIR}/cleanup-jobs-podgroups-absent.txt" 2>&1 || true
done
for podgroup_name in "${PODGROUP_NAMES[@]}"; do
  "${KUBECTL[@]}" get podgroup -n "${NAMESPACE}" "${podgroup_name}" >>"${RAW_DIR}/cleanup-jobs-podgroups-absent.txt" 2>&1 || true
done
for queue_name in "${QUEUE_NAMES[@]}"; do
  "${KUBECTL[@]}" get queue "${queue_name}" >>"${RAW_DIR}/cleanup-queues-absent.txt" 2>&1 || true
done
for priority_name in "${PRIORITY_NAMES[@]}"; do
  "${KUBECTL[@]}" get priorityclass "${priority_name}" >>"${RAW_DIR}/cleanup-priorityclasses-absent.txt" 2>&1 || true
done
"${KUBECTL[@]}" get resourcequota -n "${NAMESPACE}" -o json >"${RAW_DIR}/evaluation-resourcequota-after.json"
"${KUBECTL[@]}" get deployments -n volcano-system -o wide >"${RAW_DIR}/cleanup-volcano-health.txt"
"${KUBECTL[@]}" get deployment -n kueue-system kueue-controller-manager -o wide >"${RAW_DIR}/cleanup-kueue-health.txt"
"${KUBECTL[@]}" get configmap -n volcano-system volcano-scheduler-configmap -o yaml >"${RAW_DIR}/cleanup-scheduler-config.yaml"
record "EVALUATION_COMPLETE"
trap - EXIT
echo "RESULT smoke=true queue_capability=true gang=true preempt=${PREEMPT_EFFECTIVE} reclaim=not_tested_shared_quota_safety cleanup=true config_restored=true"
