#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./k8s/gpu-dev/submit-dense-v2-230m-4gpu.sh --render-only IMAGE_DIGEST JOB_NAME

  QUANTFM_PVC_CONFIRMED='gpu-dev/quantfm-data@dcfc08ba-7c57-466e-abeb-e60897855a39' \
    ./k8s/gpu-dev/submit-dense-v2-230m-4gpu.sh IMAGE_DIGEST JOB_NAME

IMAGE_DIGEST must be the complete immutable reference:
  registry.zs/gpu-dev/khalil-quantfm@sha256:<64 lowercase hex characters>

Set QUANTFM_SERVER_DRY_RUN_ONLY=1 to stop after all live gates and the
server-side dry-run.  --render-only performs no cluster or GPU access and emits
the still-suspended manifest to stdout.
EOF
}

mode="submit"
if [[ "${1:-}" == "--render-only" ]]; then
  mode="render-only"
  shift
fi
if (( $# != 2 )); then
  usage >&2
  exit 64
fi

image="$1"
job_name="$2"
if [[ ! "$image" =~ ^registry[.]zs/gpu-dev/khalil-quantfm@sha256:[0-9a-f]{64}$ ]]; then
  echo "IMAGE_DIGEST must be a full internal khalil-quantfm sha256 digest." >&2
  exit 64
fi
if (( ${#job_name} > 63 )) || \
   [[ ! "$job_name" =~ ^khalil-[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
  echo "JOB_NAME must be a <=63 character DNS label beginning with khalil-." >&2
  exit 64
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
template="${script_dir}/khalil-dense-v2-230m-4gpu-job.template.yaml"
kubeconfig_path="/etc/rancher/k3s/k3s.yaml"
context_name="default"
namespace="gpu-dev"
quota_name="gpu-quota"
pvc_name="quantfm-data"
expected_pvc_uid="dcfc08ba-7c57-466e-abeb-e60897855a39"
expected_pv_name="pvc-792a789f-ebf3-4a57-9c32-58f23c0c9580"
expected_pv_uid="8d66d1f8-7a0a-48ff-ae7b-e49a4e3e59fa"
expected_local_path="/data/k3s/storage/pvc-792a789f-ebf3-4a57-9c32-58f23c0c9580_khalil_quantfm-data"
expected_node="gpu-dev-01"
expected_confirmation="${namespace}/${pvc_name}@${expected_pvc_uid}"

rendered_suspended="$(mktemp /tmp/khalil-dense-v2-4gpu-suspended.XXXXXX.yaml)"
rendered_active="$(mktemp /tmp/khalil-dense-v2-4gpu-active.XXXXXX.yaml)"
quota_json="$(mktemp /tmp/khalil-dense-v2-4gpu-quota.XXXXXX.json)"
pvc_json="$(mktemp /tmp/khalil-dense-v2-4gpu-pvc.XXXXXX.json)"
pv_json="$(mktemp /tmp/khalil-dense-v2-4gpu-pv.XXXXXX.json)"
pods_json="$(mktemp /tmp/khalil-dense-v2-4gpu-pods.XXXXXX.json)"
node_json="$(mktemp /tmp/khalil-dense-v2-4gpu-node.XXXXXX.json)"
cleanup() {
  rm -f -- \
    "$rendered_suspended" "$rendered_active" \
    "$quota_json" "$pvc_json" "$pv_json" "$pods_json" "$node_json"
}
trap cleanup EXIT

sed \
  -e "s|REPLACE_IMAGE|${image}|g" \
  -e "s|khalil-dense-v2-230m-4gpu-template|${job_name}|g" \
  "$template" > "$rendered_suspended"

if rg -q 'REPLACE_|hostPath:|^[[:space:]]*schedulerName:' "$rendered_suspended"; then
  echo "Rendered Job contains a placeholder or forbidden scheduling/storage field." >&2
  exit 65
fi
if ! rg -Fq "image: ${image}" "$rendered_suspended" || \
   ! rg -q '^[[:space:]]*claimName: quantfm-data$' "$rendered_suspended" || \
   ! rg -q '^[[:space:]]*suspend: true$' "$rendered_suspended"; then
  echo "Rendered suspended Job violates its image, PVC, or suspension contract." >&2
  exit 65
fi

if [[ "$mode" == "render-only" ]]; then
  command cat "$rendered_suspended"
  exit 0
fi

if [[ "${QUANTFM_PVC_CONFIRMED:-}" != "$expected_confirmation" ]]; then
  echo "Refusing submission without the exact administrator-confirmed PVC identity." >&2
  echo "Expected QUANTFM_PVC_CONFIRMED=${expected_confirmation}" >&2
  exit 65
fi
for executable in kubectl python3 nvidia-smi; do
  if ! command -v "$executable" >/dev/null 2>&1; then
    echo "Required executable is unavailable: $executable" >&2
    exit 69
  fi
done
if [[ ! -r "$kubeconfig_path" ]]; then
  echo "Missing restricted kubeconfig: $kubeconfig_path" >&2
  exit 66
fi

# Only the actual submit path removes suspension.  The checked-in template and
# --render-only output remain inert.
sed '/^[[:space:]]*suspend: true$/d' "$rendered_suspended" > "$rendered_active"
if rg -q '^[[:space:]]*suspend:' "$rendered_active"; then
  echo "Active render unexpectedly retains a suspend field." >&2
  exit 65
fi

kubectl_base=(
  kubectl
  --kubeconfig "$kubeconfig_path"
  --context "$context_name"
)

require_permission() {
  local verb="$1"
  local resource="$2"
  shift 2
  local answer
  answer="$("${kubectl_base[@]}" auth can-i "$verb" "$resource" "$@")"
  if [[ "$answer" != "yes" ]]; then
    echo "RBAC gate failed: cannot $verb $resource $*" >&2
    exit 77
  fi
  echo "auth_can_i ${verb} ${resource} $*=yes"
}

require_permission create jobs.batch --namespace "$namespace"
require_permission get jobs.batch --namespace "$namespace"
require_permission list jobs.batch --namespace "$namespace"
require_permission watch jobs.batch --namespace "$namespace"
require_permission delete jobs.batch --namespace "$namespace"
require_permission get resourcequotas --namespace "$namespace"
require_permission get pods --namespace "$namespace"
require_permission list pods --namespace "$namespace"
require_permission get pods/log --namespace "$namespace"
require_permission list events --namespace "$namespace"
require_permission get persistentvolumeclaims --namespace "$namespace"
require_permission get persistentvolumes
require_permission get nodes

host_gpu_snapshot() {
  local processes
  processes="$(
    nvidia-smi \
      --query-compute-apps=gpu_uuid,used_gpu_memory \
      --format=csv,noheader,nounits
  )"
  if [[ -n "${processes//[[:space:]]/}" ]]; then
    echo "Host GPU compute processes are active; refusing overlap:" >&2
    printf '%s\n' "$processes" >&2
    return 1
  fi
  printf '%s\n' '{"active_compute_processes":[]}'
}

cluster_snapshot() {
  "${kubectl_base[@]}" --namespace "$namespace" \
    get resourcequota "$quota_name" -o json > "$quota_json"
  "${kubectl_base[@]}" --namespace "$namespace" \
    get persistentvolumeclaim "$pvc_name" -o json > "$pvc_json"
  "${kubectl_base[@]}" \
    get persistentvolume "$expected_pv_name" -o json > "$pv_json"
  "${kubectl_base[@]}" --namespace "$namespace" \
    get pods -o json > "$pods_json"
  "${kubectl_base[@]}" \
    get node "$expected_node" -o json > "$node_json"

  python3 - \
    "$quota_json" "$pvc_json" "$pv_json" "$pods_json" "$node_json" \
    "$namespace" "$pvc_name" "$expected_pvc_uid" \
    "$expected_pv_name" "$expected_pv_uid" "$expected_local_path" \
    "$expected_node" <<'PY'
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

(
    quota_path,
    pvc_path,
    pv_path,
    pods_path,
    node_path,
    namespace,
    pvc_name,
    expected_pvc_uid,
    expected_pv_name,
    expected_pv_uid,
    expected_local_path,
    expected_node,
) = sys.argv[1:]

quota = json.loads(Path(quota_path).read_text())
pvc = json.loads(Path(pvc_path).read_text())
pv = json.loads(Path(pv_path).read_text())
pods = json.loads(Path(pods_path).read_text())
node = json.loads(Path(node_path).read_text())


def gpu_quantity(value: object, *, field: str) -> int:
    if value is None:
        return 0
    try:
        parsed = int(str(value))
    except ValueError as exc:
        raise SystemExit(f"non-integral GPU quantity for {field}: {value!r}") from exc
    if parsed < 0:
        raise SystemExit(f"negative GPU quantity for {field}: {parsed}")
    return parsed


def cpu_millis(value: object, *, field: str) -> int:
    text = str(value)
    try:
        result = Decimal(text[:-1]) if text.endswith("m") else Decimal(text) * 1000
    except InvalidOperation as exc:
        raise SystemExit(f"invalid CPU quantity for {field}: {value!r}") from exc
    if result < 0 or result != result.to_integral_value():
        raise SystemExit(f"unsupported CPU quantity for {field}: {value!r}")
    return int(result)


def memory_bytes(value: object, *, field: str) -> int:
    text = str(value)
    factors = {
        "Ki": 2**10,
        "Mi": 2**20,
        "Gi": 2**30,
        "Ti": 2**40,
        "K": 10**3,
        "M": 10**6,
        "G": 10**9,
        "T": 10**12,
    }
    suffix = next((item for item in factors if text.endswith(item)), "")
    number = text[: -len(suffix)] if suffix else text
    try:
        result = Decimal(number) * factors.get(suffix, 1)
    except InvalidOperation as exc:
        raise SystemExit(f"invalid memory quantity for {field}: {value!r}") from exc
    if result < 0 or result != result.to_integral_value():
        raise SystemExit(f"unsupported memory quantity for {field}: {value!r}")
    return int(result)


hard = quota.get("status", {}).get("hard", {})
used = quota.get("status", {}).get("used", {})
hard_requests = gpu_quantity(
    hard.get("requests.nvidia.com/gpu"), field="quota hard requests"
)
hard_limits = gpu_quantity(
    hard.get("limits.nvidia.com/gpu"), field="quota hard limits"
)
used_requests = gpu_quantity(
    used.get("requests.nvidia.com/gpu"), field="quota used requests"
)
used_limits = gpu_quantity(
    used.get("limits.nvidia.com/gpu"), field="quota used limits"
)
if hard_requests < 4 or hard_limits < 4:
    raise SystemExit(
        f"gpu-dev quota cannot admit four GPUs: requests={hard_requests}, "
        f"limits={hard_limits}"
    )
if used_requests != 0 or used_limits != 0:
    raise SystemExit(
        f"gpu-dev GPU quota is already used: requests={used_requests}, "
        f"limits={used_limits}"
    )

quota_requirements = {
    "requests.cpu": (cpu_millis, 32_000),
    "limits.cpu": (cpu_millis, 48_000),
    "requests.memory": (memory_bytes, 64 * 2**30),
    "limits.memory": (memory_bytes, 128 * 2**30),
}
for resource, (parser, required) in quota_requirements.items():
    if resource not in hard:
        raise SystemExit(f"gpu-dev quota is missing required hard field: {resource}")
    hard_value = parser(hard[resource], field=f"quota hard {resource}")
    used_value = parser(used.get(resource, "0"), field=f"quota used {resource}")
    if hard_value - used_value < required:
        raise SystemExit(
            f"gpu-dev quota cannot admit Job {resource}: "
            f"available={hard_value - used_value}, required={required}"
        )

node_meta = node.get("metadata", {})
node_spec = node.get("spec", {})
node_status = node.get("status", {})
if node_meta.get("name") != expected_node:
    raise SystemExit("live GPU node identity changed")
ready_conditions = [
    condition
    for condition in node_status.get("conditions", [])
    if condition.get("type") == "Ready"
]
if len(ready_conditions) != 1 or ready_conditions[0].get("status") != "True":
    raise SystemExit(f"{expected_node} is not Ready")
if node_spec.get("unschedulable") is True:
    raise SystemExit(f"{expected_node} is unschedulable")
allocatable_gpu = gpu_quantity(
    node_status.get("allocatable", {}).get("nvidia.com/gpu"),
    field=f"node/{expected_node} allocatable GPU",
)
if allocatable_gpu < 4:
    raise SystemExit(
        f"{expected_node} has fewer than four allocatable GPUs: {allocatable_gpu}"
    )
labels = node_meta.get("labels", {})
if labels.get("accelerator") != "nvidia":
    raise SystemExit(f"{expected_node} lost accelerator=nvidia")
if labels.get("kubernetes.io/hostname") != expected_node:
    raise SystemExit(f"{expected_node} hostname label changed")

pvc_meta = pvc.get("metadata", {})
pvc_spec = pvc.get("spec", {})
if pvc_meta.get("namespace") != namespace or pvc_meta.get("name") != pvc_name:
    raise SystemExit("live PVC identity does not match gpu-dev/quantfm-data")
if pvc_meta.get("uid") != expected_pvc_uid:
    raise SystemExit(
        f"live PVC UID changed: {pvc_meta.get('uid')!r} != {expected_pvc_uid!r}"
    )
if pvc.get("status", {}).get("phase") != "Bound":
    raise SystemExit("gpu-dev/quantfm-data is not Bound")
if pvc_spec.get("volumeName") != expected_pv_name:
    raise SystemExit("gpu-dev/quantfm-data is bound to an unexpected PV")
if pvc_spec.get("storageClassName") != "local-path":
    raise SystemExit("gpu-dev/quantfm-data storageClass changed")
if pvc_spec.get("volumeMode", "Filesystem") != "Filesystem":
    raise SystemExit("gpu-dev/quantfm-data is not a filesystem volume")
if "ReadWriteOnce" not in pvc_spec.get("accessModes", []):
    raise SystemExit("gpu-dev/quantfm-data lost its ReadWriteOnce contract")
minimum_capacity = 500 * 2**30
for field, value in (
    (
        "PVC requested capacity",
        pvc_spec.get("resources", {}).get("requests", {}).get("storage"),
    ),
    ("PVC bound capacity", pvc.get("status", {}).get("capacity", {}).get("storage")),
):
    if value is None or memory_bytes(value, field=field) < minimum_capacity:
        raise SystemExit(f"{field} is below 500Gi: {value!r}")

pv_meta = pv.get("metadata", {})
pv_spec = pv.get("spec", {})
claim_ref = pv_spec.get("claimRef", {})
if pv_meta.get("name") != expected_pv_name or pv_meta.get("uid") != expected_pv_uid:
    raise SystemExit("live PV name/UID changed from the confirmed non-root backend")
if pv.get("status", {}).get("phase") != "Bound":
    raise SystemExit("confirmed QuantFM PV is not Bound")
if pv_spec.get("local", {}).get("path") != expected_local_path:
    raise SystemExit("confirmed QuantFM PV local path changed")
if (
    claim_ref.get("namespace") != namespace
    or claim_ref.get("name") != pvc_name
    or claim_ref.get("uid") != expected_pvc_uid
):
    raise SystemExit("confirmed QuantFM PV claimRef changed")
if pv_spec.get("persistentVolumeReclaimPolicy") != "Retain":
    raise SystemExit("confirmed QuantFM PV no longer has Retain policy")
pv_capacity = pv_spec.get("capacity", {}).get("storage")
if pv_capacity is None or memory_bytes(pv_capacity, field="PV capacity") < minimum_capacity:
    raise SystemExit(f"confirmed QuantFM PV capacity is below 500Gi: {pv_capacity!r}")
expected_node_affinity = {
    "required": {
        "nodeSelectorTerms": [
            {
                "matchExpressions": [
                    {
                        "key": "kubernetes.io/hostname",
                        "operator": "In",
                        "values": [expected_node],
                    }
                ]
            }
        ]
    }
}
if pv_spec.get("nodeAffinity") != expected_node_affinity:
    raise SystemExit(
        f"confirmed QuantFM PV nodeAffinity is not exactly bound to {expected_node}"
    )

active_gpu_pods: list[str] = []
for pod in pods.get("items", []):
    phase = pod.get("status", {}).get("phase")
    if phase in {"Succeeded", "Failed"}:
        continue
    gpu = 0
    spec = pod.get("spec", {})
    for container_kind in ("initContainers", "containers", "ephemeralContainers"):
        for container in spec.get(container_kind, []) or []:
            resources = container.get("resources", {})
            for contract in ("requests", "limits"):
                gpu = max(
                    gpu,
                    gpu_quantity(
                        resources.get(contract, {}).get("nvidia.com/gpu"),
                        field=(
                            f"pod/{pod.get('metadata', {}).get('name')} "
                            f"{container_kind}.{contract}"
                        ),
                    ),
                )
    if gpu:
        active_gpu_pods.append(
            f"{pod.get('metadata', {}).get('name')}:{phase}:{gpu}"
        )
if active_gpu_pods:
    raise SystemExit(
        "unrelated active GPU Pod requests exist: " + ", ".join(active_gpu_pods)
    )

summary = {
    "active_gpu_pods": [],
    "node": {
        "allocatable_gpu": allocatable_gpu,
        "name": expected_node,
        "ready": True,
        "resource_version": node_meta.get("resourceVersion"),
    },
    "pv": {
        "name": expected_pv_name,
        "resource_version": pv_meta.get("resourceVersion"),
        "uid": expected_pv_uid,
    },
    "pvc": {
        "name": f"{namespace}/{pvc_name}",
        "resource_version": pvc_meta.get("resourceVersion"),
        "uid": expected_pvc_uid,
    },
    "quota": {
        "hard": hard,
        "hard_limits_gpu": hard_limits,
        "hard_requests_gpu": hard_requests,
        "resource_version": quota.get("metadata", {}).get("resourceVersion"),
        "used": used,
        "used_limits_gpu": used_limits,
        "used_requests_gpu": used_requests,
    },
}
print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
PY
}

gpu_before="$(host_gpu_snapshot)"
state_before="$(cluster_snapshot)"
echo "host_gpu_before=${gpu_before}"
echo "cluster_before=${state_before}"

"${kubectl_base[@]}" --namespace "$namespace" \
  create --dry-run=server -f "$rendered_active" -o name

# Close the preflight/create race as far as a client-side submitter can: repeat
# both mutable gates and require the exact quota/PVC/PV snapshot to be stable.
gpu_after_dry_run="$(host_gpu_snapshot)"
state_after_dry_run="$(cluster_snapshot)"
echo "host_gpu_after_dry_run=${gpu_after_dry_run}"
echo "cluster_after_dry_run=${state_after_dry_run}"
if [[ "$gpu_before" != "$gpu_after_dry_run" ]] || \
   [[ "$state_before" != "$state_after_dry_run" ]]; then
  echo "Cluster/GPU baseline changed during preflight; refusing creation." >&2
  exit 75
fi

if [[ "${QUANTFM_SERVER_DRY_RUN_ONLY:-0}" == "1" ]]; then
  echo "server_dry_run=PASS; no Job was created"
  exit 0
fi

created_resource="$(
  "${kubectl_base[@]}" --namespace "$namespace" \
    create -f "$rendered_active" -o name
)"
if [[ "$created_resource" != "job.batch/${job_name}" ]]; then
  echo "Unexpected created resource: $created_resource" >&2
  exit 70
fi

echo "submitted=${created_resource}"
echo "image=${image}"
echo "logs: kubectl --kubeconfig ${kubeconfig_path} --context ${context_name} -n ${namespace} logs -f job/${job_name}"
echo "events: kubectl --kubeconfig ${kubeconfig_path} --context ${context_name} -n ${namespace} describe job/${job_name}"
echo "cleanup (after evidence capture): kubectl --kubeconfig ${kubeconfig_path} --context ${context_name} -n ${namespace} delete job/${job_name}"
