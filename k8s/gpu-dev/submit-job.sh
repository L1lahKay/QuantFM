#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  QUANTFM_PVC_CONFIRMED=gpu-dev/quantfm-data \
    ./k8s/gpu-dev/submit-job.sh cpu-smoke IMAGE_TAG

  QUANTFM_PVC_CONFIRMED=gpu-dev/quantfm-data \
    ./k8s/gpu-dev/submit-job.sh gpu-smoke IMAGE_TAG

  QUANTFM_PVC_CONFIRMED=gpu-dev/quantfm-data \
    ./k8s/gpu-dev/submit-job.sh train-smoke IMAGE_TAG

  QUANTFM_PVC_CONFIRMED=gpu-dev/quantfm-data \
    ./k8s/gpu-dev/submit-job.sh train IMAGE_TAG JOB_NAME CONFIG

Set QUANTFM_SERVER_DRY_RUN_ONLY=1 to stop after the server-side dry run.
EOF
}

if (( $# < 2 )); then
  usage >&2
  exit 64
fi

mode="$1"
image_tag="$2"
shift 2

if [[ "${QUANTFM_PVC_CONFIRMED:-}" != "gpu-dev/quantfm-data" ]]; then
  echo "Refusing submission: administrator confirmation is required." >&2
  echo "After gpu-dev/quantfm-data is confirmed Bound to the retained data-disk PV, run:" >&2
  echo "  export QUANTFM_PVC_CONFIRMED=gpu-dev/quantfm-data" >&2
  exit 65
fi

if [[ ! "${image_tag}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "Invalid container image tag: ${image_tag}" >&2
  exit 64
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
kubeconfig_path="/etc/rancher/k3s/k3s.yaml"
context_name="default"
namespace="gpu-dev"

case "${mode}" in
  cpu-smoke)
    template="${script_dir}/khalil-cpu-pvc-smoke.yaml"
    ;;
  gpu-smoke)
    template="${script_dir}/khalil-gpu-pvc-smoke.yaml"
    ;;
  train-smoke)
    template="${script_dir}/khalil-training-pvc-smoke.yaml"
    ;;
  train)
    if (( $# != 2 )); then
      usage >&2
      exit 64
    fi
    job_name="$1"
    config_path="$2"
    if (( ${#job_name} > 63 )) || \
       [[ ! "${job_name}" =~ ^khalil-[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
      echo "JOB_NAME must be a <=63 character DNS label beginning with khalil-." >&2
      exit 64
    fi
    if [[ "${config_path}" == *".."* ]] || \
       [[ ! "${config_path}" =~ ^quant_fm/pretrain/[A-Za-z0-9_.-]+\.ya?ml$ ]] || \
       [[ ! -f "${repo_root}/${config_path}" ]]; then
      echo "CONFIG must name an existing YAML file directly below quant_fm/pretrain/." >&2
      exit 64
    fi
    storage_path_pattern='^[[:space:]]+(manifest|vocab|validation_plan|train_dates_file|validation_dates_file|test_dates_file|out_dir):'
    if ! rg -q '^[[:space:]]+out_dir:[[:space:]]+(quant_fm/runs/|/mnt/quantfm/)' \
         "${repo_root}/${config_path}" || \
       rg "${storage_path_pattern}" "${repo_root}/${config_path}" \
         | rg -qv ':[[:space:]]+(quant_fm/runs/|/mnt/quantfm/)'; then
      echo "CONFIG contains a training data/output path outside the PVC contract." >&2
      echo "Use quant_fm/runs/... or /mnt/quantfm/... for every storage path." >&2
      exit 65
    fi
    template="${script_dir}/khalil-training-job.template.yaml"
    ;;
  *)
    usage >&2
    exit 64
    ;;
esac

if [[ ! -r "${kubeconfig_path}" ]]; then
  echo "Missing restricted kubeconfig: ${kubeconfig_path}" >&2
  exit 66
fi

rendered="$(mktemp /tmp/khalil-gpu-job.XXXXXX.yaml)"
trap 'rm -f "${rendered}"' EXIT

sed "s/REPLACE_TAG/${image_tag}/g" "${template}" > "${rendered}"

if [[ "${mode}" == train ]]; then
  sed -i \
    -e "s/khalil-quantfm-training-template/${job_name}/" \
    -e "s|REPLACE_CONFIG|${config_path}|" \
    -e '/^[[:space:]]*suspend: true$/d' \
    "${rendered}"
fi

if rg -q 'REPLACE_|hostPath:|/etc/rancher/k3s/k3s.yaml' "${rendered}"; then
  echo "Rendered Job contains a placeholder or forbidden storage/config reference." >&2
  rg -n 'REPLACE_|hostPath:|/etc/rancher/k3s/k3s.yaml' "${rendered}" >&2
  exit 65
fi

if ! rg -q '^[[:space:]]*image: registry\.zs/gpu-dev/' "${rendered}" || \
   ! rg -q '^[[:space:]]*claimName: quantfm-data$' "${rendered}"; then
  echo "Rendered Job violates the image or PVC contract." >&2
  exit 65
fi

kubectl \
  --kubeconfig "${kubeconfig_path}" \
  --context "${context_name}" \
  --namespace "${namespace}" \
  create --dry-run=server -f "${rendered}" -o name

if [[ "${QUANTFM_SERVER_DRY_RUN_ONLY:-0}" == 1 ]]; then
  echo "server_dry_run=PASS; no Job was created"
  exit 0
fi

created_resource="$(
  kubectl \
    --kubeconfig "${kubeconfig_path}" \
    --context "${context_name}" \
    --namespace "${namespace}" \
    create -f "${rendered}" -o name
)"
created_name="${created_resource#job.batch/}"

echo "submitted=${created_resource}"
echo "logs: kubectl --kubeconfig ${kubeconfig_path} --context ${context_name} -n ${namespace} logs -f job/${created_name}"
echo "events: kubectl --kubeconfig ${kubeconfig_path} --context ${context_name} -n ${namespace} describe job/${created_name}"
echo "cleanup: kubectl --kubeconfig ${kubeconfig_path} --context ${context_name} -n ${namespace} delete job/${created_name}"
