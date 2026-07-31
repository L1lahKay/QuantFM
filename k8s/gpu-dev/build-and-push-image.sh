#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: $0 IMAGE_TAG" >&2
  exit 64
fi

image_tag="$1"
if [[ ! "${image_tag}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]]; then
  echo "Invalid container image tag: ${image_tag}" >&2
  exit 64
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
image="registry.zs/gpu-dev/khalil-quantfm:${image_tag}"

if ! docker info >/dev/null 2>&1; then
  echo "Docker build service is not accessible to the current user." >&2
  echo "Run this script in an approved build host/CI runner; do not use sudo credentials in the script." >&2
  exit 69
fi

if ! docker system info --format '{{json .RegistryConfig.IndexConfigs}}' \
     >/dev/null 2>&1; then
  echo "Unable to inspect Docker registry configuration." >&2
  exit 69
fi

docker build \
  --pull \
  --file "${repo_root}/Dockerfile.gpu" \
  --tag "${image}" \
  "${repo_root}"

docker push "${image}"

repo_digests="$(
  docker image inspect \
    --format '{{join .RepoDigests "\n"}}' \
    "${image}"
)"

echo "image=${image}"
if [[ -n "${repo_digests}" ]]; then
  echo "repo_digests:"
  echo "${repo_digests}"
fi
