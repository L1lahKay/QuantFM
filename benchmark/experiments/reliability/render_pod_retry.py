#!/usr/bin/env python3
"""Render the bounded one-GPU Pod process-failure / Job retry experiment."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
WORKLOAD = SCRIPT_DIR / "pod_retry_workload.py"
IMAGE = (
    "registry.zs/gpu-dev/dylan-trainer@sha256:"
    "9e7f7f8dc3c15c522408d1e8da38401ac224b99ddfba363078f40403eb456574"
)
TOKEN_PATTERN = re.compile(r"^[a-z0-9]{8,12}$")


class RenderError(ValueError):
    pass


def render(run_token: str, backoff_limit: int = 1) -> dict[str, Any]:
    if not TOKEN_PATTERN.fullmatch(run_token):
        raise RenderError("run token must contain 8-12 lowercase letters or digits")
    if isinstance(backoff_limit, bool) or backoff_limit not in (0, 1, 2):
        raise RenderError("backoff limit must be 0, 1, or 2")
    payload = WORKLOAD.read_bytes()
    payload_sha = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")
    name = f"khalil-pod-retry-{run_token}"
    labels = {
        "app.kubernetes.io/name": "quantfm-pod-retry",
        "app.kubernetes.io/part-of": "quantfm-reliability-evaluation",
        "quantfm.openai/experiment": "pod-process-failure-job-retry",
        "quantfm.openai/run-token": run_token,
        "quantfm.openai/backoff-limit": str(backoff_limit),
    }
    command = "\n".join(
        (
            "umask 077",
            'mkdir -p "$HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR"',
            f"printf '%s' '{encoded}' | base64 -d > /experiment/pod_retry_workload.py",
            'test "$(sha256sum /experiment/pod_retry_workload.py | cut -d\' \' -f1)" = "$WORKLOAD_SHA256"',
            "python -u /experiment/pod_retry_workload.py",
        )
    )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": "gpu-dev",
            "labels": labels,
            "annotations": {
                "quantfm.openai/workload-sha256": payload_sha,
                "quantfm.openai/storage-mode": "synthetic-ephemeral",
                "quantfm.openai/checkpoints-allowed": "false",
                "quantfm.openai/backoff-limit": str(backoff_limit),
            },
        },
        "spec": {
            "backoffLimit": backoff_limit,
            "activeDeadlineSeconds": 300,
            "ttlSecondsAfterFinished": 3600,
            "completions": 1,
            "parallelism": 1,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "schedulerName": "default-scheduler",
                    "runtimeClassName": "nvidia",
                    "restartPolicy": "Never",
                    "automountServiceAccountToken": False,
                    "nodeSelector": {"accelerator": "nvidia"},
                    "tolerations": [
                        {
                            "key": "nvidia.com/gpu",
                            "operator": "Exists",
                            "effect": "NoSchedule",
                        }
                    ],
                    "securityContext": {
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "volumes": [
                        {
                            "name": "scratch",
                            "emptyDir": {"medium": "Memory", "sizeLimit": "512Mi"},
                        }
                    ],
                    "containers": [
                        {
                            "name": "trainer",
                            "image": IMAGE,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/sh", "-ceu"],
                            "args": [command],
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "readOnlyRootFilesystem": True,
                                "capabilities": {"drop": ["ALL"]},
                            },
                            "env": [
                                {"name": "RUN_TOKEN", "value": run_token},
                                {"name": "INJECTION_WINDOW_SECONDS", "value": "45"},
                                {"name": "WORKLOAD_SHA256", "value": payload_sha},
                                {"name": "HOME", "value": "/experiment/home"},
                                {
                                    "name": "XDG_CACHE_HOME",
                                    "value": "/experiment/cache",
                                },
                                {"name": "TORCH_HOME", "value": "/experiment/torch"},
                                {"name": "TMPDIR", "value": "/experiment/tmp"},
                                {"name": "CUDA_CACHE_DISABLE", "value": "1"},
                                {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                                {
                                    "name": "POD_NAME",
                                    "valueFrom": {
                                        "fieldRef": {"fieldPath": "metadata.name"}
                                    },
                                },
                                {
                                    "name": "POD_UID",
                                    "valueFrom": {
                                        "fieldRef": {"fieldPath": "metadata.uid"}
                                    },
                                },
                            ],
                            "volumeMounts": [
                                {"name": "scratch", "mountPath": "/experiment"},
                                {"name": "scratch", "mountPath": "/dev/shm"},
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "2",
                                    "memory": "4Gi",
                                    "nvidia.com/gpu": "1",
                                },
                                "limits": {
                                    "cpu": "2",
                                    "memory": "4Gi",
                                    "nvidia.com/gpu": "1",
                                },
                            },
                        }
                    ],
                },
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--backoff-limit", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = render(args.run_token, args.backoff_limit)
    except RenderError as exc:
        print(f"render refused: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
