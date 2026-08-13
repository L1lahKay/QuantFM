#!/usr/bin/env python3
"""Run a bounded, UID-owned cross-Pod PVC checkpoint persistence smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "benchmark/config/storage-smoke-20260806.json"
EVIDENCE_ROOT = REPO_ROOT / "benchmark/results/storage"
KUBECONFIG = "/etc/rancher/k3s/k3s.yaml"
CONTEXT = "default"
TOKEN_RE = re.compile(r"^[a-z0-9]{8,16}$")
WRITE_PREFIX = "PVC_CHECKPOINT_WRITE_JSON="
READ_PREFIX = "PVC_CHECKPOINT_READ_JSON="


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def kube(
    args: list[str], *, input_text: str | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    command = ["kubectl", "--kubeconfig", KUBECONFIG, "--context", CONTEXT, *args]
    return subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        check=check,
    )


def writer_program() -> str:
    return r"""
import hashlib
import json
import os
import pathlib
import re
import time

token = os.environ["RUN_TOKEN"]
if re.fullmatch(r"[a-z0-9]{8,16}", token) is None:
    raise SystemExit("invalid run token")
total_bytes = int(os.environ["CHECKPOINT_BYTES"])
root = pathlib.Path("/mnt/quantfm/benchmark-storage-smoke")
target = root / token
if target.exists():
    raise SystemExit(f"refusing existing target: {target}")
target.mkdir(parents=True, exist_ok=False)
temporary = target / "checkpoint.bin.partial"
checkpoint = target / "checkpoint.bin"
manifest = target / "manifest.json"
block_seed = hashlib.sha256(("quantfm-storage-smoke:" + token).encode()).digest()
block = (block_seed * ((1024 * 1024 + len(block_seed) - 1) // len(block_seed)))[:1024 * 1024]
digest = hashlib.sha256()
written = 0
started = time.monotonic()
with temporary.open("xb", buffering=0) as handle:
    while written < total_bytes:
        payload = block[: min(len(block), total_bytes - written)]
        handle.write(payload)
        digest.update(payload)
        written += len(payload)
    os.fsync(handle.fileno())
os.replace(temporary, checkpoint)
directory_fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
elapsed = time.monotonic() - started
record = {
    "schema_version": 1,
    "status": "success",
    "run_token": token,
    "pod_name": os.environ.get("HOSTNAME", "unknown"),
    "checkpoint_bytes": written,
    "checkpoint_sha256": digest.hexdigest(),
    "write_fsync_seconds": elapsed,
    "write_mib_per_second": written / 1048576 / elapsed,
}
manifest_temporary = target / "manifest.json.partial"
with manifest_temporary.open("x", encoding="utf-8") as handle:
    json.dump(record, handle, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(manifest_temporary, manifest)
directory_fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
print("PVC_CHECKPOINT_WRITE_JSON=" + json.dumps(record, sort_keys=True), flush=True)
"""


def reader_program() -> str:
    return r"""
import hashlib
import json
import os
import pathlib
import re
import time

token = os.environ["RUN_TOKEN"]
if re.fullmatch(r"[a-z0-9]{8,16}", token) is None:
    raise SystemExit("invalid run token")
root = pathlib.Path("/mnt/quantfm/benchmark-storage-smoke")
target = root / token
expected_target = root / token
if target != expected_target or target.parent != root:
    raise SystemExit("unsafe target")
manifest = target / "manifest.json"
checkpoint = target / "checkpoint.bin"
record = json.loads(manifest.read_text(encoding="utf-8"))
if record.get("run_token") != token or record.get("status") != "success":
    raise SystemExit("manifest ownership/status mismatch")
digest = hashlib.sha256()
read_bytes = 0
started = time.monotonic()
with checkpoint.open("rb", buffering=0) as handle:
    while True:
        block = handle.read(4 * 1024 * 1024)
        if not block:
            break
        digest.update(block)
        read_bytes += len(block)
elapsed = time.monotonic() - started
if read_bytes != int(record["checkpoint_bytes"]):
    raise SystemExit("checkpoint byte count mismatch")
if digest.hexdigest() != record["checkpoint_sha256"]:
    raise SystemExit("checkpoint SHA-256 mismatch")
result = {
    "schema_version": 1,
    "status": "success",
    "run_token": token,
    "writer_pod_name": record["pod_name"],
    "reader_pod_name": os.environ.get("HOSTNAME", "unknown"),
    "checkpoint_bytes": read_bytes,
    "checkpoint_sha256": digest.hexdigest(),
    "read_seconds": elapsed,
    "read_mib_per_second": read_bytes / 1048576 / elapsed,
    "cross_pod": record["pod_name"] != os.environ.get("HOSTNAME", "unknown"),
}
print("PVC_CHECKPOINT_READ_JSON=" + json.dumps(result, sort_keys=True), flush=True)
checkpoint.unlink()
manifest.unlink()
if any(target.iterdir()):
    raise SystemExit("refusing to remove non-empty target directory")
target.rmdir()
print("PVC_CHECKPOINT_CLEANUP=success", flush=True)
"""


def job_manifest(
    config: dict[str, Any], token: str, role: str, checkpoint_bytes: int
) -> dict[str, Any]:
    namespace = config["namespace"]
    limits = config["limits"]
    name = f"khalil-pvc-{role}-{token}"
    program = writer_program() if role == "write" else reader_program()
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": "khalil-pvc-persistence-smoke",
                "benchmark.quantfm.io/role": role,
                "benchmark.quantfm.io/run-token": token,
            },
        },
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": limits["active_deadline_seconds"],
            "ttlSecondsAfterFinished": limits["ttl_seconds_after_finished"],
            "template": {
                "metadata": {
                    "labels": {
                        "app.kubernetes.io/name": "khalil-pvc-persistence-smoke",
                        "benchmark.quantfm.io/role": role,
                        "benchmark.quantfm.io/run-token": token,
                    }
                },
                "spec": {
                    "automountServiceAccountToken": False,
                    "restartPolicy": "Never",
                    "nodeSelector": {
                        "accelerator": "nvidia",
                        "kubernetes.io/hostname": config["claim"]["node"],
                    },
                    "tolerations": [
                        {
                            "key": "nvidia.com/gpu",
                            "operator": "Exists",
                            "effect": "NoSchedule",
                        }
                    ],
                    "securityContext": {
                        "fsGroup": 1009,
                        "fsGroupChangePolicy": "OnRootMismatch",
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "containers": [
                        {
                            "name": role,
                            "image": config["image"],
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/sh", "-ceu"],
                            "args": ["python -u - <<'PY'\n" + program + "\nPY\n"],
                            "env": [
                                {"name": "RUN_TOKEN", "value": token},
                                {
                                    "name": "CHECKPOINT_BYTES",
                                    "value": str(checkpoint_bytes),
                                },
                            ],
                            "resources": {
                                "requests": {"cpu": "500m", "memory": "512Mi"},
                                "limits": {"cpu": "1", "memory": "1Gi"},
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "capabilities": {"drop": ["ALL"]},
                            },
                            "volumeMounts": [
                                {"name": "quantfm-data", "mountPath": "/mnt/quantfm"}
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "quantfm-data",
                            "persistentVolumeClaim": {
                                "claimName": config["claim"]["name"]
                            },
                        }
                    ],
                },
            },
        },
    }


def exact_live_preflight(config: dict[str, Any], evidence: Path) -> None:
    namespace = config["namespace"]
    required = [
        ("create", "jobs.batch"),
        ("get", "jobs.batch"),
        ("delete", "jobs.batch"),
        ("get", "pods"),
        ("list", "pods"),
        ("get", "pods/log"),
        ("list", "events"),
        ("get", "persistentvolumeclaims"),
    ]
    auth: list[dict[str, str]] = []
    for verb, resource in required:
        allowed = kube(
            ["auth", "can-i", verb, resource, "--namespace", namespace]
        ).stdout.strip()
        auth.append({"verb": verb, "resource": resource, "allowed": allowed})
        if allowed != "yes":
            raise RuntimeError(f"missing permission: {verb} {resource} in {namespace}")
    allowed = kube(
        ["auth", "can-i", "get", "persistentvolumes", "--all-namespaces"]
    ).stdout.strip()
    auth.append({"verb": "get", "resource": "persistentvolumes", "allowed": allowed})
    if allowed != "yes":
        raise RuntimeError("missing permission: get persistentvolumes")
    write_json(evidence / "auth-can-i.json", auth)

    pvc = load_json_from_kube(
        ["-n", namespace, "get", "pvc", config["claim"]["name"], "-o", "json"]
    )
    pv = load_json_from_kube(
        ["get", "pv", config["claim"]["volume_name"], "-o", "json"]
    )
    write_json(evidence / "pvc-before.json", pvc)
    write_json(evidence / "pv-before.json", pv)
    claim = config["claim"]
    checks = {
        "pvc_uid": pvc["metadata"]["uid"] == claim["uid"],
        "pvc_bound": pvc["status"]["phase"] == "Bound",
        "pvc_volume": pvc["spec"]["volumeName"] == claim["volume_name"],
        "pv_uid": pv["metadata"]["uid"] == claim["volume_uid"],
        "pv_path": pv["spec"]["local"]["path"] == claim["local_path"],
        "pv_retain": pv["spec"]["persistentVolumeReclaimPolicy"] == "Retain",
        "claim_ref_uid": pv["spec"]["claimRef"]["uid"] == claim["uid"],
        "single_node_scope": config["scope"]["single_node_persistence_allowed"] is True,
        "non_root_confirmed": config["physical_backend"]["non_root_disk_confirmed"]
        is True,
    }
    mount = subprocess.run(
        [
            "findmnt",
            "--raw",
            "--noheadings",
            "--mountpoint",
            claim["mount_target"],
            "--output",
            "SOURCE,FSTYPE",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.split()
    root_source = subprocess.run(
        ["findmnt", "--raw", "--noheadings", "--mountpoint", "/", "--output", "SOURCE"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    checks.update(
        {
            "live_mount_source": bool(mount) and mount[0] == claim["mount_source"],
            "live_mount_fstype": len(mount) == 2 and mount[1] == claim["filesystem"],
            "live_mount_is_not_root": bool(mount) and mount[0] != root_source,
            "root_source_matches_record": root_source
            == config["physical_backend"]["root_mount_source"],
        }
    )
    write_json(evidence / "storage-identity-checks.json", checks)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("storage identity gate failed: " + ", ".join(failed))

    write_json(
        evidence / "resourcequota-before.json",
        load_json_from_kube(["-n", namespace, "get", "resourcequota", "-o", "json"]),
    )
    running = load_json_from_kube(
        [
            "get",
            "pods",
            "--all-namespaces",
            "--field-selector=status.phase=Running",
            "-o",
            "json",
        ]
    )
    gpu_requests: list[dict[str, Any]] = []
    for pod in running.get("items", []):
        requested = sum(
            int(
                container.get("resources", {})
                .get("requests", {})
                .get("nvidia.com/gpu", "0")
            )
            for container in pod.get("spec", {}).get("containers", [])
        )
        if requested:
            gpu_requests.append(
                {
                    "namespace": pod["metadata"]["namespace"],
                    "name": pod["metadata"]["name"],
                    "gpu_requests": requested,
                }
            )
    write_json(evidence / "running-gpu-requests-before.json", gpu_requests)

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    (evidence / "host-gpu-processes-before.csv").write_text(
        completed.stdout, encoding="utf-8"
    )


def load_json_from_kube(args: list[str]) -> dict[str, Any]:
    return json.loads(kube(args).stdout)


def capture_job(
    namespace: str, name: str, uid: str, role: str, evidence: Path
) -> tuple[dict[str, Any], str]:
    job = load_json_from_kube(["-n", namespace, "get", "job", name, "-o", "json"])
    if job["metadata"]["uid"] != uid:
        raise RuntimeError(f"UID mismatch while capturing {name}")
    write_json(evidence / f"{role}-job.json", job)
    pods = load_json_from_kube(
        ["-n", namespace, "get", "pods", "-l", f"job-name={name}", "-o", "json"]
    )
    write_json(evidence / f"{role}-pods.json", pods)
    owned = [
        pod
        for pod in pods.get("items", [])
        if any(
            ref.get("uid") == uid
            for ref in pod.get("metadata", {}).get("ownerReferences", [])
        )
    ]
    if len(owned) != 1:
        raise RuntimeError(f"expected one UID-owned Pod for {name}, found {len(owned)}")
    pod = owned[0]
    pod_uid = pod["metadata"]["uid"]
    events = load_json_from_kube(
        [
            "-n",
            namespace,
            "get",
            "events",
            "--field-selector",
            f"involvedObject.uid={pod_uid}",
            "-o",
            "json",
        ]
    )
    write_json(evidence / f"{role}-pod-events.json", events)
    logs = kube(["-n", namespace, "logs", pod["metadata"]["name"]]).stdout
    (evidence / f"{role}-container.log").write_text(logs, encoding="utf-8")
    return pod, logs


def wait_for_job(namespace: str, name: str, timeout_seconds: int) -> str:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        job = load_json_from_kube(["-n", namespace, "get", "job", name, "-o", "json"])
        conditions = {
            condition.get("type")
            for condition in job.get("status", {}).get("conditions", [])
            if condition.get("status") == "True"
        }
        if "Complete" in conditions:
            return "Complete"
        if "Failed" in conditions:
            return "Failed"
        time.sleep(2)
    return "Timeout"


def parse_marker(logs: str, prefix: str) -> dict[str, Any]:
    rows = [
        line[len(prefix) :] for line in logs.splitlines() if line.startswith(prefix)
    ]
    if len(rows) != 1:
        raise RuntimeError(f"expected one {prefix} marker, found {len(rows)}")
    value = json.loads(rows[0])
    if value.get("status") != "success":
        raise RuntimeError(f"marker {prefix} is not successful")
    return value


def delete_exact_job(namespace: str, name: str, uid: str, evidence: Path) -> None:
    current = kube(
        ["-n", namespace, "get", "job", name, "--ignore-not-found", "-o", "json"],
        check=False,
    )
    if current.returncode != 0:
        raise RuntimeError(
            f"cannot safely inspect Job {name} before cleanup: {current.stderr.strip()}"
        )
    if not current.stdout.strip():
        return
    payload = json.loads(current.stdout)
    if payload["metadata"]["uid"] != uid:
        raise RuntimeError(f"refusing to delete UID-mismatched Job {name}")
    deleted = kube(
        ["-n", namespace, "delete", "job", name, "--wait=true", "--timeout=120s"]
    )
    with (evidence / "kubernetes-cleanup.log").open("a", encoding="utf-8") as handle:
        handle.write(deleted.stdout)


def checksum_evidence(evidence: Path) -> None:
    rows = []
    for path in sorted(evidence.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "SHA256SUMS":
            rows.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            )
    (evidence / "SHA256SUMS").write_text("".join(rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("plan", "dry-run", "run"), nargs="?", default="plan"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-token")
    parser.add_argument("--checkpoint-bytes", type=int)
    args = parser.parse_args(argv)
    config = load_json(CONFIG_PATH)
    default_bytes = int(config["limits"]["checkpoint_bytes_default"])
    checkpoint_bytes = args.checkpoint_bytes or default_bytes
    if checkpoint_bytes < 1048576 or checkpoint_bytes > int(
        config["limits"]["checkpoint_bytes_max"]
    ):
        parser.error(
            "--checkpoint-bytes must be between 1MiB and the configured maximum"
        )
    if args.action == "run" and not args.execute:
        parser.error("run requires --execute")
    if args.action != "run" and args.execute:
        parser.error("--execute is valid only with run")
    token = args.run_token or (datetime.now(timezone.utc).strftime("s%y%m%d%H%M%S"))
    if TOKEN_RE.fullmatch(token) is None:
        parser.error("run token must contain 8-16 lowercase letters/digits")
    manifests = {
        role: job_manifest(config, token, role, checkpoint_bytes)
        for role in ("write", "read")
    }
    if args.action == "plan":
        print(
            f"mutation=none\nnamespace={config['namespace']}\nclaim={config['claim']['name']}"
        )
        print(f"checkpoint_bytes={checkpoint_bytes}\nrun_token={token}\nnext=dry-run")
        return 0

    evidence = EVIDENCE_ROOT / f"pvc-persistence-{token}"
    if evidence.exists():
        raise SystemExit(f"evidence directory already exists: {evidence}")
    evidence.mkdir(parents=True)
    write_json(evidence / "config.json", config)
    for role, manifest in manifests.items():
        write_json(evidence / f"{role}-manifest.json", manifest)
    exact_live_preflight(config, evidence)
    for role, manifest in manifests.items():
        response = kube(
            ["create", "--dry-run=server", "-f", "-", "-o", "json"],
            input_text=json.dumps(manifest),
        )
        (evidence / f"{role}-server-dry-run.json").write_text(
            response.stdout, encoding="utf-8"
        )
    if args.action == "dry-run":
        checksum_evidence(evidence)
        print(f"server_dry_run=passed\nevidence={evidence}")
        return 0

    namespace = config["namespace"]
    created: list[tuple[str, str]] = []
    timeline: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "run_token": token,
        "status": "failed",
        "checkpoint_bytes": checkpoint_bytes,
        "started_at": utc_now(),
    }
    try:
        markers: dict[str, Any] = {}
        pods: dict[str, Any] = {}
        for role in ("write", "read"):
            manifest = manifests[role]
            name = manifest["metadata"]["name"]
            if kube(
                [
                    "-n",
                    namespace,
                    "get",
                    "job",
                    name,
                    "--ignore-not-found",
                    "-o",
                    "name",
                ]
            ).stdout.strip():
                raise RuntimeError(
                    f"refusing to replace existing Job {namespace}/{name}"
                )
            timeline.append({"event": f"{role}_create_begin", "at": utc_now()})
            created_response = kube(
                ["create", "-f", "-", "-o", "json"], input_text=json.dumps(manifest)
            )
            created_job = json.loads(created_response.stdout)
            uid = created_job["metadata"]["uid"]
            created.append((name, uid))
            timeline.append(
                {"event": f"{role}_create_return", "at": utc_now(), "job_uid": uid}
            )
            terminal = wait_for_job(
                namespace, name, int(config["limits"]["wait_timeout_seconds"])
            )
            timeline.append(
                {
                    "event": f"{role}_terminal_observed",
                    "at": utc_now(),
                    "terminal": terminal,
                }
            )
            pod, logs = capture_job(namespace, name, uid, role, evidence)
            pods[role] = {
                "name": pod["metadata"]["name"],
                "uid": pod["metadata"]["uid"],
                "node": pod["spec"]["nodeName"],
                "image_id": pod["status"]["containerStatuses"][0]["imageID"],
            }
            if terminal != "Complete":
                raise RuntimeError(f"Job {name} ended as {terminal}")
            markers[role] = parse_marker(
                logs, WRITE_PREFIX if role == "write" else READ_PREFIX
            )
        if not markers["read"].get("cross_pod"):
            raise RuntimeError(
                "reader marker did not prove distinct writer and reader Pods"
            )
        if (
            markers["write"]["checkpoint_sha256"]
            != markers["read"]["checkpoint_sha256"]
        ):
            raise RuntimeError("writer/reader SHA-256 mismatch")
        result.update(
            {
                "status": "passed",
                "completed_at": utc_now(),
                "writer": markers["write"],
                "reader": markers["read"],
                "pods": pods,
                "scope": config["scope"],
            }
        )
    except Exception as exc:
        result.update(
            {
                "completed_at": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    finally:
        write_json(evidence / "timeline.json", timeline)
        write_json(evidence / "result.json", result)
        for name, uid in reversed(created):
            try:
                delete_exact_job(namespace, name, uid, evidence)
            except Exception as exc:
                result["status"] = "failed"
                result.setdefault("cleanup_errors", []).append(str(exc))
        write_json(evidence / "result.json", result)
        write_json(
            evidence / "resourcequota-after.json",
            load_json_from_kube(
                ["-n", namespace, "get", "resourcequota", "-o", "json"]
            ),
        )
        checksum_evidence(evidence)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"evidence={evidence}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
