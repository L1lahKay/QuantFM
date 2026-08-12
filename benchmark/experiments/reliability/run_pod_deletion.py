#!/usr/bin/env python3
"""Delete one owned training Pod and verify Kubernetes Job reconciliation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from render_pod_deletion import render
from run_pod_retry import (
    NAMESPACE,
    RESULTS_ROOT,
    ExperimentError,
    Runner,
    utc_now,
)

EVENT_PREFIX = "POD_DELETE_EVENT_JSON="
RESULT_PREFIX = "POD_DELETE_RESULT_JSON="


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def structured_lines(log: str, prefix: str) -> list[dict[str, Any]]:
    values = []
    for line in log.splitlines():
        if not line.startswith(prefix):
            continue
        value = json.loads(line.split("=", 1)[1])
        if isinstance(value, dict):
            values.append(value)
    return values


class PodDeletionRunner(Runner):
    def __init__(self, run_token: str, output: Path) -> None:
        super().__init__(run_token, output)
        self.manifest = render(run_token)
        self.job_name = str(self.manifest["metadata"]["name"])
        self.max_owned_pods = 0
        self.max_gpu_requests = 0
        self.capacity_observations: list[dict[str, Any]] = []

    def observe_replacement_capacity(self, stage: str) -> list[dict[str, Any]]:
        pods = self.list_owned_pods()
        quota = self.json_get(["-n", NAMESPACE, "get", "resourcequota", "gpu-quota"])
        used_gpu = int(
            quota.get("status", {}).get("used", {}).get("requests.nvidia.com/gpu", "0")
        )
        self.max_owned_pods = max(self.max_owned_pods, len(pods))
        self.max_gpu_requests = max(self.max_gpu_requests, used_gpu)
        self.capacity_observations.append(
            {
                "observed_at": utc_now(),
                "stage": stage,
                "owned_pod_count": len(pods),
                "gpu_requests_used": used_gpu,
                "pods": [
                    {
                        "name": pod.get("metadata", {}).get("name"),
                        "uid": pod.get("metadata", {}).get("uid"),
                        "phase": pod.get("status", {}).get("phase"),
                        "deletion_timestamp": pod.get("metadata", {}).get(
                            "deletionTimestamp"
                        ),
                    }
                    for pod in pods
                ],
            }
        )
        self.write_json(
            "replacement-capacity-observations.json", self.capacity_observations
        )
        if len(pods) > 2 or used_gpu > 2:
            raise ExperimentError(
                f"replacement exceeded bound: pods={len(pods)} gpu_requests={used_gpu}"
            )
        return pods

    def preflight(self) -> None:
        existing_job = self.kubectl(
            ["-n", NAMESPACE, "get", "job", self.job_name, "-o", "json"],
            check=False,
        )
        if existing_job.returncode == 0:
            raise ExperimentError("preflight refused: Job name already exists")
        existing_pods = self.json_get(
            [
                "-n",
                NAMESPACE,
                "get",
                "pods",
                "-l",
                f"quantfm.openai/run-token={self.run_token}",
            ]
        )
        if existing_pods.get("items"):
            raise ExperimentError("preflight refused: run token already selects Pods")
        delete_allowed = self.kubectl(
            ["auth", "can-i", "delete", "pods", "-n", NAMESPACE]
        ).stdout.strip()
        if delete_allowed != "yes":
            raise ExperimentError("permission denied: delete pods")
        super().preflight()
        permissions = json.loads((self.output / "auth-can-i.json").read_text())
        permissions.append(
            {"verb": "delete", "resource": "pods", "allowed": delete_allowed}
        )
        self.write_json("auth-can-i.json", permissions)

    def wait_for_deletion_point(self) -> tuple[str, str, str]:
        def deletion_ready() -> tuple[str, str, str] | None:
            pods = self.list_owned_pods()
            if len(pods) != 1:
                return None
            pod = pods[0]
            metadata = pod.get("metadata", {})
            name = str(metadata.get("name") or "")
            uid = str(metadata.get("uid") or "")
            phase = str(pod.get("status", {}).get("phase") or "")
            log = self.pod_log(name)
            (self.output / f"container-{name}-before-deletion.txt").write_text(
                log, encoding="utf-8"
            )
            if phase in {"Failed", "Succeeded"}:
                raise ExperimentError("first Pod finished before deletion")
            if phase != "Running":
                return None
            markers = structured_lines(log, EVENT_PREFIX)
            marker = next(
                (
                    value
                    for value in markers
                    if value.get("event") == "deletion_ready"
                    and value.get("step") == 75
                    and value.get("pod_uid") == uid
                ),
                None,
            )
            if marker is None:
                return None
            return name, uid, str(marker["timestamp"])

        name, uid, marker_at = self.wait_for(
            "first Pod deletion marker", deletion_ready, timeout=120
        )
        snapshot = self.json_get(["-n", NAMESPACE, "get", "pod", name])
        self.write_json("first-pod-before-deletion.json", snapshot)
        self.record(
            "deletion-point-reached",
            f"pod={name}\tuid={uid}\tworkload_timestamp={marker_at}",
        )
        return name, uid, marker_at

    def delete_owned_pod(self, name: str, uid: str) -> tuple[str, str]:
        current = self.json_get(["-n", NAMESPACE, "get", "pod", name])
        metadata = current.get("metadata", {})
        if str(metadata.get("uid") or "") != uid:
            raise ExperimentError("Pod deletion refused: UID changed")
        if metadata.get("labels", {}).get("quantfm.openai/run-token") != self.run_token:
            raise ExperimentError("Pod deletion refused: run token changed")
        refs = metadata.get("ownerReferences", [])
        if not any(ref.get("uid") == self.job_uid for ref in refs):
            raise ExperimentError("Pod deletion refused: Job owner UID mismatch")
        requested_at = utc_now()
        deleted = self.kubectl(
            [
                "-n",
                NAMESPACE,
                "delete",
                "pod",
                name,
                "--wait=false",
            ],
            timeout=20,
        )
        (self.output / "pod-delete.txt").write_text(
            deleted.stdout + deleted.stderr, encoding="utf-8"
        )
        terminating = self.kubectl(
            ["-n", NAMESPACE, "get", "pod", name, "-o", "json"], check=False
        )
        if terminating.returncode == 0:
            self.write_json(
                "first-pod-terminating.json", json.loads(terminating.stdout)
            )

        def pod_absent() -> bool:
            self.observe_replacement_capacity("waiting-for-deleted-pod-absence")
            observed = self.kubectl(
                ["-n", NAMESPACE, "get", "pod", name, "-o", "json"],
                check=False,
            )
            return observed.returncode != 0

        self.wait_for("deleted Pod absence", pod_absent, timeout=60)
        confirmed_at = utc_now()
        absent = self.kubectl(
            ["-n", NAMESPACE, "get", "pod", name, "-o", "json"], check=False
        )
        (self.output / "first-pod-after-deletion.txt").write_text(
            absent.stdout + absent.stderr, encoding="utf-8"
        )
        if absent.returncode == 0:
            raise ExperimentError("deleted Pod still exists")
        self.record(
            "pod-deleted",
            f"pod={name}\tuid={uid}\trequested_at={requested_at}\tconfirmed_at={confirmed_at}",
        )
        return requested_at, confirmed_at

    def wait_for_replacement(self, deleted_uid: str) -> tuple[str, str, str]:
        def replacement_ready() -> tuple[str, str, str] | None:
            pods = self.observe_replacement_capacity("waiting-for-replacement")
            self.write_json(
                "replacement-pods-observed.json",
                {"apiVersion": "v1", "kind": "List", "items": pods},
            )
            for pod in pods:
                metadata = pod.get("metadata", {})
                uid = str(metadata.get("uid") or "")
                name = str(metadata.get("name") or "")
                if not uid or uid == deleted_uid:
                    continue
                log = self.pod_log(name)
                (self.output / f"container-{name}-replacement-observed.txt").write_text(
                    log, encoding="utf-8"
                )
                markers = structured_lines(log, EVENT_PREFIX)
                marker = next(
                    (
                        value
                        for value in markers
                        if value.get("event") == "attempt_started"
                        and value.get("pod_uid") == uid
                    ),
                    None,
                )
                if marker is not None:
                    return name, uid, str(marker["timestamp"])
            return None

        name, uid, marker_at = self.wait_for(
            "replacement Pod attempt marker", replacement_ready, timeout=150
        )
        self.write_json(
            "replacement-pod-started.json",
            self.json_get(["-n", NAMESPACE, "get", "pod", name]),
        )
        self.record(
            "replacement-running",
            f"pod={name}\tuid={uid}\tworkload_timestamp={marker_at}",
        )
        return name, uid, marker_at

    def wait_complete(self) -> None:
        waited = self.kubectl(
            [
                "-n",
                NAMESPACE,
                "wait",
                "--for=condition=complete",
                f"job/{self.job_name}",
                "--timeout=180s",
            ],
            timeout=190,
        )
        (self.output / "job-wait.txt").write_text(
            waited.stdout + waited.stderr, encoding="utf-8"
        )
        self.record("job-complete")

    def capture_and_validate_deletion(
        self,
        *,
        deleted_name: str,
        deleted_uid: str,
        deletion_marker_at: str,
        delete_requested_at: str,
        delete_confirmed_at: str,
        replacement_name: str,
        replacement_uid: str,
        replacement_marker_at: str,
    ) -> dict[str, Any]:
        job = self.json_get(["-n", NAMESPACE, "get", "job", self.job_name])
        pods = self.list_owned_pods()
        self.write_json("job-final.json", job)
        self.write_json(
            "pods-final.json", {"apiVersion": "v1", "kind": "List", "items": pods}
        )
        if len(pods) != 1:
            raise ExperimentError(f"expected one final owned Pod, found {len(pods)}")
        final_uid = str(pods[0].get("metadata", {}).get("uid") or "")
        if final_uid != replacement_uid or replacement_uid == deleted_uid:
            raise ExperimentError("replacement Pod UID validation failed")
        log = self.pod_log(replacement_name)
        (self.output / f"container-{replacement_name}.txt").write_text(
            log, encoding="utf-8"
        )
        results = structured_lines(log, RESULT_PREFIX)
        if len(results) != 1 or results[0].get("status") != "success":
            raise ExperimentError("replacement Pod did not emit one success result")
        if results[0].get("pod_uid") != replacement_uid:
            raise ExperimentError("replacement success result UID mismatch")
        statuses = pods[0].get("status", {}).get("containerStatuses", [])
        exit_code = (
            statuses[0].get("state", {}).get("terminated", {}).get("exitCode")
            if statuses
            else None
        )
        if exit_code != 0:
            raise ExperimentError(f"replacement Pod exit={exit_code}")
        conditions = job.get("status", {}).get("conditions", [])
        complete = next(
            (
                condition
                for condition in conditions
                if condition.get("type") == "Complete"
                and condition.get("status") == "True"
            ),
            None,
        )
        if complete is None:
            raise ExperimentError("Job Complete condition is missing")
        all_events = self.json_get(["-n", NAMESPACE, "get", "events"])
        target_uids = {self.job_uid, deleted_uid, replacement_uid}
        events = [
            event
            for event in all_events.get("items", [])
            if event.get("involvedObject", {}).get("uid") in target_uids
        ]
        self.write_json(
            "events.json", {"apiVersion": "v1", "kind": "List", "items": events}
        )
        creation = str(job.get("metadata", {}).get("creationTimestamp") or "")
        completion = str(
            job.get("status", {}).get("completionTime")
            or complete.get("lastTransitionTime")
            or ""
        )
        api_wall = (
            (parse_timestamp(completion) - parse_timestamp(creation)).total_seconds()
            if creation and completion
            else None
        )
        request_to_replacement = (
            parse_timestamp(replacement_marker_at)
            - parse_timestamp(delete_requested_at)
        ).total_seconds()
        confirmed_to_replacement = (
            parse_timestamp(replacement_marker_at)
            - parse_timestamp(delete_confirmed_at)
        ).total_seconds()
        result = {
            "schema_version": 1,
            "experiment": "pod-deletion-job-reconciliation",
            "status": "passed",
            "run_token": self.run_token,
            "job_name": self.job_name,
            "job_uid": self.job_uid,
            "backoff_limit": 1,
            "pod_replacement_policy": "TerminatingOrFailed",
            "maximum_owned_pods_observed": self.max_owned_pods,
            "maximum_gpu_requests_observed": self.max_gpu_requests,
            "deleted_pod": {
                "name": deleted_name,
                "uid": deleted_uid,
                "deletion_marker_at": deletion_marker_at,
                "delete_requested_at": delete_requested_at,
                "delete_confirmed_at": delete_confirmed_at,
                "final_absent": True,
            },
            "replacement_pod": {
                "name": replacement_name,
                "uid": replacement_uid,
                "attempt_started_at": replacement_marker_at,
                "exit_code": exit_code,
            },
            "delete_request_to_replacement_seconds": request_to_replacement,
            "delete_confirmation_to_replacement_seconds": confirmed_to_replacement,
            "api_wall_clock_seconds": api_wall,
            "training": results[0],
            "captured_event_count": len(events),
            "validated_at": utc_now(),
        }
        self.write_json("result.json", result)
        self.record("validation-passed")
        return result

    def capture_live_failure_state(self) -> None:
        if not self.created or not self.job_uid:
            return
        job = self.kubectl(
            ["-n", NAMESPACE, "get", "job", self.job_name, "-o", "json"],
            check=False,
        )
        if job.returncode == 0:
            self.write_json("job-at-failure.json", json.loads(job.stdout))
        pods = self.list_owned_pods()
        self.write_json(
            "pods-at-failure.json",
            {"apiVersion": "v1", "kind": "List", "items": pods},
        )
        all_events = self.json_get(["-n", NAMESPACE, "get", "events"])
        pod_uids = {str(pod.get("metadata", {}).get("uid") or "") for pod in pods}
        target_uids = {self.job_uid, *pod_uids}
        events = [
            event
            for event in all_events.get("items", [])
            if event.get("involvedObject", {}).get("uid") in target_uids
        ]
        self.write_json(
            "events-at-failure.json",
            {"apiVersion": "v1", "kind": "List", "items": events},
        )

    def cleanup(self) -> None:
        super().cleanup()
        if not self.created:
            return
        check = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        (self.output / "host-gpu-processes-after.csv").write_text(
            check.stdout, encoding="utf-8"
        )
        if check.returncode != 0 or check.stdout.strip():
            raise ExperimentError("host GPU process remained after cleanup")

    def run(self) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=False)
        self.record("experiment-start", f"run_token={self.run_token}")
        try:
            self.preflight()
            self.create()
            deleted_name, deleted_uid, marker_at = self.wait_for_deletion_point()
            requested_at, confirmed_at = self.delete_owned_pod(
                deleted_name, deleted_uid
            )
            replacement_name, replacement_uid, replacement_marker_at = (
                self.wait_for_replacement(deleted_uid)
            )
            self.wait_complete()
            return self.capture_and_validate_deletion(
                deleted_name=deleted_name,
                deleted_uid=deleted_uid,
                deletion_marker_at=marker_at,
                delete_requested_at=requested_at,
                delete_confirmed_at=confirmed_at,
                replacement_name=replacement_name,
                replacement_uid=replacement_uid,
                replacement_marker_at=replacement_marker_at,
            )
        except Exception as exc:
            try:
                self.capture_live_failure_state()
            except Exception as capture_exc:
                self.record(
                    "failure-capture-error",
                    f"{type(capture_exc).__name__}: {str(capture_exc)[:200]}",
                )
            self.write_json(
                "failure.json",
                {
                    "schema_version": 1,
                    "experiment": "pod-deletion-job-reconciliation",
                    "status": "failed",
                    "run_token": self.run_token,
                    "job_name": self.job_name,
                    "job_uid": self.job_uid,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failed_at": utc_now(),
                },
            )
            self.record("experiment-failed", f"{type(exc).__name__}: {str(exc)[:300]}")
            raise
        finally:
            self.cleanup()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"), nargs="?", default="plan")
    parser.add_argument("--run-token", default="del260806a")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = render(args.run_token)
    output = (
        args.output or (RESULTS_ROOT / f"pod-deletion-{args.run_token}")
    ).resolve()
    try:
        output.relative_to(RESULTS_ROOT.resolve())
    except ValueError:
        print("output must remain below benchmark/results/reliability", file=sys.stderr)
        return 2
    if args.action == "plan":
        print(
            json.dumps(
                {
                    "mutation": "none",
                    "experiment": "pod-deletion-job-reconciliation",
                    "job": manifest["metadata"]["name"],
                    "namespace": NAMESPACE,
                    "gpu": 1,
                    "backoffLimit": 1,
                    "podReplacementPolicy": "TerminatingOrFailed",
                    "delete_after_step": 75,
                    "total_steps": 300,
                    "output": str(output),
                },
                indent=2,
            )
        )
        return 0
    if not args.execute:
        print("run refused: add --execute", file=sys.stderr)
        return 2
    if output.exists():
        print(f"run refused: output already exists: {output}", file=sys.stderr)
        return 2
    runner = PodDeletionRunner(args.run_token, output)
    try:
        result = runner.run()
    except Exception as exc:
        print(f"experiment failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
