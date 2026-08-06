#!/usr/bin/env python3
"""Run and capture the bounded Pod process-failure / Kubernetes Job retry test."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from render_pod_retry import render


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "benchmark" / "results" / "reliability"
KUBECONFIG = "/etc/rancher/k3s/k3s.yaml"
CONTEXT = "default"
NAMESPACE = "gpu-dev"
EVENT_PREFIX = "POD_RETRY_EVENT_JSON="
RESULT_PREFIX = "POD_RETRY_RESULT_JSON="


class ExperimentError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


class Runner:
    def __init__(self, run_token: str, output: Path) -> None:
        self.run_token = run_token
        self.output = output
        self.manifest = render(run_token)
        self.job_name = str(self.manifest["metadata"]["name"])
        self.job_uid: str | None = None
        self.created = False
        self.timeline: list[str] = []

    def record(self, event: str, detail: str = "") -> None:
        line = f"{utc_now()}\t{event}"
        if detail:
            line += f"\t{detail}"
        self.timeline.append(line)
        (self.output / "timeline.tsv").write_text(
            "\n".join(self.timeline) + "\n", encoding="utf-8"
        )

    def kubectl(
        self,
        args: Sequence[str],
        *,
        check: bool = True,
        input_text: str | None = None,
        timeout: int = 30,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            "kubectl",
            "--kubeconfig",
            KUBECONFIG,
            "--context",
            CONTEXT,
            *args,
        ]
        completed = subprocess.run(
            command,
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        if check and completed.returncode != 0:
            raise ExperimentError(
                f"kubectl {' '.join(args)} failed: {completed.stderr.strip()}"
            )
        return completed

    def json_get(self, args: Sequence[str]) -> dict[str, Any]:
        completed = self.kubectl([*args, "-o", "json"])
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ExperimentError(f"kubectl returned invalid JSON for {' '.join(args)}") from exc
        if not isinstance(value, dict):
            raise ExperimentError("kubectl JSON response is not an object")
        return value

    def write_json(self, name: str, value: Any) -> None:
        (self.output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def preflight(self) -> None:
        self.record("preflight-start")
        permissions = []
        for verb, resource in (
            ("create", "jobs.batch"),
            ("get", "jobs.batch"),
            ("delete", "jobs.batch"),
            ("list", "pods"),
            ("get", "pods"),
            ("create", "pods/exec"),
            ("list", "events"),
        ):
            result = self.kubectl(
                ["auth", "can-i", verb, resource, "-n", NAMESPACE]
            ).stdout.strip()
            permissions.append({"verb": verb, "resource": resource, "allowed": result})
            if result != "yes":
                raise ExperimentError(f"permission denied: {verb} {resource}")
        self.write_json("auth-can-i.json", permissions)

        quota = self.json_get(["-n", NAMESPACE, "get", "resourcequota", "gpu-quota"])
        self.write_json("resourcequota-before.json", quota)
        used = quota.get("status", {}).get("used", {})
        if str(used.get("requests.nvidia.com/gpu", "0")) != "0":
            raise ExperimentError("gpu-dev already has used GPU requests")

        pods = self.json_get(["-n", NAMESPACE, "get", "pods"])
        running_gpu: list[dict[str, Any]] = []
        for pod in pods.get("items", []):
            if pod.get("status", {}).get("phase") != "Running":
                continue
            total = 0
            for container in pod.get("spec", {}).get("containers", []):
                resources = container.get("resources", {})
                requests = resources.get("requests", {})
                limits = resources.get("limits", {})
                total += int(requests.get("nvidia.com/gpu", limits.get("nvidia.com/gpu", 0)))
            if total:
                running_gpu.append(
                    {"namespace": NAMESPACE, "pod": pod.get("metadata", {}).get("name"), "gpu": total}
                )
        self.write_json("running-gpu-requests-before.json", running_gpu)
        if running_gpu:
            raise ExperimentError("unrelated Running GPU Pod exists")

        process_check = subprocess.run(
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
        (self.output / "host-gpu-processes-before.csv").write_text(
            process_check.stdout, encoding="utf-8"
        )
        if process_check.returncode != 0:
            raise ExperimentError("host GPU process query failed")
        if process_check.stdout.strip():
            raise ExperimentError("host GPU compute process exists")

        manifest_text = json.dumps(self.manifest, indent=2, sort_keys=True) + "\n"
        (self.output / "manifest.json").write_text(manifest_text, encoding="utf-8")
        dryrun = self.kubectl(
            ["-n", NAMESPACE, "create", "--dry-run=server", "-f", "-", "-o", "json"],
            input_text=manifest_text,
        )
        (self.output / "server-dry-run.json").write_text(dryrun.stdout, encoding="utf-8")
        self.record("preflight-complete")

    def list_owned_pods(self) -> list[dict[str, Any]]:
        pods = self.json_get(
            [
                "-n",
                NAMESPACE,
                "get",
                "pods",
                "-l",
                f"quantfm.openai/run-token={self.run_token}",
            ]
        )
        owned = []
        for pod in pods.get("items", []):
            refs = pod.get("metadata", {}).get("ownerReferences", [])
            if self.job_uid and not any(ref.get("uid") == self.job_uid for ref in refs):
                raise ExperimentError("run-token selected a Pod not owned by this Job UID")
            owned.append(pod)
        return owned

    def pod_log(self, pod_name: str) -> str:
        return self.kubectl(
            ["-n", NAMESPACE, "logs", pod_name, "-c", "trainer"],
            check=False,
        ).stdout

    def wait_for(
        self,
        description: str,
        predicate: Any,
        *,
        timeout: int,
    ) -> Any:
        deadline = time.monotonic() + timeout
        last: Any = None
        while time.monotonic() < deadline:
            last = predicate()
            if last:
                return last
            time.sleep(1)
        raise ExperimentError(f"timeout waiting for {description}; last={last!r}")

    def create(self) -> None:
        manifest_text = json.dumps(self.manifest, separators=(",", ":"))
        created = self.kubectl(
            ["-n", NAMESPACE, "create", "-f", "-", "-o", "json"],
            input_text=manifest_text,
        )
        job = json.loads(created.stdout)
        self.job_uid = str(job.get("metadata", {}).get("uid") or "")
        if not self.job_uid:
            raise ExperimentError("created Job has no UID")
        self.created = True
        self.write_json("job-created.json", job)
        self.record("job-created", f"name={self.job_name}\tuid={self.job_uid}")

    def inject_failure(self) -> tuple[str, str]:
        def first_ready() -> tuple[str, str] | None:
            pods = self.list_owned_pods()
            if not pods:
                return None
            self.write_json(
                "pods-latest-before-injection.json",
                {"apiVersion": "v1", "kind": "List", "items": pods},
            )
            pod = sorted(
                pods,
                key=lambda item: str(item.get("metadata", {}).get("creationTimestamp") or ""),
            )[0]
            name = str(pod.get("metadata", {}).get("name") or "")
            uid = str(pod.get("metadata", {}).get("uid") or "")
            phase = str(pod.get("status", {}).get("phase") or "")
            log = self.pod_log(name)
            (self.output / f"container-{name}-observed.txt").write_text(
                log, encoding="utf-8"
            )
            if phase in {"Failed", "Succeeded"}:
                statuses = pod.get("status", {}).get("containerStatuses", [])
                terminated = statuses[0].get("state", {}).get("terminated", {}) if statuses else {}
                raise ExperimentError(
                    "first Pod terminated before the injection marker: "
                    f"phase={phase} reason={terminated.get('reason')} "
                    f"exit={terminated.get('exitCode')} log={log[-500:]!r}"
                )
            if phase != "Running":
                return None
            if EVENT_PREFIX not in log or '"event":"attempt_started"' not in log:
                return None
            (self.output / f"container-{name}-before-failure.txt").write_text(
                log, encoding="utf-8"
            )
            return name, uid

        first_name, first_uid = self.wait_for(
            "first Pod attempt marker", first_ready, timeout=120
        )
        self.record("first-attempt-running", f"pod={first_name}\tuid={first_uid}")
        first_snapshot = self.json_get(["-n", NAMESPACE, "get", "pod", first_name])
        self.write_json("first-pod-before-failure.json", first_snapshot)
        inventory = self.kubectl(
            [
                "-n",
                NAMESPACE,
                "exec",
                first_name,
                "-c",
                "trainer",
                "--",
                "/bin/sh",
                "-c",
                "for p in /proc/[0-9]*/comm; do "
                "pid=${p#/proc/}; pid=${pid%/comm}; "
                "name=$(cat \"$p\" 2>/dev/null || true); "
                "printf '%s\\t%s\\n' \"$pid\" \"$name\"; done",
            ],
            check=False,
            timeout=20,
        )
        (self.output / "container-processes-before-failure.txt").write_text(
            inventory.stdout + inventory.stderr, encoding="utf-8"
        )
        injected_at = utc_now()
        killed = self.kubectl(
            [
                "-n",
                NAMESPACE,
                "exec",
                first_name,
                "-c",
                "trainer",
                "--",
                "/bin/sh",
                "-c",
                "for p in /proc/[0-9]*/comm; do "
                "pid=${p#/proc/}; pid=${pid%/comm}; "
                "name=$(cat \"$p\" 2>/dev/null || true); "
                "if [ \"$pid\" != 1 ] && { [ \"$name\" = python ] || [ \"$name\" = python3 ]; }; then "
                "kill -KILL \"$pid\"; exit 0; fi; "
                "done; exit 1",
            ],
            check=False,
            timeout=20,
        )
        (self.output / "failure-injection.txt").write_text(
            f"injected_at={injected_at}\nreturncode={killed.returncode}\n"
            f"stdout={killed.stdout}\nstderr={killed.stderr}\n",
            encoding="utf-8",
        )
        if killed.returncode != 0:
            raise ExperimentError(
                f"failure injection did not terminate the Python child: returncode={killed.returncode}"
            )
        self.record("failure-injected", f"pod={first_name}\tuid={first_uid}")
        return first_name, first_uid

    def wait_for_retry_and_completion(
        self, first_name: str, first_uid: str
    ) -> tuple[str, str]:
        def retry_started() -> tuple[str, str] | None:
            pods = self.list_owned_pods()
            self.write_json(
                "pods-latest-after-injection.json",
                {"apiVersion": "v1", "kind": "List", "items": pods},
            )
            for pod in pods:
                uid = str(pod.get("metadata", {}).get("uid") or "")
                name = str(pod.get("metadata", {}).get("name") or "")
                if uid == first_uid:
                    continue
                phase = str(pod.get("status", {}).get("phase") or "")
                log = self.pod_log(name)
                (self.output / f"container-{name}-retry-observed.txt").write_text(
                    log, encoding="utf-8"
                )
                if phase in {"Failed", "Succeeded"} and EVENT_PREFIX not in log:
                    statuses = pod.get("status", {}).get("containerStatuses", [])
                    terminated = statuses[0].get("state", {}).get("terminated", {}) if statuses else {}
                    raise ExperimentError(
                        "retry Pod terminated before its attempt marker: "
                        f"phase={phase} reason={terminated.get('reason')} "
                        f"exit={terminated.get('exitCode')} log={log[-500:]!r}"
                    )
                if phase != "Running":
                    continue
                if EVENT_PREFIX in log and '"event":"attempt_started"' in log:
                    return name, uid
            return None

        second_name, second_uid = self.wait_for(
            "second Pod retry attempt", retry_started, timeout=150
        )
        self.record("retry-attempt-running", f"pod={second_name}\tuid={second_uid}")
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
        return second_name, second_uid

    def capture_and_validate(
        self, first_name: str, first_uid: str, second_name: str, second_uid: str
    ) -> dict[str, Any]:
        job = self.json_get(["-n", NAMESPACE, "get", "job", self.job_name])
        pods = self.list_owned_pods()
        self.write_json("job-final.json", job)
        self.write_json("pods-final.json", {"apiVersion": "v1", "kind": "List", "items": pods})
        pod_uids = {str(pod.get("metadata", {}).get("uid") or "") for pod in pods}
        all_events = self.json_get(["-n", NAMESPACE, "get", "events"])
        target_uids = {self.job_uid, *pod_uids}
        events = [
            event
            for event in all_events.get("items", [])
            if event.get("involvedObject", {}).get("uid") in target_uids
        ]
        self.write_json("events.json", {"apiVersion": "v1", "kind": "List", "items": events})

        logs: dict[str, str] = {}
        for pod in pods:
            name = str(pod.get("metadata", {}).get("name") or "")
            log = self.pod_log(name)
            logs[name] = log
            (self.output / f"container-{name}.txt").write_text(log, encoding="utf-8")

        if len(pods) != 2 or pod_uids != {first_uid, second_uid}:
            raise ExperimentError(f"expected exactly two owned Pod UIDs, found {pod_uids}")
        first = next(pod for pod in pods if pod.get("metadata", {}).get("uid") == first_uid)
        second = next(pod for pod in pods if pod.get("metadata", {}).get("uid") == second_uid)
        first_state = first.get("status", {}).get("containerStatuses", [{}])[0].get("state", {})
        second_state = second.get("status", {}).get("containerStatuses", [{}])[0].get("state", {})
        first_exit = first_state.get("terminated", {}).get("exitCode")
        second_exit = second_state.get("terminated", {}).get("exitCode")
        if first_exit in (None, 0):
            raise ExperimentError(f"first Pod did not record a non-zero exit: {first_exit}")
        if second_exit != 0:
            raise ExperimentError(f"retry Pod did not exit successfully: {second_exit}")
        if RESULT_PREFIX in logs[first_name]:
            raise ExperimentError("failed first Pod unexpectedly emitted a success result")
        success_lines = [
            line
            for line in logs[second_name].splitlines()
            if line.startswith(RESULT_PREFIX)
        ]
        if len(success_lines) != 1:
            raise ExperimentError("retry Pod must emit exactly one success result")
        payload = json.loads(success_lines[0].split("=", 1)[1])
        if payload.get("status") != "success" or payload.get("pod_uid") != second_uid:
            raise ExperimentError("retry success marker does not match second Pod UID")
        conditions = job.get("status", {}).get("conditions", [])
        if not any(
            condition.get("type") == "Complete" and condition.get("status") == "True"
            for condition in conditions
        ):
            raise ExperimentError("Job Complete condition is missing")
        result = {
            "schema_version": 1,
            "experiment": "pod-process-failure-job-retry",
            "status": "passed",
            "run_token": self.run_token,
            "job_name": self.job_name,
            "job_uid": self.job_uid,
            "first_pod": {"name": first_name, "uid": first_uid, "exit_code": first_exit},
            "retry_pod": {"name": second_name, "uid": second_uid, "exit_code": second_exit},
            "retry_training": payload,
            "owned_pod_count": len(pods),
            "captured_event_count": len(events),
            "validated_at": utc_now(),
        }
        self.write_json("result.json", result)
        self.record("validation-passed")
        return result

    def cleanup(self) -> None:
        if not self.created or not self.job_uid:
            return
        current = self.kubectl(
            ["-n", NAMESPACE, "get", "job", self.job_name, "-o", "json"],
            check=False,
        )
        if current.returncode == 0:
            job = json.loads(current.stdout)
            current_uid = str(job.get("metadata", {}).get("uid") or "")
            if current_uid != self.job_uid:
                raise ExperimentError("cleanup refused: Job UID no longer matches")
            self.kubectl(
                [
                    "-n",
                    NAMESPACE,
                    "delete",
                    "job",
                    self.job_name,
                    "--cascade=foreground",
                    "--wait=true",
                    "--timeout=60s",
                ],
                timeout=70,
            )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            pods = self.list_owned_pods()
            if not pods:
                break
            time.sleep(1)
        else:
            raise ExperimentError("cleanup timeout: owned Pods remain")
        quota = self.json_get(["-n", NAMESPACE, "get", "resourcequota", "gpu-quota"])
        self.write_json("resourcequota-after.json", quota)
        used = quota.get("status", {}).get("used", {})
        if str(used.get("requests.nvidia.com/gpu", "0")) != "0":
            raise ExperimentError("GPU quota did not return to zero")
        self.record("cleanup-complete", "job_absent=true\tpods_absent=true\tgpu_requests=0")

    def run(self) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=False)
        self.record("experiment-start", f"run_token={self.run_token}")
        try:
            self.preflight()
            self.create()
            first_name, first_uid = self.inject_failure()
            second_name, second_uid = self.wait_for_retry_and_completion(
                first_name, first_uid
            )
            return self.capture_and_validate(
                first_name, first_uid, second_name, second_uid
            )
        except Exception as exc:
            self.write_json(
                "failure.json",
                {
                    "schema_version": 1,
                    "experiment": "pod-process-failure-job-retry",
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
    parser.add_argument("--run-token", default="retry260806")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = render(args.run_token)
    output = (args.output or (RESULTS_ROOT / f"pod-retry-{args.run_token}")).resolve()
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
                    "experiment": "pod-process-failure-job-retry",
                    "job": manifest["metadata"]["name"],
                    "namespace": NAMESPACE,
                    "gpu": 1,
                    "backoffLimit": 1,
                    "output": str(output),
                },
                indent=2,
            )
        )
        return 0
    if not args.execute:
        print("run refused: add --execute", file=sys.stderr)
        return 2
    runner = Runner(args.run_token, output)
    try:
        result = runner.run()
    except Exception as exc:
        print(f"experiment failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
