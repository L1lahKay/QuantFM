#!/usr/bin/env python3
"""Run the bounded one-GPU Kubernetes Job backoffLimit 0/1/2 matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from render_pod_retry import render
from run_pod_retry import (
    CONTEXT,
    EVENT_PREFIX,
    KUBECONFIG,
    NAMESPACE,
    RESULT_PREFIX,
    RESULTS_ROOT,
    ExperimentError,
    Runner,
    utc_now,
)

MATRIX_CASES = (
    {"backoff_limit": 0, "injected_failures": 1, "terminal": "Failed"},
    {"backoff_limit": 1, "injected_failures": 1, "terminal": "Complete"},
    {"backoff_limit": 2, "injected_failures": 2, "terminal": "Complete"},
)


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class BackoffCaseRunner(Runner):
    def __init__(
        self,
        run_token: str,
        output: Path,
        *,
        backoff_limit: int,
        injected_failures: int,
        terminal: str,
    ) -> None:
        super().__init__(run_token, output)
        if terminal not in {"Complete", "Failed"}:
            raise ValueError("terminal must be Complete or Failed")
        self.backoff_limit = backoff_limit
        self.injected_failures = injected_failures
        self.terminal = terminal
        self.manifest = render(run_token, backoff_limit)
        self.job_name = str(self.manifest["metadata"]["name"])

    def wait_attempt(self, excluded_uids: set[str], ordinal: int) -> tuple[str, str]:
        def attempt_ready() -> tuple[str, str] | None:
            pods = sorted(
                self.list_owned_pods(),
                key=lambda item: str(
                    item.get("metadata", {}).get("creationTimestamp") or ""
                ),
            )
            self.write_json(
                f"pods-attempt-{ordinal}-observed.json",
                {"apiVersion": "v1", "kind": "List", "items": pods},
            )
            for pod in pods:
                metadata = pod.get("metadata", {})
                uid = str(metadata.get("uid") or "")
                name = str(metadata.get("name") or "")
                if not uid or uid in excluded_uids:
                    continue
                phase = str(pod.get("status", {}).get("phase") or "")
                log = self.pod_log(name)
                (
                    self.output / f"container-attempt-{ordinal}-{name}-observed.txt"
                ).write_text(log, encoding="utf-8")
                if phase in {"Failed", "Succeeded"} and EVENT_PREFIX not in log:
                    statuses = pod.get("status", {}).get("containerStatuses", [])
                    terminated = (
                        statuses[0].get("state", {}).get("terminated", {})
                        if statuses
                        else {}
                    )
                    raise ExperimentError(
                        f"attempt {ordinal} terminated before its marker: "
                        f"phase={phase} exit={terminated.get('exitCode')}"
                    )
                if phase != "Running":
                    continue
                if EVENT_PREFIX in log and '"event":"attempt_started"' in log:
                    return name, uid
            return None

        name, uid = self.wait_for(
            f"attempt {ordinal} marker", attempt_ready, timeout=180
        )
        self.write_json(
            f"pod-attempt-{ordinal}-before-action.json",
            self.json_get(["-n", NAMESPACE, "get", "pod", name]),
        )
        self.record("attempt-running", f"ordinal={ordinal}\tpod={name}\tuid={uid}")
        return name, uid

    def inject_attempt_failure(self, name: str, uid: str, ordinal: int) -> None:
        inventory = self.kubectl(
            [
                "-n",
                NAMESPACE,
                "exec",
                name,
                "-c",
                "trainer",
                "--",
                "/bin/sh",
                "-c",
                "for p in /proc/[0-9]*/comm; do "
                "pid=${p#/proc/}; pid=${pid%/comm}; "
                'n=$(cat "$p" 2>/dev/null || true); '
                'printf \'%s\\t%s\\n\' "$pid" "$n"; done',
            ],
            check=False,
            timeout=20,
        )
        (self.output / f"container-processes-attempt-{ordinal}.txt").write_text(
            inventory.stdout + inventory.stderr, encoding="utf-8"
        )
        injected_at = utc_now()
        killed = self.kubectl(
            [
                "-n",
                NAMESPACE,
                "exec",
                name,
                "-c",
                "trainer",
                "--",
                "/bin/sh",
                "-c",
                "for p in /proc/[0-9]*/comm; do "
                "pid=${p#/proc/}; pid=${pid%/comm}; "
                'n=$(cat "$p" 2>/dev/null || true); '
                'if [ "$pid" != 1 ] && { [ "$n" = python ] || [ "$n" = python3 ]; }; then '
                'kill -KILL "$pid"; exit 0; fi; done; exit 1',
            ],
            check=False,
            timeout=20,
        )
        (self.output / f"failure-injection-attempt-{ordinal}.txt").write_text(
            f"injected_at={injected_at}\nreturncode={killed.returncode}\n"
            f"stdout={killed.stdout}\nstderr={killed.stderr}\n",
            encoding="utf-8",
        )
        if killed.returncode != 0:
            raise ExperimentError(
                f"attempt {ordinal} failure injection returned {killed.returncode}"
            )
        self.record("failure-injected", f"ordinal={ordinal}\tpod={name}\tuid={uid}")

    def wait_terminal(self) -> None:
        condition = self.terminal.lower()
        waited = self.kubectl(
            [
                "-n",
                NAMESPACE,
                "wait",
                f"--for=condition={condition}",
                f"job/{self.job_name}",
                "--timeout=240s",
            ],
            timeout=250,
        )
        (self.output / "job-wait.txt").write_text(
            waited.stdout + waited.stderr, encoding="utf-8"
        )
        self.record("job-terminal", f"condition={self.terminal}")

    @staticmethod
    def terminated_exit(pod: Mapping[str, Any]) -> int | None:
        statuses = pod.get("status", {}).get("containerStatuses", [])
        if not statuses:
            return None
        return statuses[0].get("state", {}).get("terminated", {}).get("exitCode")

    def capture_and_validate_matrix_case(
        self, attempts: list[dict[str, Any]]
    ) -> dict[str, Any]:
        job = self.json_get(["-n", NAMESPACE, "get", "job", self.job_name])
        pods = sorted(
            self.list_owned_pods(),
            key=lambda item: str(
                item.get("metadata", {}).get("creationTimestamp") or ""
            ),
        )
        self.write_json("job-final.json", job)
        self.write_json(
            "pods-final.json", {"apiVersion": "v1", "kind": "List", "items": pods}
        )
        pod_uids = {str(pod.get("metadata", {}).get("uid") or "") for pod in pods}
        expected_uids = {str(attempt["uid"]) for attempt in attempts}
        if len(pods) != len(attempts) or pod_uids != expected_uids:
            raise ExperimentError(
                f"expected attempt UIDs {expected_uids}, found {pod_uids}"
            )

        all_events = self.json_get(["-n", NAMESPACE, "get", "events"])
        target_uids = {self.job_uid, *pod_uids}
        events = [
            event
            for event in all_events.get("items", [])
            if event.get("involvedObject", {}).get("uid") in target_uids
        ]
        self.write_json(
            "events.json", {"apiVersion": "v1", "kind": "List", "items": events}
        )

        logs: dict[str, str] = {}
        pod_by_uid: dict[str, Mapping[str, Any]] = {}
        for pod in pods:
            metadata = pod.get("metadata", {})
            name = str(metadata.get("name") or "")
            uid = str(metadata.get("uid") or "")
            pod_by_uid[uid] = pod
            logs[name] = self.pod_log(name)
            (self.output / f"container-{name}.txt").write_text(
                logs[name], encoding="utf-8"
            )

        enriched_attempts: list[dict[str, Any]] = []
        success_payload: dict[str, Any] | None = None
        for attempt in attempts:
            uid = str(attempt["uid"])
            name = str(attempt["name"])
            exit_code = self.terminated_exit(pod_by_uid[uid])
            role = str(attempt["role"])
            if role == "injected-failure":
                if exit_code in (None, 0):
                    raise ExperimentError(
                        f"injected attempt {attempt['ordinal']} exit={exit_code}"
                    )
                if RESULT_PREFIX in logs[name]:
                    raise ExperimentError("injected attempt emitted a success result")
            else:
                if exit_code != 0:
                    raise ExperimentError(f"success attempt exit={exit_code}")
                success_lines = [
                    line
                    for line in logs[name].splitlines()
                    if line.startswith(RESULT_PREFIX)
                ]
                if len(success_lines) != 1:
                    raise ExperimentError("success attempt must emit one result")
                success_payload = json.loads(success_lines[0].split("=", 1)[1])
                if success_payload.get("pod_uid") != uid:
                    raise ExperimentError("success result Pod UID mismatch")
            enriched_attempts.append({**attempt, "exit_code": exit_code})

        conditions = job.get("status", {}).get("conditions", [])
        terminal_condition = next(
            (
                condition
                for condition in conditions
                if condition.get("type") == self.terminal
                and condition.get("status") == "True"
            ),
            None,
        )
        if terminal_condition is None:
            raise ExperimentError(f"Job {self.terminal} condition is missing")
        if self.terminal == "Failed":
            if terminal_condition.get("reason") != "BackoffLimitExceeded":
                raise ExperimentError(
                    f"unexpected Failed reason: {terminal_condition.get('reason')}"
                )
            if success_payload is not None:
                raise ExperimentError("failed Job unexpectedly has a success result")
        elif success_payload is None:
            raise ExperimentError("complete Job has no success result")

        creation = str(job.get("metadata", {}).get("creationTimestamp") or "")
        terminal_time = str(
            terminal_condition.get("lastTransitionTime")
            or job.get("status", {}).get("completionTime")
            or ""
        )
        api_wall = None
        if creation and terminal_time:
            api_wall = (
                parse_timestamp(terminal_time) - parse_timestamp(creation)
            ).total_seconds()
        result = {
            "schema_version": 1,
            "experiment": "kubernetes-job-backoff-limit-matrix",
            "status": "passed",
            "run_token": self.run_token,
            "job_name": self.job_name,
            "job_uid": self.job_uid,
            "backoff_limit": self.backoff_limit,
            "injected_failures": self.injected_failures,
            "expected_terminal": self.terminal,
            "terminal_reason": terminal_condition.get("reason"),
            "api_wall_clock_seconds": api_wall,
            "attempts": enriched_attempts,
            "training": success_payload,
            "owned_pod_count": len(pods),
            "captured_event_count": len(events),
            "validated_at": utc_now(),
        }
        self.write_json("result.json", result)
        self.record("validation-passed")
        return result

    def cleanup(self) -> None:
        super().cleanup()
        if not self.created:
            return
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
        (self.output / "host-gpu-processes-after.csv").write_text(
            process_check.stdout, encoding="utf-8"
        )
        if process_check.returncode != 0 or process_check.stdout.strip():
            raise ExperimentError("host GPU process remained after cleanup")

    def run(self) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=False)
        self.record(
            "experiment-start",
            f"run_token={self.run_token}\tbackoff_limit={self.backoff_limit}",
        )
        attempts: list[dict[str, Any]] = []
        excluded_uids: set[str] = set()
        try:
            self.preflight()
            self.create()
            for ordinal in range(1, self.injected_failures + 1):
                name, uid = self.wait_attempt(excluded_uids, ordinal)
                attempts.append(
                    {
                        "ordinal": ordinal,
                        "name": name,
                        "uid": uid,
                        "role": "injected-failure",
                    }
                )
                self.inject_attempt_failure(name, uid, ordinal)
                excluded_uids.add(uid)
            if self.terminal == "Complete":
                ordinal = self.injected_failures + 1
                name, uid = self.wait_attempt(excluded_uids, ordinal)
                attempts.append(
                    {
                        "ordinal": ordinal,
                        "name": name,
                        "uid": uid,
                        "role": "success",
                    }
                )
            self.wait_terminal()
            return self.capture_and_validate_matrix_case(attempts)
        except Exception as exc:
            self.write_json(
                "failure.json",
                {
                    "schema_version": 1,
                    "experiment": "kubernetes-job-backoff-limit-matrix",
                    "status": "failed",
                    "run_token": self.run_token,
                    "backoff_limit": self.backoff_limit,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failed_at": utc_now(),
                },
            )
            self.record("experiment-failed", f"{type(exc).__name__}: {str(exc)[:300]}")
            raise
        finally:
            self.cleanup()


def case_token(matrix_token: str, backoff_limit: int) -> str:
    token = f"{matrix_token}b{backoff_limit}"
    render(token, backoff_limit)
    return token


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"), nargs="?", default="plan")
    parser.add_argument("--matrix-token", default="bo260811")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = [
        {
            **case,
            "run_token": case_token(args.matrix_token, int(case["backoff_limit"])),
        }
        for case in MATRIX_CASES
    ]
    output = (
        args.output or (RESULTS_ROOT / f"backoff-matrix-{args.matrix_token}")
    ).resolve()
    try:
        output.relative_to(RESULTS_ROOT.resolve())
    except ValueError:
        print("output must remain below benchmark/results/reliability", file=sys.stderr)
        return 2
    plan = {
        "mutation": "none" if args.action == "plan" else "create bounded Jobs",
        "experiment": "kubernetes-job-backoff-limit-matrix",
        "kubeconfig": KUBECONFIG,
        "context": CONTEXT,
        "namespace": NAMESPACE,
        "gpu_per_job": 1,
        "sequential": True,
        "cases": cases,
        "output": str(output),
    }
    if args.action == "plan":
        print(json.dumps(plan, indent=2))
        return 0
    if not args.execute:
        print("run refused: add --execute", file=sys.stderr)
        return 2
    if output.exists():
        print(f"run refused: output already exists: {output}", file=sys.stderr)
        return 2

    started = utc_now()
    results: list[dict[str, Any]] = []
    try:
        for case in cases:
            runner = BackoffCaseRunner(
                str(case["run_token"]),
                output / f"backoff-{case['backoff_limit']}",
                backoff_limit=int(case["backoff_limit"]),
                injected_failures=int(case["injected_failures"]),
                terminal=str(case["terminal"]),
            )
            results.append(runner.run())
    except Exception as exc:
        print(f"matrix failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    summary = {
        "schema_version": 1,
        "experiment": "kubernetes-job-backoff-limit-matrix",
        "status": "passed",
        "matrix_token": args.matrix_token,
        "started_at": started,
        "completed_at": utc_now(),
        "case_count": len(results),
        "cases": results,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
