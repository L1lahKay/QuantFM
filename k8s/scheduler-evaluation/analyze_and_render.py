#!/usr/bin/env python3
"""Analyze raw benchmark artifacts and render terminal-style evidence images."""

from __future__ import annotations

import json
import re
import textwrap
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "docs/assets/gpu-scheduler-evaluation/raw"
CURRENT = RAW / "current"
SCREENSHOTS = ROOT / "docs/assets/gpu-scheduler-evaluation/screenshots"
RESULTS = ROOT / "docs/assets/gpu-scheduler-evaluation/bare-k8s-results.json"
KUEUE_RESULTS = ROOT / "docs/assets/gpu-scheduler-evaluation/current-kueue-results.json"
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def seconds(start: str, end: str) -> float:
    return round((parse_time(end) - parse_time(start)).total_seconds(), 3)


def log_stamp(log: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)} (\S+)$", log, re.MULTILINE)
    if not match:
        raise ValueError(f"missing timestamp: {label}")
    return match.group(1)


def clean_transcript(text: str) -> str:
    text = ANSI_RE.sub("", text).replace("\r", "")
    lines = []
    for line in text.splitlines():
        if line.startswith("Script started on") or line.startswith("Script done on"):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def section(text: str, start: str, end: str | None = None) -> str:
    start_at = text.index(start)
    end_at = text.index(end, start_at) if end else len(text)
    return text[start_at:end_at].strip()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def analyze() -> tuple[dict[str, object], dict[str, object]]:
    log = clean_transcript((RAW / "bare-k8s-benchmark.txt").read_text())
    historical_inventory = clean_transcript((RAW / "cluster-inventory.txt").read_text())
    jobs = load_json(RAW / "contention-jobs.json")["items"]
    pods = load_json(RAW / "contention-pods.json")["items"]
    preemption = load_json(RAW / "preemption-pods.json")["items"]
    quota = load_json(RAW / "quota.json")
    quota_job = load_json(RAW / "quota-job.json")

    inventory_match = re.search(
        r"^gpu-dev-01\s+True\s+(\d+)\s+nvidia\s+(\S+)$",
        historical_inventory,
        re.MULTILINE,
    )
    if not inventory_match:
        raise ValueError("historical node inventory evidence missing")
    gpu_capacity = int(inventory_match.group(1))
    kubernetes_version = inventory_match.group(2)

    jobs_by_name = {item["metadata"]["name"]: item for item in jobs}
    holder = next(
        item for item in pods if item["metadata"]["labels"]["role"] == "holder"
    )
    waiter = next(
        item for item in pods if item["metadata"]["labels"]["role"] == "waiter"
    )
    holder_gpu_request = int(
        holder["spec"]["containers"][0]["resources"]["requests"]["nvidia.com/gpu"]
    )
    waiter_gpu_request = int(
        waiter["spec"]["containers"][0]["resources"]["requests"]["nvidia.com/gpu"]
    )
    waiter_state = waiter["status"]["containerStatuses"][0]["state"]["terminated"]
    waiter_job = jobs_by_name["bare-waiter-5gpu"]
    waiter_submit = log_stamp(log, "WAITER_SUBMITTED")
    waiter_job_created = waiter_job["metadata"]["creationTimestamp"]
    waiter_pod_created = waiter["metadata"]["creationTimestamp"]
    waiter_started = waiter_state["startedAt"]
    waiter_finished = waiter_state["finishedAt"]
    waiter_complete = waiter_job["status"]["completionTime"]

    high = next(
        item
        for item in preemption
        if item["metadata"]["name"] == "bare-preempt-high-5gpu"
    )
    high_state = high["status"]["containerStatuses"][0]["state"]["terminated"]
    high_submit = log_stamp(log, "HIGH_SUBMITTED")
    high_created = high["metadata"]["creationTimestamp"]
    high_gpu_request = int(
        high["spec"]["containers"][0]["resources"]["requests"]["nvidia.com/gpu"]
    )

    quota_error = re.search(
        r"exceeded quota: bare-gpu-quota, requested: "
        r"requests\.nvidia\.com/gpu=2, used: requests\.nvidia\.com/gpu=0, "
        r"limited: requests\.nvidia\.com/gpu=1",
        log,
    )
    if not quota_error:
        raise ValueError("quota rejection evidence missing")
    if "Insufficient nvidia.com/gpu" not in log:
        raise ValueError("GPU scheduling rejection evidence missing")
    if "Preempted" not in log or high["metadata"]["uid"] not in log:
        raise ValueError("preemption victim/preemptor evidence missing")

    quota_job_gpu_request = int(
        quota_job["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"][
            "nvidia.com/gpu"
        ]
    )

    result: dict[str, object] = {
        "captured_at_utc": log_stamp(log, "BENCHMARK_END"),
        "evidence_context": "Historical privileged benchmark; not rerunnable by the current restricted account",
        "cluster": {
            "node": "gpu-dev-01",
            "gpu_capacity": gpu_capacity,
            "scheduler": "default-scheduler",
            "kubernetes": kubernetes_version,
        },
        "contention": {
            "holder_gpu_request": holder_gpu_request,
            "waiter_gpu_request": waiter_gpu_request,
            "theoretical_remaining_gpu_after_holder_request": (
                gpu_capacity - holder_gpu_request
            ),
            "global_gpu_allocation_snapshot_available": False,
            "waiter_client_apply_returned_at": waiter_submit,
            "waiter_job_created_at": waiter_job_created,
            "waiter_pod_created_at": waiter_pod_created,
            "waiter_container_started_at": waiter_started,
            "waiter_container_finished_at": waiter_finished,
            "waiter_job_completed_at": waiter_complete,
            "api_job_creation_to_container_start_seconds": seconds(
                waiter_job_created, waiter_started
            ),
            "api_pod_creation_to_container_start_seconds": seconds(
                waiter_pod_created, waiter_started
            ),
            "container_runtime_seconds": seconds(waiter_started, waiter_finished),
            "api_job_creation_to_process_finish_seconds": seconds(
                waiter_job_created, waiter_finished
            ),
            "api_job_wall_clock_seconds": seconds(waiter_job_created, waiter_complete),
            "client_observed_apply_return_to_start_seconds": seconds(
                waiter_submit, waiter_started
            ),
            "client_observed_apply_return_to_complete_seconds": seconds(
                waiter_submit, waiter_complete
            ),
            "pending_reason": "Insufficient nvidia.com/gpu",
        },
        "quota": {
            "namespace_gpu_limit": int(
                quota["status"]["hard"]["requests.nvidia.com/gpu"]
            ),
            "job_gpu_request": quota_job_gpu_request,
            "pods_created": 0,
            "enforcement": "Job FailedCreate; request was rejected rather than queued",
        },
        "preemption": {
            "victim_priority": 100,
            "preemptor_priority": 1000,
            "gpu_request_each": high_gpu_request,
            "high_client_apply_returned_at": high_submit,
            "high_pod_created_at": high_created,
            "high_container_started_at": high_state["startedAt"],
            "high_container_finished_at": high_state["finishedAt"],
            "api_pod_creation_to_start_seconds": seconds(
                high_created, high_state["startedAt"]
            ),
            "container_runtime_seconds": seconds(
                high_state["startedAt"], high_state["finishedAt"]
            ),
            "api_pod_wall_clock_seconds": seconds(
                high_created, high_state["finishedAt"]
            ),
            "client_observed_apply_return_to_start_seconds": seconds(
                high_submit, high_state["startedAt"]
            ),
            "victim_event": "Preempted",
            "victim_object_snapshot_available": False,
        },
    }
    RESULTS.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    kueue_log = clean_transcript((CURRENT / "kueue-probe.txt").read_text())
    kueue_job = load_json(CURRENT / "kueue-probe-job.json")
    kueue_pods = load_json(CURRENT / "kueue-probe-pods.json")["items"]
    kueue_events = load_json(CURRENT / "kueue-probe-events.json")["items"]
    workload_event = next(
        item for item in kueue_events if item.get("reason") == "CreatedWorkload"
    )
    client_submit = log_stamp(kueue_log, "JOB_SUBMIT_CLIENT")
    client_end = log_stamp(kueue_log, "PROBE_CLIENT_END")
    workload_created = workload_event.get("eventTime") or workload_event.get(
        "lastTimestamp"
    )
    kueue_result: dict[str, object] = {
        "captured_at_utc": client_end,
        "cluster": {
            "namespace": "gpu-dev",
            "kubernetes": "v1.35.4+k3s1",
            "kueue_api": "kueue.x-k8s.io/v1beta2",
            "kueue_controller_version": "not readable by the restricted account",
            "volcano_api_present": False,
        },
        "safe_admission_probe": {
            "queue_label_candidate": "gpu-dev",
            "job_client_submit_at": client_submit,
            "job_created_at": kueue_job["metadata"]["creationTimestamp"],
            "workload_event_at": workload_created,
            "job_suspended": kueue_job["spec"]["suspend"],
            "workload_created": True,
            "pods_created_at_snapshot": len(kueue_pods),
            "observation_wall_clock_seconds": seconds(client_submit, client_end),
            "client_submit_to_workload_event_seconds": seconds(
                client_submit, workload_created
            ),
            "admission_observed": False,
            "container_started": False,
            "job_completed": False,
            "runtime_seconds": None,
            "end_to_end_wall_clock_seconds": None,
            "result": "Kueue gating active; runnable queue/quota path not demonstrated",
        },
        "controller_catchup_diagnostic": {
            "input_explicit_suspend": False,
            "pod_created_then_deleted": True,
            "pod_bound_by_default_scheduler": True,
            "container_started": False,
            "safe_manifest_now_sets_explicit_suspend": True,
        },
        "evidence_limit": (
            "The account cannot read LocalQueue, ClusterQueue, Workload, or "
            "ResourceQuota, so the non-admission reason and quota policy are unknown."
        ),
    }
    KUEUE_RESULTS.write_text(
        json.dumps(kueue_result, indent=2, ensure_ascii=False) + "\n"
    )
    return result, kueue_result


def wrap_lines(text: str, width: int = 128) -> list[str]:
    wrapped: list[str] = []
    for line in clean_transcript(text).splitlines():
        if not line:
            wrapped.append("")
            continue
        indent = len(line) - len(line.lstrip())
        parts = textwrap.wrap(
            line,
            width=width,
            subsequent_indent=" " * min(indent + 2, 12),
            replace_whitespace=False,
            drop_whitespace=False,
        )
        wrapped.extend(parts or [""])
    return wrapped


def render_terminal(title: str, body: str, destination: Path) -> None:
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
    font = ImageFont.truetype(font_path, 18)
    bold = ImageFont.truetype(bold_path, 18)
    title_font = ImageFont.truetype(bold_path, 20)
    lines = wrap_lines(body)
    line_height = 27
    width = 1720
    height = 86 + line_height * len(lines) + 34
    image = Image.new("RGB", (width, height), "#0d1117")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 58), fill="#161b22")
    for x, color in ((25, "#ff5f56"), (53, "#ffbd2e"), (81, "#27c93f")):
        draw.ellipse((x, 20, x + 14, 34), fill=color)
    draw.text((118, 17), title, font=title_font, fill="#e6edf3")

    y = 72
    for line in lines:
        if any(
            marker in line
            for marker in ("FailedScheduling", "FailedCreate", "Preempted", "Forbidden")
        ):
            color = "#ff7b72"
            selected_font = bold
        elif any(
            marker in line
            for marker in ("Complete", "Succeeded", "VISIBLE_GPUS", "SUSPENDED_NO_POD")
        ):
            color = "#7ee787"
            selected_font = font
        elif line.startswith(
            (
                "BENCHMARK_",
                "EXPERIMENT_",
                "CONTENTTION_",
                "PREEMPTION_",
                "CAPTURED_AT",
                "KUEUE_",
                "VOLCANO_",
                "RESTRICTED_",
                "VISIBLE_",
                "DIRECT_",
            )
        ):
            color = "#79c0ff"
            selected_font = bold
        elif line.startswith("METRIC"):
            color = "#d2a8ff"
            selected_font = bold
        elif any(
            marker in line
            for marker in ("Suspended", "CreatedWorkload", " NONE", "CAN_I")
        ):
            color = "#e3b341"
            selected_font = font
        else:
            color = "#c9d1d9"
            selected_font = font
        draw.text((24, y), line, font=selected_font, fill=color)
        y += line_height

    draw.text(
        (24, height - 27),
        "Captured kubectl transcript + derived metrics; terminal-style rendering (UTC)",
        font=font,
        fill="#8b949e",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, optimize=True)


def render(result: dict[str, object], kueue_result: dict[str, object]) -> None:
    log = clean_transcript((RAW / "bare-k8s-benchmark.txt").read_text())
    historical_inventory = clean_transcript((RAW / "cluster-inventory.txt").read_text())
    current_inventory = clean_transcript((CURRENT / "cluster-access.txt").read_text())
    kueue_log = clean_transcript((CURRENT / "kueue-probe.txt").read_text())
    contention = section(log, "EXPERIMENT_CONTENTION_START", "EXPERIMENT_QUOTA_START")
    quota = section(log, "EXPERIMENT_QUOTA_START", "EXPERIMENT_PREEMPTION_START")
    preemption = section(log, "EXPERIMENT_PREEMPTION_START", "BENCHMARK_END")
    kueue_probe = section(kueue_log, "JOB_SUBMIT_CLIENT", "job.batch ")

    c = result["contention"]
    p = result["preemption"]
    k = kueue_result["safe_admission_probe"]
    contention += (
        "\nMETRIC api_job_to_start_seconds="
        + str(c["api_job_creation_to_container_start_seconds"])
        + " api_pod_to_start_seconds="
        + str(c["api_pod_creation_to_container_start_seconds"])
        + " runtime_seconds="
        + str(c["container_runtime_seconds"])
        + " api_job_wall_seconds="
        + str(c["api_job_wall_clock_seconds"])
        + " client_apply_return_to_complete_seconds="
        + str(c["client_observed_apply_return_to_complete_seconds"])
    )
    preemption += (
        "\nMETRIC api_high_pod_to_start_seconds="
        + str(p["api_pod_creation_to_start_seconds"])
        + " api_high_pod_wall_seconds="
        + str(p["api_pod_wall_clock_seconds"])
        + " runtime_seconds="
        + str(p["container_runtime_seconds"])
    )
    kueue_probe += (
        "\nMETRIC observation_wall_seconds="
        + str(k["observation_wall_clock_seconds"])
        + " client_submit_to_workload_event_seconds="
        + str(k["client_submit_to_workload_event_seconds"])
        + " pods_at_snapshot="
        + str(k["pods_created_at_snapshot"])
        + " runtime_seconds=N/A end_to_end_wall_seconds=N/A"
    )

    catchup_job_events = load_json(
        CURRENT / "kueue-controller-catchup-job-events.json"
    )["items"]
    catchup_pod_events = load_json(
        CURRENT / "kueue-controller-catchup-pod-events.json"
    )["items"]
    catchup_lines = [
        "DIAGNOSTIC_INPUT explicit_suspend=false (historical input; do not re-run)",
    ]
    all_catchup_events = sorted(
        [*catchup_job_events, *catchup_pod_events],
        # Kubernetes core Events mix second and sub-second timestamps. The API
        # server resourceVersion preserves the observed write order here.
        key=lambda item: int(item["metadata"]["resourceVersion"]),
    )
    for item in all_catchup_events:
        event_at = item.get("eventTime") or item.get("lastTimestamp")
        resource_version = item["metadata"]["resourceVersion"]
        catchup_lines.append(
            f"rv={resource_version} {event_at} {item['reason']}: {item['message']} "
            f"source={item.get('reportingComponent', 'unknown')}"
        )
    catchup_lines.extend(
        (
            "CONCLUSION the Pod was bound before controller catch-up, then deleted before container start",
            "FIX safe probe now sets spec.suspend=true; see screenshot 05 (zero Pods)",
        )
    )
    catchup = "\n".join(catchup_lines)

    render_terminal(
        "01 - Pre-Volcano restricted-account inventory (2026-07-31 06:25 UTC)",
        current_inventory,
        SCREENSHOTS / "01-current-cluster-access.png",
    )
    render_terminal(
        "01 - Pre-Volcano restricted-account inventory (2026-07-31 06:25 UTC)",
        current_inventory,
        SCREENSHOTS / "01-cluster-inventory.png",
    )
    render_terminal(
        "02 - Bare Kubernetes GPU contention and wall clock",
        contention,
        SCREENSHOTS / "02-bare-contention.png",
    )
    render_terminal(
        "03 - Bare Kubernetes ResourceQuota enforcement",
        quota,
        SCREENSHOTS / "03-bare-quota.png",
    )
    render_terminal(
        "04 - Bare Kubernetes Priority preemption",
        preemption,
        SCREENSHOTS / "04-bare-preemption.png",
    )
    render_terminal(
        "05 - Kueue admission gate (safe suspended Job)",
        kueue_probe,
        SCREENSHOTS / "05-kueue-admission-gate.png",
    )
    render_terminal(
        "06 - Kueue controller catch-up diagnostic",
        catchup,
        SCREENSHOTS / "06-kueue-controller-catchup.png",
    )
    render_terminal(
        "07 - Historical admin inventory before Kueue installation",
        historical_inventory,
        SCREENSHOTS / "07-historical-cluster-inventory.png",
    )


def main() -> None:
    result, kueue_result = analyze()
    render(result, kueue_result)
    print(
        json.dumps(
            {"bare_kubernetes": result, "current_kueue": kueue_result},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
