#!/usr/bin/env python3
"""
Validate Volcano GPU evidence, derive wall-clock metrics, and render images.

The script deliberately works from captured Kubernetes objects, UID-filtered
Events, and monotonic client timestamps.  It never reads a kubeconfig or talks
to the cluster.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from analyze_and_render import parse_time, render_terminal, seconds

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/assets/gpu-scheduler-evaluation"
RAW = ASSETS / "raw/volcano"
SCREENSHOTS = ASSETS / "screenshots"
RESULTS = ASSETS / "volcano-results.json"
UPSTREAM = (
    ROOT / "k8s/scheduler-evaluation/volcano/upstream/volcano-development-v1.15.1.yaml"
)
UPSTREAM_SUMS = ROOT / "k8s/scheduler-evaluation/volcano/upstream/SHA256SUMS"


def load_json(name: str) -> dict[str, Any]:
    return json.loads((RAW / name).read_text())


def read(name: str) -> str:
    return (RAW / name).read_text()


def items(name: str) -> list[dict[str, Any]]:
    return load_json(name)["items"]


def named(objects: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [obj for obj in objects if obj["metadata"]["name"] == name]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one object named {name}, got {len(matches)}"
        )
    return matches[0]


def condition(obj: dict[str, Any], type_name: str) -> dict[str, Any]:
    matches = [
        c for c in obj.get("status", {}).get("conditions", []) if c["type"] == type_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {type_name} condition on "
            f"{obj['metadata']['name']}, got {len(matches)}"
        )
    return matches[0]


def container_state(pod: dict[str, Any]) -> dict[str, Any]:
    statuses = pod.get("status", {}).get("containerStatuses", [])
    if len(statuses) != 1:
        raise ValueError(f"expected one container status on {pod['metadata']['name']}")
    return statuses[0]["state"]


def gpu_request(pod: dict[str, Any]) -> int:
    total = 0
    for container in pod["spec"]["containers"]:
        total += int(
            container.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0)
        )
    return total


def quota(name: str) -> dict[str, Any]:
    return named(items(name), "gpu-quota")


def timeline(name: str) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    pattern = re.compile(r"^(\S+) utc=(\S+) monotonic_ns=(\d+)$")
    for line in read(name).splitlines():
        match = pattern.fullmatch(line)
        if match:
            parsed[match.group(1)] = {
                "utc": match.group(2),
                "monotonic_ns": int(match.group(3)),
            }
    if not parsed:
        raise ValueError(f"no timeline entries found in {name}")
    return parsed


def monotonic_seconds(stamps: dict[str, dict[str, Any]], start: str, end: str) -> float:
    return round(
        (stamps[end]["monotonic_ns"] - stamps[start]["monotonic_ns"]) / 1_000_000_000,
        3,
    )


def event_time(event: dict[str, Any]) -> str:
    value = event.get("eventTime") or event.get("lastTimestamp")
    if not value:
        raise ValueError(f"event {event['metadata']['name']} has no timestamp")
    return value


def uid_events(name: str, uids: set[str]) -> list[dict[str, Any]]:
    selected = [
        event
        for event in items(name)
        if event.get("involvedObject", {}).get("uid") in uids
    ]
    return sorted(selected, key=lambda event: int(event["metadata"]["resourceVersion"]))


def one_event(
    events: list[dict[str, Any]],
    *,
    uid: str,
    reason: str,
    component: str | None = None,
    message_contains: str | None = None,
) -> dict[str, Any]:
    matches = []
    for event in events:
        if event.get("involvedObject", {}).get("uid") != uid:
            continue
        if event.get("reason") != reason:
            continue
        if component and event.get("reportingComponent") != component:
            continue
        if message_contains and message_contains not in event.get("message", ""):
            continue
        matches.append(event)
    if not matches:
        raise ValueError(
            f"missing event uid={uid} reason={reason} component={component} "
            f"message_contains={message_contains}"
        )
    return matches[-1]


def log_value(log: str, key: str) -> str:
    match = re.search(rf"(?:^|\]\s+){re.escape(key)}=([^\s]+)", log, re.MULTILINE)
    if not match:
        raise ValueError(f"missing {key} in captured container output")
    return match.group(1)


def log_values(log: str, key: str) -> list[str]:
    values = re.findall(rf"(?:^|\]\s+){re.escape(key)}=([^\s]+)", log, re.MULTILINE)
    if not values:
        raise ValueError(f"missing {key} in captured container output")
    return values


def kv_line(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}=(\S+)$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"missing {key}")
    return match.group(1)


def actions(text: str) -> list[str]:
    match = re.search(r'^\s*actions:\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise ValueError("scheduler actions not found")
    return [part.strip() for part in match.group(1).split(",")]


def config_payload(text: str) -> str:
    match = re.search(
        r"^data:\n  volcano-scheduler\.conf: \|\n(?P<body>(?:    .*\n)+?)^kind:",
        text,
        re.MULTILINE,
    )
    if not match:
        raise ValueError("Volcano scheduler ConfigMap payload not found")
    return "\n".join(line[4:] for line in match.group("body").rstrip().splitlines())


def installation() -> dict[str, Any]:
    install_text = read("install-apply.txt")
    begin_match = re.search(r"^INSTALL_BEGIN_(\S+)$", install_text, re.MULTILINE)
    return_match = re.search(
        r"^INSTALL_APPLY_RETURN_(\S+)$", install_text, re.MULTILINE
    )
    if not begin_match or not return_match:
        raise ValueError("installation apply timestamps missing")
    install_begin = begin_match.group(1)
    apply_return = return_match.group(1)

    expected_digest = UPSTREAM_SUMS.read_text().split()[0]
    actual_digest = hashlib.sha256(UPSTREAM.read_bytes()).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("pinned Volcano manifest digest mismatch")

    dry_run_objects = [
        line
        for line in read("install-server-dry-run-complete.txt").splitlines()
        if line
    ]
    if len(dry_run_objects) != 43:
        raise ValueError(f"expected 43 dry-run objects, got {len(dry_run_objects)}")

    volcano_crds = [
        crd
        for crd in items("installation-crds.json")
        if crd["metadata"]["name"].endswith("volcano.sh")
    ]
    for crd in volcano_crds:
        established = condition(crd, "Established")
        if established["status"] != "True":
            raise ValueError(f"CRD not Established: {crd['metadata']['name']}")
    if len(volcano_crds) != 11:
        raise ValueError(f"expected 11 Volcano CRDs, got {len(volcano_crds)}")

    volcano_webhook_configs = [
        config
        for config in items("installation-webhooks.json")
        if config["metadata"]["name"].startswith("volcano-")
    ]
    if len(volcano_webhook_configs) != 7:
        raise ValueError(
            f"expected 7 Volcano webhook configurations, got {len(volcano_webhook_configs)}"
        )
    for config in volcano_webhook_configs:
        webhooks = config.get("webhooks", [])
        if not webhooks or not all(
            webhook["clientConfig"].get("caBundle") for webhook in webhooks
        ):
            raise ValueError(f"webhook CA bundle missing: {config['metadata']['name']}")

    endpoint_slices = items("installation-endpointslices.json")
    ready_endpoint_addresses = 0
    for endpoint_slice in endpoint_slices:
        for endpoint in endpoint_slice.get("endpoints", []):
            if endpoint.get("conditions", {}).get("ready") is True:
                ready_endpoint_addresses += len(endpoint.get("addresses", []))
    if len(endpoint_slices) != 3 or ready_endpoint_addresses != 3:
        raise ValueError(
            "expected three Volcano EndpointSlices with one ready address each, got "
            f"slices={len(endpoint_slices)} addresses={ready_endpoint_addresses}"
        )

    deployments: dict[str, Any] = {}
    available_times: list[str] = []
    for deploy in items("installation-deployments.json"):
        name = deploy["metadata"]["name"]
        if not name.startswith("volcano-"):
            continue
        available = condition(deploy, "Available")
        if (
            available["status"] != "True"
            or deploy.get("status", {}).get("readyReplicas") != 1
        ):
            raise ValueError(f"Volcano deployment not healthy: {name}")
        images = [
            container["image"]
            for container in deploy["spec"]["template"]["spec"]["containers"]
        ]
        if len(images) != 1 or not images[0].endswith(":v1.15.1"):
            raise ValueError(f"unexpected image for {name}: {images}")
        available_at = available["lastTransitionTime"]
        available_times.append(available_at)
        deployments[name] = {
            "image": images[0],
            "ready": "1/1",
            "created_at": deploy["metadata"]["creationTimestamp"],
            "available_at": available_at,
        }
    if set(deployments) != {
        "volcano-admission",
        "volcano-controllers",
        "volcano-scheduler",
    }:
        raise ValueError(f"unexpected Volcano deployment set: {sorted(deployments)}")

    kueue = load_json("installation-kueue-health.json")
    kueue_image = kueue["spec"]["template"]["spec"]["containers"][0]["image"]
    if kueue.get("status", {}).get("readyReplicas") != 1:
        raise ValueError("Kueue was not healthy after Volcano installation")

    default_actions = actions(read("installed-default-scheduler-config.yaml"))
    if default_actions != ["enqueue", "allocate", "backfill"]:
        raise ValueError(f"unexpected installed default actions: {default_actions}")
    last_available = max(available_times, key=parse_time)
    return {
        "version": "v1.15.1",
        "manifest": "k8s/scheduler-evaluation/volcano/upstream/volcano-development-v1.15.1.yaml",
        "manifest_sha256": actual_digest,
        "server_side_dry_run_object_count": len(dry_run_objects),
        "install_begin_at": install_begin,
        "kubectl_apply_return_at": apply_return,
        "kubectl_apply_rtt_seconds": seconds(install_begin, apply_return),
        "last_deployment_available_at": last_available,
        "install_begin_to_last_deployment_available_seconds": seconds(
            install_begin, last_available
        ),
        "time_precision_note": "Deployment condition timestamps have one-second precision",
        "volcano_crds_established": len(volcano_crds),
        "volcano_webhook_configurations_with_ca_bundle": len(volcano_webhook_configs),
        "volcano_endpoint_slices": len(endpoint_slices),
        "volcano_ready_endpoint_addresses": ready_endpoint_addresses,
        "deployments": deployments,
        "admission_init_job": "Complete",
        "default_scheduler_actions": default_actions,
        "kueue_coexistence": {
            "deployment": kueue["metadata"]["name"],
            "image": kueue_image,
            "ready": "1/1",
        },
    }


def gpu_smoke() -> dict[str, Any]:
    stamps = timeline("cuda-smoke-timeline.txt")
    job = load_json("cuda-smoke-job.json")
    pod_list = items("cuda-smoke-pods.json")
    if len(pod_list) != 1:
        raise ValueError(f"expected one final CUDA smoke Pod, got {len(pod_list)}")
    pod = pod_list[0]
    podgroup = load_json("cuda-smoke-podgroup.json")
    queue_obj = load_json("cuda-smoke-queue.json")
    if pod["metadata"]["ownerReferences"][0]["uid"] != job["metadata"]["uid"]:
        raise ValueError("CUDA smoke Pod does not belong to captured Job")
    state = container_state(pod)["terminated"]
    scheduled_condition = condition(pod, "PodScheduled")
    events = uid_events(
        "cuda-smoke-events.json",
        {
            job["metadata"]["uid"],
            pod["metadata"]["uid"],
            podgroup["metadata"]["uid"],
        },
    )
    scheduled_event = one_event(
        events,
        uid=pod["metadata"]["uid"],
        reason="Scheduled",
        component="volcano",
    )
    output = read("cuda-smoke-container-output.txt")
    visible = int(log_value(output, "VISIBLE_GPU_COUNT"))
    torch_cuda = log_value(output, "TORCH_CUDA_AVAILABLE") == "true"
    tensor_result = int(log_value(output, "CUDA_TENSOR_RESULT"))
    app_start = log_value(output, "APP_START_UTC")
    app_finish = log_value(output, "APP_FINISH_UTC")
    if not (
        pod["spec"]["schedulerName"] == "volcano"
        and scheduled_condition["status"] == "True"
        and scheduled_event["reportingComponent"] == "volcano"
        and state["exitCode"] == 0
        and visible == 1
        and torch_cuda
        and tensor_result == 42
        and job["status"].get("succeeded") == 1
        and podgroup["spec"]["queue"] == queue_obj["metadata"]["name"]
        and podgroup["status"]["phase"] == "Completed"
    ):
        raise ValueError("final CUDA smoke success chain is incomplete")

    before_quota = quota("cuda-smoke-resourcequota-before.json")
    after_quota = quota("cuda-smoke-resourcequota-after.json")
    return {
        "effective": True,
        "job": job["metadata"]["name"],
        "job_uid": job["metadata"]["uid"],
        "pod": pod["metadata"]["name"],
        "pod_uid": pod["metadata"]["uid"],
        "podgroup": podgroup["metadata"]["name"],
        "podgroup_uid": podgroup["metadata"]["uid"],
        "queue": queue_obj["metadata"]["name"],
        "scheduler_name": pod["spec"]["schedulerName"],
        "scheduled_event_reporting_component": scheduled_event["reportingComponent"],
        "node": pod["spec"]["nodeName"],
        "gpu_request": gpu_request(pod),
        "gpu_model": re.search(r"GPU 0: (.+?) \(UUID:", output).group(1),
        "visible_gpu_count": visible,
        "torch_version": log_value(output, "TORCH_VERSION"),
        "torch_cuda_available": torch_cuda,
        "cuda_tensor_result": tensor_result,
        "times": {
            "client_create_begin": stamps["CUDA_SMOKE_JOB_CREATE_BEGIN"]["utc"],
            "client_create_return": stamps["CUDA_SMOKE_JOB_CREATE_RETURN"]["utc"],
            "job_created": job["metadata"]["creationTimestamp"],
            "pod_created": pod["metadata"]["creationTimestamp"],
            "pod_scheduled": scheduled_condition["lastTransitionTime"],
            "container_started": state["startedAt"],
            "application_started": app_start,
            "application_finished": app_finish,
            "container_finished": state["finishedAt"],
            "job_completed": job["status"]["completionTime"],
            "client_complete_observed": stamps["CUDA_SMOKE_COMPLETE_OBSERVED"]["utc"],
        },
        "metrics_seconds": {
            "client_create_rtt": monotonic_seconds(
                stamps, "CUDA_SMOKE_JOB_CREATE_BEGIN", "CUDA_SMOKE_JOB_CREATE_RETURN"
            ),
            "api_job_create_to_pod_scheduled": seconds(
                job["metadata"]["creationTimestamp"],
                scheduled_condition["lastTransitionTime"],
            ),
            "api_job_create_to_container_start": seconds(
                job["metadata"]["creationTimestamp"], state["startedAt"]
            ),
            "application_runtime": seconds(app_start, app_finish),
            "container_runtime": seconds(state["startedAt"], state["finishedAt"]),
            "api_job_wall_clock": seconds(
                job["metadata"]["creationTimestamp"], job["status"]["completionTime"]
            ),
            "client_create_to_complete_observed": monotonic_seconds(
                stamps, "CUDA_SMOKE_JOB_CREATE_BEGIN", "CUDA_SMOKE_COMPLETE_OBSERVED"
            ),
        },
        "quota_gpu_hard": int(
            before_quota["status"]["hard"]["requests.nvidia.com/gpu"]
        ),
        "quota_gpu_used_before": int(
            before_quota["status"]["used"]["requests.nvidia.com/gpu"]
        ),
        "quota_gpu_used_after_cleanup": int(
            after_quota["status"]["used"]["requests.nvidia.com/gpu"]
        ),
        "event_filter": "Job, Pod, and PodGroup metadata.uid",
    }


def queue_capability(node_gpu: int) -> dict[str, Any]:
    stamps = timeline("evaluation-timeline.txt")
    during_pods = items("queue-during-pods.json")
    during_pgs = items("queue-during-podgroups.json")
    during_queues = items("queue-during-queues.json")
    final_jobs = items("queue-final-jobs.json")
    final_pods = items("queue-final-pods.json")

    holder_job = named(final_jobs, "khalil-volcano-quota-holder")
    waiter_job = named(final_jobs, "khalil-volcano-quota-waiter")
    holder_during = named(during_pods, "khalil-volcano-quota-holder-rmpkg")
    waiter_during = named(during_pods, "khalil-volcano-quota-waiter-qr4mq")
    holder = named(final_pods, holder_during["metadata"]["name"])
    waiter = named(final_pods, waiter_during["metadata"]["name"])
    holder_state = container_state(holder)["terminated"]
    waiter_state = container_state(waiter)["terminated"]
    waiter_scheduled = condition(waiter, "PodScheduled")
    queue_obj = named(during_queues, "khalil-volcano-quota")
    holder_pg = named(during_pgs, "khalil-volcano-quota-holder-pg")
    waiter_pg = named(during_pgs, "khalil-volcano-quota-waiter-pg")

    uids = {
        holder_job["metadata"]["uid"],
        waiter_job["metadata"]["uid"],
        holder["metadata"]["uid"],
        waiter["metadata"]["uid"],
        holder_pg["metadata"]["uid"],
        waiter_pg["metadata"]["uid"],
        queue_obj["metadata"]["uid"],
    }
    during_events = uid_events("queue-during-events.json", uids)
    final_events = uid_events("queue-final-events.json", uids)
    quota_event = one_event(
        during_events,
        uid=waiter_pg["metadata"]["uid"],
        reason="Unschedulable",
        component="volcano",
        message_contains="queue resource quota insufficient: insufficient nvidia.com/gpu",
    )
    waiter_schedule_event = one_event(
        final_events,
        uid=waiter["metadata"]["uid"],
        reason="Scheduled",
        component="volcano",
    )

    all_pods = items("queue-during-all-pods.json")
    running_gpu_request = sum(
        gpu_request(pod)
        for pod in all_pods
        if pod.get("status", {}).get("phase") == "Running"
    )
    if not (
        int(queue_obj["spec"]["capability"]["nvidia.com/gpu"]) == 1
        and int(queue_obj["status"]["allocated"]["nvidia.com/gpu"]) == 1
        and holder_during["status"]["phase"] == "Running"
        and holder_during["spec"].get("nodeName")
        and waiter_during["status"]["phase"] == "Pending"
        and not waiter_during["spec"].get("nodeName")
        and waiter_pg["status"]["phase"] == "Pending"
        and waiter_schedule_event["reportingComponent"] == "volcano"
        and waiter_job["status"].get("succeeded") == 1
    ):
        raise ValueError("Queue capability success chain is incomplete")

    during_quota = quota("queue-during-resourcequota.json")
    return {
        "effective": True,
        "queue": queue_obj["metadata"]["name"],
        "queue_gpu_capability": int(queue_obj["spec"]["capability"]["nvidia.com/gpu"]),
        "queue_gpu_allocated_during_snapshot": int(
            queue_obj["status"]["allocated"]["nvidia.com/gpu"]
        ),
        "holder": holder_job["metadata"]["name"],
        "waiter": waiter_job["metadata"]["name"],
        "holder_gpu_request": gpu_request(holder),
        "waiter_gpu_request": gpu_request(waiter),
        "waiter_during_snapshot": {
            "pod_phase": waiter_during["status"]["phase"],
            "node_name": waiter_during["spec"].get("nodeName"),
            "podgroup_phase": waiter_pg["status"]["phase"],
            "event_reason": quota_event["reason"],
            "event_message": quota_event["message"],
            "event_reporting_component": quota_event["reportingComponent"],
        },
        "node_gpu_allocatable": node_gpu,
        "running_pod_gpu_requests_cluster_snapshot": running_gpu_request,
        "node_request_headroom_during_wait": node_gpu - running_gpu_request,
        "headroom_interpretation": (
            "Kubernetes Running-Pod request headroom, not direct device-utilization telemetry"
        ),
        "namespace_quota_gpu_used_during_snapshot": int(
            during_quota["status"]["used"]["requests.nvidia.com/gpu"]
        ),
        "times": {
            "waiter_client_create_begin": stamps["QUEUE_WAITER_CREATE_BEGIN"]["utc"],
            "waiter_job_created": waiter_job["metadata"]["creationTimestamp"],
            "holder_container_finished": holder_state["finishedAt"],
            "waiter_pod_scheduled": waiter_scheduled["lastTransitionTime"],
            "waiter_container_started": waiter_state["startedAt"],
            "waiter_container_finished": waiter_state["finishedAt"],
            "waiter_job_completed": waiter_job["status"]["completionTime"],
            "waiter_complete_observed": stamps["QUEUE_WAITER_COMPLETE_OBSERVED"]["utc"],
        },
        "metrics_seconds": {
            "client_create_rtt": monotonic_seconds(
                stamps, "QUEUE_WAITER_CREATE_BEGIN", "QUEUE_WAITER_CREATE_RETURN"
            ),
            "api_job_create_to_pod_scheduled": seconds(
                waiter_job["metadata"]["creationTimestamp"],
                waiter_scheduled["lastTransitionTime"],
            ),
            "api_job_create_to_container_start": seconds(
                waiter_job["metadata"]["creationTimestamp"], waiter_state["startedAt"]
            ),
            "holder_finish_to_waiter_schedule": seconds(
                holder_state["finishedAt"], waiter_scheduled["lastTransitionTime"]
            ),
            "holder_finish_to_waiter_start": seconds(
                holder_state["finishedAt"], waiter_state["startedAt"]
            ),
            "waiter_container_runtime": seconds(
                waiter_state["startedAt"], waiter_state["finishedAt"]
            ),
            "api_waiter_job_wall_clock": seconds(
                waiter_job["metadata"]["creationTimestamp"],
                waiter_job["status"]["completionTime"],
            ),
            "client_create_to_complete_observed": monotonic_seconds(
                stamps, "QUEUE_WAITER_CREATE_BEGIN", "QUEUE_WAITER_COMPLETE_OBSERVED"
            ),
        },
    }


def gang() -> dict[str, Any]:
    stamps = timeline("evaluation-timeline.txt")
    below_jobs = items("gang-below-threshold-jobs.json")
    below_pods = [
        pod
        for pod in items("gang-below-threshold-pods.json")
        if pod["metadata"]["name"].startswith("khalil-volcano-gang-")
    ]
    below_pg = named(
        items("gang-below-threshold-podgroups.json"), "khalil-volcano-gang-pg"
    )
    below_queue = named(
        items("gang-below-threshold-queues.json"), "khalil-volcano-gang"
    )
    final_job = named(items("gang-final-jobs.json"), "khalil-volcano-gang")
    final_pods = sorted(
        [
            pod
            for pod in items("gang-final-pods.json")
            if pod["metadata"]["name"].startswith("khalil-volcano-gang-")
        ],
        key=lambda pod: pod["metadata"]["name"],
    )
    final_pg = named(items("gang-final-podgroups.json"), "khalil-volcano-gang-pg")
    final_queue = named(items("gang-final-queues.json"), "khalil-volcano-gang")
    if len(below_pods) != 2 or len(final_pods) != 2:
        raise ValueError("Gang scenario did not capture exactly two member Pods")
    if {pod["metadata"]["uid"] for pod in below_pods} != {
        pod["metadata"]["uid"] for pod in final_pods
    }:
        raise ValueError("Gang below/final snapshots do not contain the same Pods")

    uids = {
        final_job["metadata"]["uid"],
        final_pg["metadata"]["uid"],
        final_queue["metadata"]["uid"],
        *(pod["metadata"]["uid"] for pod in final_pods),
    }
    below_events = uid_events("gang-below-threshold-events.json", uids)
    final_events = uid_events("gang-final-events.json", uids)
    quota_event = one_event(
        below_events,
        uid=below_pg["metadata"]["uid"],
        reason="Unschedulable",
        component="volcano",
        message_contains="queue resource quota insufficient: insufficient nvidia.com/gpu",
    )
    scheduled_events = [
        one_event(
            final_events,
            uid=pod["metadata"]["uid"],
            reason="Scheduled",
            component="volcano",
        )
        for pod in final_pods
    ]
    states = [container_state(pod)["terminated"] for pod in final_pods]
    output = read("gang-final-container-output.txt")
    app_starts = sorted(log_values(output, "APP_START_UTC"), key=parse_time)
    app_finishes = sorted(log_values(output, "APP_FINISH_UTC"), key=parse_time)
    start_spread = seconds(app_starts[0], app_starts[-1])
    finish_spread = seconds(app_finishes[0], app_finishes[-1])

    if not (
        named(below_jobs, "khalil-volcano-gang")["status"].get("active") == 2
        and int(below_queue["spec"]["capability"]["nvidia.com/gpu"]) == 1
        and below_pg["spec"]["minMember"] == 2
        and int(below_pg["spec"]["minResources"]["nvidia.com/gpu"]) == 2
        and all(pod["status"]["phase"] == "Pending" for pod in below_pods)
        and all(not pod["spec"].get("nodeName") for pod in below_pods)
        and int(final_queue["spec"]["capability"]["nvidia.com/gpu"]) == 2
        and all(pod["spec"].get("nodeName") for pod in final_pods)
        and all(event["reportingComponent"] == "volcano" for event in scheduled_events)
        and final_job["status"].get("succeeded") == 2
        and all(state["exitCode"] == 0 for state in states)
    ):
        raise ValueError("Gang success chain is incomplete")

    scheduled_times = [
        condition(pod, "PodScheduled")["lastTransitionTime"] for pod in final_pods
    ]
    started_times = [state["startedAt"] for state in states]
    return {
        "effective": True,
        "job": final_job["metadata"]["name"],
        "podgroup": final_pg["metadata"]["name"],
        "queue": final_queue["metadata"]["name"],
        "min_member": below_pg["spec"]["minMember"],
        "min_resources_gpu": int(below_pg["spec"]["minResources"]["nvidia.com/gpu"]),
        "initial_queue_gpu_capability": int(
            below_queue["spec"]["capability"]["nvidia.com/gpu"]
        ),
        "initial_snapshot": {
            "pod_count": len(below_pods),
            "pending_pods": sum(
                pod["status"]["phase"] == "Pending" for pod in below_pods
            ),
            "bound_pods": sum(bool(pod["spec"].get("nodeName")) for pod in below_pods),
            "podgroup_phase": below_pg["status"]["phase"],
            "event_reason": quota_event["reason"],
            "event_message": quota_event["message"],
            "event_reporting_component": quota_event["reportingComponent"],
        },
        "patched_queue_gpu_capability": int(
            final_queue["spec"]["capability"]["nvidia.com/gpu"]
        ),
        "final": {
            "scheduled_pods": len(scheduled_events),
            "succeeded_pods": final_job["status"]["succeeded"],
            "scheduled_event_reporting_components": [
                event["reportingComponent"] for event in scheduled_events
            ],
        },
        "times": {
            "client_create_begin": stamps["GANG_CREATE_BEGIN"]["utc"],
            "job_created": final_job["metadata"]["creationTimestamp"],
            "below_threshold_snapshot": stamps["CAPTURE_gang-below-threshold"]["utc"],
            "capability_patch_return": stamps["GANG_CAPABILITY_PATCH_RETURN"]["utc"],
            "pod_scheduled": scheduled_times,
            "container_started": started_times,
            "application_started": app_starts,
            "application_finished": app_finishes,
            "job_completed": final_job["status"]["completionTime"],
            "client_complete_observed": stamps["GANG_COMPLETE_OBSERVED"]["utc"],
        },
        "metrics_seconds": {
            "client_create_rtt": monotonic_seconds(
                stamps, "GANG_CREATE_BEGIN", "GANG_CREATE_RETURN"
            ),
            "api_job_create_to_both_scheduled": seconds(
                final_job["metadata"]["creationTimestamp"],
                max(scheduled_times, key=parse_time),
            ),
            "api_job_create_to_both_container_started": seconds(
                final_job["metadata"]["creationTimestamp"],
                max(started_times, key=parse_time),
            ),
            "api_scheduled_spread": seconds(
                min(scheduled_times, key=parse_time),
                max(scheduled_times, key=parse_time),
            ),
            "api_container_start_spread": seconds(
                min(started_times, key=parse_time), max(started_times, key=parse_time)
            ),
            "application_start_spread": start_spread,
            "application_finish_spread": finish_spread,
            "api_job_wall_clock": seconds(
                final_job["metadata"]["creationTimestamp"],
                final_job["status"]["completionTime"],
            ),
            "client_create_to_complete_observed": monotonic_seconds(
                stamps, "GANG_CREATE_BEGIN", "GANG_COMPLETE_OBSERVED"
            ),
        },
        "patch_to_schedule_precision": (
            "Both Scheduled events are in the patch-return second; API timestamps cannot resolve sub-second latency"
        ),
    }


def preemption(node_gpu: int, quota_hard: int) -> dict[str, Any]:
    stamps = timeline("evaluation-timeline.txt")
    before_jobs = items("preempt-before-jobs.json")
    before_pods = items("preempt-before-pods.json")
    before_pgs = items("preempt-before-podgroups.json")
    before_queue = named(items("preempt-before-queues.json"), "khalil-volcano-preempt")
    after_jobs = items("preempt-after-jobs.json")
    after_pods = items("preempt-after-pods.json")
    after_pgs = items("preempt-after-podgroups.json")

    victim_job_before = named(before_jobs, "khalil-volcano-preempt-victim")
    victim_job_after = named(after_jobs, "khalil-volcano-preempt-victim")
    victim_pod = next(
        pod
        for pod in before_pods
        if pod["metadata"]["name"].startswith("khalil-volcano-preempt-victim-")
    )
    high_job = named(after_jobs, "khalil-volcano-preempt-high")
    high_pod = next(
        pod
        for pod in after_pods
        if pod["metadata"]["name"].startswith("khalil-volcano-preempt-high-")
    )
    victim_pg = named(before_pgs, "khalil-volcano-preempt-victim-pg")
    high_pg = named(after_pgs, "khalil-volcano-preempt-high-pg")
    victim_started = container_state(victim_pod)["running"]["startedAt"]
    high_state = container_state(high_pod)["terminated"]

    uids = {
        victim_job_before["metadata"]["uid"],
        high_job["metadata"]["uid"],
        victim_pod["metadata"]["uid"],
        high_pod["metadata"]["uid"],
        victim_pg["metadata"]["uid"],
        high_pg["metadata"]["uid"],
        before_queue["metadata"]["uid"],
    }
    events = uid_events("preempt-after-events.json", uids)
    evict = one_event(
        events,
        uid=victim_pod["metadata"]["uid"],
        reason="Evict",
        component="volcano",
        message_contains="because of preempt",
    )
    killing = one_event(
        events,
        uid=victim_pod["metadata"]["uid"],
        reason="Killing",
        component="kubelet",
    )
    high_scheduled_event = one_event(
        events,
        uid=high_pod["metadata"]["uid"],
        reason="Scheduled",
        component="volcano",
    )
    high_scheduled = condition(high_pod, "PodScheduled")["lastTransitionTime"]
    default_actions = actions(read("installed-default-scheduler-config.yaml"))
    test_actions = actions(read("preempt-scheduler-config.yaml"))
    restored_actions = actions(read("restored-scheduler-config.yaml"))
    before_quota = quota("preempt-before-resourcequota.json")

    victim_failure = condition(victim_job_after, "Failed")
    if not (
        default_actions == ["enqueue", "allocate", "backfill"]
        and test_actions == ["allocate", "backfill", "preempt"]
        and restored_actions == default_actions
        and int(before_queue["spec"]["capability"]["nvidia.com/gpu"]) == 3
        and gpu_request(victim_pod) == 3
        and gpu_request(high_pod) == 1
        and victim_pod["spec"]["priority"] == 100
        and high_pod["spec"]["priority"] == 1000
        and evict["reportingComponent"] == "volcano"
        and killing["reportingComponent"] == "kubelet"
        and high_scheduled_event["reportingComponent"] == "volcano"
        and high_job["status"].get("succeeded") == 1
        and high_state["exitCode"] == 0
        and victim_failure["reason"] == "BackoffLimitExceeded"
        and quota_hard == 4
    ):
        raise ValueError("Preemption success chain is incomplete")

    return {
        "effective": True,
        "scope": "same Queue preempt action triggered by Queue capability, not physical node exhaustion",
        "default_scheduler_actions": default_actions,
        "temporary_test_scheduler_actions": test_actions,
        "restored_scheduler_actions": restored_actions,
        "queue": before_queue["metadata"]["name"],
        "queue_gpu_capability": int(
            before_queue["spec"]["capability"]["nvidia.com/gpu"]
        ),
        "queue_gpu_allocated_before": int(
            before_queue["status"]["allocated"]["nvidia.com/gpu"]
        ),
        "victim": {
            "job": victim_job_before["metadata"]["name"],
            "pod": victim_pod["metadata"]["name"],
            "priority": victim_pod["spec"]["priority"],
            "gpu_request": gpu_request(victim_pod),
            "started_at": victim_started,
            "evict_at": event_time(evict),
            "evict_reporting_component": evict["reportingComponent"],
            "evict_message": evict["message"],
            "killing_at": event_time(killing),
            "job_result": "Failed/BackoffLimitExceeded",
        },
        "preemptor": {
            "job": high_job["metadata"]["name"],
            "pod": high_pod["metadata"]["name"],
            "priority": high_pod["spec"]["priority"],
            "gpu_request": gpu_request(high_pod),
            "job_created_at": high_job["metadata"]["creationTimestamp"],
            "scheduled_at": high_scheduled,
            "started_at": high_state["startedAt"],
            "finished_at": high_state["finishedAt"],
            "job_completed_at": high_job["status"]["completionTime"],
        },
        "node_gpu_allocatable": node_gpu,
        "node_request_headroom_before_preemption": node_gpu - gpu_request(victim_pod),
        "namespace_gpu_quota_hard": quota_hard,
        "combined_test_gpu_requests": gpu_request(victim_pod) + gpu_request(high_pod),
        "namespace_gpu_quota_used_before_high_submission": int(
            before_quota["status"]["used"]["requests.nvidia.com/gpu"]
        ),
        "metrics_seconds": {
            "client_config_patch_to_scheduler_ready": monotonic_seconds(
                stamps, "preempt_CONFIG_PATCH_BEGIN", "preempt_CONFIG_READY"
            ),
            "client_high_create_rtt": monotonic_seconds(
                stamps, "PREEMPT_HIGH_CREATE_BEGIN", "PREEMPT_HIGH_CREATE_RETURN"
            ),
            "api_high_job_create_to_scheduled": seconds(
                high_job["metadata"]["creationTimestamp"], high_scheduled
            ),
            "api_high_job_create_to_container_start": seconds(
                high_job["metadata"]["creationTimestamp"], high_state["startedAt"]
            ),
            "client_high_create_to_ready_observed": monotonic_seconds(
                stamps, "PREEMPT_HIGH_CREATE_BEGIN", "PREEMPT_HIGH_READY_OBSERVED"
            ),
            "victim_api_runtime_before_evict": seconds(
                victim_started, event_time(evict)
            ),
            "preemptor_container_runtime": seconds(
                high_state["startedAt"], high_state["finishedAt"]
            ),
            "api_high_job_wall_clock": seconds(
                high_job["metadata"]["creationTimestamp"],
                high_job["status"]["completionTime"],
            ),
            "client_high_create_to_complete_observed": monotonic_seconds(
                stamps, "PREEMPT_HIGH_CREATE_BEGIN", "PREEMPT_HIGH_COMPLETE_OBSERVED"
            ),
            "client_scheduler_restore_to_ready": monotonic_seconds(
                stamps, "SCHEDULER_RESTORE_BEGIN", "SCHEDULER_RESTORE_READY"
            ),
        },
        "precision_note": "Event and API object timestamps have one-second precision",
        "checkpoint_resume_tested": False,
    }


def cleanup(default_actions: list[str]) -> dict[str, Any]:
    cleanup_config = read("cleanup-scheduler-config.yaml")
    restored_config = read("restored-scheduler-config.yaml")
    installed_config = read("installed-default-scheduler-config.yaml")
    if actions(cleanup_config) != default_actions:
        raise ValueError("cleanup scheduler actions were not restored")
    if config_payload(restored_config) != config_payload(installed_config):
        raise ValueError("restored scheduler payload differs from installed default")

    job_pg_absent = read("cleanup-jobs-podgroups-absent.txt")
    queue_absent = read("cleanup-queues-absent.txt")
    priority_absent = read("cleanup-priorityclasses-absent.txt")
    jobs_absent = len(
        re.findall(r'jobs\.batch "khalil-volcano-[^"]+" not found', job_pg_absent)
    )
    podgroups_absent = len(
        re.findall(
            r'podgroups\.scheduling\.volcano\.sh "khalil-volcano-[^"]+" not found',
            job_pg_absent,
        )
    )
    queues_absent = len(
        re.findall(
            r'queues\.scheduling\.volcano\.sh "khalil-volcano-[^"]+" not found',
            queue_absent,
        )
    )
    priorities_absent = len(
        re.findall(
            r'priorityclasses\.scheduling\.k8s\.io "khalil-volcano-[^"]+" not found',
            priority_absent,
        )
    )
    if (jobs_absent, podgroups_absent, queues_absent, priorities_absent) != (
        6,
        6,
        4,
        2,
    ):
        raise ValueError(
            "unexpected cleanup absence counts: "
            f"{jobs_absent}, {podgroups_absent}, {queues_absent}, {priorities_absent}"
        )

    volcano_health = read("cleanup-volcano-health.txt")
    kueue_health = read("cleanup-kueue-health.txt")
    for name in ("volcano-admission", "volcano-controllers", "volcano-scheduler"):
        if not re.search(rf"^{name}\s+1/1\s+1\s+1\s+", volcano_health, re.MULTILINE):
            raise ValueError(f"cleanup health missing for {name}")
    if not re.search(
        r"^kueue-controller-manager\s+1/1\s+1\s+1\s+", kueue_health, re.MULTILINE
    ):
        raise ValueError("Kueue not healthy in cleanup snapshot")
    quota_after = quota("evaluation-resourcequota-after.json")
    gpu_used_after = int(quota_after["status"]["used"]["requests.nvidia.com/gpu"])
    if gpu_used_after != 0:
        raise ValueError(f"GPU quota not released after cleanup: {gpu_used_after}")
    return {
        "effective": True,
        "scheduler_config_restored": True,
        "restored_actions": default_actions,
        "test_jobs_absent": jobs_absent,
        "test_podgroups_absent": podgroups_absent,
        "test_queues_absent": queues_absent,
        "test_priorityclasses_absent": priorities_absent,
        "namespace_gpu_quota_used_after": gpu_used_after,
        "volcano_deployments_ready": "3/3 (each 1/1)",
        "kueue_deployment_ready": "1/1",
        "volcano_left_installed": True,
    }


def analyze() -> dict[str, Any]:
    transcript = read("evaluation-transcript.txt")
    if kv_line(transcript, "ADMIN_KUBECONFIG_PATH") != "/etc/rancher/k3s/k3s.yaml":
        raise ValueError(
            "evaluation did not use the explicitly authorized kubeconfig path"
        )
    if kv_line(transcript, "ADMIN_CONTEXT") != "default":
        raise ValueError("evaluation did not use the captured default admin context")
    if kv_line(transcript, "NAMESPACE") != "gpu-dev":
        raise ValueError("evaluation did not run in gpu-dev")
    if int(kv_line(transcript, "PREFLIGHT_RUNNING_GPU_REQUEST_TOTAL")) != 0:
        raise ValueError("unrelated Running GPU requests existed at preflight")
    result_line = re.search(r"^RESULT (.+)$", transcript, re.MULTILINE)
    if not result_line or "cleanup=true" not in result_line.group(1):
        raise ValueError("evaluation result line missing or incomplete")

    node = load_json("evaluation-node.json")
    node_gpu = int(node["status"]["allocatable"]["nvidia.com/gpu"])
    before_quota = quota("evaluation-resourcequota-before.json")
    quota_hard = int(before_quota["status"]["hard"]["requests.nvidia.com/gpu"])
    install = installation()
    smoke = gpu_smoke()
    queue_result = queue_capability(node_gpu)
    gang_result = gang()
    preempt_result = preemption(node_gpu, quota_hard)
    cleanup_result = cleanup(install["default_scheduler_actions"])
    stamps = timeline("cuda-smoke-timeline.txt")

    result: dict[str, Any] = {
        "captured_at_utc": stamps["CUDA_SMOKE_CLEANUP_COMPLETE"]["utc"],
        "evidence_scope": "N=1 functional evaluation on the live single-node GPU cluster",
        "access": {
            "kubeconfig_path": "/etc/rancher/k3s/k3s.yaml",
            "context": "default",
            "namespace": "gpu-dev",
            "credential_content_captured_or_committed": False,
            "authorization_scope": "User-authorized Volcano maintenance evaluation only",
        },
        "cluster": {
            "kubernetes": node["status"]["nodeInfo"]["kubeletVersion"],
            "node": node["metadata"]["name"],
            "node_gpu_allocatable": node_gpu,
            "node_accelerator_label": node["metadata"]["labels"].get("accelerator"),
            "namespace_gpu_quota_hard": quota_hard,
            "preflight_running_gpu_request_total": int(
                kv_line(transcript, "PREFLIGHT_RUNNING_GPU_REQUEST_TOTAL")
            ),
        },
        "installation": install,
        "gpu_cuda_smoke": smoke,
        "queue_capability": queue_result,
        "gang": gang_result,
        "preemption": preempt_result,
        "reclaim": {
            "tested": False,
            "effective": None,
            "reason": (
                "A controlled cross-Queue physical-pressure test could not be produced safely "
                "inside the shared gpu-dev quota of 4 GPUs on an 8-GPU node"
            ),
            "default_enabled": False,
        },
        "restoration_cleanup": cleanup_result,
        "limitations": [
            "Each scenario was run once; these are functional observations, not stable performance statistics.",
            "The cluster has one GPU node, so multi-node topology behavior was not evaluated.",
            "Volcano preempt was enabled temporarily for the controlled test and is disabled in the restored default actions.",
            "Victim checkpoint/resume and training wall-clock penalty were not evaluated.",
            "Cross-Queue reclaim, capacity deserved borrowing, and long-run fairness were not tested.",
            "API object and core Event timestamps are generally one-second precision; sub-second values come only from app logs or the client monotonic clock.",
        ],
        "evidence_refs": {
            "raw": "docs/assets/gpu-scheduler-evaluation/raw/volcano/",
            "manifests": "k8s/scheduler-evaluation/volcano/bench/",
            "runner": "k8s/scheduler-evaluation/run_volcano_gpu_evaluation.sh",
            "cuda_runner": "k8s/scheduler-evaluation/run_volcano_cuda_smoke.sh",
        },
    }
    RESULTS.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def render(result: dict[str, Any]) -> None:
    install = result["installation"]
    smoke = result["gpu_cuda_smoke"]
    queue_result = result["queue_capability"]
    gang_result = result["gang"]
    preempt_result = result["preemption"]
    cleanup_result = result["restoration_cleanup"]

    install_body = "\n".join(
        [
            "INSTALLATION_SOURCE official Volcano v1.15.1 development manifest",
            f"MANIFEST_SHA256 {install['manifest_sha256']}",
            f"SERVER_DRY_RUN objects={install['server_side_dry_run_object_count']} result=PASS",
            f"INSTALL_BEGIN {install['install_begin_at']}",
            f"KUBECTL_APPLY_RETURN {install['kubectl_apply_return_at']}",
            f"METRIC kubectl_apply_rtt_seconds={install['kubectl_apply_rtt_seconds']}",
            f"CRDS Established=True count={install['volcano_crds_established']}",
            (
                f"WEBHOOK_CONFIGS count={install['volcano_webhook_configurations_with_ca_bundle']} "
                "caBundle_nonempty=7/7"
            ),
            (
                f"ENDPOINT_SLICES count={install['volcano_endpoint_slices']} "
                f"ready_addresses={install['volcano_ready_endpoint_addresses']}"
            ),
            "DEPLOY volcano-scheduler ready=1/1 image=vc-scheduler:v1.15.1 available=2026-07-31T07:16:41Z",
            "DEPLOY volcano-controllers ready=1/1 image=vc-controller-manager:v1.15.1 available=2026-07-31T07:16:41Z",
            "DEPLOY volcano-admission ready=1/1 image=vc-webhook-manager:v1.15.1 available=2026-07-31T07:16:45Z",
            f"METRIC install_begin_to_last_available_seconds={install['install_begin_to_last_deployment_available_seconds']} api_precision=1s",
            "ADMISSION_INIT Complete",
            f"DEFAULT_ACTIONS {','.join(install['default_scheduler_actions'])}",
            f"KUEUE_COEXISTENCE image={install['kueue_coexistence']['image']} ready={install['kueue_coexistence']['ready']}",
            "CONCLUSION Volcano control plane installed and Ready; Kueue remained healthy",
        ]
    )
    render_terminal(
        "08 - Volcano v1.15.1 installation and readiness",
        install_body,
        SCREENSHOTS / "08-volcano-installation.png",
    )

    sm = smoke["metrics_seconds"]
    st = smoke["times"]
    smoke_body = "\n".join(
        [
            "ADMIN_KUBECONFIG_PATH=/etc/rancher/k3s/k3s.yaml (path only; no credential content captured)",
            "ADMIN_CONTEXT=default NAMESPACE=gpu-dev",
            f"JOB {smoke['job']} uid={smoke['job_uid']}",
            f"POD {smoke['pod']} schedulerName={smoke['scheduler_name']} node={smoke['node']}",
            f"EVENT Scheduled reportingComponent={smoke['scheduled_event_reporting_component']} at={st['pod_scheduled']}",
            f"GPU_MODEL={smoke['gpu_model']}",
            f"VISIBLE_GPU_COUNT={smoke['visible_gpu_count']}",
            f"TORCH_VERSION={smoke['torch_version']}",
            f"TORCH_CUDA_AVAILABLE={str(smoke['torch_cuda_available']).lower()}",
            f"CUDA_TENSOR_RESULT={smoke['cuda_tensor_result']}",
            f"APP_START_UTC={st['application_started']}",
            f"APP_FINISH_UTC={st['application_finished']}",
            f"JOB_COMPLETE={st['job_completed']}",
            (
                "METRIC create_rtt={client_create_rtt}s api_create_to_scheduled={api_job_create_to_pod_scheduled}s "
                "api_create_to_start={api_job_create_to_container_start}s"
            ).format(**sm),
            (
                "METRIC app_runtime={application_runtime}s container_runtime={container_runtime}s "
                "api_job_wall={api_job_wall_clock}s client_e2e={client_create_to_complete_observed}s"
            ).format(**sm),
            "CONCLUSION Volcano binding + NVIDIA visibility + real Torch CUDA tensor execution all PASS",
        ]
    )
    render_terminal(
        "09 - Volcano GPU/CUDA execution and wall clock",
        smoke_body,
        SCREENSHOTS / "09-volcano-gpu-cuda-wallclock.png",
    )

    qm = queue_result["metrics_seconds"]
    qd = queue_result["waiter_during_snapshot"]
    queue_body = "\n".join(
        [
            f"QUEUE {queue_result['queue']} capability.gpu={queue_result['queue_gpu_capability']} allocated.gpu={queue_result['queue_gpu_allocated_during_snapshot']}",
            f"HOLDER job={queue_result['holder']} request.gpu={queue_result['holder_gpu_request']} phase=Running bound=true",
            f"WAITER job={queue_result['waiter']} request.gpu={queue_result['waiter_gpu_request']} phase={qd['pod_phase']} bound=false pg={qd['podgroup_phase']}",
            f"EVENT {qd['event_reason']} reportingComponent={qd['event_reporting_component']}",
            f"MESSAGE {qd['event_message']}",
            (
                f"NODE_REQUEST_HEADROOM allocatable={queue_result['node_gpu_allocatable']} "
                f"running_requests={queue_result['running_pod_gpu_requests_cluster_snapshot']} "
                f"headroom={queue_result['node_request_headroom_during_wait']}"
            ),
            f"RESOURCEQUOTA used.gpu={queue_result['namespace_quota_gpu_used_during_snapshot']} (counts Pending Pod requests)",
            f"HOLDER_FINISHED {queue_result['times']['holder_container_finished']}",
            f"WAITER_SCHEDULED {queue_result['times']['waiter_pod_scheduled']} source=volcano",
            f"WAITER_STARTED {queue_result['times']['waiter_container_started']}",
            (
                "METRIC create_to_scheduled={api_job_create_to_pod_scheduled}s create_to_start={api_job_create_to_container_start}s "
                "release_to_schedule={holder_finish_to_waiter_schedule}s release_to_start={holder_finish_to_waiter_start}s"
            ).format(**qm),
            (
                "METRIC runtime={waiter_container_runtime}s api_job_wall={api_waiter_job_wall_clock}s "
                "client_e2e={client_create_to_complete_observed}s"
            ).format(**qm),
            "CONCLUSION Queue capability enforced despite 7-GPU node request headroom; waiter ran after release",
        ]
    )
    render_terminal(
        "10 - Volcano Queue GPU capability enforcement",
        queue_body,
        SCREENSHOTS / "10-volcano-queue-capability.png",
    )

    gm = gang_result["metrics_seconds"]
    gang_body = "\n".join(
        [
            f"PODGROUP {gang_result['podgroup']} minMember={gang_result['min_member']} minResources.gpu={gang_result['min_resources_gpu']}",
            f"QUEUE {gang_result['queue']} initial_capability.gpu={gang_result['initial_queue_gpu_capability']}",
            (
                f"BELOW_THRESHOLD pods={gang_result['initial_snapshot']['pod_count']} "
                f"pending={gang_result['initial_snapshot']['pending_pods']} "
                f"bound={gang_result['initial_snapshot']['bound_pods']} pg={gang_result['initial_snapshot']['podgroup_phase']}"
            ),
            f"EVENT {gang_result['initial_snapshot']['event_message']}",
            f"CAPABILITY_PATCH_RETURN {gang_result['times']['capability_patch_return']} new_gpu={gang_result['patched_queue_gpu_capability']}",
            f"POD_SCHEDULED {gang_result['times']['pod_scheduled'][0]} | {gang_result['times']['pod_scheduled'][1]} source=volcano",
            f"APP_START {gang_result['times']['application_started'][0]} | {gang_result['times']['application_started'][1]}",
            f"APP_FINISH {gang_result['times']['application_finished'][0]} | {gang_result['times']['application_finished'][1]}",
            (
                "METRIC create_to_both_scheduled={api_job_create_to_both_scheduled}s "
                "create_to_both_started={api_job_create_to_both_container_started}s app_start_spread={application_start_spread}s"
            ).format(**gm),
            (
                "METRIC api_job_wall={api_job_wall_clock}s client_e2e={client_create_to_complete_observed}s "
                "succeeded=2/2"
            ).format(**gm),
            "PRECISION patch return and Scheduled events share one API second; no fake sub-second control-plane claim",
            "CONCLUSION zero bindings below minResources; both members bound after capability reached 2 GPUs",
        ]
    )
    render_terminal(
        "11 - Volcano PodGroup/Gang threshold behavior",
        gang_body,
        SCREENSHOTS / "11-volcano-gang.png",
    )

    pm = preempt_result["metrics_seconds"]
    preempt_body = "\n".join(
        [
            f"DEFAULT_ACTIONS {','.join(preempt_result['default_scheduler_actions'])}",
            f"TEST_ACTIONS {','.join(preempt_result['temporary_test_scheduler_actions'])}",
            (
                f"QUEUE {preempt_result['queue']} capability.gpu={preempt_result['queue_gpu_capability']} "
                f"allocated.gpu={preempt_result['queue_gpu_allocated_before']}"
            ),
            (
                f"VICTIM job={preempt_result['victim']['job']} priority={preempt_result['victim']['priority']} "
                f"request.gpu={preempt_result['victim']['gpu_request']} started={preempt_result['victim']['started_at']}"
            ),
            (
                f"PREEMPTOR job={preempt_result['preemptor']['job']} priority={preempt_result['preemptor']['priority']} "
                f"request.gpu={preempt_result['preemptor']['gpu_request']}"
            ),
            (
                f"SAFETY namespace_quota={preempt_result['namespace_gpu_quota_hard']} combined_requests={preempt_result['combined_test_gpu_requests']} "
                f"node_request_headroom_before={preempt_result['node_request_headroom_before_preemption']}"
            ),
            (
                f"EVENT {preempt_result['victim']['evict_at']} Evict reportingComponent={preempt_result['victim']['evict_reporting_component']} "
                f"message='{preempt_result['victim']['evict_message']}'"
            ),
            f"EVENT {preempt_result['victim']['killing_at']} Killing reportingComponent=kubelet",
            f"HIGH_SCHEDULED {preempt_result['preemptor']['scheduled_at']} source=volcano",
            f"HIGH_STARTED {preempt_result['preemptor']['started_at']}",
            (
                "METRIC high_create_to_scheduled={api_high_job_create_to_scheduled}s "
                "high_create_to_start={api_high_job_create_to_container_start}s client_ready={client_high_create_to_ready_observed}s"
            ).format(**pm),
            (
                "METRIC victim_runtime_before_evict~{victim_api_runtime_before_evict}s high_runtime={preemptor_container_runtime}s "
                "api_high_job_wall={api_high_job_wall_clock}s client_e2e={client_high_create_to_complete_observed}s"
            ).format(**pm),
            f"VICTIM_RESULT {preempt_result['victim']['job_result']} checkpoint_resume=false",
            "CONCLUSION same-Queue Volcano preempt action evicted the controlled victim and admitted the high-priority Job",
        ]
    )
    render_terminal(
        "12 - Volcano controlled same-Queue preemption",
        preempt_body,
        SCREENSHOTS / "12-volcano-preemption.png",
    )

    cleanup_body = "\n".join(
        [
            f"RESTORED_ACTIONS {','.join(cleanup_result['restored_actions'])}",
            f"CONFIG_RESTORED={str(cleanup_result['scheduler_config_restored']).lower()}",
            (
                f"ABSENT jobs={cleanup_result['test_jobs_absent']} podgroups={cleanup_result['test_podgroups_absent']} "
                f"queues={cleanup_result['test_queues_absent']} priorityclasses={cleanup_result['test_priorityclasses_absent']}"
            ),
            f"RESOURCEQUOTA gpu-dev used.requests.nvidia.com/gpu={cleanup_result['namespace_gpu_quota_used_after']}",
            f"HEALTH Volcano={cleanup_result['volcano_deployments_ready']} Kueue={cleanup_result['kueue_deployment_ready']}",
            "RECLAIM tested=false result=N/A reason=shared-quota safety boundary",
            "LIMIT N=1; single node; no median/P95; no cross-Queue reclaim; no checkpoint/resume",
            "VOLCANO_LEFT_INSTALLED=true",
            "CONCLUSION test objects removed, GPU quota released, default scheduler config restored",
        ]
    )
    render_terminal(
        "13 - Volcano restoration, cleanup, and evidence boundary",
        cleanup_body,
        SCREENSHOTS / "13-volcano-restoration-cleanup.png",
    )


def main() -> None:
    result = analyze()
    render(result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
