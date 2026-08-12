#!/usr/bin/env python3
"""Render the bounded one-GPU Pod deletion / Job reconciliation experiment."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from render_pod_retry import RenderError
from render_pod_retry import render as render_retry

SCRIPT_DIR = Path(__file__).resolve().parent
WORKLOAD = SCRIPT_DIR / "pod_deletion_workload.py"


def render(run_token: str) -> dict[str, Any]:
    manifest = copy.deepcopy(render_retry(run_token, 1))
    payload = WORKLOAD.read_bytes()
    payload_sha = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")
    name = f"khalil-pod-delete-{run_token}"
    labels = {
        "app.kubernetes.io/name": "quantfm-pod-deletion",
        "app.kubernetes.io/part-of": "quantfm-reliability-evaluation",
        "quantfm.openai/experiment": "pod-deletion-job-reconciliation",
        "quantfm.openai/run-token": run_token,
    }
    manifest["metadata"]["name"] = name
    manifest["metadata"]["labels"] = labels
    manifest["metadata"]["annotations"].update(
        {
            "quantfm.openai/workload-sha256": payload_sha,
            "quantfm.openai/failure-action": "delete-owned-pod",
            "quantfm.openai/checkpoints-allowed": "false",
        }
    )
    manifest["spec"]["backoffLimit"] = 1
    manifest["spec"]["podReplacementPolicy"] = "TerminatingOrFailed"
    manifest["spec"]["activeDeadlineSeconds"] = 240
    manifest["spec"]["template"]["metadata"]["labels"] = labels
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    container["args"] = [
        "\n".join(
            (
                "umask 077",
                'mkdir -p "$HOME" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TMPDIR"',
                f"printf '%s' '{encoded}' | base64 -d > /experiment/pod_deletion_workload.py",
                'test "$(sha256sum /experiment/pod_deletion_workload.py | cut -d\' \' -f1)" = "$WORKLOAD_SHA256"',
                "python -u /experiment/pod_deletion_workload.py",
            )
        )
    ]
    container["env"] = [
        item
        for item in container["env"]
        if item.get("name") not in {"INJECTION_WINDOW_SECONDS", "WORKLOAD_SHA256"}
    ]
    container["env"].extend(
        [
            {"name": "WORKLOAD_SHA256", "value": payload_sha},
            {"name": "TOTAL_STEPS", "value": "300"},
            {"name": "DELETE_AFTER_STEP", "value": "75"},
            {"name": "STEP_DELAY_SECONDS", "value": "0.03"},
        ]
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = render(args.run_token)
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
