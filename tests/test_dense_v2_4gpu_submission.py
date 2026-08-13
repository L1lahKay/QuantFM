from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "quant_fm/pretrain/config_v2_230m.yaml"
FOUR_GPU_CONFIG = ROOT / "quant_fm/pretrain/config_v2_230m_4gpu.yaml"
TEMPLATE = ROOT / "k8s/gpu-dev/khalil-dense-v2-230m-4gpu-job.template.yaml"
SUBMITTER = ROOT / "k8s/gpu-dev/submit-dense-v2-230m-4gpu.sh"
IMAGE = "registry.zs/gpu-dev/khalil-quantfm@sha256:" + "a" * 64
JOB_NAME = "khalil-dense-v2-230m-4gpu-test"
DATES = ROOT / "quant_fm/data/medium_60_dates.txt"
SYMBOLS_SZ = ROOT / "quant_fm/data/medium_symbols_sz.txt"
SYMBOLS_SH = ROOT / "quant_fm/data/medium_symbols_sh.txt"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _template_preflight_python() -> str:
    manifest = _load_yaml(TEMPLATE)
    command = manifest["spec"]["template"]["spec"]["containers"][0]["args"][0]
    marker = "python - <<'PY'\n"
    start = command.index(marker) + len(marker)
    return command[start : command.index("\nPY\n", start)]


def test_four_gpu_config_preserves_dense_230m_contract_and_global_batch() -> None:
    base = _load_yaml(BASE_CONFIG)
    candidate = _load_yaml(FOUR_GPU_CONFIG)
    expected = copy.deepcopy(base)
    expected["optim"]["grad_accum"] = 16

    assert candidate == expected
    base_global_batch = (
        base["optim"]["micro_batch_size"] * base["optim"]["grad_accum"] * 8
    )
    four_gpu_global_batch = (
        candidate["optim"]["micro_batch_size"] * candidate["optim"]["grad_accum"] * 4
    )
    assert base_global_batch == four_gpu_global_batch == 128
    assert candidate["data"]["manifest"].startswith("quant_fm/runs/")
    assert candidate["data"]["vocab"].startswith("quant_fm/runs/")
    assert candidate["data"]["validation_plan"].startswith("quant_fm/runs/")
    assert candidate["runtime"]["out_dir"].startswith("quant_fm/runs/")


def test_frozen_full_market_plan_defines_symbol_day_coverage_denominator() -> None:
    dates = DATES.read_text(encoding="utf-8").splitlines()
    symbols_sz = SYMBOLS_SZ.read_text(encoding="utf-8").splitlines()
    symbols_sh = SYMBOLS_SH.read_text(encoding="utf-8").splitlines()

    assert len(dates) == len(set(dates)) == 60
    assert len(symbols_sz) == len(set(symbols_sz)) == 2836
    assert len(symbols_sh) == len(set(symbols_sh)) == 2269
    assert (len(symbols_sz) + len(symbols_sh)) * len(dates) == 306_300


def test_full_market_preflight_allows_symbol_day_gaps_but_not_fake_universe(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "quant_fm/data"
    artifact_dir = tmp_path / "quant_fm/runs/v2_shared/data"
    data_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    (tmp_path / "quant_fm/runs/v2_shared/artifact_audit.json").write_text(
        json.dumps(
            {
                "checked_all_paths": True,
                "contract_ready": True,
                "failed_symbol_gaps": [],
                "issues": [],
            }
        ),
        encoding="utf-8",
    )
    for source in (DATES, SYMBOLS_SZ, SYMBOLS_SH):
        (data_dir / source.name).write_bytes(source.read_bytes())

    dates = DATES.read_text(encoding="utf-8").splitlines()
    symbols = [
        (market, symbol)
        for market, source in (("SZ", SYMBOLS_SZ), ("SH", SYMBOLS_SH))
        for symbol in source.read_text(encoding="utf-8").splitlines()
    ]
    sparse_shards = [
        {"date": dates[index % len(dates)], "market": market, "symbol": symbol}
        for index, (market, symbol) in enumerate(symbols)
    ]
    (artifact_dir / "manifest.json").write_text(
        json.dumps({"shards": sparse_shards}),
        encoding="utf-8",
    )
    preflight = _template_preflight_python()
    accepted = subprocess.run(
        [sys.executable, "-c", preflight],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "V2_FULL_MARKET_PREFLIGHT_PASS" in accepted.stdout
    assert "missing_symbol_days=" in accepted.stdout
    assert "coverage=1.000000" not in accepted.stdout

    fake_shards = [
        {"date": date, "market": "SZ", "symbol": symbols[0][1]} for date in dates
    ]
    (artifact_dir / "manifest.json").write_text(
        json.dumps({"shards": fake_shards}),
        encoding="utf-8",
    )
    rejected = subprocess.run(
        [sys.executable, "-c", preflight],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "market/symbol union does not match" in rejected.stderr


def test_checked_in_four_gpu_job_template_is_suspended_and_scoped() -> None:
    manifest = _load_yaml(TEMPLATE)
    spec = manifest["spec"]
    pod_spec = spec["template"]["spec"]
    container = pod_spec["containers"][0]

    assert manifest["apiVersion"] == "batch/v1"
    assert manifest["kind"] == "Job"
    assert manifest["metadata"]["namespace"] == "gpu-dev"
    assert manifest["metadata"]["name"].startswith("khalil-")
    assert spec["suspend"] is True
    assert spec["activeDeadlineSeconds"] == 7 * 24 * 60 * 60
    assert spec["backoffLimit"] == 0
    assert spec["ttlSecondsAfterFinished"] == 86400

    assert pod_spec["runtimeClassName"] == "nvidia"
    assert pod_spec["nodeSelector"] == {
        "accelerator": "nvidia",
        "kubernetes.io/hostname": "gpu-dev-01",
    }
    assert pod_spec["restartPolicy"] == "Never"
    assert pod_spec["automountServiceAccountToken"] is False
    assert pod_spec["securityContext"]["runAsNonRoot"] is True
    assert "schedulerName" not in pod_spec

    assert container["image"] == "REPLACE_IMAGE"
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "4"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "4"
    assert container["resources"]["requests"]["cpu"] == "32"
    assert container["resources"]["requests"]["memory"] == "64Gi"
    assert container["resources"]["limits"]["cpu"] == "48"
    assert container["resources"]["limits"]["memory"] == "128Gi"
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]

    command = container["args"][0]
    assert "V2_FULL_MARKET_PREFLIGHT_PASS" in command
    assert "medium_60_dates.txt" in command
    assert "medium_symbols_sz.txt" in command
    assert "medium_symbols_sh.txt" in command
    assert "len(expected_dates) != 60" in command
    assert 'expected_symbol_counts = {"SZ": 2836, "SH": 2269}' in command
    assert "allowed_keys" in command
    assert "duplicate date/market/symbol shards" in command
    assert "outside the frozen date/universe plan" in command
    assert "market/symbol union does not match the frozen universe" in command
    assert "missing_symbol_days" in command
    assert "coverage={coverage:.6f}" in command
    assert "104857600" in command
    assert "python -m torch.distributed.run" in command
    assert "--nproc_per_node=4" in command
    assert "config_v2_230m_4gpu.yaml" in command
    assert "--resume auto" in command
    subprocess.run(
        ["/bin/sh", "-n", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )

    pvc = pod_spec["volumes"][0]["persistentVolumeClaim"]
    assert pvc["claimName"] == "quantfm-data"
    dshm = next(volume for volume in pod_spec["volumes"] if volume["name"] == "dshm")
    assert dshm["emptyDir"] == {"medium": "Memory", "sizeLimit": "8Gi"}
    dshm_mount = next(
        mount for mount in container["volumeMounts"] if mount["name"] == "dshm"
    )
    assert dshm_mount["mountPath"] == "/dev/shm"
    assert "hostPath" not in TEMPLATE.read_text(encoding="utf-8")


def test_local_renderer_accepts_only_digest_and_keeps_job_suspended() -> None:
    rendered = subprocess.run(
        [str(SUBMITTER), "--render-only", IMAGE, JOB_NAME],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = yaml.safe_load(rendered.stdout)
    assert manifest["metadata"]["name"] == JOB_NAME
    assert manifest["spec"]["suspend"] is True
    assert manifest["spec"]["template"]["spec"]["containers"][0]["image"] == IMAGE
    assert "REPLACE_" not in rendered.stdout

    invalid = subprocess.run(
        [
            str(SUBMITTER),
            "--render-only",
            "registry.zs/gpu-dev/khalil-quantfm:mutable",
            JOB_NAME,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert invalid.returncode == 64
    assert "full internal" in invalid.stderr


def test_submitter_rechecks_live_safety_gates_before_create() -> None:
    source = SUBMITTER.read_text(encoding="utf-8")

    assert "/etc/rancher/k3s/k3s.yaml" in source
    assert 'context_name="default"' in source
    assert 'namespace="gpu-dev"' in source
    assert "dcfc08ba-7c57-466e-abeb-e60897855a39" in source
    assert "8d66d1f8-7a0a-48ff-ae7b-e49a4e3e59fa" in source
    assert "get resourcequota" in source
    assert "get persistentvolumeclaim" in source
    assert "get persistentvolume" in source
    assert 'get node "$expected_node" -o json' in source
    assert "get pods -o json" in source
    assert "unrelated active GPU Pod requests exist" in source
    assert "--query-compute-apps=gpu_uuid,used_gpu_memory" in source
    assert "gpu_uuid,pid" not in source
    assert "has fewer than four allocatable GPUs" in source
    assert "PV nodeAffinity is not exactly bound" in source
    assert "PVC bound capacity" in source
    assert "PV capacity" in source
    for permission in (
        'require_permission get jobs.batch --namespace "$namespace"',
        'require_permission list jobs.batch --namespace "$namespace"',
        'require_permission watch jobs.batch --namespace "$namespace"',
        'require_permission delete jobs.batch --namespace "$namespace"',
        'require_permission get pods --namespace "$namespace"',
        'require_permission list pods --namespace "$namespace"',
        'require_permission get pods/log --namespace "$namespace"',
        'require_permission list events --namespace "$namespace"',
    ):
        assert permission in source
    assert source.count('gpu_before="$(host_gpu_snapshot)"') == 1
    assert source.count('state_before="$(cluster_snapshot)"') == 1
    assert source.count('gpu_after_dry_run="$(host_gpu_snapshot)"') == 1
    assert source.count('state_after_dry_run="$(cluster_snapshot)"') == 1

    dry_run = source.index('create --dry-run=server -f "$rendered_active"')
    create = source.index('create -f "$rendered_active" -o name', dry_run + 1)
    second_snapshot = source.index('state_after_dry_run="$(cluster_snapshot)"')
    assert dry_run < second_snapshot < create
    assert "QUANTFM_SERVER_DRY_RUN_ONLY" in source[dry_run:create]
