from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = REPO_ROOT / "benchmark" / "experiments" / "reliability"


def load_module(name: str, path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RENDER = load_module("render_pod_retry", EXPERIMENT / "render_pod_retry.py")
DELETE_RENDER = load_module(
    "render_pod_deletion", EXPERIMENT / "render_pod_deletion.py"
)


class PodRetryExperimentTests(unittest.TestCase):
    def test_manifest_is_bounded_and_uses_normal_job_retry(self) -> None:
        manifest = RENDER.render("retry1234")
        self.assertEqual(manifest["apiVersion"], "batch/v1")
        self.assertEqual(manifest["kind"], "Job")
        self.assertEqual(manifest["metadata"]["namespace"], "gpu-dev")
        self.assertTrue(manifest["metadata"]["name"].startswith("khalil-"))
        self.assertEqual(manifest["spec"]["backoffLimit"], 1)
        self.assertEqual(manifest["spec"]["activeDeadlineSeconds"], 300)
        pod = manifest["spec"]["template"]["spec"]
        self.assertEqual(pod["runtimeClassName"], "nvidia")
        self.assertFalse(pod["automountServiceAccountToken"])
        self.assertNotIn("runAsUser", pod["securityContext"])
        self.assertEqual(
            pod["securityContext"]["seccompProfile"]["type"], "RuntimeDefault"
        )
        container = pod["containers"][0]
        self.assertIn("@sha256:", container["image"])
        command = container["args"][0]
        self.assertIn("python -u", command)
        self.assertNotIn("exec python", command)
        self.assertEqual(container["resources"]["requests"]["nvidia.com/gpu"], "1")
        self.assertEqual(container["resources"]["limits"]["nvidia.com/gpu"], "1")
        self.assertNotIn("persistentVolumeClaim", str(manifest))

    def test_renderer_rejects_unbounded_token(self) -> None:
        for token in ("short", "UPPER1234", "contains-dash", "a" * 13):
            with self.assertRaises(RENDER.RenderError):
                RENDER.render(token)

    def test_renderer_bounds_backoff_limit(self) -> None:
        self.assertEqual(RENDER.render("retry1234", 0)["spec"]["backoffLimit"], 0)
        self.assertEqual(RENDER.render("retry1234", 2)["spec"]["backoffLimit"], 2)
        for value in (-1, 3, True):
            with self.assertRaises(RENDER.RenderError):
                RENDER.render("retry1234", value)

    def test_runner_defaults_to_non_mutating_plan(self) -> None:
        completed = subprocess.run(
            ["python3", str(EXPERIMENT / "run_pod_retry.py")],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('"mutation": "none"', completed.stdout)
        self.assertIn('"gpu": 1', completed.stdout)

    def test_runner_rejects_output_outside_evidence_root(self) -> None:
        completed = subprocess.run(
            [
                "python3",
                str(EXPERIMENT / "run_pod_retry.py"),
                "plan",
                "--output",
                "/tmp/pod-retry-evidence",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("output must remain", completed.stderr)

    def test_runner_fixes_identity_and_uid_cleanup(self) -> None:
        source = (EXPERIMENT / "run_pod_retry.py").read_text(encoding="utf-8")
        self.assertIn('KUBECONFIG = "/etc/rancher/k3s/k3s.yaml"', source)
        self.assertIn('CONTEXT = "default"', source)
        self.assertIn("current_uid != self.job_uid", source)
        self.assertIn('[ \\"$name\\" = python ]', source)
        self.assertIn('kill -KILL \\"$pid\\"', source)
        self.assertIn("if killed.returncode != 0", source)

    def test_backoff_matrix_defaults_to_non_mutating_plan(self) -> None:
        completed = subprocess.run(
            ["python3", str(EXPERIMENT / "run_backoff_matrix.py")],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        plan = completed.stdout
        self.assertIn('"mutation": "none"', plan)
        self.assertIn('"backoff_limit": 0', plan)
        self.assertIn('"backoff_limit": 1', plan)
        self.assertIn('"backoff_limit": 2', plan)
        self.assertIn('"sequential": true', plan)

    def test_pod_deletion_manifest_is_bounded(self) -> None:
        manifest = DELETE_RENDER.render("delete1234")
        self.assertEqual(manifest["metadata"]["name"], "khalil-pod-delete-delete1234")
        self.assertEqual(manifest["metadata"]["namespace"], "gpu-dev")
        self.assertEqual(manifest["spec"]["backoffLimit"], 1)
        self.assertEqual(
            manifest["spec"]["podReplacementPolicy"], "TerminatingOrFailed"
        )
        self.assertEqual(manifest["spec"]["activeDeadlineSeconds"], 240)
        pod = manifest["spec"]["template"]["spec"]
        self.assertEqual(pod["restartPolicy"], "Never")
        self.assertEqual(pod["runtimeClassName"], "nvidia")
        self.assertNotIn("persistentVolumeClaim", str(manifest))
        env = {item["name"]: item.get("value") for item in pod["containers"][0]["env"]}
        self.assertEqual(env["TOTAL_STEPS"], "300")
        self.assertEqual(env["DELETE_AFTER_STEP"], "75")

    def test_pod_deletion_runner_defaults_to_plan_and_validates_owner(self) -> None:
        completed = subprocess.run(
            ["python3", str(EXPERIMENT / "run_pod_deletion.py")],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('"mutation": "none"', completed.stdout)
        self.assertIn('"delete_after_step": 75', completed.stdout)
        self.assertIn('"podReplacementPolicy": "TerminatingOrFailed"', completed.stdout)
        source = (EXPERIMENT / "run_pod_deletion.py").read_text(encoding="utf-8")
        self.assertIn("Pod deletion refused: UID changed", source)
        self.assertIn("Pod deletion refused: Job owner UID mismatch", source)
        self.assertIn('"delete",\n                "pod",', source)
        self.assertIn('"--wait=false"', source)
        self.assertNotIn('"--force"', source)
        self.assertIn("replacement exceeded bound", source)


if __name__ == "__main__":
    unittest.main()
