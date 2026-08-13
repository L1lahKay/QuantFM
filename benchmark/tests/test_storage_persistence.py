from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "benchmark/experiments/storage/run_pvc_persistence.py"
SPEC = importlib.util.spec_from_file_location("run_pvc_persistence", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class StoragePersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (ROOT / "benchmark/config/storage-smoke-20260806.json").read_text(
                encoding="utf-8"
            )
        )

    def test_plan_is_non_mutating_without_cluster_access(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(MODULE_PATH), "plan", "--run-token", "test260806"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("mutation=none", completed.stdout)
        self.assertIn("claim=quantfm-data", completed.stdout)

    def test_jobs_are_bounded_cpu_only_and_mount_only_exact_claim(self) -> None:
        for role in ("write", "read"):
            manifest = MODULE.job_manifest(
                self.config, "test260806", role, 64 * 1024 * 1024
            )
            self.assertEqual(manifest["apiVersion"], "batch/v1")
            self.assertEqual(manifest["kind"], "Job")
            self.assertTrue(manifest["metadata"]["name"].startswith("khalil-"))
            self.assertEqual(manifest["metadata"]["namespace"], "gpu-dev")
            self.assertEqual(manifest["spec"]["backoffLimit"], 0)
            self.assertLessEqual(manifest["spec"]["activeDeadlineSeconds"], 300)
            pod = manifest["spec"]["template"]["spec"]
            self.assertFalse(pod["automountServiceAccountToken"])
            self.assertEqual(
                pod["nodeSelector"]["kubernetes.io/hostname"], "gpu-dev-01"
            )
            container = pod["containers"][0]
            self.assertRegex(
                container["image"], r"^registry\.zs/gpu-dev/.+@sha256:[0-9a-f]{64}$"
            )
            resources = container["resources"]
            self.assertNotIn("nvidia.com/gpu", resources["requests"])
            self.assertNotIn("nvidia.com/gpu", resources["limits"])
            self.assertEqual(
                pod["volumes"],
                [
                    {
                        "name": "quantfm-data",
                        "persistentVolumeClaim": {"claimName": "quantfm-data"},
                    }
                ],
            )
            self.assertNotIn("hostPath", json.dumps(manifest))

    def test_reader_cleanup_is_file_whitelisted(self) -> None:
        program = MODULE.reader_program()
        self.assertIn('checkpoint = target / "checkpoint.bin"', program)
        self.assertIn('manifest = target / "manifest.json"', program)
        self.assertIn("checkpoint.unlink()", program)
        self.assertIn("manifest.unlink()", program)
        self.assertIn("refusing to remove non-empty target directory", program)
        self.assertNotIn("rmtree", program)
        self.assertNotIn("rm -rf", program)


if __name__ == "__main__":
    unittest.main()
