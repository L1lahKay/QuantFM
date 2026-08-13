from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "benchmark" / "scripts" / "evaluation_readiness.py"
SPEC = importlib.util.spec_from_file_location("evaluation_readiness", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


class EvaluationReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = load("benchmark/config/runtime.json")
        self.matrix = load("benchmark/config/current-safe-matrix.json")
        self.inventory = load("benchmark/results/raw/cluster-inventory/summary.json")
        self.requirements = load("benchmark/config/evaluation-requirements.json")
        self.results = json.loads(
            (ROOT / "benchmark/report/benchmark_results.json").read_text(
                encoding="utf-8"
            )
        )

    def test_current_state_uses_ephemeral_contract_but_keeps_live_cluster_gates(
        self,
    ) -> None:
        result = MODULE.evaluate(
            self.runtime,
            self.matrix,
            self.inventory,
            self.requirements,
            self.results,
        )
        self.assertFalse(result["execution_ready"])
        self.assertFalse(result["final_report_ready"])
        codes = {item["code"] for item in result["execution_blockers"]}
        self.assertIn("SYNTHETIC_EPHEMERAL_AUTHORIZATION_CONSUMED", codes)
        self.assertNotIn("STORAGE_ON_ROOT_DISK", codes)
        self.assertNotIn("STORAGE_NOT_CONFIRMED", codes)
        self.assertNotIn("KUEUE_NOT_AUTHORIZED", codes)
        self.assertNotIn("VOLCANO_NOT_AUTHORIZED", codes)
        self.assertIn("KUEUE_LOCALQUEUE_ABSENT", codes)
        self.assertIn("VOLCANO_QUEUE_ABSENT", codes)
        self.assertIn("GPU_CONTAMINATION_DETECTED", codes)
        self.assertEqual(result["storage_mode"], "synthetic_ephemeral")
        final_codes = {item["code"] for item in result["final_report_blockers"]}
        self.assertIn("MONITORING_CHAIN_INCOMPLETE", final_codes)
        self.assertIn("GPU_NODE_TOPOLOGY_MISMATCH", final_codes)
        self.assertNotIn("MODEL_MATRIX_INCOMPLETE", final_codes)
        # The checked-in report is a live integration fixture. The completed
        # K8s batch plus the 2026-08-06 Kueue/Volcano batch now contribute at
        # least three exact-contract runs to all 15 current-safe cells.
        self.assertEqual(result["performance_evidence"]["ready_matrix_cells"], 15)

    def test_k8s_subset_can_be_execution_ready_without_monitoring(self) -> None:
        runtime = copy.deepcopy(self.runtime)
        runtime["policy"]["training_storage_backend_confirmed"] = True
        runtime["policy"]["storage_backend_non_root_disk"] = True
        inventory = copy.deepcopy(self.inventory)
        inventory["host_gpu_process_snapshot"][
            "external_compute_processes_detected"
        ] = 0
        result = MODULE.evaluate(
            runtime,
            self.matrix,
            inventory,
            self.requirements,
            self.results,
            schedulers=["k8s"],
            scenarios=["nn-single-gpu1"],
        )
        self.assertFalse(result["execution_ready"])
        self.assertFalse(result["final_report_ready"])
        execution_codes = {
            item["code"] for item in result["execution_blockers"]
        }
        self.assertEqual(
            execution_codes,
            {"SYNTHETIC_EPHEMERAL_AUTHORIZATION_CONSUMED"},
        )
        final_codes = {item["code"] for item in result["final_report_blockers"]}
        self.assertIn("MONITORING_CHAIN_INCOMPLETE", final_codes)

    def test_target_matrix_reports_runner_and_namespace_gpu_limit(self) -> None:
        target = load("benchmark/config/target-matrix.json")
        result = MODULE.evaluate(
            self.runtime,
            target,
            self.inventory,
            self.requirements,
            self.results,
            schedulers=["k8s"],
            scenarios=["transformer-single-gpu8"],
        )
        codes = {item["code"] for item in result["execution_blockers"]}
        self.assertIn("MATRIX_SCENARIO_DISABLED", codes)
        self.assertIn("RUNNER_GPU_LIMIT_EXCEEDED", codes)
        self.assertIn("NAMESPACE_GPU_QUOTA_EXCEEDED", codes)
        self.assertIn("STORAGE_ON_ROOT_DISK", codes)

    def test_latest_observation_can_clear_only_the_volatile_gpu_process_blocker(
        self,
    ) -> None:
        latest = {
            "observed_at_utc": "2026-08-03T06:27:43.130Z",
            "host_gpu_compute_processes": {
                "query_succeeded": True,
                "process_count": 0,
                "items": [],
            },
            "gpu_dev": {
                "gpu_quota_hard": 4,
                "gpu_quota_used": 0,
                "local_queues": [],
            },
            "volcano_open_queues": ["default", "root"],
        }
        inventory = MODULE.merge_latest_observation(self.inventory, latest)
        result = MODULE.evaluate(
            self.runtime,
            self.matrix,
            inventory,
            self.requirements,
            self.results,
        )
        codes = {item["code"] for item in result["execution_blockers"]}
        self.assertNotIn("GPU_CONTAMINATION_DETECTED", codes)
        self.assertNotIn("STORAGE_ON_ROOT_DISK", codes)
        self.assertIn("KUEUE_LOCALQUEUE_ABSENT", codes)

    def test_synthetic_mode_refuses_missing_exact_authorization(self) -> None:
        runtime = copy.deepcopy(self.runtime)
        runtime["policy"]["synthetic_ephemeral_authorized"] = False
        inventory = copy.deepcopy(self.inventory)
        inventory["host_gpu_process_snapshot"][
            "external_compute_processes_detected"
        ] = 0
        result = MODULE.evaluate(
            runtime,
            self.matrix,
            inventory,
            self.requirements,
            self.results,
            schedulers=["k8s"],
            scenarios=["nn-single-gpu1"],
        )
        codes = {item["code"] for item in result["execution_blockers"]}
        self.assertIn("SYNTHETIC_EPHEMERAL_CONTRACT_INVALID", codes)
        self.assertNotIn("STORAGE_ON_ROOT_DISK", codes)

    def test_current_safe_scheduler_authorization_does_not_leak_to_target(self) -> None:
        target = load("benchmark/config/target-matrix.json")
        inventory = copy.deepcopy(self.inventory)
        inventory["host_gpu_process_snapshot"][
            "external_compute_processes_detected"
        ] = 0
        result = MODULE.evaluate(
            self.runtime,
            target,
            inventory,
            self.requirements,
            self.results,
            schedulers=["kueue", "volcano"],
            scenarios=["nn-single-gpu1"],
        )
        codes = {item["code"] for item in result["execution_blockers"]}
        self.assertIn("KUEUE_NOT_AUTHORIZED", codes)
        self.assertIn("VOLCANO_NOT_AUTHORIZED", codes)
        self.assertIn("STORAGE_ON_ROOT_DISK", codes)

    def test_only_validated_completed_rows_count_toward_repetitions(self) -> None:
        rows = [
            {
                "scenario_id": "nn-single-gpu1",
                "scheduler": "K8s",
                "status": "completed",
                "execution_stage": "completed",
            },
            {
                "scenario_id": "nn-single-gpu1",
                "scheduler": "K8s",
                "status": "dry-run-complete",
                "execution_stage": "server-dry-run",
            },
        ]
        result = MODULE.evaluate(
            self.runtime,
            self.matrix,
            self.inventory,
            self.requirements,
            rows,
            schedulers=["k8s"],
            scenarios=["nn-single-gpu1"],
        )
        evidence = result["performance_evidence"]
        self.assertEqual(evidence["ready_matrix_cells"], 0)
        self.assertEqual(
            evidence["missing_or_under_repeated_cells"][0]["completed_repetitions"],
            1,
        )

    def test_cli_gate_returns_distinct_failure_status_and_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "readiness.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--gate",
                    "final-report",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 11, completed.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(payload["final_report_ready"])


if __name__ == "__main__":
    unittest.main()
