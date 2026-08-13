import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / "k8s" / "scheduler-evaluation" / "runtime-setup"


class RuntimeSetupSafetyTests(unittest.TestCase):
    def test_scheduler_objects_are_exact_and_bounded(self) -> None:
        resource_flavor = (SETUP / "00-kueue-resourceflavor.yaml").read_text()
        cluster_queue = (SETUP / "01-kueue-clusterqueue.yaml").read_text()
        local_queue = (SETUP / "02-kueue-localqueue.yaml").read_text()
        volcano_queue = (SETUP / "03-volcano-queue.yaml").read_text()

        self.assertIn("name: khalil-kueue-nvidia", resource_flavor)
        self.assertIn("accelerator: nvidia", resource_flavor)
        self.assertIn("name: khalil-kueue-eval", cluster_queue)
        self.assertIn("queueingStrategy: StrictFIFO", cluster_queue)
        self.assertIn('nominalQuota: "4"', cluster_queue)
        self.assertIn("namespace: gpu-dev", local_queue)
        self.assertIn("clusterQueue: khalil-kueue-eval", local_queue)
        self.assertIn("name: khalil-volcano-smoke", volcano_queue)
        self.assertIn("reclaimable: false", volcano_queue)
        self.assertIn('nvidia.com/gpu: "4"', volcano_queue)

        policy_text = cluster_queue.lower()
        for forbidden in ("cohort", "preemption", "fairsharing", "borrowing"):
            self.assertNotIn(forbidden, policy_text)

    def test_helper_fixes_identity_and_checks_references(self) -> None:
        helper = (SETUP / "manage.sh").read_text()

        self.assertIn("KUBECONFIG_PATH=/etc/rancher/k3s/k3s.yaml", helper)
        self.assertIn("KUBE_CONTEXT=default", helper)
        self.assertNotIn("${KUBECONFIG_PATH:-", helper)
        self.assertNotIn("${KUBE_CONTEXT:-", helper)
        self.assertIn("EVIDENCE_ROOT=", helper)
        self.assertIn("assert_dedicated_queues_unused", helper)
        self.assertIn("limits.nvidia.com/gpu", helper)
        self.assertIn("get podgroups.scheduling.volcano.sh --all-namespaces", helper)


if __name__ == "__main__":
    unittest.main()
