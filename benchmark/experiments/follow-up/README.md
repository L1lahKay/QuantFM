# Follow-up experiment workspace

This directory contains preparation for the scheduler, backend, topology,
failure, monitoring, and storage follow-up experiments. Nothing here mutates a
cluster by default.

`capture_baseline.sh` records a privacy-minimized, read-only snapshot. It fixes
the Kubernetes identity to `/etc/rancher/k3s/k3s.yaml`, context `default`, and
never reads Secrets or kubeconfig contents. The capture includes node/GPU/NUMA
topology, exact permissions, GPU compute capability, privacy-minimized
NIC/RDMA state, scheduler configuration, quota, monitoring inventory, block
devices, and exact PVC-local-path-to-mount provenance:

```bash
benchmark/experiments/follow-up/capture_baseline.sh
benchmark/experiments/follow-up/capture_baseline.sh capture --execute
```

The execution runbook and acceptance matrix live in
`EXPERIMENT_PLAN.md`. Cluster mutations in that document are future
administrator steps, not authorization granted by the files themselves.

The corrected storage evidence is in
`../../results/follow-up-baseline/20260806T090745Z/`. The bounded cross-Pod
persistence runner and its scope are documented in `../storage/README.md`.
