# Bounded scheduler benchmark queues

This directory contains exactly the four scheduler objects required by
`benchmark/config/runtime.json`:

- Kueue `ResourceFlavor/khalil-kueue-nvidia`;
- Kueue `ClusterQueue/khalil-kueue-eval`;
- Kueue `LocalQueue/gpu-dev/khalil-kueue-eval`;
- Volcano `Queue/khalil-volcano-smoke`.

The quota/capability is deliberately capped at 4 GPUs, 8 CPUs and 16 GiB.  It
matches the largest single cell in `current-safe-matrix.json` and does not
bypass the existing `gpu-dev` four-GPU `ResourceQuota`.  No cohort, borrowing,
reclaim or preemption policy is configured.

The helper defaults to a non-mutating plan and uses server-side dry-run.  An
actual apply or cleanup additionally requires `--execute`:

```bash
k8s/scheduler-evaluation/runtime-setup/manage.sh plan
k8s/scheduler-evaluation/runtime-setup/manage.sh apply --execute
k8s/scheduler-evaluation/runtime-setup/manage.sh cleanup --execute
```

The helper fixes the Kubernetes identity to `/etc/rancher/k3s/k3s.yaml`,
context `default`, and accepts evidence directories only below
`benchmark/results/scheduler-setup`. Environment variables cannot redirect it
to another cluster.

The script refuses to reconcile an existing exact-name object unless both
ownership labels match. Cleanup first proves that no benchmark Job, Workload,
PodGroup, or benchmark Pod remains and that no Workload or PodGroup references
the dedicated queues; it then deletes only the four exact names and verifies
absence. Both requested and limited namespace GPU usage must be zero before a
mutation.

Before `apply`, the helper repeats a privacy-minimized host query and proceeds
only when `nvidia-smi` succeeds and reports no compute processes.  Evidence
contains only GPU UUID and used memory; PID, user and command are never queried.
Cleanup records the same observation but is not blocked by unrelated host GPU
processes, because removing idle queue objects does not consume or preempt GPU
work.

This helper does not authorize a maintenance window, change scheduler system
components, alter the namespace ResourceQuota, or run a benchmark.  The active
`AGENTS.md` authorization and maintenance gates remain authoritative.
