# Reliability experiments

`run_pod_retry.py` verifies one bounded part of the failure-recovery plan: a
container process is killed in the first Pod of a one-GPU Kubernetes Job, the
Job controller creates a second Pod because `backoffLimit: 1`, and the second
Pod completes real CUDA training.

The default action is non-mutating:

```bash
python3 benchmark/experiments/reliability/run_pod_retry.py plan
```

An authorized execution uses the fixed `/etc/rancher/k3s/k3s.yaml` identity,
context `default`, namespace `gpu-dev`, an internal digest image, one GPU,
memory-backed scratch, no service-account token, no PVC, and exact-name/UID
cleanup. Evidence is written below `benchmark/results/reliability/`.

This experiment proves Pod process-failure retry only. It does not prove node
failure, checkpoint persistence, checkpoint resume, or preemption recovery.

The 2026-08-06 run `retry260810` passed: the first Pod exited 137 after the
training process was killed, a second Pod with a different UID was created,
and that Pod exited 0 after 80 timed CUDA steps. See the
[experiment report](../../../docs/gpu-scheduler-reports/pod-retry-20260806/POD_RETRY_EXPERIMENT_REPORT.md)
and [machine-readable result](../../results/reliability/pod-retry-retry260810/result.json).
