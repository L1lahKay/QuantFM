# Storage persistence smoke

`run_pvc_persistence.py` verifies the already rebound
`gpu-dev/quantfm-data` claim without modifying its PV, PVC, StorageClass or
existing data. The experiment writes a bounded checkpoint below a fresh
run-token directory, fsyncs and atomically renames it, then mounts the same RWO
claim in a different Pod, checks the SHA-256, and removes only the two files and
directory created by that token.

The default action is non-mutating:

```bash
python3 benchmark/experiments/storage/run_pvc_persistence.py plan
```

Use `dry-run` for API validation and `run --execute` only after the live
preflight passes. Evidence is stored under `benchmark/results/storage/`.

This experiment proves same-node, cross-Pod persistence only. It does not
prove multi-node attach, fencing, node-failure recovery, backup, or filesystem
quota enforcement.
