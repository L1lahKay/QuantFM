# Historical bare-Kubernetes benchmark inputs

These manifests explain the raw evidence captured on 2026-07-30. They are not
valid submissions under the current `gpu-dev-khalil` operating rules: they
create namespaces and cluster-scoped PriorityClasses, include direct Pods,
request five GPUs, and use a public image.

Do not apply these files to the current GPU cluster. The current compliant,
single-GPU Kueue probe is in `../gpu-dev/kueue-probe.yaml`.
