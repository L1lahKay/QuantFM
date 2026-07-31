# 发给 Reinhard 的确认消息

@Reinhard 你好，我在按要求评估 GPU 集群的 Kueue/Volcano 队列、配额和抢占能力，同时准备把 QuantFM 数据从 `/home` 迁出。

当前已只读确认：

- 集群为 k3s `v1.35.4+k3s1`，单节点 `gpu-dev-01`，8 × RTX 5090。
- 只有 `default-scheduler`，未安装 Kueue/Volcano，原生 Workload/PodGroup API 未启用。
- `gpu-dev` 命名空间限 4 GPU；`khalil` 没有 GPU ResourceQuota，当前可申请 8 GPU。
- `khalil` 没有 PVC。QuantFM `quant_fm/runs` 约 289G，仍在 `/home`；根盘已用 90%。
- `/data/k3s` 约 2TiB，仅用 2%；现有 `local-path` StorageClass 底层使用该盘，但是 RWO 单节点存储，回收策略是 `Delete`。

请帮忙确认：

1. 是否允许在当前集群串行安装 Kueue `v0.19.0` 和 Volcano `v1.15.0` 做隔离实验？建议先测 Kueue、卸载/验收后再测 Volcano。
2. 是否允许在临时命名空间执行 GPU 队列、份额借用/回收和抢占实验？是否有指定测试时间窗？
3. `khalil` 的正式 GPU 份额应设为多少？当前不受限可用 8 GPU 是否符合预期？
4. QuantFM 数据的指定位置是哪一种：`local-path` PVC（`/data/k3s/storage`）、`/data/...` hostPath、MinIO，还是其他 NFS/Ceph 存储？
5. 如果使用 PVC，申请 500Gi 还是 1Ti？是否需要新建 `Retain` StorageClass，避免删除 PVC 时底层数据一起删除？
6. 迁移前是否需要做 MinIO/离线备份？旧 `/home` 数据应保留多少天再清理？
7. 当前是否有 Prometheus/Grafana 或其他指定监控系统？后续需记录队列等待、准入、抢占和 wall-clock median/P95。

我已完成裸 Kubernetes 基线实测和截图，得到回复后会按相同口径执行 Kueue/Volcano 对比和存储迁移。
