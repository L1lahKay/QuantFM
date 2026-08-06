# GPU 调度评估与实验报告

本目录集中保存 GPU 调度调研结论和后续独立实验报告。调研报告给出选型结论，
独立实验报告记录具体实验设计、结果和证据边界。

| 文档 | 内容 |
|---|---|
| [GPU 集群训练后端选型报告](GPU集群训练后端执行选型评估报告.md) | Native Kubernetes、Kueue、Volcano 的选型结论和生产门槛 |
| [NN 与 Transformer 三调度器对比实验](current-safe-20260806/CURRENT_SAFE_N3_EXPERIMENT_REPORT.md) | 5 个场景、3 个调度器、每格 N=3，共 45 次运行 |
| [Pod 进程故障与 Job 重试实验](pod-retry-20260806/POD_RETRY_EXPERIMENT_REPORT.md) | 训练进程退出后由 Job controller 创建新 Pod 并完成 CUDA 训练 |
| [Kubernetes Job backoffLimit 0/1/2 对照实验](backoff-matrix-20260806/JOB_BACKOFF_MATRIX_EXPERIMENT_REPORT.md) | 对照零次、一次和两次重试预算下的 Pod 数量、退出码与 Job 终态 |

机器可读结果、Kubernetes 对象、Events 和容器日志保留在 `benchmark/results/`，
报告中的“原始记录”链接直接指向对应证据文件。
