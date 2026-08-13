# QuantFM PVC 跨 Pod 持久性烟测（2026-08-06）

## 结论

`gpu-dev/quantfm-data` 的精确 PV path 位于独立的
`/dev/mapper/data--vg-k3s` XFS，而不是根文件系统
`/dev/mapper/ubuntu--vg-ubuntu--lv`。因此无需仅为了“离开根盘”再次替换该
claim。

64 MiB checkpoint 跨 Pod persistence smoke 已通过：写 Pod 完成 `fsync` 和
原子 rename，读 Pod 使用同一 PVC 重新挂载后得到完全相同的 SHA-256，并删除了
本 run-token 创建的 checkpoint、manifest 和空目录。两个 Job 随后按 UID 精确
清理，`gpu-dev` ResourceQuota used 恢复为 CPU=0、memory=0、GPU=0。

## 实测结果

| 项目 | 结果 |
|---|---:|
| Run token | `stor260806f` |
| Checkpoint | 67,108,864 bytes |
| SHA-256 | `71e28cf5f5255681d9313620a5854ce9a675238f72c9304f543de176d020e7d2` |
| 写入 + fsync | 0.131570 s |
| 应用层写入 | 486.43 MiB/s |
| 读取 + SHA-256 | 0.097039 s |
| 应用层读取 | 659.53 MiB/s |
| Writer / Reader Pod | 不同 Pod UID，均在 `gpu-dev-01` |
| 镜像 | 同一内部 digest `dylan-trainer@sha256:9e7f…574` |
| 清理 | checkpoint/manifest/run directory/两个 Job 均已清理 |

该 64 MiB 结果可能命中 page cache，只用于路径和持久性语义烟测，不作为磁盘
吞吐基准。正式性能结论仍需固定 fio 镜像、4 KiB/1 MiB、iodepth 1/16/64、单 Pod/
并发 4 Pod、N≥3，并同时记录 direct-I/O 能力和缓存策略。

## 证据链

- 精确 PVC/PV/块设备基线：
  `benchmark/results/follow-up-baseline/20260806T090745Z/`
- 成功运行、Job/Pod UID、Events、日志、server-side dry-run、配额与 checksum：
  `benchmark/results/storage/pvc-persistence-stor260806f/`
- 可复用 fail-closed runner：
  `benchmark/experiments/storage/run_pvc_persistence.py`

两次前置失败 (`stor260806b`、`stor260806d`) 都发生在容器执行 Python 前；第一次
为直接 OCI exec 找不到 `python`，第二次为普通 UID 无权执行镜像内 Python。
两次均未产生 PVC checkpoint，失败 Job 已清理。最终运行采用该镜像历史环境探针
已验证的 shell/Python 入口和镜像默认用户，同时保留 seccomp、drop ALL
capabilities、禁止 privilege escalation 以及文件白名单清理。

## 未覆盖范围

- 该 claim 是 RWO、node-affine local-path；未证明多节点 attach、fencing、
  reattach 或 Node 故障恢复。
- 未运行训练应用的模型/优化器/RNG checkpoint resume，也未证明抢占后恢复。
- `/data/k3s` 当前显示 XFS `noquota`；不能把 500Gi PVC request 当成已执行的文件
  系统硬限额，也不得用填满共享卷的方式测试。
- 多节点与 Node 故障主结果仍需共享/分布式 CSI、第二 GPU 节点和维护窗口。
