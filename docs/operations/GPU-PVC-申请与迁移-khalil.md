# Khalil QuantFM GPU PVC 申请与迁移清单

更新日期：2026-07-31

## 当前风险

- 本地训练目录：`/home/khalil/DataCleaning7.3/QuantFM/quant_fm/runs`
- 当前占用：约 310 GiB
- 所在文件系统：约 996 GiB，已使用 92%，仅余约 81 GiB
- `gpu-dev-khalil` 无权查看、创建或修改 PVC
- 2026-07-30 的时间线强烈表明 `khalil/quantfm-data` 已成功创建：PVC
  清单写入时间为 08:45，`/data/k3s/storage` 在 08:47 新增第二个卷目录
- 管理员已确认该 PVC 为 500 GiB、RWO、`local-path`、Bound；对应 PV 为
  `pvc-792a789f-ebf3-4a57-9c32-58f23c0c9580`，实际目录位于独立数据盘
  `/data/k3s/storage/pvc-792a789f-ebf3-4a57-9c32-58f23c0c9580_khalil_quantfm-data`
- PV 回收策略已从 `Delete` 修改为 `Retain`
- 该 PVC 位于旧命名空间 `khalil`，不能被现在 `gpu-dev` 中的 Job 跨命名空间挂载
- 旧 GPU Job 仍以 `hostPath` 使用节点根盘，不应继续用于新训练

在 PVC 完成分配和验证前，不提交新的长时间训练，不把输出切到一个未经
确认的 claim，也不删除本地原始数据。

## 请管理员确认/提供

| 项目 | 请求 |
|---|---|
| Namespace | `gpu-dev` |
| 现有 claim | 已确认为 `khalil/quantfm-data`，状态 Bound |
| 新空间 claim | 将现有数据保留并暴露为 `gpu-dev/quantfm-data`；如名称不同请返回精确名称 |
| 容量 | 最低 500 GiB；考虑现有 310 GiB 与后续 checkpoint，建议 1 TiB |
| 后端 | 当前 `local-path` 基目录为独立 2 TiB XFS 数据盘 `/data/k3s/storage`，可继续使用；需确认没有回落到根盘 |
| AccessMode | 单节点训练可用 RWO；如需跨节点并发请提供 RWX |
| 文件权限 | UID 1006、GID/fsGroup 1009 可读写 |
| 保留策略 | 删除 Job 不删除数据；PVC 回收策略和备份策略需明确 |
| 调度约束 | 返回 StorageClass、卷拓扑和必要 node affinity |

管理员还需允许 Khalil 至少 `get` 指定 PVC，或书面返回 PVC 的 Bound 状态、
实际容量和 StorageClass，以便提交前验证。

## 迁移顺序

1. 已确认 `khalil/quantfm-data`、对应 PV、实际目录、容量和已有数据状态。
2. 已将 PV 回收策略从 `Delete` 改为 `Retain`。
3. 已停止所有已知本地写入者；Dense230M 主训练完成，并停止遗留的
   `monitor_training` 写入进程。
4. 用无 GPU 的一次性 Job 挂载源目录为只读、PVC 为读写；先复制到
   `runs.migrating-20260731`，不直接覆盖正式 `runs`。
5. 对比总字节数、文件数，并对 checkpoint、manifest、vocab 和交付产物做
   SHA-256 抽样/关键文件全量校验。
6. 管理员在保留底层数据的前提下，将该 PV 从
   `khalil/quantfm-data` 重新绑定为 `gpu-dev/quantfm-data`。
7. 使用挂载 PVC 的暂停 GPU Job 做只读冒烟检查，再启动短训练验证 checkpoint
   能写入和恢复。
8. 所有新 Job 切换到 PVC；本地副本保留到验收与独立备份完成后再决定是否清理。

## 容器挂载合同

- 镜像内工作目录：`/workspace/QuantFM`
- PVC 挂载点：`/workspace/QuantFM/quant_fm/runs`
- 数据、缓存、日志、checkpoint 和输出必须位于该挂载点下
- 代码和 Python 环境放入 `registry.zs/gpu-dev/` 私有镜像
- 不使用 `hostPath` 持久化训练产物

暂停的规范模板见 `k8s/gpu-dev/khalil-training-job.template.yaml`。
