# GPU 调度、训练后端与可靠性后续实验计划

更新日期：2026-08-06（UTC）

## 1. 目标、边界和当前结论

这份计划把后续工作拆成可独立验收、可精确清理的实验。所有实际 GPU
训练仍必须是 `gpu-dev` 中以 `khalil-` 开头的 `batch/v1 Job`，使用内部
digest 镜像、NVIDIA runtime、匹配的 GPU request/limit，并服从共享
ResourceQuota。任何持久数据、checkpoint、缓存或训练输出，都必须先由管理员
确认一个不在根盘上的 PVC 及其物理后端。

2026-08-06 09:07 UTC 的只读基线已确认：

- 集群只有节点 `gpu-dev-01`，物理上有 8 张 RTX 5090；`gpu-dev` 的 GPU
  request/limit 配额均为 4，所以当前不能进行 8-GPU 或多节点运行。
- `nvidia-smi topo -m` 没有 NVLink，GPU/CPU NUMA 亲和分组为 3/2/3：
  GPU 0–2 属于 NUMA 0，GPU 3–4 属于 NUMA 1，GPU 5–7 属于 NUMA 2；
  跨组链路为 `SYS`。
- 节点没有 zone/rack/block TAS 标签。Kueue 配置中的
  `waitForPodsReady`、`fairSharing` 和 `admissionFairSharing` 均未启用。
- Volcano 当前 actions 是 `enqueue, allocate, backfill`，gang 和 DRF 的
  `enablePreemptable` 都是 `false`，因此当前配置不能声称已经验证
  `preempt/reclaim`。
- 只发现 standalone NVIDIA device plugin，未发现 GPU Operator、DCGM
  Exporter、Prometheus、Grafana 或相关监控 CRD。
- `quantfm-data` 仍由 `local-path` 提供，但旧结论中用 `findmnt -T /data`
  判断后端是根盘的方法错误。对 PV 的精确 `spec.local.path` 解析后，路径落在
  `/data/k3s`，其后端为 `/dev/mapper/data--vg-k3s` XFS，物理盘是独立的 7TB
  Intel NVMe；根文件系统则是 `/dev/mapper/ubuntu--vg-ubuntu--lv`。仓库中的
  管理员迁移记录与当前 PVC/PV UID、Retain、path 和 node affinity 完全匹配。
- 这纠正了“必须为了离开根盘而替换当前 claim”的前提，但不把 local-path
  提升为共享存储：该卷仍是单节点 RWO、node-affine，不能证明多节点 reattach、
  fencing 或节点故障恢复；这些目标仍需共享/分布式 CSI。

最新基线证据目录为
`benchmark/results/follow-up-baseline/20260806T090745Z/`，其中的
`SHA256SUMS` 已逐项校验通过。

## 2. 已完成的同构 NN/Transformer N=3 批次

已经以同一个 digest 镜像、同一个 runtime 配置、同一个矩阵和进程内确定性
数据，随机化执行了 5 个场景 × 3 个调度器 × 3 次重复，共 45/45 次成功：

- Matrix SHA-256：
  `eb9f41067008490d3894dfa619f759f5e086d08df5bc24f5d6f0a8f3506d70df`
- Runtime SHA-256：
  `7b3d662d2f84f2373e4a2e6d71932b83856840cc58c68b01c2e7a00534071e6b`
- 镜像：
  `registry.zs/gpu-dev/dylan-trainer@sha256:9e7f7f8dc3c15c522408d1e8da38401ac224b99ddfba363078f40403eb456574`
- Kueue/Volcano orchestration：
  `benchmark/results/orchestration/current-safe-kv-n3-20260806a/summary.json`
- Native Kubernetes orchestration：
  `benchmark/results/orchestration/current-safe-k8s-n3-20260806b/summary.json`

下面是这个精确 45-run 批次的中位数。`training` 取训练进程标记的全局训练
区间；`wall` 取客户端提交到 Job 完成；`queue` 取提交到首次调度/admission
边界。Kubernetes Event 只有整秒时间戳时，小于 1 秒的负差按左删失 0 记录，
而不是伪造负排队时间。

| 场景 | 调度器 | N | training (s) | wall (s) | queue (s) |
|---|---:|---:|---:|---:|---:|
| NN 1 GPU / Single Pod | K8s | 3 | 0.314301 | 11.379 | 0.312151 |
| NN 1 GPU / Single Pod | Kueue | 3 | 0.291717 | 10.827 | 0.442177 |
| NN 1 GPU / Single Pod | Volcano | 3 | 0.295610 | 11.791 | 0.791000 |
| NN 4 GPU / Multi Pod | K8s | 3 | 1.122089 | 21.621 | 0.692917 |
| NN 4 GPU / Multi Pod | Kueue | 3 | 1.142873 | 21.271 | 0.823537 |
| NN 4 GPU / Multi Pod | Volcano | 3 | 1.136167 | 20.780 | 0.760000 |
| Transformer 1 GPU / Single Pod | K8s | 3 | 8.201485 | 23.540 | 0.302469 |
| Transformer 1 GPU / Single Pod | Kueue | 3 | 8.029038 | 24.885 | 0.438548 |
| Transformer 1 GPU / Single Pod | Volcano | 3 | 7.948987 | 23.881 | 0.000000* |
| Transformer 4 GPU / Single Pod | K8s | 3 | 8.946993 | 27.808 | 0.284381 |
| Transformer 4 GPU / Single Pod | Kueue | 3 | 9.128663 | 26.928 | 0.435902 |
| Transformer 4 GPU / Single Pod | Volcano | 3 | 9.120512 | 28.760 | 0.833000 |
| Transformer 4 GPU / Multi Pod | K8s | 3 | 26.062220 | 49.939 | 0.672835 |
| Transformer 4 GPU / Multi Pod | Kueue | 3 | 35.170762 | 59.116 | 0.824342 |
| Transformer 4 GPU / Multi Pod | Volcano | 3 | 34.698093 | 58.738 | 1.093000 |

`*` 三次 Volcano queue 值为 0、0、0.114 秒，中位数 0 属于时间戳精度造成的
左删失结果。这里的 Multi Pod 是对调度和多进程启动路径的真实运行，但当前
工作负载不是跨节点 NCCL/DDP 训练；后者必须按第 5 节单独验收。

临时 ResourceFlavor、ClusterQueue、LocalQueue、Volcano Queue 已按名称和 UID
确认后清理，Workload、PodGroup、GPU request 和 GPU 进程都已恢复为空。原先的
一次性 current-safe 运行授权至此用完，不能扩展解释成下面高级功能的变更授权。

## 3. 统一实验协议

### 3.1 固定变量

每个可比较实验块必须记录并校验以下指纹；任一项变化就建立新 series，不能与
旧结果混合求中位数：

1. 内部镜像完整 digest、OCI labels、CUDA/driver/NCCL/LightGBM/PyTorch 版本；
2. 数据 manifest digest、样本数、切分、特征和预处理 digest；合成数据还要记录
   生成器实现 digest、seed 和生成位置；
3. 完整参数 JSON digest，包括 batch size、steps/epochs、精度、优化器、线程、
   workers、world size、backend、checkpoint 周期；
4. Job/Pod 模板规范化 digest、scheduler、queue、priority、Pod 数与每 Pod GPU；
5. 节点 UID、GPU UUID/PCI bus、NUMA、网卡/RDMA 接口和 Pod→node→GPU 布局；
6. ResourceQuota、Queue/ClusterQueue/LocalQueue/PodGroup/Workload 的快照；
7. 每个 block 至少 N=3，按固定 seed 随机化执行顺序。性能结论报告 median、
   min/max，并在 N≥5 时增加 p95/置信区间。

### 3.2 时间定义

所有时间均保存原始 UTC 时间戳和派生秒数：

| 边界 | 记号 | 证据来源 |
|---|---|---|
| 客户端开始提交 | T0 | runner 单调时钟 + UTC |
| API 创建 Job | T1 | Job `creationTimestamp` 与 UID |
| 队列 admission | T2 | Kueue Workload admission / Volcano PodGroup、Queue Event |
| 所有成员 Pod 已绑定 | T3 | PodScheduled condition/Event |
| 所有训练容器已启动 | T4 | container `state.running.startedAt` |
| 全局训练开始/结束 | T5/T6 | rank 0 结构化 marker；全 rank 汇聚验证 |
| Job Complete | T7 | Job condition/completionTime + 客户端观察时间 |

派生指标为：API latency=`T1-T0`，admission latency=`T2-T1`，binding latency=
`T3-T2`，startup=`T4-T3`，training=`T6-T5`，wall clock=`T7-T0`，以及
non-training overhead=`wall-training`。Multi-Pod 必须同时记录 first/last Pod
差值，不能只看 rank 0。

### 3.3 每次变更的强制门禁

- **G0，只读基线：** 固定 kubeconfig/context，核验 `auth can-i`、ResourceQuota、
  无无关 Running GPU request、隐私最小化 GPU 进程、节点 Ready、镜像 digest。
- **G1，管理员批准：** 修改 Kueue/Volcano 配置、scheduler actions、CRD/RBAC、
  GPU Operator、监控组件、StorageClass/CSI、节点 label/cordon/drain 前，必须有
  人工确认的维护窗口和精确资源清单。
- **G2，dry-run 与回滚：** 保存 before 对象和 resourceVersion；先做客户端渲染、
  schema 验证、server-side dry-run；写明回滚目标和健康检查。
- **G3，运行期守卫：** 只清理本实验创建且 UID 匹配的对象；出现无关 GPU
  workload、配额漂移、节点 NotReady、镜像不一致即停止新提交。
- **G4，证据闭环：** capture Job/Pod/Workload/PodGroup/Queue/Event/log/GPU 样本，
  完成 checksum 后才允许清理；清理后再次确认配额和进程归零。

## 4. LightGBM CPU、OpenCL 和 CUDA

### 4.1 构建与烟测

三个 backend 必须来自 LightGBM 4.6.0，同一训练程序、数据生成器和参数；但因为
编译特性不同，应使用三个独立的 digest 镜像：

- CPU：`device_type=cpu`。固定 16 CPU、同一 NUMA/CPU pinning，在没有 GPU
  request 的 Job 中做真实 boosting。
- OpenCL GPU：从官方 4.6.0 sdist（SHA-256
  `cb1c59720eb569389c0ba74d14f52351b573af489f230032a1c9f314f8bab7fe`）以
  `USE_GPU=ON` 构建，运行时使用 `device_type=gpu`。烟测必须同时看到 NVIDIA
  OpenCL ICD、恰好一张可见 GPU、真实 boosting 和有限预测值。
- CUDA：同一 sdist 以 `USE_CUDA=ON` 和目标 GPU 的已核验 CMake architecture
  构建，运行时使用 `device_type=cuda`。烟测同样必须做真实 boosting，不能以
  `import lightgbm` 或 build 成功代替。

OpenCL 与 CUDA 的 fail-closed 构建、渲染和烟测构件已准备在 `benchmark/lgb/`；
二者默认只输出 plan。实际 build/push 尚未执行，因为还缺管理员审核过的 builder/
runtime digest、registry 写入凭据和最终内部 digest。本机当前用户也无 Docker/
containerd socket 权限。存储的物理非根盘条件已经纠正并通过单节点烟测，LGB 的
剩余前置阻塞以镜像 build/provenance 为主。

### 4.2 性能矩阵与验收

先用 `rows=65,536, features=100, rounds=10` 做 backend smoke；通过后运行正式矩阵
`rows=1,000,000, features=100, rounds=100, max_bin=63, num_leaves=63,
learning_rate=0.1, seed=20260803`，每后端 N=3。CPU 固定 16 threads，GPU 后端
也保留相同 host thread 参数。为了区分缓存和编译效应，每个 backend 先做一次不
入统计的 warm-up，再随机化 9 次正式运行。

通过条件：

- 每次都只有一个成功的 `BENCHMARK_RESULT_JSON`；marker 中 backend 与镜像
  build provenance 一致，训练 100 rounds、预测有限、Job Complete；
- CPU Job 无 GPU request；OpenCL/CUDA Job 各 request/limit 1 GPU，且 DCGM/
  `nvidia-smi` 证明训练区间内有对应进程和非零利用率；
- 输出 training、wall、queue、rows/s、峰值内存/GPU 显存、平均/峰值 GPU
  utilization 和能耗；结果按 backend 独立报告，不把 OpenCL 称作 CUDA；
- 若某 backend 失败，保留结构化错误和日志并标记 blocked/failed，绝不回退到
  CPU 后仍把行标为 GPU。

## 5. 8-GPU、多节点物理拓扑和真实 NCCL/DDP

### 5.1 执行前条件

当前只有一个 GPU 节点且 namespace hard quota=4，这一阶段现在是 **blocked**。
启动前至少要满足：

- 两个 Ready GPU 节点，合计可分配 GPU≥8；管理员确认临时 GPU quota=8，且不
  影响其他团队；节点标签、GPU UUID/PCIe/NUMA、网卡速度/MTU/RDMA、路由和
  firewall 都已记录；
- 使用同一 digest 镜像，内含版本固定且实际可用的 NCCL、PyTorch 与
  `nccl-tests`；集群 DNS、rendezvous Service 和 NetworkPolicy 允许 worker
  通信；
- 明确 `NCCL_SOCKET_IFNAME`、IB/RDMA 是否启用、NCCL debug/version，禁止通过
  隐式 interface 选择得到不可复现实验。

### 5.2 布局矩阵

按下列顺序验证，避免把单节点 PCIe 结果误写成多节点网络结果：

| 布局 | world size | Pod 布局 | 目的 |
|---|---:|---|---|
| A | 8 | 1 Pod × 8 GPU，单节点 | 单节点上界；记录 3/2/3 NUMA 跨组代价 |
| B（主结果） | 8 | 2 Pod × 4 GPU，2 节点各 1 Pod | 真实跨节点 NCCL/DDP |
| C | 8 | 4 Pod × 2 GPU，至少 2 节点 | Pod/rendezvous 敏感性 |
| D | 4 | 2 Pod × 2 GPU，同节点与跨节点各一组 | 隔离 PCIe 与网络开销 |

TAS/affinity 必须把 B 固定为两个 host domain，A 固定为一个 host domain；保存
实际 Pod→node→GPU UUID，而不是只保存期望 spec。B/C 每个节点的 CPU 和内存
请求对称，rank 与 local rank 显式映射。

### 5.3 两级验收

1. **NCCL functional/performance：** `all_reduce_perf` 从 8 MiB 扫到 1 GiB，
   warm-up 5、正式 20；N=3，保存 algbw/busbw、错误计数和每 rank 日志。所有
   rank 退出 0、无 timeout/fallback/async error，才进入 DDP。
2. **真实 DDP：** NN 和 Transformer 使用同一数据 manifest/参数，1、4、8 GPU
   各 N=3；固定 global batch（主要结论）并另做固定 per-GPU batch（吞吐伸缩）。
   保存每 step all-reduce 时间、samples/tokens/s、training/wall、GPU/网络利用率；
   验证梯度/最终 checkpoint checksum 语义一致。

报告分别给出 strong-scaling efficiency 与 weak-scaling throughput，不能用
当前的多进程启动 smoke 替代 NCCL/DDP 结论。

## 6. 多项目并发、长期排队与公平性

建立三个仅属于本实验的项目身份和标签：team-a Transformer、team-b NN、team-c
LGB。每个 Job 带 `quantfm/team`、`quantfm/project`、`quantfm/queue`、run token，
三方使用独立 LocalQueue/Queue，但共同受不超过 namespace ResourceQuota 的总量
约束。LGB CPU Job 用 CPU 配额竞争实验；LGB GPU 后端通过后再加入 1-GPU 竞争。

实验分两层：

1. **确定性 burst：** 先用 4-GPU holder 填满配额，再按 Latin-square 顺序提交
   三团队的 1-GPU short/long Job。释放 holder 后，分别验证 FIFO/priority/fair
   配置下 admission 顺序、队头阻塞、完成后配额释放和下一个 Job 启动。每个提交
   顺序轮换 N=3。
2. **2–4 小时 soak：** 三团队以预先生成、可重放的到达时间序列持续提交 NN、
   Transformer、LGB short/long mix；每分钟保存 Queue/Workload/Pod/Quota 快照，
   不靠 sleep 顺序推断。设置总 Job 上限和 active deadline，任何团队都不能创建
   无界 backlog。

核心指标：submitted/admitted/running/completed/failed，queue wait p50/p95/max，
GPU-seconds allocated/used，head-of-line blocking，quota release latency，抢占次数，
每团队 slowdown=`wall/isolated-wall`，以及 Jain 公平指数
`(Σx_i)^2/(n·Σx_i^2)`。通过条件是：

- 实际 admission 次序能由配置规则解释；同优先级 FIFO 不倒序；高优先级行为与
  明确策略一致；
- Job Complete/Failed 后，其 quota 在 30 秒内释放，随后等待 Job 可 admission；
- 2–4 小时内无 orphan Workload/PodGroup、无无限 Pending、无控制器错误增长；
- Fair Sharing 1:1 稳态 GPU-seconds 差异≤20%、Jain≥0.95；2:1 权重按 share
  归一化后差异≤20%。短窗口不用于证明长期公平。

## 7. Kueue TAS、waitForPodsReady、Fair Sharing 和受控抢占

这一节会修改 Kueue 配置/队列资源，必须单独获得维护窗口、精确对象授权和回滚
批准。当前一次性 Kueue evaluation 权限不覆盖这些动作。

### 7.1 TAS

管理员先为至少两个节点定义稳定的 region/zone/rack/hostname 标签，再创建实验
专用 Topology 与关联 `ResourceFlavor.spec.topologyName`。分别运行：

- required hostname：单 Pod 多 GPU 必须同 host；资源不足时保持 inadmissible；
- required rack + preferred hostname：2×4 DDP 必须在指定 rack 内，优先紧凑；
- unconstrained：作为对照。

必须同时核对 Workload
`status.admission.podSetAssignments[].topologyAssignment`、实际 Pod nodeName 和
node labels。只看 Pod 最终位置不足以证明 TAS 生效。当前单节点、无拓扑标签时只
能做渲染/dry-run，不能给出 TAS 功能结论。

### 7.2 waitForPodsReady

维护窗口内启用有界参数，例如 timeout=120s、recoveryTimeout=60s、
blockAdmission=true、requeue timestamp=Eviction、backoff 10–60s，并保留 before
配置用于回滚。测试四个 case：全员按时 Ready、故意缺一个 member 超时、Ready
后删除一个 Pod 并在 recoveryTimeout 内恢复、超过 recoveryTimeout。

通过条件：超时 workload 被 Evicted/Requeued，配额释放；blockAdmission 时后续
workload 不越过未 Ready workload；恢复窗口内重建不误释放，超窗则按配置重新
排队；所有 Event/condition 与 wall-clock 链一致。完成后恢复原配置并验证 controller
ready。

### 7.3 Fair Sharing

先验证 cohort 内两个实验 ClusterQueue，weight 1:1；再改为 2:1。只运行本实验
Job，nominal quota 总量不超过授权值，borrowing 上限显式设置。若还要测同一
ClusterQueue 内多团队，另开 Admission Fair Sharing case：两个 LocalQueue 设置
1:1 和 2:1 权重，记录历史 usage 半衰期及采样间隔。

使用第 6 节 soak 指标验收。Queue 空闲时允许另一方利用闲置资源；竞争恢复后，
share 应在规定窗口内回归目标。必须区分传统 cohort fair sharing、Admission
Fair Sharing 和 priority FIFO，不把它们混成一个结论。

### 7.4 受控抢占

只允许抢占同一实验创建的 low-priority victim。先让 victim 获取 1–2 GPU 并写出
可校验 checkpoint，再提交资源不足的 high-priority Job。保存 Kueue Workload
preemption/eviction reason、victim Pod termination、preemptor admission/start，
以及 victim 重新排队和从 checkpoint 恢复的 global step。

通过条件：没有非实验对象被影响；victim eviction 到 high Pod start 的链条完整；
资源释放≤30 秒；high 完成后 victim 被重新 admission；恢复后的最终模型/step 与
无故障对照在约定容差内。若没有真实 eviction，只能写“逻辑配置存在”，不能写
“抢占已生效”。

## 8. Volcano Gang、Queue 和 preempt/reclaim 回归

Volcano 当前配置只支持 `enqueue, allocate, backfill` 路径。最终 actions 需要先由
管理员明确选择并批准，保存原 ConfigMap digest、Deployment rollout 状态和回滚
命令，server-side dry-run 后再更新。若采用 Volcano 1.15 的 gang-aware
`gangPreempt/gangReclaim` alpha 路径，必须与传统 `preempt/reclaim` 分成不同
series，并记录 feature/config；不能把两者结果合并。

回归矩阵：

- **Gang below threshold：** `minMember=4` 时只提供 3 个可调度成员，确认没有
  部分绑定/运行；补齐第 4 个后全部成员成组启动。另测 `minResources` 不足/足够。
- **Queue：** 两个实验 Queue 分别设置 capability/weight，先填满 holder，再提交
  waiter；核对 Queue status、PodGroup condition、排队和释放后 admission。
- **Preempt：** 同一 Queue 内 low victim 对 high Job；必须捕获 victim eviction
  与 preemptor start。
- **Reclaim：** 两个 Queue 共享资源，空闲时 A 使用 B 的 share；B 提交后回收
  A 的实验 victim 并启动 B。只有完整 eviction→start 链才算 effective。

每个 case N=3，victim 仅为本轮创建对象。配置更新后先做 CPU/零 GPU smoke，再
做 1-GPU，最后才做 gang 4-GPU；出现 admission webhook/scheduler/controller
异常立即回滚。回归完成后恢复最终批准的 actions（若批准的是临时实验配置则恢复
原配置），精确删除 PriorityClass/Queue/PodGroup/Job 并验证无残留。

## 9. Pod/Node 故障、重试、checkpoint/resume 和抢占后恢复

这一阶段依赖存储能力与监控先通过。当前 claim 已通过单节点跨 Pod 持久性烟测，
但 Node 故障和多节点恢复仍依赖共享/分布式 CSI。

### 9.1 已完成：Pod 进程故障与 Job 重试

2026-08-06 已完成一个不依赖持久存储的 1-GPU 子实验：首个 Pod 的训练 Python
进程被终止并以 exit 137 结束；`backoffLimit: 1` 的 Job 创建第二个、UID 不同的
Pod，后者以 exit 0 完成 80 步 CUDA 训练。故障注入至重试 Pod 进入实验窗口为
13.586 秒，Job 创建至 Complete 为 70.561 秒。证据位于
`benchmark/results/reliability/pod-retry-retry260810/`。

该结果只覆盖 Pod 进程故障和 Job retry，不覆盖训练应用的 checkpoint/resume、
节点故障或抢占后恢复；后三项仍受应用镜像、第二 GPU 节点、共享存储和维护窗口
约束。

| 故障注入 | 对照/操作 | 预期与验收 |
|---|---|---|
| Pod 删除 | 在 step 25–50% 删除一个实验 Pod | Job/控制器按策略重建；恢复 step≥最后完整 checkpoint；RTO、重复计算量可量化 |
| 进程失败 | 指定 rank 在固定 step 退出非零 | `backoffLimit=0/1/2` 三组语义与 Job condition 一致，无无限重试 |
| 节点 cordon/drain | 管理员只操作预先批准的实验节点 | Pod 转移到另一 GPU 节点；卷可重新挂载；非实验 Pod 不受影响 |
| 节点失联 | 维护窗口内由管理员注入 | 检测时间、Pod eviction、卷 fencing/reattach、RTO 全链闭环 |
| 抢占 | 第 7/8 节的本实验 victim | checkpoint 完整、重新 admission、resume marker 和最终结果一致 |

checkpoint 使用临时文件→`fsync`→原子 rename，文件名含 global step；manifest
记录模型/优化器/RNG/sampler/world-size/config digest 和内容 SHA-256。每次恢复
必须输出 `resumed_from_step`、checkpoint digest、丢失/重算 steps。通过条件：

- 最新完整 checkpoint 可读且 checksum 正确，故意截断文件不会被选择；
- 恢复后的 global step 单调、不重复计入样本；最终 loss/metric 与无故障对照在
  预定义容差内；
- RPO≤一个 checkpoint interval，Pod 故障 RTO≤5 分钟；节点故障 RTO 单独报告，
  不因单节点/本地卷不可迁移而伪报成功；
- 清理后 PVC 中只保留明确批准的 durable output，临时故障文件按 run token 精确
  删除。

## 10. GPU Operator、DCGM Exporter、Prometheus 和 Grafana

当前节点已有工作中的 NVIDIA driver/runtime/device plugin。直接安装 GPU Operator
可能接管这些组件，存在驱动重启和 GPU workload 中断风险。维护窗口中必须先确定
以下二选一方案：

1. **推荐的低变更方案：** 保留现有 driver/toolkit/device plugin，只部署版本固定
   的 DCGM Exporter，再部署 Prometheus Operator/kube-state-metrics/Grafana；
2. **完整 GPU Operator：** 明确把已有 driver/toolkit/device-plugin 设为 operator
   不管理或安排迁移，逐项验证兼容性和回滚。没有维护窗口不得采用。

部署前记录 chart/manifest 版本、digest、镜像 digest、CRD/RBAC diff，server-side
dry-run，且不得使用 latest。DCGM Exporter 启用 Kubernetes pod mapping；通过
kube-state-metrics 的 Pod UID/Job owner 数据，将标准 namespace/pod/container
GPU 指标与以下低基数标签关联：`job`、`quantfm_team`、`quantfm_project`、
`quantfm_queue`、`scheduler`。run token 只用于短期明细，不进入长期高基数 dashboard。

必须至少采集 GPU utilization、显存、功耗、温度、PCIe/NVLink（若存在）、XID/
ECC，以及 CPU/memory/network/PVC I/O、Kueue Workload/ClusterQueue/LocalQueue、
Volcano Queue/PodGroup 和 Kubernetes Job 状态。Dashboard 至少有：

- Job drill-down：training 区间、每 GPU/rank 利用率、显存、功耗、错误；
- 团队/项目：GPU-seconds、成功率、queue p50/p95、slowdown；
- 队列：pending/admitted/running、nominal/borrowed share、preemption/reclaim、配额；
- 节点/存储：GPU 健康、网络、PVC 吞吐/延迟和容量。

验收用一个 1-GPU Job：Prometheus 能按 Pod UID 找到连续样本，Grafana 能从 Job
下钻到 team/queue，Job 完成后指标标签仍能在保留期内查询；人工保存带时间范围、
query 和 dashboard revision 的截图。还要做 exporter/Prometheus Pod 重启，确认
无目标丢失和告警恢复。只有部署对象 Ready、抓取 target up 且查询/截图闭环，才
把“监控部署”标为完成。

## 11. 存储现状纠正、单节点验收与多节点升级

2026-08-06 的精确路径证据表明，`quantfm-data` 并不在根 LV 上：PV path 位于
`/data/k3s/storage/...`，最长 mount prefix 为 `/data/k3s`，后端是
`/dev/mapper/data--vg-k3s` XFS；根 LV 是另一个设备。因此不应为了“离开根盘”
再次迁移或重绑当前 claim，更不能触碰同 namespace 中不属于本项目的
`dylan-data`。

当前卷仍是 `local-path`、RWO、绑定 `gpu-dev-01`。若目标只是单节点训练，可在
管理员既有确认范围内继续验证；若目标是多节点 checkpoint/resume 或节点故障
恢复，则仍应并行创建共享/分布式 CSI 的新 PVC，验证后显式 cutover，并保留旧
claim 作为限时回滚点。不要原地改当前 PV 的 path、StorageClass 或 node affinity。

后端选择顺序：

- 多节点 checkpoint/resume 主结果优先采用有明确持久性、fencing 和多节点 attach
  语义的共享/分布式 CSI；
- 若只采用节点本地 NVMe，必须物理确认不是根盘，使用 Local PV 静态绑定，并明确
  标注它不能支持跨节点恢复；不能把路径名 `/data` 当作非根盘证明。

多节点升级步骤：

1. 管理员提供新共享后端的容量、冗余/备份、fencing、StorageClass provisioner、
   reclaimPolicy、volumeBindingMode 和 quota 方案；证明其支持目标节点间 attach。
2. 创建新、精确命名的 StorageClass/PVC；运行只读/写入 smoke，确认 node/zone
   拓扑、ownership、fsGroup、容量和 expansion 行为。
3. 训练停写并记录源 manifest；使用一次性迁移 Job 复制到新 PVC，生成逐文件大小/
   SHA-256 manifest，再做第二次只读校验。迁移 Job 同时挂载两卷，但不包含 Secret
   输出。
4. 先让一个 canary Job 使用新 claim；通过后修改工作负载配置指向新 claim。旧
   claim 保持只读、设定回滚期限；不得立即删除。
5. 验证吞吐、checkpoint、quota、Pod 重建和节点故障；全部通过且管理员确认备份后，
   才另行批准旧卷退役。

2026-08-06 已完成一个 64 MiB 单节点跨 Pod persistence smoke：写 Pod 完成
`fsync` 和原子 rename，读 Pod（不同 UID）重新挂载后得到相同 SHA-256，随后只
删除该 run-token 的 checkpoint、manifest 和空目录。该次 smoke 的应用层写入为
486.43 MiB/s、读取为 659.53 MiB/s；它只作路径和语义烟测，样本太小且可能命中
page cache，不能作为磁盘性能结论。证据位于
`benchmark/results/storage/pvc-persistence-stor260806f/`。

正式性能测试分 4 KiB random read/write（IOPS/latency）与 1 MiB sequential read/write
（MiB/s），iodepth 1/16/64，单 Pod 和并发 4 Pod，各 N=3；使用受限文件而不是整盘
破坏性测试。应用层另测 1/5/20 GiB checkpoint 的写入、`fsync`、加载和 checksum。
通过条件由管理员根据后端能力先定阈值；最低硬条件是 checksum 0 差异、PVC quota
拒绝行为可解释、Pod 重建后可读、共享 CSI 在节点故障后能安全 reattach。当前
`/data/k3s` XFS mount 显示 `noquota`，所以不能假设 500Gi PVC request 会产生文件
系统硬限额；不得用填满共享卷的破坏性方法验证。若要验收 storage quota，管理员
需启用并绑定 XFS project quota，或提供会实际执行容量限制的 CSI。

## 12. 推荐实施顺序和状态

依赖关系决定了建议顺序，而不是原清单的编号顺序：

| 阶段 | 内容 | 当前状态 | 解锁条件 |
|---|---|---|---|
| P0 | 只读基线、时间模型、同构 NN/Transformer N=3 | **完成** | 45/45、报告和精确清理已完成 |
| P1 | LGB CPU/OpenCL/CUDA build + smoke + N=3 | **已准备，未运行** | 三个内部 digest、builder/registry 权限；GPU smoke |
| P2 | 存储物理取证与单节点 persistence | **部分完成** | 独立盘与 64MiB 跨 Pod smoke 已通过；正式 fio N=3、quota、共享 CSI 待补 |
| P3 | DCGM/Prometheus/Grafana | **blocked** | 组件选择、版本/digest、维护窗口 |
| P4 | 8-GPU/多节点 NCCL/DDP | **blocked** | ≥2 GPU 节点、quota=8、网络/拓扑确认 |
| P5 | Kueue TAS/wait/Fair Sharing/preemption | **blocked** | Kueue 配置与对象的专项授权、维护窗口 |
| P6 | Volcano Gang/Queue/preempt/reclaim | **blocked** | 最终 actions、专项授权、维护窗口 |
| P7 | 并发与 2–4 小时公平性 soak | **设计完成** | P1、P3、相应调度策略通过；长期窗口 |
| P8 | Pod/Node 故障与 checkpoint 恢复 | **部分完成** | Pod retry 与跨 Pod checkpoint 文件持久性已通过；应用 resume/Node 故障仍依赖 P2、P3、第二节点及批准 |

实际执行建议为 P1 image smoke → P2 正式存储/共享 CSI → P3 → P4 → P5/P6 →
P7 → P8。每完成一个阶段
都独立生成 evidence index、checksum、结果表和 cleanup receipt；后续阶段不能覆盖
前一阶段的原始证据。
