# GPU 调度评估：Kueue / Volcano / 裸 Kubernetes

> 核验日期：2026-07-31 UTC  
> 集群：k3s `v1.35.4+k3s1`，单节点 `gpu-dev-01`，8 张 NVIDIA GPU  
> 测试 namespace：`gpu-dev`，GPU ResourceQuota hard limit = 4  
> Volcano：`v1.15.1`，已安装并完成单次 GPU 功能评估  
> 管理访问：`/etc/rancher/k3s/k3s.yaml`，context `default`；本文和证据库均未保存凭据内容

## 1. 结论

1. **Volcano 已安装成功，控制面实际可用。** 官方 `v1.15.1` manifest 经 43 个对象 server-side dry-run 后应用；11 个 Volcano CRD 均 `Established=True`，7 个 webhook 配置均有 CA bundle，scheduler/controller/admission 三个 Deployment 均 `1/1 Ready`。从 apply 开始到最后一个 Deployment Available 约 `19.2s`；这是安装就绪时间，不是作业 wall-clock。
2. **Volcano 确实完成了 GPU binding 和 CUDA 执行。** 原生 `batch/v1 Job` 的 Pod 由 `reportingComponent=volcano` 绑定到 `gpu-dev-01`；容器看到 1 张 RTX 5090，`torch.cuda.is_available()=true`，CUDA tensor 计算结果为 42。该次 Job API wall-clock 为 `14s`，客户端提交开始到观察 Complete 为 `14.200s`。
3. **Queue `capability` 硬上限实际生效。** Queue GPU capability=1、holder 已占 1 时，waiter 保持 Pending 且无 `nodeName`，事件明确为 `queue resource quota insufficient: insufficient nvidia.com/gpu`。当时按全群 Running Pod request 口径节点仍有 7 GPU headroom，故不是节点物理 request 不足；holder 释放后 waiter 才绑定和运行。waiter Job API wall-clock 为 `44s`。
4. **PodGroup/Gang 的绑定门槛实际生效。** `minMember=2`、`minResources=2 GPU`，Queue capability=1 时两个成员均 Pending、零 binding；capability 提升到 2 后两 Pod 同一 API 秒被 Volcano 绑定。应用实际启动时间差 `0.099s`，Job API wall-clock 为 `24s`。这证明 Gang 门槛，不代表容器原子同时启动。
5. **同 Queue 抢占在显式启用后实际生效。** 官方默认 actions 只有 `enqueue, allocate, backfill`，默认并不执行 `preempt` 或 `reclaim`。本轮临时改为 `allocate, backfill, preempt`：Priority 100、3 GPU victim 被 Volcano 发出 `Evict: preempt`，随后 Priority 1000、1 GPU Job 绑定并启动；高优先级 Job API wall-clock 为 `16s`。实验后已恢复默认配置。
6. **跨 Queue `reclaim`、借用和公平份额没有实测。** 单节点有 8 GPU，但共享 namespace quota 只有 4 GPU；在不绕过 quota、不影响其他用户的前提下，无法构造可信的跨 Queue 物理资源压力。因此只能陈述官方能力，不能写“已生效”。
7. **Kueue 当前只完成 gate 级验证。** Kueue `v0.19.0` 仍为 `1/1 Ready`，且没有被 Volcano 安装或测试修改。此前探针证明 Job suspend/Workload 创建门控生效，但未获得 GPU Admission；其 quota、borrowing、fair sharing、Workload preemption 仍未在本集群端到端验证。
8. **裸 Kubernetes 的 GPU 资源约束、ResourceQuota 和 Pod Priority 抢占已有历史单次证据，但没有批作业队列语义。** 这些历史用例与本次 Volcano 的 GPU 数、镜像和业务时长不同，不能直接比较时延优劣。

总判断：**在当前单节点集群中，Volcano 的 GPU 调度、Queue capability、Gang 和显式同 Queue preempt 均有完整实际证据链；Kueue 只有 admission gate 证据；裸 K8s 有基础约束和 Pod 抢占证据。任何方案都没有足够样本支持吞吐、median/P95 或“谁更快”的结论。**

## 2. 当前状态与证据等级

| 能力 | 裸 Kubernetes | Kueue | Volcano |
|---|---|---|---|
| GPU request 约束 | 历史单次实测生效 | Admission 后仍由默认 scheduler 执行；本轮未到此阶段 | **实测生效**：Volcano Scheduled + CUDA tensor |
| 显式批队列 | 不支持 | LocalQueue/ClusterQueue；仅 gate 被验证 | **Queue 实测生效** |
| GPU 配额/份额 | ResourceQuota 硬上限实测 | nominalQuota 未验证 | **capability 实测**；deserved/guarantee 未测 |
| 排队后释放运行 | 历史节点资源等待实测 | 未成功 Admission | **实测生效** |
| Gang | 未验证 | quota all-or-nothing；严格放置未验证 | **PodGroup 门槛实测生效** |
| 抢占 | Pod Priority 历史实测 | Workload preemption 未验证 | **同 Queue preempt 实测生效，但默认关闭** |
| 跨队列回收 | 不支持 | Cohort borrowing/preemption 未验证 | `reclaim` 能力存在，**未实测** |
| 公平分享 | 无批作业公平策略 | 未验证 | capacity/proportion 未验证 |
| 性能统计 | N=1 历史样本 | 右删失探针 | 每场景 N=1 |

下图是 Volcano 安装前 2026-07-31 06:25 UTC 的受限账号快照。图中的 `VOLCANO_API NONE` 是当时事实，不代表安装后的现状；`current-kueue-results.json` 也作为安装前历史证据原样保留。

![Volcano 安装前的受限账号快照](assets/gpu-scheduler-evaluation/screenshots/01-current-cluster-access.png)

## 3. 三种方案的机制与限制

| 维度 | 裸 Kubernetes | Kueue | Volcano |
|---|---|---|---|
| 定位 | 通用 Pod scheduler | 作业准入、逻辑配额与队列 | 独立批处理 scheduler |
| 调度链路 | Job → Pod → `default-scheduler` | Job → Workload Admission → `default-scheduler` | Job/PodGroup → `volcano` scheduler |
| 用户队列 | 无 | LocalQueue → ClusterQueue | cluster-scoped Queue |
| GPU 配额 | namespace ResourceQuota hard limit | ResourceFlavor + nominalQuota | `capability` / `deserved` / `guarantee.resource` |
| 超限行为 | Pod 创建被 quota 拒绝，或 Pending | Workload 未准入，Job 保持挂起 | PodGroup/Job 在 Queue 内等待 |
| 借用 | 无 | Cohort borrowing/lending | deserved 超额借用 + `reclaim` |
| 公平分享 | 无批作业级策略 | ClusterQueue/LocalQueue fair sharing | capacity 或 proportion/DRF 等插件 |
| 抢占粒度 | Pod Priority | Workload | 同 Queue `preempt`；跨 Queue `reclaim` |
| Gang | 本集群未验证 | 整体 quota reservation；物理就绪还需 TAS/`waitForPodsReady` | PodGroup `spec.minMember` / `minResources` |
| Backfill | 无批语义 | 排序策略可跳过暂时不可准入 Workload | `backfill` action 主要处理未声明 request 的 BestEffort Pod |
| 运维面 | 最小 | CRD + controller + webhook | CRD + scheduler + controller + webhook + 全局 scheduler 配置 |
| 当前集群状态 | 在用 | v0.19.0，1/1 Ready | v1.15.1，三个 Deployment 1/1 Ready |

### 3.1 裸 Kubernetes

ResourceQuota 与队列不是同一件事：

- ResourceQuota 只表达 namespace 的硬上限；永久超限时，Job 对象可以存在，但 Job Controller 创建 Pod 会反复 `FailedCreate`。
- `default-scheduler` 只决定当前 Pod 能否放到某节点；它的内部 pending queue 不是租户可配置的批作业队列。
- Pod Priority 抢占不知道训练作业边界。victim 的 CUDA 上下文会丢失；checkpoint、恢复、重试和 wasted compute 都由应用承担。

官方语义见 [Kubernetes ResourceQuota](https://kubernetes.io/docs/concepts/policy/resource-quotas/) 与 [Pod Priority and Preemption](https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/)。

### 3.2 Kueue

Kueue 不替换 kube-scheduler。它先对 Workload 做逻辑 quota reservation/admission，之后才解除 Job 挂起，节点放置仍由默认 scheduler 完成。[Kueue Overview](https://kueue.sigs.k8s.io/docs/overview/)

它对 GPU 的主要增量是 LocalQueue/ClusterQueue、Cohort borrowing/lending、fair sharing 和 Workload 级 preemption。但 `Admitted=True` 不自动等于节点可放置或 CUDA 可用。例如一个 Pod 请求 8 GPU、两台节点各有 4 GPU时，若没有合适的拓扑约束，逻辑总量满足也无法落地；相关能力见 [Topology Aware Scheduling](https://kueue.sigs.k8s.io/v0.19/docs/concepts/topology_aware_scheduling/)。

Kueue 的 all-or-nothing 首先是 quota admission，不是 kube-scheduler 的原子启动事务。严格多 Pod 就绪通常还需 TAS、`waitForPodsReady`、超时和 requeue。[All-or-nothing](https://kueue.sigs.k8s.io/docs/concepts/all_or_nothing/)

### 3.3 Volcano

Volcano Queue 字段含义不同：

- `capability`：Queue 硬上限，本轮已实测；
- `deserved`：capacity plugin 使用的软权益，可借用，资源紧张时超出部分可被回收；
- `guarantee.resource`：不可借给其他 Queue 的保留资源，隔离强但可能降低利用率；
- `weight`：proportion plugin 计算动态份额时使用。

Queue schema 和语义见 [v1.15.1 Queue API 类型](https://github.com/volcano-sh/volcano/blob/v1.15.1/staging/src/volcano.sh/apis/pkg/apis/scheduling/v1beta1/types.go) 与 [Queue Resource Management](https://volcano.sh/docs/keyfeatures/queueresourcemanagement/)。capacity 与 proportion 是不同份额模型，不能把字段混为一谈。

原生 `batch/v1 Job` 关联显式 PodGroup 时，本轮采用：

```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: PodGroup
spec:
  queue: khalil-volcano-gang
  minMember: 2
  minResources:
    nvidia.com/gpu: "2"
---
apiVersion: batch/v1
kind: Job
spec:
  template:
    metadata:
      annotations:
        scheduling.k8s.io/group-name: khalil-volcano-gang-pg
    spec:
      schedulerName: volcano
```

关键点是 scheduler 实际读取 `scheduling.k8s.io/group-name`；Queue 由 `PodGroup.spec.queue` 指定，不能把 `scheduling.volcano.sh/group-name` 当成等价替代。[Volcano v1.15.1 `getJobID`](https://github.com/volcano-sh/volcano/blob/v1.15.1/pkg/scheduler/api/job_info.go)

官方安装默认 scheduler 配置为：

```yaml
actions: "enqueue, allocate, backfill"
```

因此默认不执行 `preempt` 或 `reclaim`；PriorityClass 和 `reclaimable: true` 单独存在不足以证明抢占/回收已启用。[v1.15.1 默认配置](https://github.com/volcano-sh/volcano/blob/v1.15.1/installer/helm/chart/volcano/config/volcano-scheduler.conf)

## 4. Volcano 安装与变更控制

本轮在用户明确确认的维护窗口中执行，并按照用户后续指令使用：

```text
kubeconfig: /etc/rancher/k3s/k3s.yaml
context:    default
```

没有复制、打印、提交或写入该 kubeconfig 内容。它是本次评估的一次性管理员例外，不是日常 GPU Job 提交方式；报告完成后不得复用于无关任务。检查时该文件权限为 `0644 root:root`，本轮未擅自修改；建议管理员评估后收紧至 `0600`。

| 安装项目 | 实际结果 |
|---|---:|
| 版本 | Volcano v1.15.1 |
| 官方 manifest SHA-256 | `b5fb45a57bcd1132be7055354999d244266873710609afd1f162005772d15c41` |
| server-side dry-run | 43/43 对象通过 |
| `kubectl apply` 客户端 RTT | `2.505s` |
| install begin → 最后 Deployment Available | 约 `19.231s`，Deployment condition 为秒级精度 |
| Volcano CRD | 11/11 `Established=True` |
| Volcano webhook 配置 | 7/7 CA bundle 非空 |
| Volcano service EndpointSlice | 3 个，3 个 ready address |
| scheduler/controller/admission | 各 `1/1 Ready`，镜像均 v1.15.1 |
| admission init Job | Complete |
| Kueue 共存状态 | v0.19.0，`1/1 Ready` |

![Volcano v1.15.1 安装与就绪证据](assets/gpu-scheduler-evaluation/screenshots/08-volcano-installation.png)

治理文件 [AGENTS.md](../AGENTS.md) 已明确：允许的标准 Volcano 系统资源与标准名称、原生 `batch/v1 Job` 限制，以及测试用 `khalil-` Queue/PodGroup/PriorityClass；所有 GPU 测试仍必须位于 `gpu-dev`、受共享 quota 约束。

## 5. Volcano GPU 实测

### 5.1 测试边界

所有最终验收场景均满足：

- namespace=`gpu-dev`，原生 `batch/v1 Job`，项目对象使用 `khalil-` 前缀；
- 内部镜像 `registry.zs/gpu-dev/...`；
- `runtimeClassName: nvidia`、`accelerator: nvidia`、NVIDIA toleration；
- CPU/内存/GPU requests 和 limits 明确，GPU request=limit；
- 单 Job 最大请求 3 GPU，合计不超过 namespace hard quota 4；
- `backoffLimit: 0`、TTL 和 active deadline；
- 测试前 Running GPU request=0，测试 victim 只使用本轮自建对象，没有抢占其他用户；
- 探针只写 stdout，无训练数据、checkpoint 或持久输出，因此不使用 PVC。

### 5.2 GPU binding、CUDA 和 wall-clock

最终成功样本按 Job、Pod、PodGroup UID 过滤事件，避免混入早先同名对象。

| 阶段 | UTC / 时长 |
|---|---:|
| client create begin → return | `0.684s` |
| Job 创建 | `07:48:02` |
| Pod Scheduled by Volcano | `07:48:03` |
| 容器开始 | `07:48:05` |
| 应用日志开始 → 结束 | `07:48:05.736` → `07:48:13.634`，`7.898s` |
| 容器 API runtime | `8s` |
| Job Complete | `07:48:16` |
| Job API wall-clock | `14s` |
| client create begin → Complete observed | `14.200s` |

容器实际输出：RTX 5090、`VISIBLE_GPU_COUNT=1`、Torch `2.13.0+cu130`、`TORCH_CUDA_AVAILABLE=true`、`CUDA_TENSOR_RESULT=42`。

![Volcano GPU/CUDA 与 wall-clock](assets/gpu-scheduler-evaluation/screenshots/09-volcano-gpu-cuda-wallclock.png)

结论：**资源调度和 CUDA 执行均实际生效。** 该样本使用已缓存镜像和极短 tensor 探针，不代表训练吞吐。

### 5.3 Queue capability

Queue `khalil-volcano-quota` capability=1 GPU；holder 和 waiter 各请求 1 GPU。

| 观察 | 实际证据 |
|---|---|
| holder 运行时 | Queue `status.allocated.gpu=1` |
| waiter | Pod Pending、`nodeName` 为空、PodGroup Pending |
| Volcano 事件 | `queue resource quota insufficient: insufficient nvidia.com/gpu` |
| 节点 request 口径 | 8 allocatable，Running Pod GPU request 合计 1，headroom=7 |
| namespace quota | hard=4；快照 used=2，因为 Pending Pod request 也计入 ResourceQuota |
| 资源释放 | holder 容器 `07:41:24` 结束 |
| 后续 | waiter `07:41:26` Scheduled，`07:41:29` Started |

waiter API create→Scheduled `32s`、create→Started `35s`、runtime `6s`、Job wall-clock `44s`；client create→Complete observed `44.190s`。holder finish→waiter Scheduled 为 `2s`，finish→Started 为 `5s`。

![Volcano Queue capability 等待与释放](assets/gpu-scheduler-evaluation/screenshots/10-volcano-queue-capability.png)

结论：**Queue capability 硬上限实际生效，而且等待不是由节点 GPU request 不足造成。** “headroom=7”是 Kubernetes request 快照，不是 `nvidia-smi` 利用率遥测。

### 5.4 PodGroup/Gang

`khalil-volcano-gang-pg` 配置 `minMember=2`、`minResources.gpu=2`；两个成员各请求 1 GPU。

1. Queue capability=1 时，提交约 5.7s 后快照仍为 2 Pending、0 binding，PodGroup phase=Pending，并有 Queue GPU quota insufficient 事件。
2. `07:41:51.129` 客户端观察到 capability patch 返回；两个 Pod 的 API Scheduled 时间均为 `07:41:51`。因 Kubernetes Event/condition 只有秒级精度，不能声称 patch 后具体多少毫秒完成 binding。
3. 两容器 API startedAt 均为 `07:41:55`；应用日志实际开始为 `55.690` 和 `55.789`，spread=`0.099s`。
4. Job API wall-clock=`24s`，client create→Complete observed=`24.362s`，2/2 成功。

![Volcano PodGroup/Gang 阈值行为](assets/gpu-scheduler-evaluation/screenshots/11-volcano-gang.png)

结论：**低于 `minResources` 时零 binding，达到门槛后两成员均被绑定，Gang 调度门槛实际生效。** 最终 PodGroup conditions 会保留旧的 Unschedulable 条件，判断完成状态应结合 phase、Pod Scheduled 事件和 Job 状态，不能把 condition 列表当成只含最新状态。

### 5.5 同 Queue preempt

默认配置不含 preempt。为受控实测，临时全局配置为：

```yaml
actions: "allocate, backfill, preempt"
```

scheduler rollout ready 用时 `3.879s`。实验 Queue capability=3：

- victim：Priority 100，请求 3 GPU；
- preemptor：Priority 1000，请求 1 GPU；
- 节点仍有 5 GPU request headroom；两者合计 4，恰好不超过 namespace quota；
- high Job 创建同秒，Volcano 对 victim Pod 发出 `Evict: Pod is evicted, because of preempt`，kubelet 发出 `Killing`；
- high Pod 下一 API 秒由 Volcano Scheduled，创建后 `5s` 容器启动；
- victim 因 `backoffLimit:0` 最终 `Failed/BackoffLimitExceeded`，没有 replacement 或 checkpoint/resume。

| 指标 | 单次结果 |
|---|---:|
| high Job create → Scheduled | `1s` |
| high Job create → container start | `5s` |
| client create → Ready observed | `4.992s` |
| victim API start → Evict | 约 `4s`，事件秒级精度 |
| high container runtime | `8s` |
| high Job API wall-clock | `16s` |
| client create → Complete observed | `16.071s` |

![Volcano 显式同 Queue 抢占](assets/gpu-scheduler-evaluation/screenshots/12-volcano-preemption.png)

结论：**临时启用 `preempt` 后，同 Queue 抢占实际生效。** 这是 Queue capability 触发的逻辑抢占，不是节点物理耗尽。它不能外推为默认配置已经启用抢占，也不能证明真实训练可无损恢复。

### 5.6 Reclaim、借用和公平分享

本轮没有测试跨 Queue `reclaim`、capacity `deserved` 借用、`guarantee.resource` 保留、capacity/proportion/DRF 长期公平份额、GPU backfill、多节点拓扑或分布式训练。

原因不是功能失败，而是共享 quota=4、节点=8 的安全边界无法在不绕过 quota 的情况下构造可信跨 Queue 物理压力。报告将其标记为 N/A，而不是 Pass。

实验后配置恢复到 `enqueue, allocate, backfill`；6 个 Jobs、6 个 PodGroups、4 个 Queues、2 个 PriorityClasses 均逐名 NotFound，GPU quota used 回到 0。Volcano 三个 Deployment与 Kueue 均健康；Volcano 系统本身保留安装。

![Volcano 配置恢复与测试资源清理](assets/gpu-scheduler-evaluation/screenshots/13-volcano-restoration-cleanup.png)

## 6. 裸 Kubernetes 历史实测

这些实验在 2026-07-30 由管理员完成。历史清单包含单独 namespace、直接 Pod、5 GPU 请求和公网镜像，不符合当前 khalil 规则，已在 runner 中阻止重跑；这里只作为存档证据。

### 6.1 GPU 竞争

holder、waiter 都请求 5 GPU；holder 先运行约 30s，waiter 业务运行 10s。waiter 出现 `FailedScheduling: Insufficient nvidia.com/gpu`，holder 结束后才启动。

| 指标 | API 对象口径 |
|---|---:|
| waiter Job 创建 → 容器开始 | `34s` |
| waiter Pod 创建 → 容器开始 | `33s` |
| 容器运行 | `10s` |
| Job 创建 → Complete | `46s` |

![裸 Kubernetes GPU 竞争](assets/gpu-scheduler-evaluation/screenshots/02-bare-contention.png)

### 6.2 ResourceQuota

namespace GPU hard limit=1、Job 请求=2：Job 对象创建成功，但 Pod 创建被 API Server 拒绝；Pod 数为 0，并出现 `FailedCreate: exceeded quota`。

![裸 Kubernetes ResourceQuota](assets/gpu-scheduler-evaluation/screenshots/03-bare-quota.png)

这证明硬拒绝，不等于排队；也没有队列排序、公平分享或借用语义。

### 6.3 Pod Priority 抢占

Priority 1000、请求 5 GPU 的 Pod 抢占 Priority 100、请求 5 GPU 的 Pod；高优先级 Pod create→start `2s`，victim 出现 `Preempted/Killing`。

![裸 Kubernetes Pod Priority 抢占](assets/gpu-scheduler-evaluation/screenshots/04-bare-preemption.png)

victim 删除前的完整 startedAt 快照没有保存，无法严谨计算 wasted runtime；N=1 的 `2s` 也不是稳定延迟。

## 7. Kueue 当前实测边界

### 7.1 安全 gate 探针

符合当前规则的 1 GPU `batch/v1 Job` 显式 `suspend:true` 并带候选 queue label。实际产生 `CreatedWorkload`，观察 `10.448s` 内 Job 始终 suspended、Pod 数为 0；没有 Admission、容器启动或 CUDA 输出。

![Kueue admission gate](assets/gpu-scheduler-evaluation/screenshots/05-kueue-admission-gate.png)

因此只可写：**Kueue Controller、Workload 创建和准入前门控生效。** `10.448s` 是右删失等待下界，runtime 和 end-to-end wall-clock 均为 N/A。

### 7.2 Controller 接管时序诊断

早先未显式设置 `suspend:true` 的输入出现：default scheduler 先绑定 Pod，随后 Kueue Controller 创建 Workload 并删除 Pod；容器未启动。安全模板已改为显式挂起。

![Kueue Controller 接管时序](assets/gpu-scheduler-evaluation/screenshots/06-kueue-controller-catchup.png)

正式启用前仍需管理员核验 LocalQueue、ClusterQueue、ResourceFlavor、GPU nominalQuota、Cohort、fair sharing/preemption 配置，以及 webhook 对 `batch/v1 Job` 的接管顺序。Volcano 安装后 Kueue v0.19.0 仍为 `1/1 Ready`，但本轮没有重新执行其 GPU Admission 测试。

## 8. Wall-clock 汇总与可比性

| 方案/场景 | 排队/准入 | container runtime | API Job/Pod wall | client wall | 结论 |
|---|---:|---:|---:|---:|---|
| Volcano CUDA smoke | create→start `3s` | `8s`；app `7.898s` | Job `14s` | `14.200s` | GPU/CUDA Pass |
| Volcano Queue waiter | create→start `35s` | `6s` | Job `44s` | `44.190s` | Queue cap Pass |
| Volcano Gang 2×1 GPU | create→both start `13s` | 约 `8s` | Job `24s` | `24.362s` | Gang threshold Pass |
| Volcano preemptor | create→start `5s` | `8s` | Job `16s` | `16.071s` | 显式 same-Queue preempt Pass |
| 裸 K8s 历史 waiter 5 GPU | create→start `34s` | `10s` | Job `46s` | apply-return→Complete 约 `45s` | GPU wait Pass |
| 裸 K8s 历史 preemptor 5 GPU | create→start `2s` | `10s` | Pod `12s` | — | Pod preempt Pass |
| Kueue gate | 至少 `10.448s`，未 Admission | N/A | N/A | 右删失 `10.448s` | gate Pass，GPU path 未验收 |

口径说明：

- Kubernetes Job/Pod/condition/core Event 多为秒级时间戳；表中的整数秒来自 API 对象。
- `client wall` 来自同一进程的 `time.monotonic_ns()`，用于消除系统时钟跳变。
- Gang 的 `0.099s` spread 与 CUDA `7.898s` runtime 来自容器应用日志的毫秒时间戳。
- 裸 K8s 历史用例请求 5 GPU，Volcano 用例为 1–3 GPU，工作时长也不同；不能据此宣称某 scheduler 更快。
- 所有场景各 N=1，镜像已缓存，且没有真实训练、PVC/IO、checkpoint 或多节点通信；没有 median/P95 和吞吐结论。

## 9. 统一能力矩阵与后续缺口

| 实验 | 裸 K8s | Kueue | Volcano |
|---|---|---|---|
| GPU 真可用 | 历史 `nvidia-smi` | 未到 Admission | **Torch CUDA tensor 已完成** |
| 超额等待/释放 | 节点不足等待已测 | 未测 | **Queue capability 已测** |
| namespace 硬 quota | 已测 | 应继续保留 ResourceQuota | 测试受 hard=4 保护 |
| Gang | 未测 | 未测 | **2 成员门槛已测** |
| 优先级抢占 | Pod 级已测 | Workload 级未测 | **同 Queue 显式 preempt 已测** |
| borrowing/reclaim | 不支持 | 未测 | 未测 |
| fair sharing | 不支持 | 未测 | 未测 |
| 多节点拓扑 | 未测 | 未测 | 未测 |
| checkpoint/resume penalty | 未测 | 未测 | 未测 |

下一阶段若要做生产选择，应使用同一镜像、同一 1-GPU/多-GPU 业务、同一数据/PVC、同一运行时长，每场景至少 5 次报告 median/range；经验 P95 建议至少 20 次。还需记录 victim checkpoint、恢复次数、最终 wall-clock 惩罚和 GPU 利用率。

## 10. 推荐方案

- **只需要单 Pod 训练和最低运维面：** 裸 K8s + ResourceQuota + 保守 PriorityClass 足够，但没有批队列、公平份额或作业级抢占。
- **需要原生 Job 的逻辑 quota、跨 namespace 队列和 Cohort：** Kueue 的模型更贴合，但当前必须先补齐 LocalQueue/ClusterQueue 可见性和成功 GPU Admission 闭环。
- **需要严格 PodGroup/Gang 或 Volcano 插件链：** Volcano 已证明在本集群可工作；Queue/Gang/preempt 的调度语义强于裸 K8s，但引入独立 scheduler/webhook 和全局配置运维。

对 QuantFM 的建议是保留 Volcano 作为已安装试点，但不要仅凭 N=1 时延替换现有默认路径。先明确未来训练是否转为多 Pod、是否刚性需要 Gang、是否能完成 checkpoint/resume；若答案为是，再把 Volcano 纳入生产模板。Kueue 已存在且未被破坏，应由管理员完成可运行 Queue/Admission 验收后，再用统一负载比较两者。

抢占默认保持关闭。真实训练在开启任何 Kueue/Volcano 抢占前，必须通过 checkpoint/resume 和 victim wall-clock penalty 验收。ResourceQuota 继续作为 `gpu-dev` 最终硬护栏。

## 11. 证据、截图和复现

截图是由捕获的真实 Kubernetes JSON/Event、容器 stdout 和客户端 monotonic timeline 确定性渲染的 terminal-style 证据图，不是 GUI 像素截屏。所有事件按对象 UID 过滤；原始对象保存在 `raw/`，派生结果在 JSON 中。

主要文件：

```text
AGENTS.md
k8s/scheduler-evaluation/analyze_and_render.py
k8s/scheduler-evaluation/analyze_volcano.py
k8s/scheduler-evaluation/run_volcano_gpu_evaluation.sh
k8s/scheduler-evaluation/run_volcano_cuda_smoke.sh
k8s/scheduler-evaluation/volcano/bench/
k8s/scheduler-evaluation/volcano/upstream/volcano-development-v1.15.1.yaml
docs/assets/gpu-scheduler-evaluation/raw/volcano/
docs/assets/gpu-scheduler-evaluation/volcano-results.json
docs/assets/gpu-scheduler-evaluation/screenshots/
docs/assets/gpu-scheduler-evaluation/SHA256SUMS
```

离线重新分析与渲染不会访问集群或读取 kubeconfig：

```bash
.venv/bin/python k8s/scheduler-evaluation/analyze_and_render.py
.venv/bin/python k8s/scheduler-evaluation/analyze_volcano.py
cd docs/assets/gpu-scheduler-evaluation
sha256sum -c SHA256SUMS
```

历史裸 K8s runner 已被阻断，不应在当前 GPU 环境重跑。Volcano runner 会改全局 scheduler 配置，也只能在新维护窗口、明确授权和相同安全边界下执行。

## 12. 官方参考

- [Kueue Overview](https://kueue.sigs.k8s.io/docs/overview/)
- [Kueue ClusterQueue](https://kueue.sigs.k8s.io/v0.19/docs/concepts/cluster_queue/)
- [Kueue LocalQueue](https://kueue.sigs.k8s.io/v0.19/docs/concepts/local_queue/)
- [Kueue Preemption](https://kueue.sigs.k8s.io/v0.19/docs/concepts/preemption/)
- [Kueue Admission Fair Sharing](https://kueue.sigs.k8s.io/docs/concepts/admission_fair_sharing/)
- [Kueue Kubernetes Job integration](https://kueue.sigs.k8s.io/v0.19/docs/tasks/run/jobs/)
- [Volcano Queue](https://volcano.sh/docs/concepts/queue/)
- [Volcano Queue Resource Management](https://volcano.sh/docs/keyfeatures/queueresourcemanagement/)
- [Volcano Scheduling Actions](https://volcano.sh/docs/scheduler/actions/)
- [Volcano PodGroup](https://volcano.sh/docs/concepts/podgroup/)
- [Volcano Gang plugin](https://volcano.sh/docs/scheduler/plugins/gang/)
- [Volcano v1.15.1 release](https://github.com/volcano-sh/volcano/releases/tag/v1.15.1)
- [Volcano v1.15.1 default scheduler config](https://github.com/volcano-sh/volcano/blob/v1.15.1/installer/helm/chart/volcano/config/volcano-scheduler.conf)
- [Kubernetes ResourceQuota](https://kubernetes.io/docs/concepts/policy/resource-quotas/)
- [Kubernetes Pod Priority and Preemption](https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/)
