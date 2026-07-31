# GPU K8s 集群使用与测试手册（khalil）

> 用途：团队内部使用 GPU 集群时的登录、作业提交、冒烟测试和排障。  
> 环境核对日期：2026-07-30。  
> 本文不含 kubeconfig、Token、SSH 私钥、密码或业务密钥。

## 1. 已验证环境

| 项目 | 当前值 |
|---|---|
| SSH 入口 | `120.253.243.127:2226` |
| SSH 用户 | `khalil` |
| 服务器主机名 | `zhisui` |
| Kubernetes | k3s `v1.35.4+k3s1` |
| Kubernetes API | 在服务器上使用 `https://127.0.0.1:6443` |
| 工作命名空间 | `khalil` |
| GPU 节点 | `gpu-dev-01`，`Ready=True` |
| GPU | 8 × NVIDIA GeForce RTX 5090 |
| 单卡显存 | 32607 MiB |
| NVIDIA 驱动 | `595.71.05` |
| 已验证 PyTorch | `2.13.0+cu130` |
| RuntimeClass | `nvidia` |
| GPU 资源名 | `nvidia.com/gpu` |
| GPU 节点标签 | `accelerator=nvidia` |
| 默认 StorageClass | `local-path` |

当前是单 GPU 节点、8 卡共享环境。作业申请的 GPU 数之和超过剩余卡数时，新作业会保持 `Pending`。

## 2. 访问与安全

### 2.1 SSH 登录

在本地终端执行：

```bash
ssh -p 2226 khalil@120.253.243.127
```

登录后进入项目：

```bash
cd /home/khalil/DataCleaning7.3/QuantFM
```

### 2.2 kubeconfig 安全约束

服务器现有 `/etc/rancher/k3s/k3s.yaml` 是管理员 kubeconfig，不得上传 Git，不得作为群文件转发，不得复制其中的客户端证书或私钥。

团队成员长期使用时，应由管理员按人签发只能访问指定命名空间的凭据，并通过私密渠道分发。不要让多人共用一个 Token。

已获授权的服务器终端可临时设置：

```bash
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
export GPU_NS=khalil
```

每次操作都要显式带命名空间：

```bash
kubectl -n "$GPU_NS" get pods,jobs
```

## 3. 提交前检查

```bash
kubectl cluster-info
kubectl get node gpu-dev-01 -L accelerator
kubectl get runtimeclass nvidia
kubectl get node gpu-dev-01 \
  -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
kubectl -n "$GPU_NS" auth can-i create jobs.batch
kubectl -n "$GPU_NS" get pods,jobs -o wide
```

预期：

- 节点 `gpu-dev-01` 为 `Ready`；
- `accelerator` 标签为 `nvidia`；
- GPU 可分配数为 `8`；
- RuntimeClass `nvidia` 存在；
- 当前身份对 `khalil` 命名空间可创建 Job。

查看当前 GPU 是否已被其他 Pod 申请：

```bash
kubectl get pods -A -o custom-columns=\
'NS:.metadata.namespace,POD:.metadata.name,NODE:.spec.nodeName,GPU:.spec.containers[*].resources.limits.nvidia\.com/gpu,STATUS:.status.phase'
```

## 4. 一卡冒烟测试

配置文件：

```text
k8s/gpu-cluster/gpu-usage-config.yaml
```

这个 Job 会申请 1 张 GPU，验证：

1. `nvidia` RuntimeClass 能否创建容器；
2. `nvidia.com/gpu: 1` 能否被正确调度和隔离；
3. `nvidia-smi` 能否读取 GPU；
4. 当前 PyTorch 能否识别 CUDA；
5. GPU FP16 矩阵乘法能否完成并产生有限数值。

为防止异常作业长期占用 GPU，该 Job 设置了 5 分钟硬超时。

### 4.1 语法检查

```bash
kubectl apply --dry-run=server \
  -f k8s/gpu-cluster/gpu-usage-config.yaml
```

### 4.2 提交和查看

```bash
kubectl -n "$GPU_NS" delete job gpu-usage-smoke --ignore-not-found
kubectl apply -f k8s/gpu-cluster/gpu-usage-config.yaml
kubectl -n "$GPU_NS" get pod -l app=gpu-usage-smoke -w
```

Pod 进入 `Completed` 后按 `Ctrl+C`，然后查看日志：

```bash
kubectl -n "$GPU_NS" logs job/gpu-usage-smoke
kubectl -n "$GPU_NS" describe job gpu-usage-smoke
```

通过标准：

- Job 显示 `Complete`；
- Pod 显示 `Completed`；
- 日志显示 `visible_gpus=1`；
- 日志最后出现 `GPU_SMOKE_PASS`；
- Events 中没有 `FailedScheduling`、`FailedCreatePodSandBox` 或 `FailedMount`。

完成后可立即清理；如不清理，Job 也会在完成 1 小时后自动删除：

```bash
kubectl -n "$GPU_NS" delete job gpu-usage-smoke --ignore-not-found
```

## 5. 改成自己的 GPU Job

复制配置后，至少修改以下项：

- `metadata.name`：使用唯一、可识别的作业名称；
- `image`：改为包含代码和依赖的固定版本镜像；
- `command`/`args`：改为业务启动命令；
- CPU、内存和 GPU 的 `requests`/`limits`；
- 数据卷、输出卷与挂载路径；
- `owner`、`project`、`purpose` 等标签。

GPU 数量必须在 `requests` 和 `limits` 中保持一致，例如 2 卡：

```yaml
resources:
  requests:
    cpu: "4"
    memory: 32Gi
    nvidia.com/gpu: "2"
  limits:
    cpu: "8"
    memory: 64Gi
    nvidia.com/gpu: "2"
```

保留以下调度配置：

```yaml
runtimeClassName: nvidia
nodeSelector:
  accelerator: nvidia
```

### 5.1 镜像和 hostPath

冒烟文件为了复用当前已验证的 PyTorch 环境，挂载了：

```text
/home/khalil/DataCleaning7.3/QuantFM
/home/khalil/.local/share/uv/python
```

这是单节点上的 `hostPath`，只适合当前 `gpu-dev-01`。正式共享作业建议将代码和依赖固化到内部镜像，将业务数据放在 PVC/对象存储中，不要默认其他节点也有相同宿主机目录。

## 6. 业务作业观测

```bash
# 作业和 Pod
kubectl -n "$GPU_NS" get jobs,pods -o wide

# 持续日志
kubectl -n "$GPU_NS" logs -f job/<job-name>

# 上一个容器的失败日志
kubectl -n "$GPU_NS" logs <pod-name> --previous

# 调度、挂载、镜像和容器事件
kubectl -n "$GPU_NS" describe pod <pod-name>
kubectl -n "$GPU_NS" get events --sort-by=.lastTimestamp

# CPU/内存使用率（需 metrics-server 正常）
kubectl -n "$GPU_NS" top pod <pod-name>
```

在授权的 Pod 内检查 GPU：

```bash
kubectl -n "$GPU_NS" exec <pod-name> -- nvidia-smi
```

## 7. 常见故障

### Pod 一直 `Pending`

先执行：

```bash
kubectl -n "$GPU_NS" describe pod <pod-name>
```

常见原因：GPU 已被占满、CPU/内存不足、节点标签不匹配、命名空间配额不足。请减小资源申请或等待现有作业结束，不要直接删除他人作业。

### `torch.cuda.is_available() == False`

检查：

- 是否声明 `runtimeClassName: nvidia`；
- 是否申请 `nvidia.com/gpu`；
- PyTorch 是否为 CUDA 版；
- PyTorch CUDA 版本与当前 NVIDIA 驱动是否兼容；
- `nvidia-smi` 在同一容器中是否可用。

### `FailedMount` 或 `hostPath type check failed`

配置中的宿主机路径在目标节点不存在，或容器 UID/GID 无权读取。确认 Pod 被调度到 `gpu-dev-01`，并核对 `runAsUser: 1006` / `runAsGroup: 1009`。

### `CUDA out of memory`

这是单张已分配 GPU 内的显存不足，不是 Kubernetes 内存 `limits` 报错。减小 batch/context，启用 bf16、梯度累积、activation checkpointing 或 FSDP。

### `ImagePullBackOff`

确认镜像地址和 tag 存在，节点可访问镜像仓库，私有仓库的 `imagePullSecrets` 已在 `khalil` 命名空间中配置。不要把仓库密码直接写入 YAML。

## 8. 共享与使用约定

- 群里可发本文和无凭据 YAML，不发 kubeconfig、Token、私钥或业务密钥。
- 长任务启动前在群里说明卡数和预计结束时间。
- 每个 Job 都设置 `owner` 和 `purpose` 标签，名称不要复用他人的 Job。
- 非必要不申请 8 卡；调试先用 1 卡和小数据。
- 作业结束后清理无用 Job/Pod；删除 PVC 前必须确认数据已备份。
- 禁止执行跨命名空间的批量删除命令。

## 9. 最小验收记录

每次集群、驱动、RuntimeClass 或 PyTorch 大版本升级后，建议在群里回复：

```text
测试时间：YYYY-MM-DD HH:MM
测试人：<name>
Job：gpu-usage-smoke
节点：gpu-dev-01
GPU：NVIDIA GeForce RTX 5090
PyTorch/CUDA：<torch version> / <cuda build>
结果：PASS / FAIL
日志末行：GPU_SMOKE_PASS / <error summary>
```
