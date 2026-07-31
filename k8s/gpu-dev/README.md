# QuantFM：`gpu-dev-khalil` 的固定运行方案

最终结构只有这一条链路：

```text
gpu-dev-khalil
  -> gpu-dev/batch Job
     -> registry.zs/gpu-dev/khalil-quantfm:<immutable-tag>（代码和 Python 环境）
     -> 显式申请 GPU / CPU / memory
     -> gpu-dev/quantfm-data -> 保留的 500 GiB PV（独立 /data 数据盘）
```

Job 没有 `hostPath`，也不读取节点上的 `/home`。代码和 `.venv` 在镜像中；
PVC 整卷挂到 `/mnt/quantfm`，数据、缓存、临时文件、日志、checkpoint 和输出
都写到该卷。

## 关于“没有 PVC 权限”

`gpu-dev-khalil` 不能 `get/list/create` PVC，但这**不妨碍** Job 挂载同一命名
空间中管理员已经创建好的 PVC。挂载由 Kubernetes/kubelet 完成，提交者不需要
PVC API 权限。

当前真正的门槛是旧 claim 位于 `khalil/quantfm-data`。PVC 不能跨 namespace
挂载，因此管理员必须先把保留的 PV 安全重绑为 `gpu-dev/quantfm-data`。这是
一次性动作，不是以后每次训练都要做。

## 一次性准备

### 1. 管理员完成存储切换

管理员必须确认以下三条成功标记：

```text
MIGRATION_COPY_PASS
MIGRATION_VERIFY_PASS
MIGRATION_ACTIVATE_PASS
```

随后在保持 PV 为 `Retain`、不删除 PV 和底层目录的前提下，将
`pvc-792a789f-ebf3-4a57-9c32-58f23c0c9580` 从
`khalil/quantfm-data` 重绑到 `gpu-dev/quantfm-data`。目标 PVC 清单是：

```text
k8s/storage/quantfm-data-gpu-dev-rebind.yaml
```

管理员最终应书面确认：新 PVC 为 `Bound`、仍指向上述 PV、容量 `500Gi`、
`RWO`、`local-path`，底层仍是 `/data/k3s/storage/...` 的独立数据盘，并且卷
顶层存在已激活的 `runs/`。

### 2. 构建并推送代码镜像

构建上下文采用白名单；`quant_fm/runs`、`.venv`、`.git` 和带凭据的 handout
都不会进入构建上下文。

构建机/CI 需要有 Docker build 权限，并按 handout 预先登录 `registry.zs`；
凭据不得写入 Dockerfile、脚本参数或镜像层。当前 `khalil` 登录账号没有
`/var/run/docker.sock` 权限，因此必须由获准的构建机或 CI 执行下面的脚本。

```bash
TAG="$(git rev-parse --short=12 HEAD)-$(date -u +%Y%m%d%H%M%S)-py312-torch213-cu130"

./k8s/gpu-dev/build-and-push-image.sh "${TAG}"
```

镜像内有两个指向 PVC 的兼容链接：

```text
/workspace/QuantFM/quant_fm/runs -> /mnt/quantfm/runs
/home/khalil/DataCleaning7.3/QuantFM/quant_fm/runs -> /mnt/quantfm/runs
```

第二条只是容器内链接，用来兼容旧 manifest 中保存的绝对路径；它不是
`hostPath`，不会访问节点 `/home`。后续重建 manifest 时可以统一改成
`/mnt/quantfm/runs/...`，但首次切换不需要冒险改动 48 万个文件的数据清单。

## 第一次验收

收到管理员的 Bound 确认后：

```bash
export QUANTFM_PVC_CONFIRMED=gpu-dev/quantfm-data
TAG='<上一步推送的不可变 tag>'

./k8s/gpu-dev/submit-job.sh cpu-smoke "${TAG}"
./k8s/gpu-dev/submit-job.sh gpu-smoke "${TAG}"
./k8s/gpu-dev/submit-job.sh train-smoke "${TAG}"
```

CPU 检查必须输出 `CPU_PVC_SMOKE_PASS`；GPU 检查必须同时打印 RTX GPU、Torch
CUDA 信息和 `GPU_PVC_SMOKE_PASS`；短训练必须输出
`TRAINING_CHECKPOINT_RESUME_PASS`，证明 checkpoint 写入 PVC 后能够恢复。提交脚本
固定使用：

```text
~/.kube/config-gpu
gpu-dev-khalil@k3s-gpu
gpu-dev
```

并且每次先做服务端 dry-run，再创建 Job。

## 日常训练

以单卡为默认值，Job 名必须以 `khalil-` 开头：

```bash
export QUANTFM_PVC_CONFIRMED=gpu-dev/quantfm-data
TAG='<已推送的不可变 tag>'

./k8s/gpu-dev/submit-job.sh train \
  "${TAG}" \
  khalil-backbone-moe-20260731 \
  quant_fm/pretrain/config_v2_backbone_moe.yaml
```

训练模板显式申请 1 GPU、4/8 CPU 和 8/16 GiB 内存。按实际程序调整资源；
GPU request/limit 必须相等，只能为 1–4，优先 1。完成后及时删除 Job，PVC
中的数据不会随 Job 删除。

## 删除旧 `/home` 副本的门槛

只有在全量迁移验证、激活、重绑、CPU/GPU 读写检查，以及一次短训练的
checkpoint 写入和恢复全部通过后，才可以清理原来的
`/home/khalil/DataCleaning7.3/QuantFM/quant_fm/runs`。在此之前它仍是唯一回退
副本，不能删除。
