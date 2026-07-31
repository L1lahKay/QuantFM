# GPU 集群群公告文案

1. `gpu-usage-config.yaml`：1 卡 GPU 作业配置兼冒烟测试。
2. `GPU-K8S-集群使用与测试手册-khalil.md`：登录、提交、验收、排障和使用约定。

当前已核对环境：`gpu-dev-01`，8 × RTX 5090，`nvidia` RuntimeClass，GPU 资源名为 `nvidia.com/gpu`，工作命名空间为 `khalil`。

首次使用请先按手册运行 1 卡冒烟测试，日志出现 `GPU_SMOKE_PASS` 才算通过。调试请优先使用 1 卡；长时间或多卡任务启动前，请在群里说明卡数和预计结束时间。

安全提醒：群附件不包含 kubeconfig、Token、SSH 私钥或业务密钥。访问凭据请按人申请，并通过私密渠道分发，不要在群里转发管理员 kubeconfig。
