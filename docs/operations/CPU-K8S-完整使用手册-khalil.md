# CPU k8s 集群完整使用手册（khalil）

> 适用对象：第一次在公司 CPU k3s 集群上运行自己代码的开发者。  
> 最后核对日期：2026-07-24。  
> 本文以用户 `khalil`、工作命名空间 `khalil`、应用 `myapp` 为例；若管理员分配了其他命名空间，请全文替换。

## 1. 已验证的环境

当前已经实际验证：

- SSH 入口：`120.253.243.127:2226`
- SSH 用户：`khalil`
- SSH 登录后的服务器主机名：`zhisui`
- `kubectl`：`/usr/local/bin/kubectl`
- `k3s`：`/usr/local/bin/k3s`
- kubeconfig：`/etc/rancher/k3s/k3s.yaml`
- Kubernetes API：在服务器上通过 `https://127.0.0.1:6443` 访问
- 当前 kubeconfig 身份：`system:admin`
- 当前权限组：`system:masters`，即集群最高管理员权限
- CPU 节点：`cpu-server-01`（`192.168.2.14`）和 `cpu-worker-01`（`192.168.2.13`）
- 默认 StorageClass：`local-path`
- CPU 镜像仓库：`registry.cpu.zs:32443`
- Ingress 入口：`192.168.2.14:80`，通过不同 `*.zs` Host 路由

当前集群中还没有 `khalil` 命名空间。本文会把创建独立命名空间作为第一次使用的准备步骤。

### 1.1 重要安全说明

`/etc/rancher/k3s/k3s.yaml` 是管理员 kubeconfig。使用它时可以查看、修改和删除整个集群的资源，因此：

1. 日常操作始终显式指定 `-n khalil`。
2. 不要执行不理解的 `kubectl delete`、`kubectl replace --force` 或集群级命令。
3. 不要复制、上传或提交 `/etc/rancher/k3s/k3s.yaml` 到 Git。
4. 长期使用应让管理员签发只允许访问个人命名空间的受限 kubeconfig。
5. 该文件当前曾显示为 `644` 权限，普通用户可以读取管理员凭据；应由管理员改为更严格的权限并提供受限凭据。

---

## 2. 每次使用集群：登录和初始化

### 2.1 从 Windows 登录服务器

在 Windows PowerShell 中执行：

```powershell
ssh -p 2226 khalil@120.253.243.127
```

如果平时使用 SSH 密钥，登录过程可能不会询问密码。若出现私钥口令提示，它是用于解锁本机私钥的口令；若出现 Linux 密码提示，则输入 `khalil` 的 Linux 账户密码。

### 2.2 在服务器上启用 kubeconfig

SSH 登录成功后执行：

```bash
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
```

这个环境变量只对当前终端有效。每次重新 SSH 登录都要执行一次。

验证连接：

```bash
kubectl cluster-info
kubectl auth whoami
kubectl get nodes -o wide
kubectl get pods -A
```

预期：

- `cluster-info` 显示控制面地址。
- `auth whoami` 当前显示 `system:admin`。
- 两个 CPU 节点应为 `Ready`。
- 系统 Pod 大部分应为 `Running` 或 `Completed`。

为了减少重复输入，可以在当前终端定义：

```bash
export NS=khalil
alias k='kubectl -n khalil'
```

后面既可以写 `kubectl -n "$NS" get pods`，也可以写 `k get pods`。

---

## 3. 第一次使用：建立独立命名空间

先查看是否已经存在：

```bash
kubectl get namespace khalil
```

如果返回 `NotFound`，在确认该名称符合团队约定后创建一次：

```bash
kubectl create namespace khalil
kubectl get namespace khalil
```

以后所有业务资源都放在这里。不要把自己的应用部署到 `default`、`kube-system`、`argocd`、`platform` 等公共或系统命名空间。

查看该命名空间现有资源：

```bash
kubectl -n khalil get all
kubectl -n khalil get pvc,configmap,secret,ingress
```

如果管理员已经配置配额，查看方法是：

```bash
kubectl -n khalil get resourcequota
kubectl -n khalil describe resourcequota
kubectl -n khalil get limitrange
```

无论有没有默认配额，自己的工作负载都应显式填写 CPU 和内存的 `requests`、`limits`。

### 3.1 先运行一个最小 CPU 冒烟任务

创建 `cpu-smoke.yaml`：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: cpu-smoke
  namespace: khalil
spec:
  restartPolicy: Never
  containers:
    - name: cpu-smoke
      image: registry.cpu.zs:32443/test/alpine:1
      command: ["sh", "-c"]
      args:
        - |
          echo "running on: $HOSTNAME"
          echo "visible CPU count: $(nproc)"
          echo "CPU smoke test finished"
      resources:
        requests:
          cpu: "100m"
          memory: "64Mi"
        limits:
          cpu: "1"
          memory: "256Mi"
```

提交并查看输出：

```bash
kubectl apply -f cpu-smoke.yaml
kubectl -n khalil get pod cpu-smoke -w
# Pod 变为 Completed 后按 Ctrl+C
kubectl -n khalil logs cpu-smoke
kubectl -n khalil describe pod cpu-smoke
```

日志中出现 `CPU smoke test finished`，并且 Pod 状态为 `Completed`，说明镜像拉取、CPU 调度、容器启动和日志读取整条链路都正常。确认后清理：

```bash
kubectl -n khalil delete pod cpu-smoke
```

---

## 4. 代码在集群上运行的完整链路

一个应用从代码到集群通常经历：

```text
本地代码
  → Dockerfile 构建容器镜像
  → 推送到 registry.cpu.zs:32443
  → Deployment 或 Job 引用该镜像
  → Kubernetes 调度到 CPU 节点
  → kubectl 查看状态和日志
  → Service/Ingress（如需提供网络服务）
```

Kubernetes 不直接运行源码目录。最通用的方法是先把代码做成容器镜像，再由集群拉取镜像。

### 4.1 选择工作负载类型

| 场景 | Kubernetes 对象 | 特点 |
|---|---|---|
| Web API、网站、常驻进程 | `Deployment` | 长期运行，崩溃自动重启，支持滚动更新 |
| 一次性数据处理、批量推理 | `Job` | 执行完成后退出，可判断成功或失败 |
| 定时数据处理 | `CronJob` | 按 cron 表达式定时创建 Job |
| 需要稳定身份和独立磁盘的有状态应用 | `StatefulSet` | 比 Deployment 复杂，确认需要后再用 |

---

## 5. 准备一个最小示例项目

下面的 Python 示例没有第三方依赖，便于先跑通全链路。自己的项目可以保留相同目录结构，再替换代码和启动命令。

```text
myapp/
├── main.py
├── Dockerfile
└── k8s/
    ├── deployment.yaml
    ├── service.yaml
    ├── ingress.yaml
    └── job.yaml
```

### 5.1 `main.py`

```python
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(
            {
                "status": "ok",
                "app": "myapp",
                "hostname": os.getenv("HOSTNAME", "unknown"),
                "path": self.path,
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
```

### 5.2 `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY main.py /app/main.py

EXPOSE 8080
CMD ["python", "/app/main.py"]
```

如果构建机器访问不了 Docker Hub，请让管理员把 `python:3.11-slim` 同步到内部仓库，并把 `FROM` 改成平台给出的内部地址，例如：

```dockerfile
FROM registry.cpu.zs:32443/library/python:3.11-slim
```

不要随意猜内部基础镜像地址；先用仓库 UI 或向管理员确认该 tag 是否存在。

### 5.3 将本地代码传到服务器（可选）

如果代码在 Windows 本地，可以在 PowerShell 中执行：

```powershell
scp -P 2226 -r "C:\path\to\myapp" khalil@120.253.243.127:~/projects/
```

也可以在服务器上通过 Git 获取代码：

```bash
mkdir -p ~/projects
cd ~/projects
git clone <你的仓库SSH地址> myapp
cd myapp
```

不要把密码、API Key、数据库口令或 kubeconfig 放进项目目录或 Git 仓库。

---

## 6. 构建并推送镜像

可以在已配置 Docker 和仓库证书的开发机上构建，也可以在 `.14` 服务器上构建。先检查当前服务器是否具备 Docker 权限：

```bash
command -v docker
docker version
docker info
```

如果出现 Docker socket 的 `permission denied`，不要猜 sudo 密码；让管理员把 `khalil` 加入允许构建镜像的用户组，或改用 CI/CD。

### 6.1 登录内部镜像仓库

```bash
docker login registry.cpu.zs:32443
```

账号和密码向管理员获取。不要把密码直接写进脚本、命令行参数或 Git。

如果报 `x509: certificate signed by unknown authority`，说明构建机没有信任管理员提供的 `cpu-registry-ca.crt`。Linux Docker 的配置方式：

```bash
sudo mkdir -p /etc/docker/certs.d/registry.cpu.zs:32443
sudo cp cpu-registry-ca.crt /etc/docker/certs.d/registry.cpu.zs:32443/ca.crt
sudo systemctl restart docker
```

这一步需要系统管理员权限；服务器若已经配置好则不用重复执行。

### 6.2 构建并推送唯一 tag

进入项目目录：

```bash
cd ~/projects/myapp
```

Git 项目推荐用提交 SHA 作为 tag：

```bash
TAG=$(git rev-parse --short HEAD)
IMAGE="registry.cpu.zs:32443/khalil/myapp:$TAG"

docker build -t "$IMAGE" .
docker push "$IMAGE"
echo "$IMAGE"
```

没有 Git 时可使用时间戳：

```bash
TAG=$(date +%Y%m%d-%H%M%S)
IMAGE="registry.cpu.zs:32443/khalil/myapp:$TAG"
docker build -t "$IMAGE" .
docker push "$IMAGE"
```

不要长期使用 `latest`。唯一 tag 能明确知道集群正在运行哪个版本，也方便回滚。

---

## 7. 运行长期服务：Deployment

创建 `k8s/deployment.yaml`：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  namespace: khalil
  labels:
    app: myapp
spec:
  replicas: 1
  revisionHistoryLimit: 5
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
        - name: myapp
          image: registry.cpu.zs:32443/khalil/myapp:REPLACE_TAG
          imagePullPolicy: IfNotPresent
          ports:
            - name: http
              containerPort: 8080
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2"
              memory: "2Gi"
          readinessProbe:
            httpGet:
              path: /healthz
              port: http
            initialDelaySeconds: 3
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /healthz
              port: http
            initialDelaySeconds: 10
            periodSeconds: 20
```

资源字段含义：

- `requests`：调度时预留的最低资源。
- `limits`：容器允许使用的上限。
- `500m` CPU：半个 CPU 核。
- `2` CPU：两个 CPU 核。
- 内存超过 limit 时，容器可能被 `OOMKilled`。

不要通过 `nodeSelector` 强行固定节点，除非确实依赖该节点的本地数据或管理员明确要求。默认让调度器选择 `.13` 或 `.14`。

### 7.1 部署

```bash
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
export NS=khalil
TAG=$(git rev-parse --short HEAD)

sed "s/REPLACE_TAG/$TAG/g" k8s/deployment.yaml | kubectl apply -f -
kubectl -n "$NS" rollout status deployment/myapp --timeout=180s
kubectl -n "$NS" get pods -l app=myapp -o wide
```

如果 tag 是时间戳，请把 `TAG` 改成构建时实际使用的值。

### 7.2 查看日志和进入容器

```bash
kubectl -n khalil logs deployment/myapp --tail=100
kubectl -n khalil logs -f deployment/myapp
kubectl -n khalil exec -it deployment/myapp -- sh
```

`logs -f` 用 `Ctrl+C` 退出，不会停止应用。

---

## 8. 给长期服务提供稳定地址：Service

创建 `k8s/service.yaml`：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: myapp
  namespace: khalil
spec:
  selector:
    app: myapp
  ports:
    - name: http
      port: 80
      targetPort: http
```

应用：

```bash
kubectl apply -f k8s/service.yaml
kubectl -n khalil get service myapp
kubectl -n khalil get endpoints myapp
```

集群内其他 Pod 可通过以下地址访问：

```text
http://myapp.khalil.svc.cluster.local:80
```

同一命名空间内可以简写为：

```text
http://myapp
```

集群内自测：

```bash
kubectl -n khalil run nettest --rm -it --restart=Never \
  --image=registry.cpu.zs:32443/test/alpine:1 -- \
  wget -qO- http://myapp/
```

如果 `Service` 没有 endpoints，通常是 `Service.spec.selector` 与 Pod 的 label 不一致。

---

## 9. 从集群外访问：Ingress

创建 `k8s/ingress.yaml`：

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: myapp
  namespace: khalil
spec:
  ingressClassName: nginx
  rules:
    - host: khalil-myapp.zs
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: myapp
                port:
                  number: 80
```

应用并检查：

```bash
kubectl apply -f k8s/ingress.yaml
kubectl -n khalil get ingress myapp
```

### 9.1 从 Windows 通过 SSH 隧道测试

在新的 PowerShell 窗口保持运行：

```powershell
ssh -p 2226 -N -L 8080:192.168.2.14:80 khalil@120.253.243.127
```

再开一个 PowerShell：

```powershell
curl.exe -H "Host: khalil-myapp.zs" http://127.0.0.1:8080/
```

预期返回：

```json
{"status": "ok", "app": "myapp", ...}
```

如果公司 DNS 或本机 hosts 已把 `khalil-myapp.zs` 指向 `192.168.2.14`，并且电脑能直接访问该内网，也可以直接打开：

```text
http://khalil-myapp.zs/
```

---

## 10. 运行一次性 CPU 任务：Job

对于数据清洗、离线计算、批量推理等运行完成后退出的代码，使用 Job，不要用 Deployment。

创建 `k8s/job.yaml`：

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: myapp-job
  namespace: khalil
spec:
  backoffLimit: 1
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels:
        app: myapp-job
    spec:
      restartPolicy: Never
      containers:
        - name: worker
          image: registry.cpu.zs:32443/khalil/myapp:REPLACE_TAG
          command: ["python", "-c"]
          args:
            - |
              import os
              print("CPU job started on", os.getenv("HOSTNAME"))
              print(sum(i * i for i in range(1000000)))
          resources:
            requests:
              cpu: "2"
              memory: "2Gi"
            limits:
              cpu: "8"
              memory: "8Gi"
```

提交与查看：

```bash
TAG=$(git rev-parse --short HEAD)
sed "s/REPLACE_TAG/$TAG/g" k8s/job.yaml | kubectl apply -f -

kubectl -n khalil get jobs,pods
kubectl -n khalil logs -f job/myapp-job
kubectl -n khalil describe job myapp-job
```

Job 名称不可重复创建。重新运行时，先确认旧任务结果和数据已经保存，再删除旧 Job：

```bash
kubectl -n khalil delete job myapp-job
```

更安全的做法是让每次 Job 使用唯一名称，例如 `myapp-job-20260724-153000`。

---

## 11. 配置和敏感信息

### 11.1 普通配置：ConfigMap

```bash
kubectl -n khalil create configmap myapp-config \
  --from-literal=LOG_LEVEL=INFO \
  --from-literal=WORKERS=4
```

Deployment 中引用：

```yaml
envFrom:
  - configMapRef:
      name: myapp-config
```

### 11.2 密码和 API Key：Secret

不要把真实 Secret 写入 Git。可以在终端安全读取后创建：

```bash
read -s -p "API key: " MY_API_KEY; echo
kubectl -n khalil create secret generic myapp-secret \
  --from-literal=API_KEY="$MY_API_KEY"
unset MY_API_KEY
```

Deployment 中引用：

```yaml
envFrom:
  - secretRef:
      name: myapp-secret
```

注意：Kubernetes Secret 默认只是 base64 编码，不等于强加密。不要给无关用户 Secret 读取权限。

---

## 12. 持久化数据：PVC

需要保存上传文件、任务结果或数据库数据时，可创建 PVC。创建 `k8s/pvc.yaml`：

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: myapp-data
  namespace: khalil
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: local-path
  resources:
    requests:
      storage: 20Gi
```

应用：

```bash
kubectl apply -f k8s/pvc.yaml
kubectl -n khalil get pvc
```

在 Deployment 的容器中加入：

```yaml
volumeMounts:
  - name: data
    mountPath: /data
```

在 Pod spec 中加入：

```yaml
volumes:
  - name: data
    persistentVolumeClaim:
      claimName: myapp-data
```

### 12.1 `local-path` 的重要限制

- 数据存储在 Pod 所在节点的本地磁盘。
- 它不是多节点高可用存储。
- 节点或磁盘损坏时可能丢失数据。
- 删除 PVC 可能同时删除实际数据。
- 重要数据必须另做备份，不能只依赖该 PVC。

不要把 PVC 和 Deployment 放在同一个清理命令里随意删除。

---

## 13. 更新、查看版本和回滚

### 13.1 发布新版本

```bash
TAG=$(git rev-parse --short HEAD)
IMAGE="registry.cpu.zs:32443/khalil/myapp:$TAG"

docker build -t "$IMAGE" .
docker push "$IMAGE"

kubectl -n khalil set image deployment/myapp myapp="$IMAGE"
kubectl -n khalil rollout status deployment/myapp --timeout=180s
```

### 13.2 确认当前镜像

```bash
kubectl -n khalil get deployment myapp \
  -o jsonpath='{.spec.template.spec.containers[0].image}{"\n"}'
```

### 13.3 查看发布历史并回滚

```bash
kubectl -n khalil rollout history deployment/myapp
kubectl -n khalil rollout undo deployment/myapp
kubectl -n khalil rollout status deployment/myapp
```

回滚只恢复 Deployment 模板，不会自动回滚数据库结构或 PVC 中的数据。

---

## 14. 日常监控命令

```bash
# 节点状态和资源使用
kubectl get nodes -o wide
kubectl top nodes

# 自己命名空间的资源
kubectl -n khalil get pods,deployments,jobs,services,ingress,pvc -o wide
kubectl -n khalil top pods

# 实时观察 Pod 变化
kubectl -n khalil get pods -w

# 日志
kubectl -n khalil logs deployment/myapp --tail=200
kubectl -n khalil logs -f deployment/myapp

# 上一次崩溃的日志
kubectl -n khalil logs <pod名称> --previous

# 事件，排调度和拉镜像问题最有用
kubectl -n khalil get events --sort-by=.lastTimestamp

# 查看对象详细状态
kubectl -n khalil describe pod <pod名称>
kubectl -n khalil describe deployment myapp
```

如果 `kubectl top` 报 Metrics API 不可用，先检查 metrics-server；它已经出现在本集群的 `cluster-info` 中，但仍可能暂时未就绪。

---

## 15. 常见故障排查

### 15.1 `ImagePullBackOff`

常见原因：

- 镜像没有成功 push。
- Deployment 中的 tag 拼错。
- 镜像仓库证书或认证问题。
- 镜像路径不符合 `registry.cpu.zs:32443/khalil/<app>:<tag>`。

检查：

```bash
kubectl -n khalil describe pod <pod名称>
docker push registry.cpu.zs:32443/khalil/myapp:<tag>
```

### 15.2 `Pending`

常见原因：

- CPU 或内存 request 大于节点可用资源。
- 超过 namespace 配额。
- PVC 无法绑定。
- 使用了不存在的节点标签。

检查：

```bash
kubectl -n khalil describe pod <pod名称>
kubectl -n khalil describe resourcequota
kubectl get nodes
```

### 15.3 `CrashLoopBackOff`

说明容器启动后反复退出：

```bash
kubectl -n khalil logs <pod名称>
kubectl -n khalil logs <pod名称> --previous
kubectl -n khalil describe pod <pod名称>
```

重点检查启动命令、环境变量、文件路径、端口以及应用自身异常。

### 15.4 `OOMKilled`

容器超过了内存 limit：

```bash
kubectl -n khalil describe pod <pod名称>
kubectl -n khalil top pod <pod名称>
```

先确认代码是否存在内存泄漏，再合理提高 `resources.limits.memory`。

### 15.5 Service 访问失败

```bash
kubectl -n khalil get pods --show-labels
kubectl -n khalil get service myapp -o yaml
kubectl -n khalil get endpoints myapp
```

确认：

- Service selector 与 Pod label 一致。
- `targetPort` 与容器监听端口一致。
- 应用监听 `0.0.0.0`，而不是只监听 `127.0.0.1`。

### 15.6 Ingress 返回 404/502/503

- 404：Host 或 path 没匹配路由。
- 502/503：Service、endpoint 或后端端口有问题。

```bash
kubectl -n khalil describe ingress myapp
kubectl -n khalil get service,endpoints
curl -H 'Host: khalil-myapp.zs' http://192.168.2.14/
```

---

## 16. 清理资源

只删除无状态应用对象：

```bash
kubectl -n khalil delete ingress myapp
kubectl -n khalil delete service myapp
kubectl -n khalil delete deployment myapp
```

删除 Job：

```bash
kubectl -n khalil delete job myapp-job
```

删除 PVC 前必须确认数据已经备份：

```bash
kubectl -n khalil get pvc myapp-data
# 确认无误后才执行：
kubectl -n khalil delete pvc myapp-data
```

不要使用以下宽泛命令：

```text
kubectl delete all --all -A
kubectl delete namespace kube-system
kubectl delete namespace platform
```

---

## 17. 可选：接入自动 CI/CD

当手动流程已经跑通后，可以使用 Woodpecker + Harbor + Argo CD：

```text
git push
  → Woodpecker 使用 Kaniko 构建镜像
  → 推送 Harbor
  → CI 回写 deployment.yaml 中的 commit SHA tag
  → Argo CD 监听 Git
  → 自动同步到 k8s
```

当前试用环境入口：

- Woodpecker：`https://120.253.243.127:8443`
- Harbor UI：`http://192.168.2.14:30002`
- Argo CD：`https://192.168.2.14:30808`

CI/CD 使用的是 `192.168.2.14:30002/library/...` Harbor 路径；手动部署 Quickstart 使用的是 `registry.cpu.zs:32443/...`。两套路径不要混用。

完整接入步骤见平台配套文档 `git-cicd-quickstart.md`（本仓库未收录）。

---

## 18. 集群内可复用的平台服务

自己的 Pod 可以通过 Kubernetes DNS 访问平台服务：

| 服务 | 集群内地址 | 用途 |
|---|---|---|
| Nacos | `nacos.nacos.svc.cluster.local:8848` | 配置中心、服务发现 |
| Milvus | `milvus.milvus.svc.cluster.local:19530` | 向量存储和检索 |
| Langfuse | `http://langfuse-web.langfuse:3000` | AI 调用追踪 |
| bge-m3 Embedding | `bge-embedding.model-serving.svc.cluster.local` | 文本转 1024 维向量 |

使用前先通过平台配套的 `nacos-quickstart.md`、`milvus-quickstart.md`、
`embedding-quickstart.md`、`langfuse-quickstart.md` 和
`api-gateway-quickstart.md` 确认服务状态、认证方式和 API 格式；这些手册不在本仓库中。

不要把平台服务凭据硬编码到镜像；使用 Kubernetes Secret 注入。

---

## 19. 每日使用速查

### 开始工作

```powershell
ssh -p 2226 khalil@120.253.243.127
```

```bash
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
export NS=khalil

kubectl get nodes
kubectl -n "$NS" get pods -o wide
```

### 发布服务

```bash
cd ~/projects/myapp
TAG=$(git rev-parse --short HEAD)
IMAGE="registry.cpu.zs:32443/khalil/myapp:$TAG"

docker build -t "$IMAGE" .
docker push "$IMAGE"
kubectl -n khalil set image deployment/myapp myapp="$IMAGE"
kubectl -n khalil rollout status deployment/myapp
kubectl -n khalil logs -f deployment/myapp
```

### 提交一次性任务

```bash
TAG=$(git rev-parse --short HEAD)
sed "s/REPLACE_TAG/$TAG/g" k8s/job.yaml | kubectl apply -f -
kubectl -n khalil logs -f job/myapp-job
```

### 下班前检查

```bash
kubectl -n khalil get pods,jobs,pvc
kubectl -n khalil get events --sort-by=.lastTimestamp | tail -20
```

确认没有意外的 `CrashLoopBackOff`、`ImagePullBackOff`、长期 `Pending` 或重复运行的高资源 Job。

---

## 20. 上线前检查清单

- [ ] 应用镜像使用唯一 tag，而不是 `latest`
- [ ] Deployment/Job 明确设置 CPU 和内存 requests/limits
- [ ] 应用监听 `0.0.0.0`
- [ ] Web 服务具备 readiness/liveness probe
- [ ] 密码和 API Key 使用 Secret，没有进入 Git
- [ ] Service selector 与 Pod label 一致
- [ ] PVC 数据有独立备份
- [ ] 更新前记录当前镜像版本
- [ ] 能通过 `kubectl logs` 看到有效运行日志
- [ ] 已验证回滚方法
- [ ] 所有命令都显式指定 `-n khalil`
- [ ] 没有修改系统命名空间或集群级资源

完成这些检查后，即可比较安全、可重复地在 CPU k8s 集群上运行自己的代码。
