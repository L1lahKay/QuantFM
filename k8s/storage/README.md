# QuantFM 存储切换：管理员一次性操作

此目录中的迁移、校验、激活和重绑资源只用于 2026-07-31 的一次性切换。
它们不是未来训练模板。未来 `gpu-dev` Job 只允许挂载 PVC，不允许使用
`hostPath`。

以下操作需要集群管理员身份；`gpu-dev-khalil` 不具备、也不需要这些权限。

## 已锁定对象

```text
旧 PVC:  khalil/quantfm-data
目标 PVC: gpu-dev/quantfm-data
PV:      pvc-792a789f-ebf3-4a57-9c32-58f23c0c9580
容量:    500Gi
模式:    RWO / Filesystem
SC:      local-path
回收:    Retain
节点:    gpu-dev-01
后端:    /data/k3s/storage/pvc-792a789f-ebf3-4a57-9c32-58f23c0c9580_khalil_quantfm-data
```

`df` 显示的约 2 TiB 是共享底层 XFS；PVC 声明容量仍是 500 GiB。

## 1. 完成并保存迁移验收证据

不得只凭文件数/字节数相同继续。管理员必须看到并保存以下三个最终标记：

```text
MIGRATION_COPY_PASS
MIGRATION_VERIFY_PASS
MIGRATION_ACTIVATE_PASS
```

校验 Job 必须结束为 `Complete`，并且日志中的 source/destination 全量哈希相同。
激活完成后，PV 顶层必须存在正式的 `runs/`，而不是仅有
`runs.migrating-20260731/`。

若校验尚未完成，停止在这里。不要删除 PVC、PV 或本地源数据。

## 2. 切换前只读检查

```bash
PV_NAME=pvc-792a789f-ebf3-4a57-9c32-58f23c0c9580

kubectl get pv "${PV_NAME}" -o yaml > /tmp/quantfm-pv-before-rebind.yaml
kubectl -n khalil get pvc quantfm-data -o yaml \
  > /tmp/quantfm-old-pvc-before-rebind.yaml

kubectl get pv "${PV_NAME}" \
  -o jsonpath='reclaim={.spec.persistentVolumeReclaimPolicy}{"\n"}claim={.spec.claimRef.namespace}/{.spec.claimRef.name}{"\n"}path={.spec.local.path}{"\n"}'
kubectl -n khalil describe pvc quantfm-data
kubectl -n gpu-dev get pvc quantfm-data
```

继续前必须同时满足：

- reclaim 为 `Retain`；
- claim 为 `khalil/quantfm-data`；
- `Used By` 为空，没有 Pod 继续挂载旧 PVC；
- 后端路径、容量、StorageClass、access mode 和 node affinity 未变化；
- `gpu-dev/quantfm-data` 返回 NotFound。

## 3. 预创建目标 claim

目标清单用 `volumeName` 精确指定原 PV，不允许动态创建第二个 PV：

```bash
kubectl create --dry-run=server \
  -f k8s/storage/quantfm-data-gpu-dev-rebind.yaml -o yaml
kubectl create -f k8s/storage/quantfm-data-gpu-dev-rebind.yaml
```

旧 PVC 尚在时，目标 PVC 保持 `Pending` 是预期行为。若它绑定到了其他 PV，
立即停止，不执行下一步。

## 4. 在 `Retain` 保护下切换 claimRef

这是破坏性边界。再次确认第 1、2 节全部通过后才执行：

```bash
kubectl delete pvc -n khalil quantfm-data
kubectl wait --for=delete pvc/quantfm-data -n khalil --timeout=5m
kubectl get pv "${PV_NAME}"
```

PV 必须进入 `Released`，且仍为 `Retain`。PVC 如果卡在 `Terminating`，应查找
仍在使用它的 Pod；不要移除 PVC protection finalizer。

然后以测试操作保护关键字段，并完整替换旧 claimRef：

```bash
kubectl patch pv "${PV_NAME}" --type=json -p='[
  {"op":"test","path":"/spec/persistentVolumeReclaimPolicy","value":"Retain"},
  {"op":"test","path":"/spec/claimRef/namespace","value":"khalil"},
  {"op":"test","path":"/spec/claimRef/name","value":"quantfm-data"},
  {"op":"replace","path":"/spec/claimRef","value":{
    "apiVersion":"v1",
    "kind":"PersistentVolumeClaim",
    "namespace":"gpu-dev",
    "name":"quantfm-data"
  }}
]'
```

## 5. 最终验收

```bash
kubectl wait --for=jsonpath='{.status.phase}'=Bound \
  pvc/quantfm-data -n gpu-dev --timeout=5m

kubectl -n gpu-dev get pvc quantfm-data -o wide
kubectl get pv "${PV_NAME}" \
  -o jsonpath='reclaim={.spec.persistentVolumeReclaimPolicy}{"\n"}claim={.spec.claimRef.namespace}/{.spec.claimRef.name}{"\n"}claimUID={.spec.claimRef.uid}{"\n"}path={.spec.local.path}{"\n"}'
kubectl -n gpu-dev get pvc quantfm-data \
  -o jsonpath='phase={.status.phase}{"\n"}volume={.spec.volumeName}{"\n"}uid={.metadata.uid}{"\n"}'
```

PV claim UID 必须等于新 PVC UID；PV 名、`Retain`、底层路径和 node affinity
必须保持不变。管理员把这份输出交给 Khalil，即完成后续所有 Job 所需的存储
确认。

若任一步失败，保持 PV 为 `Released + Retain` 并停止；不要删除 PV、不要移动
底层目录、不要恢复 `Delete`。

之后由 `gpu-dev-khalil` 依次运行 CPU/PVC 和 GPU/PVC 冒烟 Job。短训练的
checkpoint 写入及 `--resume auto` 恢复也通过前，不得删除原 `/home` 副本。
