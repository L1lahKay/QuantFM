# MinIO 读写指南

本文档说明 quant_fm 如何从 MinIO **读原始 L2**、如何将 **events/tokens 写回 MinIO**，以及相关配置与命令。

> 版本边界：本文的一键命令当前只编排 V1 产物。V2 `BookState/cn_l2_v2/vocab_v2/token+scalar` API 虽已落地，原始 MinIO 回放到 V2 分片仍需显式捕获事件前/后状态，尚未接入 `run_pilot.py` / `run_medium.py`。V1/V2 词表、tokens、manifest 和 checkpoint 必须使用独立路径。

---

## 1. 读写分离总览

同一套 Access Key，**读写的 bucket 与端口不同**：

```text
┌─────────────────────────────────────────────────────────────────┐
│  读（只读）                                                      │
│  S3 API   : 192.168.2.11:9000                                   │
│  Bucket   : zeus-cn-quote                                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │  Polars 流式读
                            ▼
                   本地 transient 处理
                            │
                            │  mc cp / upload_to_minio.py
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  写（可写）                                                      │
│  S3 API   : 192.168.2.11:9100                                   │
│  Bucket   : model-cache                                          │
└─────────────────────────────────────────────────────────────────┘
```

| | **读** | **写** |
|---|--------|--------|
| S3 API Endpoint | `192.168.2.11:9000` | `192.168.2.11:9100` |
| Bucket | `zeus-cn-quote` | `model-cache` |

默认值定义在 `quant_fm/scripts/minio_config.py`。

---

## 2. 凭据配置（只需一次）

读写共用同一对 Key；端口/bucket 已在代码里写死。

### 方式 A：`mc alias myminio`（零文件）

```bash
mc alias set myminio http://192.168.2.11:9000 YOUR_ACCESS_KEY YOUR_SECRET_KEY
```

代码会从 `~/.mc/config.json` 读取 key，读用 `:9000`，写用 `:9100`。

### 方式 B：环境变量文件（推荐）

```bash
cp quant_fm/scripts/minio_env.example.sh ~/.minio_fm_env.sh
chmod 600 ~/.minio_fm_env.sh
vim ~/.minio_fm_env.sh    # 只改 MINIO_ACCESS_KEY / MINIO_SECRET_KEY
source ~/.minio_fm_env.sh
```

`minio_env.example.sh` 已包含读写 endpoint 与 bucket 默认值，一般**不用改**。

### 自检

```bash
python -m quant_fm.scripts.check_minio
```

期望输出：

```text
read  s3://zeus-cn-quote @ 192.168.2.11:9000   OK (200)
write s3://model-cache @ 192.168.2.11:9100      OK (200)
```

---

## 3. 读功能

### 3.1 读什么

| 项目 | 值 |
|------|-----|
| URI 示例 | `s3://zeus-cn-quote/HDS/SOURCE=zeus/DOMAIN=quote/DATASET=china_stock/2025.01/2025.01.02/default/2/all.parquet` |
| 成交 | `.../default/2/all.parquet` |
| 委托 | `.../default/3/all.parquet` |
| 不读 | `default/1`（十档快照，PyLOB 不需要） |

### 3.2 谁在读

| 脚本 / 模块 | 函数 | 说明 |
|-------------|------|------|
| `order_book/pylob/pipeline/workflow.py` | `build_clean_dataset()` | 订单簿重建入口 |
| `order_book/pylob/pipeline/s3_io.py` | `PolarsS3Reader.read_object_keys()` | Polars 流式读 S3 |
| `quant_fm/scripts/run_medium.py` | `clean_one_day()` | 按日批量清洗 |
| `quant_fm/scripts/run_pilot.py` | `clean_one_day()` | 试点清洗 |
| `examples/run_zeus_clean.py` | `main()` | 单实验清洗 |

内部均调用 `load_read_config()` + `read_bucket()`。

### 3.3 读相关环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MINIO_READ_ENDPOINT` | `192.168.2.11:9000` | 读 API 地址 |
| `MINIO_BUCKET` | `zeus-cn-quote` | 原始 L2 bucket |
| `MINIO_ACCESS_KEY` | — | 凭据（必填） |
| `MINIO_SECRET_KEY` | — | 凭据（必填） |

### 3.4 Python 读示例

```python
import os
import polars as pl
from quant_fm.scripts.minio_config import load_read_config, read_bucket, storage_options_for_read

# 方式 1：封装好的 storage_options
storage = storage_options_for_read()
bucket = read_bucket()

path = (
    f"s3://{bucket}/HDS/SOURCE=zeus/DOMAIN=quote/DATASET=china_stock/"
    "2025.01/2025.01.02/default/2/all.parquet"
)
df = pl.scan_parquet(path, storage_options=storage).head(5).collect()
print(df)

# 方式 2：手动指定（与之前 Polars 脚本用法一致，endpoint 为 9000）
storage = {
    "aws_access_key_id": os.environ["MINIO_ACCESS_KEY"],
    "aws_secret_access_key": os.environ["MINIO_SECRET_KEY"],
    "aws_endpoint_url": "http://192.168.2.11:9000",
    "aws_allow_http": "true",
}
```

### 3.5 命令行读（mc）

```bash
# 列出原始数据目录
mc ls myminio/zeus-cn-quote/HDS/SOURCE=zeus/DOMAIN=quote/DATASET=china_stock/2025.01/

# 查看单日对象
mc ls myminio/zeus-cn-quote/HDS/SOURCE=zeus/DOMAIN=quote/DATASET=china_stock/2025.01/2025.01.02/default/
```

### 3.6 触发读的 Makefile / 命令

```bash
# 试点：3 股 × 5 天（读 MinIO，写本地）
make pilot

# medium 试跑 / 全量（读 MinIO，写本地 + 可选写回 MinIO）
make minio-pipeline
make minio-pipeline-full

# 300M：22 日全市场 + 并行清洗/规范化 + 断点续跑
CLEAN_WORKERS=32 CANON_WORKERS=16 SKIP_UPLOAD=1 bash quant_fm/scripts/run_minio_300m_pipeline.sh

# 查询数据阶段进度
uv run python -m quant_fm.scripts.check_pipeline_progress

# 底层命令
CLEAN_WORKERS=32 python -m quant_fm.scripts.run_medium \
  --workdir quant_fm/runs/medium \
  --drop-clean --drop-events --resume
```

`--resume` 会跳过已完成日期，并跳过已有 `events.parquet` / `<date>.parquet` 的标的；`CLEAN_WORKERS` 控制并行洗股进程数，`CANON_WORKERS` 控制并行规范化进程数。

读操作**不会**修改 `zeus-cn-quote` 内任何对象。

---

## 4. 写功能

### 4.1 写什么

| 路径（相对 prefix） | 说明 | 是否必须 |
|-------------------|------|----------|
| `tokens/{market}/{symbol}/{date}.parquet` | 分词后整数列，训练直接读 | 是 |
| `data/vocab.json` | 冻结词表（分箱边界 + 类别表） | 是 |
| `data/manifest.json` | 分片清单 + train/val/test 切分 | 是 |
| `events/...` | 规范 cn_l2_v1 事件 | 否（默认不上传，加 `--upload-events`） |

完整 URI 示例：

```text
s3://model-cache/fm-pretrain/<user>/medium_try/tokens/SZ/000001/2025-01-02.parquet
s3://model-cache/fm-pretrain/<user>/medium_try/data/vocab.json
s3://model-cache/fm-pretrain/<user>/medium_try/data/manifest.json
```

**不要**写入 `zeus-cn-quote/HDS/...`，避免与原始数据混淆。

### 4.2 谁在写

| 脚本 / 模块 | 函数 / 参数 | 说明 |
|-------------|-------------|------|
| `quant_fm/scripts/upload_to_minio.py` | `upload_workdir()` | 独立上传工具 |
| `quant_fm/scripts/run_medium.py` | `--upload-minio` | 流水线结束后自动上传 |
| `quant_fm/scripts/run_minio_data_pipeline.sh` | 内置 `--upload-minio` | 读→处理→写 一键脚本 |
| `quant_fm/scripts/wait_upload_minio.sh` | 轮询 manifest 后上传 | 异步补传 |

内部均调用 `load_write_config()` + `output_bucket()` + `output_prefix()`。

### 4.3 写相关环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MINIO_WRITE_ENDPOINT` | `192.168.2.11:9100` | 写 API 地址 |
| `MINIO_OUTPUT_BUCKET` | `model-cache` | 目标 bucket |
| `MINIO_OUTPUT_PREFIX` | `fm-pretrain/<user>` | 对象 key 前缀（建议含用户名） |
| `MINIO_ACCESS_KEY` | — | 与读相同 |
| `MINIO_SECRET_KEY` | — | 与读相同 |

上传 tag（子目录）由 `--upload-tag` 或 `--tag` 指定，例如 `medium_try`、`medium`。

### 4.4 写命令

**一键读→处理→写（无训练）：**

```bash
source ~/.minio_fm_env.sh
make minio-pipeline          # 试跑 5日×30股 → model-cache/.../medium_try/
make minio-pipeline-full     # 60日×全市场 → model-cache/.../medium/
```

**一键读→tokens→写→8 卡训练（推荐完整流水线）：**

```bash
make minio-full-pipeline           # 试跑 + 训练
make minio-full-pipeline-full      # 60日×全市场 + 训练
# 数据保留本地供训练；同时上传到 model-cache 备份
```

**流水线内自动上传：**

```bash
python -m quant_fm.scripts.run_medium \
  --workdir quant_fm/runs/medium_try \
  --drop-clean --drop-events \
  --upload-minio \
  --upload-tag medium_try \
  --delete-local-after-upload
```

**手动上传已有本地产物：**

```bash
python -m quant_fm.scripts.upload_to_minio \
  --workdir quant_fm/runs/medium_try \
  --tag medium_try \
  --delete-local \
  --verify
```

**mc 手动写：**

```bash
mc alias set fmwrite http://192.168.2.11:9100 YOUR_ACCESS_KEY YOUR_SECRET_KEY

mc cp --recursive quant_fm/runs/medium_try/tokens/ \
  fmwrite/model-cache/fm-pretrain/<user>/medium_try/tokens/

mc cp quant_fm/runs/medium_try/data/vocab.json \
  fmwrite/model-cache/fm-pretrain/<user>/medium_try/data/vocab.json

mc cp quant_fm/runs/medium_try/data/manifest.json \
  fmwrite/model-cache/fm-pretrain/<user>/medium_try/data/manifest.json
```

### 4.5 验证写入

```bash
python -m quant_fm.scripts.upload_to_minio \
  --workdir quant_fm/runs/medium_try --tag medium_try --verify

# 或 mc
mc ls fmwrite/model-cache/fm-pretrain/<user>/medium_try/tokens/ --recursive | head
mc du fmwrite/model-cache/fm-pretrain/<user>/medium_try/
```

---

## 5. 完整 MinIO 流水线（读 + 写 + 训练）

```bash
cd ~/DataCleaning7.3/QuantFM
source .venv/bin/activate
source ~/.minio_fm_env.sh

python -m quant_fm.scripts.check_minio

# 读→tokens→写→训练（试跑）
make minio-full-pipeline
# 等价：bash quant_fm/scripts/run_minio_full_pipeline.sh

# 读→tokens→写→训练（60日×全市场）
make minio-full-pipeline-full

# 只要数据不要训练
make minio-pipeline / make minio-pipeline-full
```

日志：`quant_fm/runs/minio_full_pipeline.log`（完整）或 `minio_pipeline.log`（仅数据）。

流水线步骤：

1. **读** `:9000/zeus-cn-quote` — 按日流式拉取 default/2+3
2. **本地处理** — PyLOB 清洗 → cn_l2_v1 events → fit_bins → tokens → manifest
3. **删中间产物** — `--drop-clean --drop-events` 控制磁盘峰值
4. **写** `:9100/model-cache` — `--upload-minio` 上传 tokens + vocab + manifest
5. **验证** — 统计远端 parquet 数量

---

## 6. 代码 API 速查

```python
from quant_fm.scripts.minio_config import (
    load_read_config,       # MinioConfig，endpoint=9000
    load_write_config,      # MinioConfig，endpoint=9100
    read_bucket,            # "zeus-cn-quote"
    output_bucket,          # "model-cache"
    output_prefix,          # "fm-pretrain/<user>[/tag]"
    storage_options_for_read,
    storage_options_for_write,
    describe_config,
)
from quant_fm.scripts.upload_to_minio import upload_workdir, remote_uri, verify_upload

# 上传
uri = upload_workdir(
    Path("quant_fm/runs/medium_try"),
    tag="medium_try",
    delete_local=True,
)
print(uri)  # s3://model-cache/fm-pretrain/<user>/medium_try/

# 验证
verify_upload("medium_try")
```

---

## 7. 常见问题

**Q：为什么读 9000、写 9100？**  
A：管理员分配：原始 L2 在 `:9000` 只读；你的缓存空间 `model-cache` 在 `:9100` 可写。

**Q：会覆盖原始数据吗？**  
A：不会。读是只读；写只在 `model-cache` 下你的 prefix 里。

**Q：和旧 `myminio`（9000）冲突吗？**  
A：不冲突。凭据可复用；代码按读写分别指定 endpoint，无需改 mc 别名 URL。

**Q：本地还要占磁盘吗？**  
A：处理时仍需 transient 空间；上传后可用 `--delete-local-after-upload` 删本地 tokens。

**Q：训练怎么读 MinIO 上的 tokens？**  
A：训练读**本地** `tokens/` + `manifest.json`（路径是构建机绝对路径）。完整流水线会先写 MinIO 备份并**保留本地**再训。若本地已删：`make download-medium && make train-medium-8gpu`。直接从 S3 URI 训练尚未做。

---

## 8. 相关文档

- [raw_to_events_tokens.md](./raw_to_events_tokens.md) — events/tokens 生成步骤
- [minio_data_workflow.md](./minio_data_workflow.md) — 磁盘批次与方案对比
- [QuantFM.md](./QuantFM.md) — 项目总览
