# MinIO 数据清洗与训练工作流

> **读写分离**：读 `zeus-cn-quote` @ `:9000`，写 `model-cache` @ `:9100`。  
> 完整读写 API 与命令见 **[minio_setup.md](./minio_setup.md)**。

## 背景：MinIO 上有什么

| | Endpoint | Bucket | 权限 |
|---|----------|--------|------|
| **读** 原始 L2 | `192.168.2.11:9000` | `zeus-cn-quote` | 只读 |
| **写** 清洗产物 | `192.168.2.11:9100` | `model-cache` | 可写 |

| 项目 | 说明 |
|------|------|
| 原始路径前缀 | `HDS/SOURCE=zeus/DOMAIN=quote/DATASET=china_stock/` |
| 写入路径前缀 | `fm-pretrain/<user>/{tag}/`（tag 如 `medium`、`medium_try`） |
| 时间跨度 | 约 **2024.01 ~ 2026.06**（31 个月，400~600 个交易日） |
| 原始总大小 | 约 **4.0 TiB**（71,664 个对象） |

### 单日目录结构（`YYYY.MM/YYYY.MM.DD/default/`）

| 分区 | 文件 | 内容 | 单日大小（约） |
|------|------|------|----------------|
| `1/` | `all.parquet` | 行情快照（十档） | ~1.5 GiB |
| `2/` | `all.parquet` | **逐笔成交** | ~3.9 GiB |
| `3/` | `all.parquet` | **逐笔委托** | ~3.6 GiB |

PyLOB 撮合只需 **`default/2` + `default/3`**（约 **7.5 GiB/天**，全市场）。

### 清洗输出（本地实测）

| 范围 | 大小 |
|------|------|
| 2 只股票 × 1 天 × 3 个 parquet | ~28 MB |
| 单只股票 × 1 天 | ~12 MB |

---

## 方案一：一键完整流水线（推荐）

**思路**：从 MinIO 读原始 L2 → 本地 transient 清洗并打成 tokens → 写回 `model-cache` → 本机用本地 tokens 做 8 卡训练（MinIO 作备份；训练仍读本地路径）。

```text
MinIO 原始 (:9000 / zeus-cn-quote)
        │  Polars 流式读
        ▼
   run_medium：clean → cn_l2_v1 events → vocab → tokens → manifest
        │
        ├─► 本地 quant_fm/runs/{medium,medium_try}/...
        │
        └─► 写 MinIO (:9100 / model-cache / fm-pretrain/<user>/{tag}/)
                    │
                    ▼
              train_medium_8gpu（读本地 tokens）
```

### 操作

```bash
cd ~/DataCleaning7.3/QuantFM
source .venv/bin/activate
source ~/.minio_fm_env.sh
make check-minio

# 试跑（5 日 × 每市场 30 股）→ 上传 → 8 卡训练
make minio-full-pipeline

# 「全量」约定：60 个均匀交易日 × 沪深全市场（≈ 总事件 1/10）+ 训练
make minio-full-pipeline-full

# ~302M 正式：22 日全市场（Chinchilla）+ 并行清洗 + 断点续跑
CLEAN_WORKERS=32 CANON_WORKERS=16 SKIP_UPLOAD=1 bash quant_fm/scripts/run_minio_300m_pipeline.sh

# 进度
uv run python -m quant_fm.scripts.check_pipeline_progress

# 只要数据不要训练
SKIP_TRAIN=1 MODE=full bash quant_fm/scripts/run_minio_full_pipeline.sh

# 数据已在 MinIO、本地被删：拉回再训
make download-medium
make train-medium-8gpu
```

规模说明：

- 仓库默认「全量」是 **medium = 60 日 × 全市场**，不是 MinIO 上全部 ~400–600 个交易日；
- **300M** 使用 `medium_300m_22_dates.txt`（约 22 日全市场，匹配 ~6B 训练事件）；
- 数据阶段支持日期/标的级续跑；训练阶段支持 `--resume auto`。

远程目录：

```text
model-cache/fm-pretrain/<user>/{tag}/
  tokens/...
  data/vocab.json
  data/manifest.json
```

---

## 方案一（历史参考）：手动清洗脚本

> 下列 `run_zeus_clean.py` 示例仍可用，但推荐改用上方 `make minio-full-pipeline*`。

**1. 清洗（从 MinIO 读，按股票过滤）**

```bash
cd ~/DataCleaning7.3/QuantFM
source .venv/bin/activate

export MINIO_ENDPOINT=192.168.2.11:9000
export MINIO_ACCESS_KEY="..."
export MINIO_SECRET_KEY="..."
export MINIO_BUCKET=zeus-cn-quote
export MINIO_OUTPUT_BUCKET=model-cache

export PYLOB_DATE=2026-02-02
export PYLOB_SYMBOLS=000001,000002
export PYLOB_MARKET=SZ
export PYLOB_OUTPUT_DIR=data/clean/2026-02-02

python examples/run_zeus_clean.py
```

**2. 上传清洗结果**

```bash
mc cp --recursive \
  data/clean/2026-02-02/ \
  fm/model-cache/fm-pretrain/<user>/cleaned/2026-02-02/
```

**3. 训练时读取**

```python
import polars as pl

storage_options = {
    "aws_access_key_id": "...",
    "aws_secret_access_key": "...",
    "aws_endpoint_url": "http://192.168.2.11:9100",
}

df = pl.read_parquet(
    "s3://model-cache/fm-pretrain/<user>/pilot/tokens/SZ/000001/2026-02-02.parquet",
    storage_options=storage_options,
)
```

### 训练用哪个文件

| 文件 | 用途 |
|------|------|
| `events.parquet` | 标准化事件流（ADD/CANCEL/TRADE），分析友好 |
| `tokens.parquet` | 离散化 token，适合 next-event 预训练 |
| `market_rows.parquet` | 撮合引擎消费的原始回放行，调试用 |

### 优点

- 本地磁盘占用极小（仅清洗过程中的内存 + 少量临时文件）
- 原始 4TB 不必落地
- 清洗结果集中管理，训练多机共享
- 与当前 `run_zeus_clean.py`（`zeus_default` 布局 + symbol 过滤）一致

### 注意

- 需要 MinIO **写权限**（`PutObject`）才能上传清洗结果
- 上传前确认路径不与原始 `HDS/...` 冲突

---

## 方案二：分批次拉取 → 清洗 → 释放原始批次

**思路**：每次只把一小批原始 parquet 拉到本地，清洗完成后删除原始文件，仅保留（或上传）清洗结果，再处理下一批。

### 流程

```text
For each batch (按天 / 按股票组):
    mc cp 原始 default/2,3  →  /tmp/raw/{date}/
            │
            ▼
    PyLOB 清洗 → data/clean/{date}/...
            │
            ▼
    rm -rf /tmp/raw/{date}/          # 释放原始占用
    mc cp 清洗结果 → MinIO（可选）
```

### 磁盘峰值估算

峰值 ≈ **一批原始大小 + 一批清洗结果大小**（若不上传、全留本地则清洗结果会累积）。

| 批次粒度 | 原始临时占用 | 清洗结果（粗算） | 本机 33GB 是否可行 |
|----------|-------------|------------------|-------------------|
| 1 天全市场 raw (2+3) | ~7.5 GiB | 全市场 ~5000 股 × 6MB ≈ **30 GiB** | 紧张 |
| 1 天 + 10 只股票 | ~7.5 GiB* | ~120 MB | 可行 |
| 1 天 + 50 只股票 | ~7.5 GiB* | ~600 MB | 可行 |
| 仅 cp 2+3 不 filter | 7.5 GiB | — | 单天 raw 可放进 33GB |

\* 若用 `mc cp` 拉全文件，即使只洗 10 只股票，仍需 **7.5 GiB** 临时盘存放完整 parquet。

**关键结论**：

1. **分批次确实能降低磁盘压力**，相对「4TB 全量落地」完全不可比。
2. 若批次 = 「整天下载 default/2 + default/3」，单批仍需 **~7.5 GiB** 临时空间。
3. 若批次 = 「按股票组、且从 MinIO 流式过滤读取」（当前 `PolarsS3Reader` 做法），**不必先 mc cp 全文件**，磁盘压力比方案二的传统 download 更小，更接近方案一。
4. 若清洗 **全市场所有股票** 并全部留本地，单日清洗结果仍可能 **~30 GiB**，33GB 磁盘依然不够长期累积。

### 方案二推荐批次策略

| 策略 | 做法 | 磁盘峰值 |
|------|------|----------|
| **A. 流式按 symbol 批（推荐）** | 不 mc cp；`PYLOB_SYMBOLS=一批股票`；`run_zeus_clean.py`；可选上传 MinIO | 内存 + ~MB 级输出 |
| **B. 按天下载 raw 再洗** | `mc cp` 当日 2+3 → `/tmp` → 本地脚本读 `/tmp` → 删除 raw | ~7.5 GiB + 清洗结果 |
| **C. 按天 + 上传后删本地** | A 或 B 清洗 → `mc cp` 到 MinIO → `rm` 本地 clean | 仅临时峰值 |

**最省磁盘**：**策略 A + 清洗完上传 MinIO + 删除本地 clean**（与方案一结合）。

### 方案二示例：按日期循环（流式，不下载全量 raw）

```bash
#!/usr/bin/env bash
# 示例：批量清洗多个交易日（流式读 MinIO，不 mc cp raw）
DATES="2026-02-02 2026-02-03 2026-02-04"
SYMBOLS="000001,000002,000003,000004,000005"

for DATE in $DATES; do
  export PYLOB_DATE=$DATE
  export PYLOB_SYMBOLS=$SYMBOLS
  export PYLOB_OUTPUT_DIR="data/clean/${DATE}"
  python examples/run_zeus_clean.py

  # 可选：上传并删除本地，进一步释放磁盘
  mc cp --recursive "data/clean/${DATE}/" \
    "myminio/zeus-cn-quote/cleaned/pylob/${DATE}/"
  rm -rf "data/clean/${DATE}"
done
```

### 方案二示例：按天下载 raw（传统分批）

```bash
DATE=2026-02-02
BASE="HDS/SOURCE=zeus/DOMAIN=quote/DATASET=china_stock/2026.02/2026.02.02/default"
TMP=/tmp/pylob_raw/${DATE}
mkdir -p "$TMP"

mc cp "myminio/zeus-cn-quote/${BASE}/2/all.parquet" "$TMP/trade.parquet"
mc cp "myminio/zeus-cn-quote/${BASE}/3/all.parquet" "$TMP/order.parquet"

# 需额外脚本从本地 parquet 清洗（当前 run_zeus_clean.py 从 MinIO 读，可改 MINIO 为本地路径或扩展 pipeline）
# ... 清洗 ...

rm -rf "$TMP"    # 释放 ~7.5 GiB
```

> 当前仓库默认脚本走 MinIO 流式读取；若要坚持「先下载再洗」，需增加本地文件数据源或先用 Polars 读本地 parquet。

---

## 方案对比

| 维度 | 方案一（清洗结果回 MinIO） | 方案二（分批拉 raw） |
|------|---------------------------|---------------------|
| 原始 4TB 是否落地 | 否 | 否（按批临时） |
| 单批磁盘峰值 | 低（MB~GB 级输出） | 中（7.5 GiB/天若 cp 全量 raw） |
| 实现复杂度 | 低（已有脚本 + mc cp） | 中（需批次调度 + 清理） |
| 训练读数据 | 直接从 MinIO 读 clean | 同左（建议仍回 MinIO） |
| 多机协作 | 好 | 好（若结果上传 MinIO） |
| 推荐度 | **首选** | 作补充（离线/无网络读 MinIO 时） |

---

## 本机磁盘现状（参考）

| 项目 | 数值 |
|------|------|
| 总容量 | 504 GB |
| 可用 | ~33 GB |
| MinIO 全量 | ~4096 GB |

**不可**全量拉取原始数据。应使用：**按 symbol / 按 date 流式清洗 + 结果存 MinIO**。

---

## 相关脚本与配置

| 文件 | 说明 |
|------|------|
| `examples/run_zeus_clean.py` | 一键清洗（`PYLOB_LAYOUT=zeus_default`） |
| `examples/minio_clean_pipeline.py` | 通用入口，支持探路 `--ls-only` |
| `PYLOB_DATE` | 交易日，如 `2026-02-02` |
| `PYLOB_SYMBOLS` | 逗号分隔股票代码 |
| `PYLOB_OUTPUT_DIR` | 本地输出目录 |

原始 object key（自动推导）：

```text
.../default/2/all.parquet   # 逐笔成交
.../default/3/all.parquet   # 逐笔委托
```
