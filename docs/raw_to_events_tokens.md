# 从原始 L2 数据生成 Events 与 Tokens

本文档说明如何从 MinIO 上的 A 股 Level-2 原始 parquet，一步步生成预训练所需的 **规范 events** 与 **tokens**。

> 本文第 1–4 节是当前已接入 MinIO 编排的 **V1 稳定路径**。V2 的因果盘口、`cn_l2_v2`、分层分箱和 token+scalar 底层 API 已实现，但还没有被 `run_pilot.py` / `run_medium.py` 自动编排。不得将 V1 events 直接改名为 V2 产物。

---

## 1. 总览

```text
【读】MinIO :9000 / zeus-cn-quote — 原始 L2（default/2 成交 + default/3 委托）
        │
        ▼  Step 1  PyLOB 订单簿重建（Polars 流式读，不落全量 raw）
clean/{date}/{market}/{symbol}/events.parquet     ← 中间产物（可 --drop-clean 删除）
        │
        ▼  Step 2  投影到 cn_l2_v1 规范
events/{market}/{symbol}/{date}.parquet           ← 规范 events（可 --drop-events 删除）
        │
        ▼  Step 3  仅在训练日期拟合分箱
data/vocab.json
        │
        ▼  Step 4  确定性分词
tokens/{market}/{symbol}/{date}.parquet
        │
        ▼  Step 5  构建 manifest
data/manifest.json
        │
        ▼  【写】MinIO :9100 / model-cache — upload_minio（--upload-minio）
s3://model-cache/fm-pretrain/<user>/{tag}/tokens/...
```

读写详细说明见 **[minio_setup.md](./minio_setup.md)**。

**一行事件 = 一个市场动作**（ADD 挂单 / CANCEL 撤单 / EXEC 成交）。  
**一行 token = 同一事件的多字段整数编码**，供 Transformer 读取。

---

## 2. 前置条件

### 2.1 环境

```bash
cd ~/DataCleaning7.3/QuantFM
source .venv/bin/activate
uv sync --extra fm   # 若尚未安装 polars / pyarrow 等
```

### 2.2 MinIO 读写配置

| | Endpoint | Bucket | 权限 |
|---|----------|--------|------|
| **读** 原始 L2 | `192.168.2.11:9000` | `zeus-cn-quote` | 只读 |
| **写** tokens 产物 | `192.168.2.11:9100` | `model-cache` | 可写 |

完整读写 API、命令、Python 示例见 **[minio_setup.md](./minio_setup.md)**。

凭据只需配置一次：

```bash
cp quant_fm/scripts/minio_env.example.sh ~/.minio_fm_env.sh
vim ~/.minio_fm_env.sh    # 只填 ACCESS_KEY / SECRET_KEY
source ~/.minio_fm_env.sh
python -m quant_fm.scripts.check_minio
```

**读→处理→写 一键（无训练）：**

```bash
make minio-pipeline          # 试跑
make minio-pipeline-full     # 60日×全市场
```

**读→tokens→写→训练 完整流水线：**

```bash
make minio-full-pipeline           # 试跑 + 8卡训练
make minio-full-pipeline-full      # 60日×全市场 + 8卡训练
# 等价：MODE=full bash quant_fm/scripts/run_minio_full_pipeline.sh
```

### 2.3 原始数据路径（zeus-cn-quote）

```text
s3://zeus-cn-quote/HDS/SOURCE=zeus/DOMAIN=quote/DATASET=china_stock/
  {YYYY.MM}/{YYYY.MM.DD}/default/
    2/all.parquet    # 逐笔成交（PyLOB 需要）
    3/all.parquet    # 逐笔委托（PyLOB 需要）
```

PyLOB **不读** `default/1`（十档快照）；预训练事件流来自成交 + 委托撮合。

---

## 3. 推荐方式：一键编排（试点）

适合少量日期 + 少量股票验证流水线。

### 3.1 Makefile

```bash
# 默认：3 只股票 × 5 天（SZ）
make pilot
```

等价于：

```bash
python -m quant_fm.scripts.run_pilot \
  --dates 2026-02-02,2026-02-03,2026-02-04,2026-02-05,2026-02-06 \
  --symbols 000001,000002,300750 \
  --market SZ \
  --workdir quant_fm/runs/pilot \
  --train-end 2026-02-04 \
  --val-end 2026-02-05 \
  --n-bins 32
```

### 3.2 输出目录

```text
quant_fm/runs/pilot/
  clean/2026-02-02/SZ/000001/events.parquet    # Step 1 中间产物
  events/SZ/000001/2026-02-02.parquet           # Step 2 规范 events
  tokens/SZ/000001/2026-02-02.parquet           # Step 4 tokens
  data/vocab.json                               # Step 3 词表
  data/manifest.json                            # Step 5 清单
```

完成后即可训练：

```bash
make train-pilot      # 单卡
make train-8gpu       # 8 卡 FSDP
```

---

## 4. 分步手动执行（理解每一步）

以下以 **2026-02-02、000001、深圳 SZ** 为例。工作目录设为 `quant_fm/runs/demo`。

### Step 1：PyLOB 清洗 → `clean/.../events.parquet`

从 MinIO **流式读取** 当日 `default/2` + `default/3`，重建订单簿，输出 PyLOB 事件流。

```bash
export PYLOB_DATE=2026-02-02
export PYLOB_SYMBOLS=000001
export PYLOB_MARKET=SZ
export PYLOB_OUTPUT_DIR=quant_fm/runs/demo/clean/2026-02-02

python examples/run_zeus_clean.py
```

**输出文件（每个 symbol 三个 parquet）：**

| 文件 | 说明 |
|------|------|
| `market_rows.parquet` | 撮合引擎消费的合并行（调试/回放用） |
| `events.parquet` | **PyLOB 事件流**，Step 2 的输入 |
| `tokens.parquet` | PyLOB 内置字段 token（**quant_fm 不使用**，请用 Step 4 产物） |

**PyLOB `events.parquet` 主要列：**

```text
symbol, market, event_idx, int_time, local_time, serial,
delta_t, session_phase, event_type, side, price, volume, log_volume, ...
```

`event_type` 取值：`ADD` / `CANCEL` / `TRADE`（后续规范化为 `EXEC`）。

**底层入口：** `pylob.pipeline.workflow.build_clean_dataset()`

---

### Step 2：规范化 → `events/{market}/{symbol}/{date}.parquet`

将 PyLOB 事件投影到 **cn_l2_v1** 统一 schema（沪深字段对齐，不做分箱）。

```bash
python -m quant_fm.lob_rebuild.export_events \
  --clean-dir quant_fm/runs/demo/clean/2026-02-02 \
  --out-dir quant_fm/runs/demo/events \
  --date 2026-02-02 \
  --markets SZ \
  --symbols 000001
```

**输出路径：**

```text
quant_fm/runs/demo/events/SZ/000001/2026-02-02.parquet
```

**规范 events 列（22 列，见 `quant_fm/schema/cn_l2_v1.py`）：**

```text
schema_version, date, exchange, market, symbol, security_id, board,
session, event_source, evt_type, side, price, qty, amount, order_type,
level, delta_t, int_time, local_time, source_seqnum, event_idx, quality_flag
```

要点：

- `price` 单位为**元**（PyLOB 整数 ÷ 10000）
- `evt_type`：`ADD` / `CANCEL` / `EXEC` / ...
- `symbol`、`date` 仅作索引，**不进模型词表**

---

### Step 3：拟合词表 → `data/vocab.json`

**只在训练日期的事件上**拟合连续字段分箱边界；类别字段使用固定词表。

```python
from pathlib import Path
from quant_fm.tokenizer.fit_bins import fit_bins

train_paths = list(Path("quant_fm/runs/demo/events").rglob("*.parquet"))
# 实际应按日期过滤，只保留 split=train 的文件

vocab = fit_bins(
    train_paths,
    n_bins=32,
    fit_dates=["2026-02-02"],  # 仅训练日
)
vocab.save(Path("quant_fm/runs/demo/data/vocab.json"))
```

**词表内容：**

- 类别字段：`evt_type`, `side`, `session`, `board`, `order_type`, `event_source`
- 连续字段分箱：`price_rel`, `log_volume`, `log_delta_t`（默认 32 bins）
- `PAD_ID = 0` 保留给 padding

⚠️ **数据泄漏规则：** 验证集、测试集日期绝不能参与 `fit_bins`。`run_pilot.py` 会在保存后调用 `assert_no_leakage()` 检查。

---

### Step 4：分词 → `tokens/{market}/{symbol}/{date}.parquet`

用冻结的 `vocab.json` 把规范 events 转为整数列。

```python
from pathlib import Path
from quant_fm.tokenizer.vocab import Vocab
from quant_fm.tokenizer.tokenize_events import tokenize_path

vocab = Vocab.load("quant_fm/runs/demo/data/vocab.json")

src = Path("quant_fm/runs/demo/events/SZ/000001/2026-02-02.parquet")
dst = Path("quant_fm/runs/demo/tokens/SZ/000001/2026-02-02.parquet")
tokenize_path(src, dst, vocab)
```

**tokens 列：**

| 列名 | 类型 | 源字段 |
|------|------|--------|
| `tok_evt_type` | 类别 id | `evt_type` |
| `tok_side` | 类别 id | `side` |
| `tok_session` | 类别 id | `session` |
| `tok_board` | 类别 id | `board` |
| `tok_order_type` | 类别 id | `order_type` |
| `tok_event_source` | 类别 id | `event_source` |
| `tok_price_bin` | 分箱 id | `price_rel`（相对价） |
| `tok_volume_bin` | 分箱 id | `log_volume` |
| `tok_delta_t_bin` | 分箱 id | `log_delta_t` |

另保留索引列：`symbol`, `security_id`, `date`, `int_time`, `event_idx`（**不进模型 embedding 求和**，仅用于对齐与下游聚合）。

**CLI 等价：** 编排脚本内部即循环调用 `tokenize_path()`。

---

### Step 5：构建 manifest → `data/manifest.json`

训练器通过 manifest 知道读哪些 token 文件、属于哪个切分。

```python
from pathlib import Path
from quant_fm.manifest.build_manifest import build_manifest

manifest = build_manifest(
    Path("quant_fm/runs/demo/tokens"),
    train_end="2026-02-04",
    val_end="2026-02-05",
    markets=("SZ",),
    vocab_path="quant_fm/runs/demo/data/vocab.json",
)
manifest.save(Path("quant_fm/runs/demo/data/manifest.json"))
```

每个 shard 记录：`market`, `symbol`, `date`, `path`, `rows`, `sha256`, `split`。

---

### 4.6 V2 产物链（底层 API 已实现，批量编排待接入）

V2 不是在 V1 token 上多加几列即可。每个事件必须按交易所序号回放，在 apply 事件前/后分别捕获状态，再生成行对齐特征：

```text
PyLOB event stream
  → iter_book_state_transitions(..., apply_event=...)
  → transitions_to_feature_frame(...)
  → cn_l2_v2.events_to_canonical(..., book_features=...)
  → fit_vocab_v2(train_event_paths, field_specs=FULL_FIELD_SPECS_V2)
  → tokenize_path_v2(...)
  → build_manifest(..., vocab_path=".../vocab_v2.json")
  → manifest.schema_version = vocab.schema_version
  → manifest.save(...)
```

关键契约：

- `book_features` 必须与 events 等长；少了必需 `*_post` 列时 `cn_l2_v2` 会直接报错。
- 事件通用状态使用 `post_event_state(t)` 来预测 `t+1`；事件价格距离明确使用 `event_price_distance_ticks_pre`。
- V2 特殊 ID 是 `PAD=0, UNK=1, NA=2, BOS=3, EOS=4, SESSION_BREAK=5`；真实 0 值不得编码为 `NA`。
- `fit_vocab_v2()` 只能读训练日；分层 reservoir 的 seed、`FieldSpec` 顺序、schema version 和训练日应固化进 `vocab_v2.json`。
- V2 tokens 同时包含 `tok_*` 和 `val_*` 列。下游 `EventWindowDatasetV2` 仅从冻结 `FieldSpec` 派生字段顺序，不猜测 parquet 列。
- `build_manifest()` 默认仍写 `cn_l2_v1`；V2 在保存 manifest 前必须显式执行
  `manifest.schema_version = vocab.schema_version`，不能仅把 `vocab_path` 改成
  `vocab_v2.json`。
- 产物应写入 `quant_fm/runs/v2_shared/`等独立根目录；禁止覆盖 V1 `events/`、`tokens/` 或 `vocab.json`。

完整代码级步骤和验收命令见 [模型底层 V2 代码改造指导](./模型底层v2代码改造指导.md)。

---

## 5. 验证产物

```bash
source .venv/bin/activate
python3 << 'PY'
import polars as pl
from pathlib import Path

base = Path("quant_fm/runs/pilot")  # 或 demo

# 1. 行数一致：clean events → canonical events → tokens
e = pl.read_parquet(base / "events/SZ/000001/2026-02-02.parquet")
t = pl.read_parquet(base / "tokens/SZ/000001/2026-02-02.parquet")
print(f"events rows: {e.height:,}")
print(f"tokens rows: {t.height:,}")
assert e.height == t.height, "events 与 tokens 行数必须相同"

# 2. token 列范围
for col in [c for c in t.columns if c.startswith("tok_")]:
    print(f"  {col}: min={t[col].min()} max={t[col].max()}")

# 3. manifest 统计
import json
m = json.loads((base / "data/manifest.json").read_text())
from collections import Counter
print("splits:", Counter(s["split"] for s in m["shards"]))
PY
```

**试点实测参考（000001 / 2026-02-02）：**

| 阶段 | 行数 |
|------|------|
| PyLOB clean events | 201,014 |
| 规范 events | 201,014 |
| tokens | 201,014 |

---

## 6. 中等规模（多日期 × 全市场）

```bash
# 仅估算规模
make medium-estimate

# 试跑：每市场 50 只股票 × 60 天
make medium-smoke

# 全量（需大磁盘或 MinIO 存储）
make medium
```

脚本：`quant_fm/scripts/run_medium.py`  
默认 60 交易日列表：`quant_fm/data/medium_60_dates.txt`  
默认标的列表：`quant_fm/data/medium_symbols_{sz,sh}.txt`

增量选项（省磁盘 / 加速）：

```bash
CLEAN_WORKERS=16 python -m quant_fm.scripts.run_medium \
  --workdir quant_fm/runs/medium \
  --drop-clean \      # canonicalize 后删 clean/
  --drop-events \     # tokenize 后删 events/
  --resume            # 日期级跳过 + 标的级跳过已有 events/tokens
```

`--resume` 行为：

- 已有 `data/.done/<date>`：整日跳过；
- 已有 `data/.clean_done/<date>`：复用当日 clean；
- clean 阶段：`skip_existing` 跳过已有 `events.parquet` 的标的；
- canonicalize / tokenize：跳过已写出的 parquet。

`CLEAN_WORKERS`（或 `PipelineConfig.n_workers`）控制订单簿重建并行度；`CANON_WORKERS` 控制 `canonicalize_clean_dir` 并行度。

查询进度：

```bash
uv run python -m quant_fm.scripts.check_pipeline_progress --workdir quant_fm/runs/medium_300m
```

### 300M 正式流水线

```bash
source ~/.minio_fm_env.sh
CLEAN_WORKERS=32 CANON_WORKERS=16 SKIP_UPLOAD=1 bash quant_fm/scripts/run_minio_300m_pipeline.sh
```

日期列表：`quant_fm/data/medium_300m_22_dates.txt`  
训练配置：`quant_fm/pretrain/config_medium_300m_8gpu.yaml`（数据就绪后自动 `--resume auto` 开训）

---

## 7. MinIO 读→写流水线（核心）

**读** `zeus-cn-quote` @ `:9000` → 本地 transient 处理 → **写** `model-cache` @ `:9100`

### 7.1 一键跑通（无训练）

```bash
make minio-pipeline          # 试跑：5日×30股
make minio-pipeline-full     # 60日×全市场
```

或：

```bash
python -m quant_fm.scripts.run_medium \
  --workdir quant_fm/runs/medium_try \
  --drop-clean --drop-events \
  --upload-minio --upload-tag medium_try \
  --delete-local-after-upload
```

### 7.2 写入路径

```text
s3://model-cache/fm-pretrain/<user>/{tag}/
  data/vocab.json
  data/manifest.json
  tokens/{market}/{symbol}/{date}.parquet
```

### 7.3 验证上传

```bash
python -m quant_fm.scripts.upload_to_minio \
  --workdir quant_fm/runs/medium_try --tag medium_try --verify
```

详见 [minio_setup.md](./minio_setup.md)。

---

## 8. 关键脚本与模块对照

| 步骤 | 脚本 / 模块 | 函数或入口 |
|------|-------------|------------|
| 一键编排 | `quant_fm/scripts/run_pilot.py` | `run()` |
| 中等规模 | `quant_fm/scripts/run_medium.py` | `run()` + `--upload-minio` |
| **MinIO 读→写** | `quant_fm/scripts/run_minio_data_pipeline.sh` | 无训练全流程 |
| **300M 正式** | `quant_fm/scripts/run_minio_300m_pipeline.sh` | 并行清洗 + 续跑 + 8 卡训 |
| **读配置** | `quant_fm/scripts/minio_config.py` | `load_read_config()` |
| **写/upload** | `quant_fm/scripts/upload_to_minio.py` | `upload_workdir()` |
| **读写自检** | `quant_fm/scripts/check_minio.py` | 连通性 + 默认端口 |
| **进度查询** | `quant_fm/scripts/check_pipeline_progress.py` | 完成天数、当日阶段、ETA |
| Step 1 清洗 | `examples/run_zeus_clean.py` | `build_clean_dataset()` |
| Step 1 核心 | `order_book/pylob/pipeline/workflow.py` | `build_clean_dataset()`（`skip_existing` / `n_workers`） |
| Step 1 事件流 | `order_book/pylob/pipeline/events.py` | `build_event_stream()` |
| Step 2 规范化 | `quant_fm/lob_rebuild/export_events.py` | `canonicalize_clean_dir()`（`skip_existing` / `CANON_WORKERS`） |
| Step 2 schema | `quant_fm/schema/cn_l2_v1.py` | `events_to_canonical()` |
| Step 3 分箱 | `quant_fm/tokenizer/fit_bins.py` | `fit_bins()` |
| Step 4 分词 | `quant_fm/tokenizer/tokenize_events.py` | `tokenize_path()` |
| Step 5 清单 | `quant_fm/manifest/build_manifest.py` | `build_manifest()` |
| V2 状态捕获 | `order_book/pylob/book_state.py` | `iter_book_state_transitions()` |
| V2 盘口转换 | `quant_fm/tokenizer/lob_transforms.py` | `transitions_to_feature_frame()` |
| V2 schema | `quant_fm/schema/cn_l2_v2.py` | `events_to_canonical()` |
| V2 分箱 / 词表 | `quant_fm/tokenizer/fit_bins_v2.py` | `fit_vocab_v2()` |
| V2 分词 | `quant_fm/tokenizer/tokenize_events_v2.py` | `tokenize_path_v2()` |

---

## 9. 常见问题

**Q：`clean/.../tokens.parquet` 和 `quant_fm/.../tokens/` 有什么区别？**  
A：前者是 PyLOB 内置 tokenizer 产物，不是 QuantFM 训练输入。QuantFM V1 读本文生成的 `cn_l2_v1 + vocab.json` tokens；V2 则读 `cn_l2_v2 + vocab_v2.json` 的 token+scalar shards，两者不能混用。

**Q：为什么要分 clean events 和 canonical events 两步？**  
A：PyLOB 输出是撮合引擎内部格式；cn_l2_v1 统一沪深字段、板块、会话等，便于跨市场训练与审计。

**Q：`--skip-clean` 有什么用？**  
A：复用已有 `clean/` 目录，跳过 MinIO 读取，只跑 Step 2–5。

**Q：多只股票、多天怎么切 train/val/test？**  
A：按 **日期** 切，不是按股票。例如 `--train-end 2026-02-04 --val-end 2026-02-05` 表示 ≤2/4 为 train，2/5 为 val，之后为 test。

**Q：磁盘不够怎么办？**  
A：按日处理 + `--drop-clean --drop-events`，结果上传 MinIO 后删本地。见 [minio_data_workflow.md](./minio_data_workflow.md)。

**Q：如何确认 MinIO 原始数据可读？**

先 `source ~/.minio_fm_env.sh`，再运行 [minio_setup.md](./minio_setup.md) 第 7 节连通性自检。

---

## 10. 下一步

events + tokens + manifest 就绪后：

```bash
# 预训练
python -m quant_fm.pretrain.train --config quant_fm/pretrain/config_pilot.yaml

# V2（先按 §4.6 生成 v2_shared/data 产物）
python -m quant_fm.pretrain.train --config quant_fm/pretrain/config_v2_25m.yaml

# 抽训练期 embedding
python -m quant_fm.embedding.extract_hidden \
  --checkpoint quant_fm/runs/pilot/run/best.pt \
  --manifest quant_fm/runs/pilot/data/manifest.json \
  --split train \
  --out quant_fm/runs/pilot/embeddings/train.parquet \
  --device cpu --dtype fp32

# 离线训练并冻结 Ranker（train_ranker.py 本身是库模块，不是 CLI）
python -m quant_fm.signal.train \
  --embeddings quant_fm/runs/pilot/embeddings/train.parquet \
  --panel quant_fm/runs/pilot/panel/daily_panel.parquet \
  --out-dir quant_fm/runs/pilot/signal_artifact \
  --device cpu

# 研究裁判 / 回测
python -m quant_fm.downstream.run_judge \
  --workdir quant_fm/runs/pilot \
  --checkpoint quant_fm/runs/pilot/run/best.pt \
  --device cpu
```

更多概念说明见 [QuantFM.md](./QuantFM.md)。
