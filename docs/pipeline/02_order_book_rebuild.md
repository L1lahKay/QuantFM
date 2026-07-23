# 阶段 2：订单簿重建与清洗

## 目标

将交易所逐笔委托和逐笔成交按时间顺序回放，恢复订单簿状态，并导出统一的清洗事件。

## 核心模块

| 文件 | 责任 |
|------|------|
| `order_book/pylob/matching_engine.py` | 公共撮合、撤单、集合竞价和状态管理 |
| `order_book/pylob/orderbook_builder_sh.py` | 上海逐笔规则与 trade/order 还原 |
| `order_book/pylob/orderbook_builder_sz.py` | 深圳限价、市价、本地最优及撤单规则 |
| `order_book/pylob/result_mixin.py` | 快照、结果比对与导出 |
| `order_book/pylob/pipeline/standardize.py` | 原始列标准化 |
| `order_book/pylob/pipeline/workflow.py` | MinIO → 回放 → parquet 总入口 |

Python 公共 API 保持为：

```python
from pylob import OrderBookSH, OrderBookSZ
from pylob.pipeline.workflow import build_clean_dataset
```

## 数据流

```text
trade/order parquet
  → 字段标准化
  → 按 symbol 过滤
  → 合并并按 local_time/序号排序
  → OrderBookSH / OrderBookSZ 逐条回放
  → ADD / CANCEL / TRADE 事件与订单簿结果
```

### 交易阶段

- `int_time < 09:30:00`：开盘集合竞价；
- `09:30:00 <= int_time < 14:57:00`：连续竞价；
- `int_time >= 14:57:00`：收盘集合竞价。

连续竞价使用价格优先、时间优先，同价位 FIFO；集合竞价按最大可成交量选择价格。

## 运行入口

Pilot 和 Medium 都通过 `PipelineConfig(layout="zeus_default")` 调用：

```python
build_clean_dataset(load_read_config(), pipeline_config)
```

命令入口：

```bash
make pilot
# 或
uv run python -m quant_fm.scripts.run_medium ...
# 300M 正式流水线（含并行清洗）
CLEAN_WORKERS=32 CANON_WORKERS=16 bash quant_fm/scripts/run_minio_300m_pipeline.sh
```

## 加速与断点续跑

`order_book/pylob/pipeline/workflow.py` 现支持：

| 能力 | 说明 |
|------|------|
| `skip_existing` | 若 `<market>/<symbol>/events.parquet` 已存在则跳过该标的 |
| `n_workers` / `CLEAN_WORKERS` | 按标的切分后多进程并行撮合（默认 `min(32, CPU/2)`） |
| `write_debug_artifacts=False` | medium/300M 流水线只写 `events.parquet`，不写 `market_rows`/`tokens`（PyLOB 内置 token 不被 quant_fm 使用） |

并行路径：父进程一次性读入并标准化当日全市场 trade/order → `partition_by(symbol)` → `ProcessPoolExecutor(spawn)` 并行回放。重启任务时配合 `--resume`，已洗标的不会重做。

## 高性能清洗路径 `--fast-clean`（P0/P1 重构）

**动机**：旧路径 `run_medium` 对 SZ、SH **各调用一次** `clean_one_day → build_clean_dataset`，
每次都完整读取 `default/2`（trade）+ `default/3`（order）两个**全市场日文件**（合计约 2.8 亿行）。
于是单日要读 **4 次** MinIO；且 clean 无本地缓存，进程中断/看门狗重启后会**从头重下**。

新增模块 `quant_fm/lob_rebuild/clean_day_fast.py`，通过 `run_medium --fast-clean` 启用，
产物与旧路径**逐字节一致**（已用小样本 old vs new 等价性回归验证），关键改动：

| 优化 | 效果 |
|------|------|
| **单次读取 + 双市场同池** | 一次性读 trade+order（union = SZ∪SH），SZ 用 `OrderBookSZ`、SH 用 `OrderBookSH` 在**同一个** `ProcessPoolExecutor(spawn)` 并行 → 每日 MinIO 读取 **4 次 → 1 次** |
| **本地 raw 缓存** | 过滤+投影后的原始帧落 `<workdir>/data/raw_cache/<date>/{trade,order}.parquet`（原子写）；中断/重启**秒级续跑、绝不重下**。当日 `clean_marker` 落地后自动 `drop_day_cache` 释放磁盘 |
| **列投影** | `scan_parquet` 只 `select` `standardize.REQUIRED_COLUMNS` 中实际存在的列（schema 探测失败则安全退回全列），再减少下载字节 |

数据流（fast 路径）：

```text
读一次 trade+order（union 过滤 + 列投影, 带本地缓存）
  → 一次 standardize
  → 一次 partition_by(symbol)（覆盖两市场）
  → 单个 spawn 进程池：SZ→OrderBookSZ / SH→OrderBookSH 并行回放
  → clean/<date>/<market>/<symbol>/events.parquet（与旧路径同构）
```

命令示例：

```bash
uv run python -m quant_fm.scripts.run_medium \
  --dates-file quant_fm/data/oos2026_dates.txt \
  --workdir quant_fm/runs/oos2026 \
  --reuse-vocab quant_fm/runs/medium_300m/data/vocab.json \
  --symbols-sz-file quant_fm/data/oos2026_liquid_sz.txt \
  --symbols-sh-file quant_fm/data/oos2026_liquid_sh.txt \
  --fast-clean --drop-clean --drop-events --resume
```

> **兼容性**：`--fast-clean` 为 opt-in；不带该标志时行为与旧路径完全一致。
> 等价性回归见 `git log` 中 `clean_day_fast` 提交（临时脚本 `_validate_fast_clean.py` 已删除，逻辑为 old vs new 逐 symbol `events.parquet` 逐字节比对）。

## 输入与输出

输入：

- 原始 trade/order DataFrame；
- `market`、`symbols`、`date`；
- 输出目录。

输出目录以具体 `PipelineConfig` 为准。medium/300M 默认产物为：

```text
clean/<date>/<market>/<symbol>/events.parquet
```

若开启 `write_debug_artifacts=True`（默认，兼容 pilot/调试），还会写出 `market_rows.parquet` 与 `tokens.parquet`。

## 正确性验证

1. 单元与回归测试：

   ```bash
   uv run python -m pytest tests/test_call_auction.py \
     tests/test_continuous_auction.py \
     tests/test_shanghai_trade_order.py -q
   ```

2. 快照一致性：使用 `quant_fm/lob_rebuild/snapshot_check.py` 将重建盘口与交易所快照逐档比较。

3. 结果比对：调用 `compare_df()` 检查回放成交/撤单与输入记录的一致性。

## 设计边界

PyLOB 是研究与离线验证引擎，不是生产级低延迟撮合服务。当前集合竞价实现覆盖核心最大成交量规则，但不宣称完整实现交易所全部 tie-break 细则。
