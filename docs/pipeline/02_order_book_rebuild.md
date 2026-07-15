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
```

## 输入与输出

输入：

- 原始 trade/order DataFrame；
- `market`、`symbols`、`date`；
- 输出目录。

输出目录以具体 `PipelineConfig` 为准，供下一阶段读取，包含标准化事件和必要的回放中间结果。

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
