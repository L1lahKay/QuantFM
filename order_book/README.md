# PyLOB 订单簿引擎

本目录包含 QuantFM 使用的沪深逐笔订单簿撮合、回放与清洗流水线。

## 功能

- 上海、深圳逐笔委托与成交回放；
- 开盘/收盘集合竞价与连续竞价；
- 价格优先、时间优先、同价 FIFO；
- 深圳限价单、市价单与本地最优单；
- 上海 trade + order 委托还原；
- 撤单同时更新价格档 deque 与活动订单索引，保持同价 FIFO 和盘口一致性；
- 盘口快照、成交、撤单导出与结果比对；
- 为模型底层 v2 逐事件捕获严格因果的 pre/post 盘口状态；
- MinIO/S3 原始数据读取、标准化和清洗编排。

## 目录

```text
order_book/
├── pylob/              # 可安装的 `pylob` Python 包
│   ├── pipeline/       # MinIO/S3 数据读取、标准化与清洗编排
│   ├── book_state.py   # 不修改撮合簿的紧凑快照与逐事件 transition
│   ├── matching_engine.py
│   ├── orderbook_builder_sh.py
│   ├── orderbook_builder_sz.py
│   └── result_mixin.py
└── pyproject.toml      # uv workspace 子项目
```

根项目通过 uv workspace 依赖本子项目，因此外部 Python API 保持不变：

```python
from pylob import (
    MatchingEngine,
    OrderBookSH,
    OrderBookSZ,
    capture_book_transition,
    snapshot_book_state,
)
from pylob.pipeline.workflow import build_clean_dataset
```

## 模型 v2 的盘口契约

[`pylob.book_state`](pylob/book_state.py) 只读取活动订单索引和价格档，不改变撮合引擎。`BookState` 提供 bid/ask 一档、5/10 档深度、spread、imbalance 和 microprice delta；`BookStateTransition` 明确区分当前事件处理前后的状态。

```python
transition = capture_book_transition(
    book,
    lambda: apply_one_exchange_event(event),
    tick_size=100,
)

# event t 的输入可使用 post，预测目标必须是 t+1；
# 当前事件价格相对盘口的位置则从 pre 计算。
state_for_next_event = transition.post_event_state
```

调用方必须先按交易所时间和 exchange/source sequence 稳定排序；不得用可能带接收延迟的 `local_time` 排序，也不得在回放前预计算未来盘口。特征转换入口为 [`quant_fm/tokenizer/lob_transforms.py`](../quant_fm/tokenizer/lob_transforms.py)，v2 schema 入口为 [`quant_fm/schema/cn_l2_v2.py`](../quant_fm/schema/cn_l2_v2.py)。`cn_l2_v2.events_to_canonical()` 强制接收逐行对齐的真实 `book_features`，不会接受占位盘口列。

请在仓库根目录执行依赖安装、测试及数据流水线命令：

```bash
uv sync --dev
uv run python -m pytest tests/test_call_auction.py \
  tests/test_continuous_auction.py \
  tests/test_shanghai_trade_order.py \
  tests/test_order_book_cancel_consistency.py \
  tests/test_book_state_causality.py -q
```

全仓当前基线（2026-07-24）为 `243 passed, 2 skipped, 1 xfailed`。真实快照/真实数据测试依赖本机数据，缺失时会按测试约定跳过；上述结果不能替代正式交易日的逐档快照一致性验收。

完整流水线说明见：

- [订单簿重建阶段](../docs/pipeline/02_order_book_rebuild.md)
- [原始 L2 到 events/tokens](../docs/raw_to_events_tokens.md)
- [QuantFM 文档索引](../docs/README.md)
