# PyLOB 订单簿引擎

本目录包含 QuantFM 使用的沪深逐笔订单簿撮合、回放与清洗流水线。

## 功能

- 上海、深圳逐笔委托与成交回放；
- 开盘/收盘集合竞价与连续竞价；
- 价格优先、时间优先、同价 FIFO；
- 深圳限价单、市价单与本地最优单；
- 上海 trade + order 委托还原；
- 盘口快照、成交、撤单导出与结果比对；
- MinIO/S3 原始数据读取、标准化和清洗编排。

## 目录

```text
order_book/
├── pylob/              # 可安装的 `pylob` Python 包
│   ├── pipeline/       # MinIO/S3 数据读取、标准化与清洗编排
│   ├── matching_engine.py
│   ├── orderbook_builder_sh.py
│   ├── orderbook_builder_sz.py
│   └── result_mixin.py
└── pyproject.toml      # uv workspace 子项目
```

根项目通过 uv workspace 依赖本子项目，因此外部 Python API 保持不变：

```python
from pylob import MatchingEngine, OrderBookSH, OrderBookSZ
from pylob.pipeline.workflow import build_clean_dataset
```

请在仓库根目录执行依赖安装、测试及数据流水线命令：

```bash
uv sync --dev
uv run python -m pytest tests/test_call_auction.py \
  tests/test_continuous_auction.py \
  tests/test_shanghai_trade_order.py -q
```

完整流水线说明见：

- [订单簿重建阶段](../docs/pipeline/02_order_book_rebuild.md)
- [原始 L2 到 events/tokens](../docs/raw_to_events_tokens.md)
- [QuantFM 文档索引](../docs/README.md)
