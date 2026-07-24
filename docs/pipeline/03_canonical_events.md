# 阶段 3：cn_l2_v1 / cn_l2_v2 事件规范化

> 当前状态（2026-07）：`cn_l2_v1` 仍是现有清洗流水线的稳定生产协议；
> `quant_fm/schema/cn_l2_v2.py` 的 schema 与转换 API 已实现，但 v2 raw replay/
> canonicalize 编排尚未接入。v1 与 v2 产物必须写入不同目录，禁止原地覆盖。

## 目标

将 PyLOB 清洗输出转换为与交易所实现无关的统一事件协议，使 Tokenizer 和模型无需感知上海/深圳字段差异。

## 核心代码

| 文件 | 责任 |
|------|------|
| `quant_fm/schema/cn_l2_v1.py` | schema、固定类别、类型与列顺序 |
| `quant_fm/schema/cn_l2_v2.py` | v1 基础列 + 版本化因果盘口列及严格 Arrow schema |
| `quant_fm/lob_rebuild/export_events.py` | 清洗目录 → canonical stock-day shards |
| `quant_fm/tokenizer/lob_transforms.py` | transition → 明确带 `_pre` / `_post` 后缀的盘口特征 |

编排器调用：

```python
canonicalize_clean_dir(
    clean_dir,
    events_dir,
    date=date,
    markets=("SZ", "SH"),
    symbols=symbols,
    skip_existing=resume,  # 已有 <date>.parquet 则跳过该标的
    n_workers=None,        # 默认 CANON_WORKERS 或 min(16, CPU/4)
)
```

`--resume` 时开启 `skip_existing`，避免中断后重写已规范化的股日分片。

上述 `canonicalize_clean_dir()` 当前调用的是 v1 导出逻辑。它不会因为目录里存在盘口
状态就自动升级为 v2，也不会自行重新回放 raw event。

全市场单日约 5100 股，`canonicalize_clean_dir` 使用多进程并行读 clean parquet、写 canonical 分片。环境变量 `CANON_WORKERS` 控制并行度（与洗股阶段的 `CLEAN_WORKERS` 独立）。

## 统一事件

一行代表一个市场事件。主要字段包括：

- 索引：`date`、`exchange`、`symbol`、`security_id`；
- 市场状态：`board`、`session`、`event_source`；
- 事件：`evt_type`、`side`、`order_type`；
- 连续值：`price`、`qty`、`amount`、`delta_t`；
- 审计：`int_time`、`local_time`、`source_seqnum`、`event_idx`、`quality_flag`。

事件类型固定为：

```text
ADD / CANCEL / EXEC / SNAP / STATUS / UNKNOWN
```

价格统一为元，原始整数价格按 `PRICE_SCALE = 10000` 转换。

## cn_l2_v2 扩展

v2 的公共转换入口为：

```python
from quant_fm.schema.cn_l2_v2 import events_to_canonical
from quant_fm.tokenizer.lob_transforms import transitions_to_feature_frame

book_features = transitions_to_feature_frame(
    transitions,
    event_prices=event_prices_in_integer_price_units,
    tick_size=100,
)
canonical_v2 = events_to_canonical(
    events,
    date="2026-01-05",
    market="SZ",
    book_features=book_features,
)
```

`events_to_canonical()` 提供结构性防线：它要求 `book_features` 与事件行数完全一致、
必需 post-event 列存在，并拒绝 `book_valid_post` 含 null。该检查本身不能证明这些特征
确实来自逐事件回放；例如其他数值列在类型允许时仍可为 null，整列
`book_valid_post=False` 也不违反转换接口。真实盘口 provenance 必须由显式 transition
capture、事件/transition/feature 一一对应、prefix-causality、coverage 和真实快照逐档
一致性共同验收。当前扩展列为：

```text
exchange_seqnum
time_of_day_ms
book_valid_post
spread_ticks_post
microprice_delta_ticks_post
imbalance_l1_post
imbalance_l5_post
imbalance_l10_post
log_bid_depth_l5_post
log_ask_depth_l5_post
event_price_distance_ticks_pre
```

其中通用盘口输入来自事件 `t` 执行后的状态，用于预测 `t+1`；事件价格距离只使用
事件 `t` 执行前的 midpoint。`schema_version` 固定为 `cn_l2_v2`。

## 输入与输出

输入：

```text
<workdir>/clean/<date>/...
```

v1 输出：

```text
<workdir>/events/<market>/<symbol>/<date>.parquet
```

每个 parquet 是单标的、单交易日事件流，并按事件顺序排列。

v2 尚无独立 CLI 或自动目录编排；接入时建议写到新的 `events_v2/` 或
`runs/*_v2/data/events/`，并先验证 raw replay capture 的一一对应关系。

## 验证条件

- 输出列符合 `CANONICAL_COLUMNS`；
- 字段类型符合 PyArrow schema；
- `event_idx` 单调递增；
- `evt_type`、`side`、`session` 等属于固定枚举；
- 每个分片仅包含一个 `date + symbol`。

v2 还必须验证：

- `schema_version == "cn_l2_v2"`；
- `exchange_seqnum` 与交易所时间共同保持稳定顺序；
- `book_valid_post` 无 null，其他盘口数值允许在空簿/单边簿时为 null；
- 改写未来事件不会改变当前及过去行的特征；
- 事件数、transition 数和 `book_features` 行数完全相同。
- 正式数据还应记录 raw replay 输入与代码版本，并检查 `book_valid_post` 覆盖率及真实
  快照逐档一致率；通过 schema 转换不等于已证明盘口来源真实。

更详细的原始字段映射见 [raw_to_events_tokens.md](../raw_to_events_tokens.md)。

## 存储策略

`events/` 是 v1 Tokenizer 的直接输入。Medium 流程可在 token 化后使用
`--drop-events` 删除，以降低数百 GB 级磁盘峰值；重新生成时必须使用相同代码版本
和原始数据。v2 事件不可复用 v1 目录的 resume marker；其 schema、vocab、token、
checkpoint 必须作为一组独立 artifact 管理。
