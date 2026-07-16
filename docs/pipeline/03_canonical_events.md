# 阶段 3：cn_l2_v1 事件规范化

## 目标

将 PyLOB 清洗输出转换为与交易所实现无关的统一事件协议，使 Tokenizer 和模型无需感知上海/深圳字段差异。

## 核心代码

| 文件 | 责任 |
|------|------|
| `quant_fm/schema/cn_l2_v1.py` | schema、固定类别、类型与列顺序 |
| `quant_fm/lob_rebuild/export_events.py` | 清洗目录 → canonical stock-day shards |

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

## 输入与输出

输入：

```text
<workdir>/clean/<date>/...
```

输出：

```text
<workdir>/events/<market>/<symbol>/<date>.parquet
```

每个 parquet 是单标的、单交易日事件流，并按事件顺序排列。

## 验证条件

- 输出列符合 `CANONICAL_COLUMNS`；
- 字段类型符合 PyArrow schema；
- `event_idx` 单调递增；
- `evt_type`、`side`、`session` 等属于固定枚举；
- 每个分片仅包含一个 `date + symbol`。

更详细的原始字段映射见 [raw_to_events_tokens.md](../raw_to_events_tokens.md)。

## 存储策略

`events/` 是 Tokenizer 的直接输入。Medium 流程可在 token 化后使用 `--drop-events` 删除，以降低数百 GB 级磁盘峰值；重新生成时必须使用相同代码版本和原始数据。
