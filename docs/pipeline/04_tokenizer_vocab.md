# 阶段 4：Tokenizer 与字段级词表

## 目标

把连续、异构的市场事件转换为模型可消费的并行整数 token 通道，同时保持因果性、跨股票尺度可比性和时间切分无泄漏。

## Token 通道

| Token | 来源 | 编码 |
|-------|------|------|
| `tok_evt_type` | 事件类型 | 固定类别 |
| `tok_side` | 买/卖/中性 | 固定类别 |
| `tok_session` | 交易时段 | 固定类别 |
| `tok_board` | 板块 | 固定类别 |
| `tok_order_type` | 委托类型 | 固定类别 |
| `tok_event_source` | 事件来源 | 固定类别 |
| `tok_price_bin` | `price_rel` | 32 个分位数 bin |
| `tok_volume_bin` | `log_volume` | 32 个分位数 bin |
| `tok_delta_t_bin` | `log_delta_t` | 32 个分位数 bin |

每个字段拥有独立 id 空间，`PAD=0`。不存在类别字段与连续字段的笛卡尔积大词表。

## 因果连续特征

核心实现：`quant_fm/tokenizer/transforms.py`。

- `price_rel = log(price / causal_EW_VWAP_mid)`；
- `log_volume = log1p(qty / 100)`；
- `log_delta_t = log1p(delta_ms)`。

EW-VWAP 仅使用当前及历史成交，半衰期默认为 5 秒，不使用未来事件。

## 分箱拟合

核心实现：`quant_fm/tokenizer/fit_bins.py`。

1. 仅扫描训练日期的 canonical events；
2. 每字段最多采样 5,000,000 个有限值；
3. `price_rel` 做 1%/99% 双侧缩尾；
4. 非负字段保留下界并做 99% 上尾缩尾；
5. 计算 31 条分位数边界形成 32 个 bin；
6. 将边界和 `fit_dates` 冻结到 `vocab.json`。

如果分位数重复导致边界坍缩，则使用线性间隔补齐。

## Tokenize

核心实现：`quant_fm/tokenizer/tokenize_events.py`。

```python
vocab = fit_bins(train_paths, n_bins=32, fit_dates=train_dates)
vocab.save(vocab_path)
assert_no_leakage(vocab, val_dates, test_dates)
tokenize_path(event_path, token_path, vocab)
```

输出：

```text
<workdir>/tokens/<market>/<symbol>/<date>.parquet
<workdir>/data/vocab.json
```

## 验证闸门

- `assert_no_leakage()`：任何 val/test 日期出现在 `fit_dates` 都立即失败；
- `coverage_report()`：统计连续字段首尾极端 bin 占比和类别字段 UNKNOWN 率；
- 确定性：同一输入 parquet + 同一 `vocab.json` 应生成相同 token。

## 设计取舍

字段级词表便于解释和扩展，并避免组合词表爆炸。代价是字段间联合关系由 Transformer 隐状态学习，而不是在单个 token id 中显式表达。
