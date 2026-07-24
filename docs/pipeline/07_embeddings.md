# 阶段 7：股日 Embedding、日内聚合与跨股票上下文

> 当前状态（2026-07）：跨 chunk 的 mean/last/last-k 语义已修复，固定多尺度池化、
> learned `IntradayAggregator` 以及同步跨股票基础设施已实现并有测试。默认 embedding
> CLI 已接入前三种和固定 `multi_scale`；learned aggregator 与 cross-asset 模型尚未
> 接入生产评分和标准训练/推理编排。

## 核心代码

| 文件 | 责任 |
|------|------|
| `quant_fm/embedding/extract_hidden.py` | checkpoint 加载、分块/批量编码和 parquet 输出 |
| `quant_fm/embedding/pool_stock_day.py` | 正确跨 chunk 池化与固定多尺度累加器 |
| `quant_fm/embedding/intraday_aggregator.py` | 独立的因果 learned chunk 聚合器 |
| `quant_fm/pretrain/train.py::load_checkpoint` | 重建 v1/v2 模型并执行 v2 artifact 校验 |
| `quant_fm/cross_asset/clock_grid.py` | 异步事件时间映射到不跨午休的交易 interval |
| `quant_fm/cross_asset/context_pool.py` | O(N) market/industry leave-one-out context |
| `quant_fm/cross_asset/dataset.py` | 严格 PIT 行业 join 和 `[T,N,D]` 面板对齐 |
| `quant_fm/cross_asset/model.py` | interval 级 O(T×N) 上下文编码与因果 GRU |

## 已修复的跨 chunk 池化

一个股日仍按 `context` 独立送入 FM；chunk 之间没有 event-transformer attention 或
KV cache。区别在于 `StockDayPoolAccumulator` 现在按时间顺序跨所有 chunk 累积：

- `mean`：所有有效事件 hidden 的严格全日均值；
- `last`：只取全日最后一个有效 hidden，不再平均每个 chunk 的 last；
- `lastk_mean`：跨 chunk 保留全日最后 `--last-k` 个 hidden；
- 空输入返回零向量，内部累积统一使用 CPU float32。

公共 API：

```python
from quant_fm.embedding.pool_stock_day import (
    StockDayPoolAccumulator,
    pool_hidden,
    pool_hidden_chunks,
)
```

## 固定多尺度池化

`--pooling multi_scale` 使用 `int_time` 计算日内毫秒，并输出八个 `d_model` 向量：

```text
mean_all / last_256 / last_1024 / open_call
continuous_am / continuous_pm / close_call / close_30m
```

最后追加一个原始 `event_count` 标量，因此输出宽度为 `8 * d_model + 1`。没有事件的
阶段窗口使用零向量。这个方法不训练额外参数，可直接给现有 Ranker 做无重训消融。

## Embedding CLI

v1 示例：

```bash
uv run python -m quant_fm.embedding.extract_hidden \
  --checkpoint quant_fm/runs/pilot/run/best.pt \
  --manifest quant_fm/runs/pilot/data/manifest.json \
  --split test \
  --out quant_fm/runs/pilot/embeddings.parquet \
  --context 2048 --pooling mean --device auto
```

v2 必须额外提供生成 checkpoint 时的 vocab，以校验 hash/schema/字段顺序：

```bash
uv run python -m quant_fm.embedding.extract_hidden \
  --checkpoint quant_fm/runs/v2_25m/run/best.pt \
  --vocab quant_fm/runs/v2_shared/data/vocab_v2.json \
  --manifest quant_fm/runs/v2_shared/data/manifest.json \
  --split test \
  --out quant_fm/runs/v2_25m/embeddings_multiscale.parquet \
  --context 2048 --pooling multi_scale \
  --batch-size 16 --dtype bf16 --device auto
```

可用 pooling 值为 `mean`、`last`、`lastk_mean` 和 `multi_scale`。`--num-parts` 与
`--part-index` 可按 token row 数近似均衡切分 shard，供多进程/多 GPU 独立提取；
各 part 的输出需要由外部显式合并。

普通 pooling 输出：

```text
date | symbol | market | emb_0 | ... | emb_<d_model-1>
```

`multi_scale` 输出相同列名，但 embedding 宽度为 `8*d_model+1`。生成 parquet 不应
提交到 GitHub。

## Learned 日内聚合器

`IntradayAggregator` 已实现为 1–2 层因果 `GRUCell`，输入为：

```python
outputs = aggregator(
    chunk_summaries,  # [B, C, input_dim]
    chunk_time,       # [B, C]，真实日内时间
    chunk_session,    # [B, C]
    chunk_mask,       # [B, C]
)
```

输出键固定为：

```text
full_day_summary
close_summary
intraday_trend_summary
activity_summary
```

有效 chunk 序号不受 padding 布局影响，`encode_chunks()` 可用于检查逐位置因果性。
当前 `extract_hidden.py` 不会创建、训练或加载该聚合器；要使用它，需要新增 chunk
summary dataset、优化目标、checkpoint 和提取编排。在完成该集成前，生产输出仍使用
固定 pooling。

## 跨股票上下文基础设施

实现遵循“先单股汇总，再同步”的边界，不接受全市场异步 raw event：

```text
单股 interval embedding
  → clock_interval_id / add_clock_interval（默认 5 分钟，不跨午休）
  → join_pit_industry（effective_time 必须严格早于 prediction_time）
  → align_interval_embeddings -> [T, N, D] + active_mask
  → build_synchronous_context（market + industry leave-one-out）
  → LinearCrossAssetModel（O(T*N*D) + 每股因果 GRU）
```

`industry_leave_one_out` 明确扣除自身；行业无同伴时输出零并设置
`industry_has_peer=False`。缺失/停牌位置不进入 market 或 industry pool，也不会更新
GRU 状态。主要 API：

```python
from quant_fm.cross_asset.clock_grid import add_clock_interval
from quant_fm.cross_asset.dataset import build_cross_asset_panel
from quant_fm.cross_asset.model import CrossAssetModelConfig, LinearCrossAssetModel
```

这些模块尚未接入 `extract_hidden`、Ranker 的默认数据集、`run_judge` 或 score 生产
入口；现阶段属于可测试的研究基础设施，不能声称已经改善生产 score。

## 验证条件

- 输出行数等于所选 split 的非空股日 shard 数；
- 普通 pooling 宽度为 checkpoint `d_model`，multi-scale 为 `8*d_model+1`；
- 不含 NaN/Inf；
- 相同 checkpoint、vocab、输入和 pooling 配置产生稳定结果；
- v2 checkpoint 缺少 `--vocab` 或 hash/schema/字段顺序不匹配时 fail fast；
- `last`/`lastk_mean` 必须通过跨 chunk 单元测试；
- learned/cross-asset 只能读取当前和过去 interval，PIT 行业生效时间严格 `<` 预测时刻。

相关测试包括 `test_hierarchical_pooling.py`、`test_intraday_aggregator.py`、
`test_cross_asset_causality.py` 和 `test_cross_asset_dataset_model.py`。截至本次改造，
全仓结果为 `243 passed, 2 skipped, 1 xfailed`。这证明代码级语义和防泄漏约束通过，
不代表 learned pooling、cross-asset 或新 FM 已完成真实训练和 OOS 收益验收。
