# 阶段 7：股日 Embedding、日内聚合与跨股票上下文

> 当前状态（2026-07）：跨 chunk 的 mean/last/last-k 语义、因果重叠窗口和版本化
> 多尺度池化均已实现。默认 V2 为 `context=2048, stride=512`，learned
> `IntradayAggregator` 与 cross-asset 模型仍未接入生产评分编排。

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

旧 checkpoint 的默认语义仍是每个 `context` 独立编码。新 V2 使用 `stride<context`
的重叠窗口：后续窗口携带最多 `context-stride` 个历史事件作为 attention 前缀，但只将
首次出现的后缀送入 pooling，因此每个事件严格计入一次。这不是跨全日 KV cache，契约
名称为 `causal_overlap_unique_emit_v2`。`StockDayPoolAccumulator` 再按时间顺序累积：

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

`--pooling multi_scale` 使用 `int_time` 计算日内毫秒。新
`hierarchical_selected_v2` 严格输出配置声明的四个 `d_model` 向量：

```text
mean_all / last_256 / continuous_pm / close_30m
```

输出宽度为 `4*d_model`，不追加未标准化的 event count。历史 `hierarchical_v1` 仍被
明确解释为 `8*d_model+1`，仅用于读取旧契约，不能冒充新表示。

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
  --context 2048 --stride 512 --pooling multi_scale \
  --batch-size 16 --dtype bf16 --device auto
```

可用 pooling 值为 `mean`、`last`、`lastk_mean` 和 `multi_scale`。`--num-parts` 与
`--part-index` 可按 token row 数近似均衡切分 shard，供多进程/多 GPU 独立提取；
各 part 的输出需要由外部显式合并。

`extract_embeddings_parallel.sh`、K8s 抽取器和 judge 入口在未显式设置时不再
下发 `context/pooling/stride`，而是使用 checkpoint 冻结值。环境变量或 CLI
仍可用于研究覆盖，但正式 Top-K gate 会精确要求 `cn_l2_v2 + post_event +
context=2048 + stride=512 + hierarchical_selected_v2`，不会仅因“也是因果重叠”就放行。

普通 pooling 输出：

```text
date | symbol | market | emb_0 | ... | emb_<d_model-1>
```

V2 `multi_scale` 输出相同列名，宽度由 checkpoint 冻结的 ordered outputs 推导；当前
默认是 `4*d_model`。生成 parquet 不应提交到 GitHub。

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
- 普通 pooling 宽度为 checkpoint `d_model`，当前 V2 multi-scale 为 `4*d_model`；
- 不含 NaN/Inf；
- 相同 checkpoint、vocab、输入和 pooling 配置产生稳定结果；
- v2 checkpoint 缺少 `--vocab` 或 hash/schema/字段顺序不匹配时 fail fast；
- `last`/`lastk_mean` 必须通过跨 chunk 单元测试；
- learned/cross-asset 只能读取当前和过去 interval，PIT 行业生效时间严格 `<` 预测时刻。

相关测试包括 `test_hierarchical_pooling.py`、`test_intraday_aggregator.py`、
`test_cross_asset_causality.py` 和 `test_cross_asset_dataset_model.py`。截至本次改造，
全仓结果为 `396 passed, 2 skipped, 1 xfailed`。这证明代码级语义和防泄漏约束通过，
不代表 learned pooling、cross-asset 或新 FM 已完成真实训练和 OOS 收益验收。
