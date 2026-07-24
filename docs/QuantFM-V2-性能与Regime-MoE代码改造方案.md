# QuantFM V2：性能、预测质量与 Regime-MoE 代码改造方案

> 状态：分阶段实施中（代码快照：2026-07-24）
> 基线：`medium_300m`，302.30M，`d_model=1024 × 18 layers`
> 目标：提高严格时序 OOS RankIC/ICIR，同时提高有效 token 吞吐、降低 embedding 推理时延
> 生产契约：保持 `date, symbol, score` 不变
> 兼容原则：v1 token、vocab、checkpoint、embedding 只读保留；V2 使用独立版本和目录

### 当前落地边界

| 模块 | 当前代码状态 | 还需完成 |
|---|---|---|
| 订单簿 / Schema / Tokenizer V2 | 撤单队列一致性已修复；已有因果 `BookState`、`cn_l2_v2`、`FieldSpec`、分层 reservoir、token+scalar 产物 | 将真实回放批量产物接入现有 MinIO 编排并生成 V2 manifest |
| 字段融合 / Loss | `legacy_sum/scaled_sum/gated_sum/concat_mlp`、field dropout、entropy-normalized CE、ordinal 辅助损失和 applicability mask 已实现 | 运行 25M/100M 多 seed 消融，不得把接口测试当成质量结论 |
| 表征聚合 | 跨 chunk `last/last-k`、多尺度固定池化和 `IntradayAggregator` 模块已实现 | 聚合器尚未接入默认生产 score 路径，需独立训练/回测 |
| 股票间影响 | 已有 5 分钟时钟对齐、PIT 行业 join、leave-one-out 上下文与 O(N) 模型 | 仍是可组合研究模块，未替换当前默认 Ranker |
| 严格 OOS 评估 | 已有明确 ReturnSpec、execution panel、缓冲 Top-K/成本、风险归因和独立 score 研究入口 | 2026 连续 60 日已参与方案形成，只是 architecture validation；最终结论必须用新的 untouched OOS |
| 训练性能 / MoE | fixed validation sampler 已实现 | micro/update/token 计数、attention fast path、推理 checkpoint 拆分、Regime-MoE 和 Backbone-MoE 尚未实现 |

因此，本文后续 PR 节仍是「目标设计 + 验收门槛」，不代表所列项目全部已进入主流程。可直接运行的底层 V2 路径见 [模型底层 V2 代码改造指导](./模型底层v2代码改造指导.md)，严格 score 研究边界见 [严格 OOS 研究回测](./严格OOS研究回测.md)。

## 1. 总体结论

不建议直接将当前 18 层 Dense Transformer 全部替换为 token-level MoE。推荐按四层推进：

1. 修复训练 step、RoPE、causal mask、数据局部性和推理 checkpoint；
2. 修复跨 chunk 池化，引入多尺度日内表征和轻量 temporal aggregator；
3. 构建真实因果盘口、连续时间、有序 Loss 和约 230M 高效主干的 Dense V2；
4. 先在股日聚合/Ranker 引入 Regime-MoE，通过后再替换顶部少量 FFN。

核心判断：

- 当前 RMSNorm、RoPE、causal attention、SwiGLU 主干方向合理；
- 当前 302M 模型实际欠训练，不能据此判断 Dense 容量已经用尽；
- 对次日横截面预测，真实盘口输入、目标对齐和日内池化比盲目增加参数更重要；
- MoE 不会自动形成牛/熊专家，专家性取决于路由粒度、因果 regime 输入和行情覆盖；
- 当前 embedding 小 batch 下，主干 MoE 可能因 dispatch/all-to-all 比 Dense 更慢。

## 2. 已确认的基线事实

### 2.1 训练预算语义错误

当前 `quant_fm/pretrain/train.py` 每个 micro-batch 增加 `state.step`，但 `grad_accum=2` 时每两个 micro-step 才更新一次参数。LR、验证、checkpoint 和 `max_steps` 却都使用同一个 step。

```text
train windows                = 2,348,341
world_size × batch_size      = 8 × 4 = 32 windows / micro-step
configured max_steps         = 40,000 micro-steps
processed windows            ≈ 1,280,000
actual optimizer updates     = 20,000
effective epoch              ≈ 0.545
```

配置注释把 40,000 当 optimizer updates，和代码实际行为相差约 2 倍。验证 loss 从约 5.93 持续下降到最终约 5.33，尚未明显平台。

### 2.2 Loss 与表征出口问题

现有验证 CE：

```text
session       ≈ 0.00063
event_type    ≈ 0.144
price         ≈ 0.696
delta_time    ≈ 1.112
volume        ≈ 2.147
```

`session` 几乎是复制任务，高熵 volume/delta-time 主导原始 CE 总和。当前股日又按 2048 events 独立编码后全日均值，上午、下午、尾盘顺序及状态切换被抹平。

### 2.3 订单簿与 OOS 边界

改造前的全量测试为：

```text
132 passed, 8 failed, 2 skipped, 1 xfailed
```

8 个失败均涉及撤单后零数量订单仍残留在价格档队列。当前代码已修复物理 deque 删除、空价格档清理和 FIFO 一致性，并为盘口转换增加了 prefix-causality 测试。截至 2026-07-24，仓库全量测试为 `215 passed, 2 skipped, 1 xfailed`；这只证明代码契约通过，不等于 V2 已完成训练或 OOS 验证。

2026 连续 60 日当前 `mean RankIC≈0.0774`、`ICIR≈0.526`、随机约 `0.0033`。该区间已经参与方案形成，只能作为 architecture validation；最终方案需使用新的 untouched 日期测试。

## 3. 目标架构

```text
正确的因果订单簿回放
    ↓
cn_l2_v2 event + compact post-event book state
    ↓
Tokenizer V2
    ├── categorical token
    ├── ordinal bin + normalized raw scalar
    ├── continuous time-of-day
    └── PIT stock/day context
    ↓
EventFieldEncoder（scaled/gated sum；concat_mlp 作为消融）
    ↓
Efficient Dense Causal Event Transformer
    ↓
chunk summaries + clock/session/activity
    ↓
Intraday Temporal Aggregator
    ↓
Causal Regime Router
    ├── shared expert
    ├── 4 routed experts
    └── Top-2 soft routing
    ↓
O(N) market / PIT industry leave-one-out context
    ↓
Cross-sectional Ranker
    ↓
date, symbol, score
```

## 4. 推荐配置

### 4.1 Dense V2

```yaml
model:
  d_model: 1024
  n_layers: 18
  n_heads: 16
  n_kv_heads: 16          # 首轮保持 MHA；GQA 单独消融
  ffn_hidden: 2816        # 替代当前 SwiGLU 4096
  dropout: 0.05           # 与 0.0 单独消融
  max_seq_len: 2048
  attention: full
```

约 231M 参数，相对当前 302M 参数减少约 23%，`L=2048,d=1024` 下理论主干乘加量减少约 19%。第一轮不同时引入 GQA、local attention 和 MoE。

### 4.2 轻量 Regime-MoE

```yaml
regime_moe:
  enabled: true
  placement: temporal_aggregator
  n_experts: 4
  top_k: 2
  shared_expert: true
  expert_hidden: 256
  router_hidden: 128
  router_temperature: 1.0
  load_balance_weight: 0.01
  router_z_loss_weight: 0.001
```

### 4.3 顶部 Sparse-MoE（后续候选）

```yaml
backbone_moe:
  enabled: false
  layer_indices: [14, 15, 16, 17]
  n_routed_experts: 4
  top_k: 1
  shared_expert_hidden: 1024
  routed_expert_hidden: 1792
  capacity_factor: 1.25
```

每个 token 激活的 FFN hidden 总量约 `1024+1792=2816`，active FFN 计算接近 Dense V2。

## 5. PR-0：实验契约与基准工具（部分落地）

### 新增文件

```text
quant_fm/benchmark/model_benchmark.py
quant_fm/benchmark/embedding_benchmark.py
quant_fm/pretrain/validation_sampler.py
quant_fm/experiments/registry.py
tests/test_validation_sampler.py
tests/test_experiment_registry.py
```

禁止只报告 `step/s`。统一记录：

```text
non_pad_tokens_per_second
windows_per_second
optimizer_updates_per_second
mean/p95_step_time
peak_allocated/reserved_gpu_memory
checkpoint_pause_seconds
embedding_stock_days_per_second
cold_checkpoint_load_seconds
warm_inference_latency_p50/p95
```

所有吞吐按真实 `attention_mask.sum()` 计算。质量统一记录字段 CE/accuracy/entropy-normalized CE、paired daily IC、Newey-West t、block bootstrap、分组单调性、扣成本 Top-bottom 和换手。

验证窗口按以下层次固定并写入 artifact：

```text
date × exchange × board × liquidity_bucket × activity_bucket
```

## 6. PR-1：训练语义与数据吞吐（待实施）

### 修改文件

```text
quant_fm/pretrain/train.py
quant_fm/pretrain/dataset.py
quant_fm/pretrain/config*.yaml
tests/test_training_schedule.py
tests/test_gradient_accumulation.py
tests/test_shard_aware_sampler.py
```

拆分训练状态：

```python
@dataclass(slots=True)
class TrainState:
    micro_step: int = 0
    update_step: int = 0
    samples_seen: int = 0
    non_pad_tokens_seen: int = 0
    best_val: float = float("inf")
    best_update_step: int = -1
```

规则：

- 每个 batch 增加 `micro_step`；
- 只有 `optimizer.step()` 成功后增加 `update_step`；
- LR、warmup、eval、best checkpoint 基于 `update_step`；
- 训练终止优先基于 `max_train_tokens`；
- resume 保存并恢复全部计数器，LR phase 连续；
- 优先测试 `micro_batch=8, grad_accum=1`，保持 global batch 64；
- 必须累积时评估 FSDP `no_sync()` 的显存与通信收益。

新增 shard-aware sampler：先打乱 shard，在 shard 内小范围连续消费 window，再按总 token 行数分配 rank；短尾窗口按长度 bucket。DataLoader 开启 `persistent_workers`、合理 `prefetch_factor`、pinned memory，GPU copy 使用 `non_blocking=True`。

验收：不同 `grad_accum` 在相同 global batch/token 数下 update 数一致；一个 epoch token 误差 <0.1%；resume LR 连续；sampler 不重复、不遗漏。

## 7. PR-2：无损注意力与推理性能（待实施）

### 修改文件

```text
quant_fm/pretrain/model.py
quant_fm/pretrain/train.py
quant_fm/embedding/extract_hidden.py
quant_fm/signal/artifact.py
tests/test_attention_fast_path.py
tests/test_rope_cache.py
tests/test_inference_checkpoint.py
```

### 7.1 RoPE cache

当前 `_rope` 成员未实际使用。改为按 `(device, dtype, max_seq_len)` 惰性缓存：频率使用 FP32 计算，cos/sin 转成 q/k dtype 后复用。cache 不进入 checkpoint，模型 `.to()` 后允许重建。

```python
def _get_rope(self, length, device, dtype):
    key = (device.type, device.index, dtype)
    cached = self._rope.get(key)
    if cached is None or cached[0].size(0) < length:
        cos, sin = build_rope_fp32(self.cfg.max_seq_len, ...)
        self._rope[key] = (cos.to(dtype), sin.to(dtype))
    cos, sin = self._rope[key]
    return cos[:length], sin[:length]
```

### 7.2 causal fast path

```python
if bool(key_mask.all()):
    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=True,
    )
else:
    out = padded_causal_attention(q, k, v, key_mask, dropout_p)
```

长度分桶后应让绝大多数完整窗口进入 `is_causal=True`，避免每层构造 `B×L×L` mask。必须验证 FP32/BF16 数值等价、padding 有效位一致，以及未来 token 扰动不改变过去输出。

### 7.3 fused op 与 compile

依次独立 benchmark：

1. `torch.nn.functional.rms_norm`；
2. fused/foreach AdamW；
3. `torch.compile` 裸模型；
4. block-level FSDP 的 compile 兼容性；
5. activation checkpointing。

全部配置化，不把未验证组合设为默认。

### 7.4 推理 checkpoint

当前 `best.pt` 含完整 AdamW 状态，约 3.6GB。拆分为：

```text
run/resume_step*.pt       # FP32 model + optimizer + scaler + TrainState
run/best_model.pt         # model + config + metadata
run/best_model_bf16.pt    # 可选推理权重
run/final_model.pt
```

推理先 `map_location="cpu"`，只读取 model/metadata，再一次性转 device/dtype，使用 `torch.inference_mode()`，并校验 checkpoint/vocab/schema hash。

## 8. PR-3：跨 chunk 正确池化与时间聚合（代码已落地，实验待做）

这是第一项可复用现有 300M checkpoint 验证预测收益的改造。

### 修改/新增文件

```text
quant_fm/embedding/pool_stock_day.py
quant_fm/embedding/extract_hidden.py
quant_fm/embedding/intraday_aggregator.py
quant_fm/embedding/schema.py
tests/test_multichunk_pooling.py
tests/test_hierarchical_pooling.py
```

正确语义：

```text
mean_all      = 全部有效事件 hidden 的严格加权平均
last          = 最后一个 chunk 的最后有效 hidden
last_k        = 跨 chunk 收集全日最后 K 个 hidden
session_mean  = 按真实 session/time mask 聚合
```

第一轮固定输出：

```text
mean_all
last_256
last_1024
continuous_am
continuous_pm
close_30m
event_count
chunk_count
```

研究期允许完整拼接；生产候选通过每个 pool 的小投影控制 embedding 尺寸。

```python
class IntradayAggregator(nn.Module):
    def forward(
        self,
        chunk_hidden: Tensor,       # [B, C, D]
        chunk_time: Tensor,         # [B, C]
        chunk_session: Tensor,      # [B, C]
        chunk_activity: Tensor,     # [B, C, A]
        chunk_mask: Tensor,         # [B, C]
    ) -> Tensor:
        ...
```

首选 1–2 层 gated MLP/GRU，小规模对照 1 层 Transformer。聚合计算控制在 FM backbone 的 2% 以内。

## 9. PR-4：订单簿、Schema V2 与 Tokenizer V2（核心代码已落地）

详细字段规范参见 `docs/模型底层v2代码改造指导.md`。实施顺序不可倒置。

### 9.1 订单簿硬前置

修改：

```text
order_book/pylob/matching_engine.py
order_book/pylob/orderbook_builder_sz.py
order_book/pylob/orderbook_builder_sh.py
```

必须保证：

- `orders` 不存在的订单不残留在价格档 deque；
- `quantity <= 0` 及时清理；
- 空价格档删除；
- 撤销中间订单不破坏 FIFO；
- SH/SZ 行为一致。

当前 8 个一致性失败已消除；生成盘口训练数据时仍必须通过回放顺序、行数对齐和 prefix-causality 门控。

### 9.2 紧凑因果盘口字段

第一版控制在 8–10 个字段：

```text
book_valid_post
spread_ticks_post
microprice_delta_ticks_post
imbalance_l1_post
imbalance_l5_post
log_bid_depth_l5_post
log_ask_depth_l5_post
event_price_distance_ticks_pre
time_of_day_ms
price_limit_distance_ticks
```

通用状态使用 `post_event_state(t)` 作为事件 t 表示的一部分预测 t+1；queue/price distance 使用明确命名的 pre-event state。

### 9.3 Tokenizer V2

新增：

```text
quant_fm/tokenizer/field_spec.py
quant_fm/tokenizer/lob_transforms.py
quant_fm/tokenizer/vocab_v2.py
```

要求：

- NA 与真实 0 使用不同 token；
- 连续字段同时保留 ordinal bin 和标准化 raw scalar；
- same-timestamp、spread 小整数、volume 整手等质量点单独编码；
- imbalance 使用对称区间；
- 分箱覆盖全训练期，使用确定性 reservoir/分层采样；
- 删除确定性冗余 `event_source`；
- `session` 保留输入但不作为主预测目标；
- 未恢复真实交易所 `order_type` 前不使用伪字段。

## 10. PR-5：字段融合、Dense V2 与 Loss（底层代码已落地）

### 10.1 EventFieldEncoder

新增 `quant_fm/pretrain/field_fusion.py` 和 `tests/test_field_fusion.py`，按顺序消融：

1. `sum / sqrt(valid_field_count) + input RMSNorm`；
2. gated sum + field dropout；
3. 每字段 16–64 维编码后 concat MLP 到 `d_model`。

不要拼接九个完整 1024 维 embedding。

### 10.2 TargetSpec 与 Loss

```python
@dataclass(frozen=True, slots=True)
class TargetSpec:
    name: str
    loss_type: str
    weight: float
    train_entropy: float
    ordinal_weight: float = 0.0
    applicable_events: tuple[str, ...] = ()
```

```yaml
loss:
  normalize_by_train_entropy: true
  targets:
    tok_evt_type:    {type: ce,         weight: 1.0}
    tok_side:        {type: ce,         weight: 1.0}
    tok_price_bin:   {type: ordinal_ce, weight: 1.0, ordinal_weight: 0.5}
    tok_volume_bin:  {type: ordinal_ce, weight: 0.5, ordinal_weight: 0.25}
    tok_delta_t_bin: {type: ordinal_ce, weight: 0.5, ordinal_weight: 0.25}
    tok_session:     {type: ce,         weight: 0.0}
```

每个目标拥有 applicability mask。全 NA 或无适用样本的任务返回零损失，不产生 NaN。

仅作为 target 增加：

```text
future_microprice_move_10/50/200_events
future_mid_move_1s/5s
future_signed_ofi_5s
future_spread_change_5s
future_realized_vol_30s
```

不在主预训练直接加入 T+1 日收益；日级目标继续由 Ranker或小规模 fine-tuning 学习。

## 11. PR-6：轻量 Regime-MoE（待实施）

第一版放在 chunk/stock-day temporal aggregator 或 Ranker，不放在逐事件 FFN。

### 新增文件

```text
quant_fm/moe/config.py
quant_fm/moe/router.py
quant_fm/moe/regime_features.py
quant_fm/moe/temporal_moe.py
quant_fm/moe/telemetry.py
tests/test_moe_router.py
tests/test_moe_causality.py
tests/test_moe_load_balance.py
tests/test_moe_permutation.py
```

### 11.1 因果 Router 输入

对 `score(T)`，只允许使用截至 T 已知的信息：

```text
market_return_5d/20d/60d through T
market_realized_vol and breadth through T
market turnover/activity T
PIT industry relative strength through T
stock spread/depth/OFI/activity through T
stock-minus-market / stock-minus-industry state
intraday chunk summary and session/time-of-day
```

禁止使用 T+1 收益定义牛熊、全样本/OOS 统计量、未来行业成分或未来相关图。

### 11.2 接口

```python
@dataclass(slots=True)
class MoEOutput:
    hidden: torch.Tensor
    router_probs: torch.Tensor
    topk_indices: torch.Tensor
    load_balance_loss: torch.Tensor
    router_z_loss: torch.Tensor


class TemporalRegimeMoE(nn.Module):
    def forward(
        self,
        hidden: torch.Tensor,          # [B, D]
        regime_features: torch.Tensor, # [B, R]
    ) -> MoEOutput:
        ...
```

```text
shared = SharedExpert(hidden)
router_probs = softmax(Router(regime_features))
top2 = sparse_top2(router_probs)
routed = Σ gate_e × Expert_e(hidden)
output = hidden + shared + routed
```

必须保留 residual 和 shared expert，避免行情边界硬切换。

### 11.3 Router 正则与遥测

```text
L = L_task
  + λ_balance × L_load_balance
  + λ_z × L_router_z
  + λ_entropy × L_entropy_floor
```

记录：

```text
expert_fraction / by_date / by_regime_bucket
router_entropy and top1_probability
expert_output_cosine_similarity
overflow_rate
expert_rankic_by_regime
```

单专家长期超过 80%、专家输出高度相同、router 只学习 board/symbol、收益仅来自单月，均视为失败。负载均衡只防塌缩，不强制每个日期绝对均匀。

当前 15 个 train dates 足以学习事件/流动性专家，不足以证明宏观牛熊专家。宏观 Regime-MoE 正式晋级要求至少 252 个连续交易日，并覆盖不同趋势、波动和流动性状态。

## 12. PR-7：O(N) 市场/行业上下文与 Ranker（独立模块已落地）

先比较现有 `use_attention=False/True` 的 3–5 seeds；没有稳定增量时，替换 O(N²) dense attention：

```text
market_mean = sum(stock_hidden) / count
industry_mean_loo = (industry_sum - own) / max(industry_count - 1, 1)

ranker_input = concat(
    own,
    market_mean,
    industry_mean_loo,
    own - market_mean,
    own - industry_mean_loo,
    regime_router_probs,
)
```

修改/新增：

```text
quant_fm/downstream/train_ranker.py
quant_fm/downstream/context.py
quant_fm/downstream/regime_ranker.py
tests/test_cross_section_permutation.py
tests/test_leave_one_out_context.py
tests/test_ranker_no_future.py
```

要求股票顺序置换后 score 同步置换；leave-one-out 不包含自身；行业使用 PIT effective-date join；Router/Ranker artifact 固化特征列和统计量。

## 13. PR-8：顶部 Sparse Backbone-MoE（待实施）

只有轻量 Regime-MoE 在 80–120M、多个 seeds 下稳定通过后才执行。

### 13.1 放置策略

- 底部 12–14 层保持 Dense，学习通用事件语法和微观结构；
- 顶部 4–6 层将 SwiGLU FFN 替换为 `SharedExpert + RoutedExperts`；
- attention 保持共享；
- 第一轮采用 Top-1 routed expert + shared expert；
- Router 输入为 token hidden 加 causal regime conditioning。

纯 token router 通常先按 `EXEC/ADD/CANCEL`、流动性或 session 分工。若目标是行情专家，应使用：

```text
router_input = token_hidden + project(causal_regime_context)
```

并分别报告按事件类型、日期和 regime 的专家使用率；控制事件类型后，专家仍应体现行情差异。

### 13.2 分布式策略

当前单一 root FSDP 不适合直接承载大规模 sparse experts。依次 benchmark：

1. 小专家本卡复制；
2. block-level FSDP；
3. expert-parallel process group；
4. all-to-all dispatch 与容量溢出；
5. expert shard checkpoint 合并与推理加载。

没有稳定 expert-parallel 实现前，不把 Backbone-MoE 作为生产依赖。

MoE 不能只报告总参数量，必须同时报告：

```text
total/active parameters per token
active FFN hidden per token
non-pad tokens/s
all-to-all and dispatch/combine time
expert GEMM utilization
```

若 active FLOPs 相同但吞吐下降超过 10%，必须有显著且稳定的 OOS 增量才能晋级。

## 14. Artifact 与兼容性

版本：

```text
schema_version: cn_l2_v2
vocab_version: 2.0
fm_artifact_version: 2.0
embedding_schema_version: 2.0
ranker_artifact_version: 2.0
moe_router_version: 1.0
```

checkpoint 至少保存：

```json
{
  "schema_version": "cn_l2_v2",
  "vocab_sha256": "...",
  "field_specs": [],
  "target_specs": [],
  "field_fusion": {},
  "attention_config": {},
  "pooling_config": {},
  "regime_feature_specs": [],
  "moe_config": {},
  "continuous_normalizers": {},
  "book_state_timing": "post_event",
  "train_state": {
    "micro_step": 0,
    "update_step": 0,
    "samples_seen": 0,
    "non_pad_tokens_seen": 0
  }
}
```

目录：

```text
quant_fm/runs/medium_300m/       # v1，只读
quant_fm/runs/v2_dense_25m/
quant_fm/runs/v2_dense_100m/
quant_fm/runs/v2_dense_230m/
quant_fm/runs/v2_regime_moe/
quant_fm/runs/v2_backbone_moe/
```

v1 loader 继续通过 `EventFieldFusion(method="legacy_sum")` 保持原始直接求和语义。V2 resume/load 会校验 schema、vocab hash、field order 和 target specs，不匹配必须报错，禁止静默兼容。

## 15. 消融实验矩阵

每次只改变一个因素，并固定有效 token、验证窗口和 Ranker seeds。

| 实验 | 唯一变化 | 重训 FM |
|---|---|---:|
| B0 | 当前 v1 基线 | 否 |
| B1 | 正确 pooling 语义 | 否 |
| B2 | 多尺度固定池化 | 否 |
| B3 | Ranker attention off/on | 否 |
| B4 | 轻量 temporal aggregator | 否或仅训聚合器 |
| D0 | 正确 step + fast path，原 Dense | 是 |
| D1 | FFN 4096 → 2816 | 是 |
| D2 | dropout 0.1 → 0.05/0.0 | 是 |
| D3 | Tokenizer/盘口 V2 | 是 |
| D4 | entropy-normalized + ordinal Loss | 是 |
| D5 | gated field fusion | 是 |
| M0 | temporal Regime-MoE，4 experts Top-2 | 否或仅训聚合器 |
| M1 | O(N) market/industry context | 否或仅训 Ranker |
| M2 | 顶部 4 层 Sparse-MoE | 是 |
| A0 | 3 local + 1 global attention 周期 | 是 |
| A1 | GQA `n_kv_heads=4/8` | 是 |

不允许第一轮同时合并 D1、D3、D4、M0。

## 16. 分阶段晋级规则

### Stage 0：不重训 FM

执行 B1–B4、M0、M1，复用现有 `best.pt`，运行 3–5 Ranker seeds。

晋级条件：

- 平均 ΔRankIC ≥ 0.005；
- paired daily IC bootstrap 80% 区间下界 > 0；
- 至少 3/5 seeds 为正；
- 推理额外开销 <5%，或能由 projection/DeepSets 抵消。

### Stage 1：25M 筛选

```text
d_model=256, n_layers=6
固定相同有效 token 或 5k update steps
每个候选 2 seeds
```

要求两个 seed 的验证 ΔRankIC 均为正；normalized NLL 不恶化超过 1%；无 NaN、token/router collapse。

### Stage 2：80–120M 复验

```text
d_model=512, n_layers=10
20k update steps
3 seeds
```

晋级条件：

- 至少 2/3 seeds 的 ΔRankIC 为正；
- 平均 ΔRankIC ≥0.005，理想 ≥0.01；
- paired IC Newey-West t ≥2；
- 分组单调性和扣成本 Top-bottom 改善；
- 换手增幅 ≤10%；
- Dense 改造吞吐不下降；
- MoE 吞吐下降 >10% 时需要更高的质量增量补偿。

### Stage 3：230M Dense V2

只组合前两阶段 winner：

```text
correct training/runtime
→ best tokenizer/book
→ best loss/fusion
→ best pooling/aggregator
→ optional lightweight Regime-MoE
```

### Stage 4：Backbone-MoE

只与最终 Dense V2 比较，固定 active FLOPs 或墙钟预算、有效 token 数、router features 和下游 seeds。

## 17. 测试清单

### 训练与性能

```text
tests/test_training_schedule.py
tests/test_gradient_accumulation.py
tests/test_shard_aware_sampler.py
tests/test_attention_fast_path.py
tests/test_rope_cache.py
tests/test_inference_checkpoint.py
```

### 数据与因果性

```text
tests/test_book_state_causality.py
tests/test_orderbook_consistency.py
tests/test_tokenizer_v2.py
tests/test_fit_bins_stratified.py
tests/test_target_applicability.py
```

### 表征与 MoE

```text
tests/test_multichunk_pooling.py
tests/test_hierarchical_pooling.py
tests/test_field_fusion.py
tests/test_multitask_loss_v2.py
tests/test_moe_router.py
tests/test_moe_causality.py
tests/test_moe_load_balance.py
tests/test_cross_section_permutation.py
tests/test_leave_one_out_context.py
```

每个 causal 测试都必须验证：修改未来记录，不改变当前及过去输出。

## 18. 推荐 PR 依赖图

```text
PR-0 benchmark contract
 ├── PR-1 train semantics + loader
 ├── PR-2 attention/inference fast path
 └── PR-3 pooling semantics

PR-4 orderbook/schema/tokenizer
 └── PR-5 fusion/loss + Dense V2

PR-3 + PR-5
 └── PR-6 lightweight Regime-MoE
      └── PR-7 O(N) context/ranker
           └── PR-8 sparse backbone MoE
```

PR-1、PR-2、PR-3 可并行开发，但合并后必须重建统一 baseline。PR-4 是盘口 V2 的硬前置；PR-8 不得绕过 PR-6 的实证闸门。

## 19. 完成定义

- training state 正确区分 micro/update/token 计数；
- 一个 epoch 的有效 token 数可复核；
- 全量订单簿一致性测试通过；
- 新盘口输入通过 prefix causality；
- Tokenizer V2 不混淆 NA 与 0；
- fast attention 与旧路径数值等价；
- 多 chunk `last/last-k` 语义正确；
- V1/V2 artifact 严格隔离；
- 推理 checkpoint 不携带 optimizer state；
- 至少一个 Dense V2 在 80–120M、3 seeds 下稳定提高 paired OOS IC；
- Regime-MoE 无专家塌缩，并在多个市场状态下产生可解释增量；
- 最终方案通过新的 untouched OOS；
- 生产交付仍严格为 `date, symbol, score`。

## 20. 暂不实施

- 一次性把 18 层全部换成 MoE；
- 将 token-level expert 使用率直接解释为牛熊专家；
- 在 15 个训练日上训练宏观 regime 专家并宣称泛化；
- 把所有股票异步逐笔事件拼成一条全市场序列；
- 全字段笛卡尔积词表；
- 使用未来收益、未来波动率或 OOS 统计量参与 Router；
- 用全样本相关矩阵构建股票图；
- 在盘口一致性测试失败时生成盘口训练特征；
- 只看 total CE、单 seed RankIC 或短样本年化决定架构；
- 为追求总参数量而忽略 active FLOPs、all-to-all 和实际 tokens/s。

最终推荐路线：

> **正确训练预算 + 无损快速 Dense 主干 + 真实因果盘口 + 多尺度日内聚合 + 轻量 Regime-MoE + O(N) 市场/行业上下文。**

顶部 Sparse-MoE 是可选扩展，不是 V2 的前置条件。
