# QuantFM MOE 架构完整结构

> 文档基线：2026-07-31。MOE 在本文中专指 Mixture of Experts。当前仓库同时包含轻量 Temporal Regime-MoE 和 Transformer 内部 Backbone-MoE，两者的路由粒度、训练成本和成熟度不同。

## 1. 总体结论

推荐采用两级、分阶段架构：

```text
事件级 Token
    ↓
OrderFlowFM Dense / Backbone-MoE
    ↓
多尺度股日 embedding
    ↓
Temporal Regime-MoE（轻量适配，可选）
    ↓
市场/行业上下文 + 横截面 Ranker
    ↓
date, symbol, score
```

- **Temporal Regime-MoE** 位于股日/日内聚合层之后，按同一时点可用的 regime 特征选择专家；成本低、可复用 Dense checkpoint，应该先验证。
- **Backbone-MoE** 把 Transformer 顶部若干层的 Dense FFN 换为稀疏专家；必须重新训练 FM，成本高，属于后续扩展。
- 当前 Temporal Regime-MoE 只有模块、normalizer、telemetry 和部分 artifact，尚无标准训练/评分链路，也没有正式 checkpoint。
- 当前 Backbone-MoE V1 已完成 50,000 updates，并存在 `best.pt`、`final.pt` 和 `final_resume.pt`；但逐层路由 telemetry、吞吐和严格下游 OOS 仍未补齐，不能列为已验收胜出的模型。
- 300/60/100 日严格 V2 MoE 只有配置和数据计划，尚未启动；存储安全控制面仍标记为 `IMPLEMENTATION_REQUIRED`。

## 2. 两类 MOE 的职责边界

| 项目 | Temporal Regime-MoE | Backbone-MoE |
|---|---|---|
| 位置 | 股日/日内聚合后 | Transformer block 内部 |
| 路由单位 | 股日或时间摘要 | 每个有效事件 token |
| Router 输入 | 显式 regime 特征 | 当前层 hidden state |
| 默认专家 | 4 routed + 1 shared | 4 routed + 1 shared |
| 默认 Top-K | Top-2 | Top-1 |
| 是否复用 Dense FM | 是 | 否，需重训 |
| 主要风险 | regime 泄漏、特征 join、样本少 | collapse、overflow、吞吐、FSDP 成本 |
| 当前状态 | 代码完成，未正式训练 | V1 训练预算已完成，路由/吞吐/OOS 未验收 |

不要把 token-level 专家使用率直接解释成“牛市专家/熊市专家”。只有 Temporal Router 明确读取可审计的 regime 特征；Backbone Router 学到的是隐状态分工，语义需要额外分析。

## 3. 公共 Top-K Router

两类 MOE 复用 `TopKRouter`。

### 3.1 路由计算

对输入 \(x\)：

\[
z=\frac{f_{router}(x)}{T},\qquad p=\operatorname{softmax}(z)
\]

选择概率最高的 \(K\) 个专家，并在 Top-K 内重新归一化：

\[
\mathcal I=\operatorname{TopK}(p),\qquad
w_i=\frac{p_i}{\sum_{j\in\mathcal I}p_j},\;i\in\mathcal I
\]

Router logits 和 softmax 强制使用 FP32，降低 bf16/fp16 下的路由抖动。

### 3.2 Router 结构

- Backbone Router：单层无 bias 线性投影 `d_model → n_experts`。
- Temporal Router：`LayerNorm → Linear → SiLU → Linear`，输入是 regime feature。
- `temperature` 控制概率尖锐程度；当前 Backbone 配置固定 1.0，Temporal 配置可调。

### 3.3 辅助损失

Router 输出两项正则：

\[
L_{balance}=E\sum_{e=1}^{E}I_e\,F_e
\]

其中 \(I_e\) 是平均 soft probability，\(F_e\) 是 detach 后的硬路由负载。

\[
L_z=\mathbb E\left[\log\sum_e\exp(z_e)\right]^2
\]

聚合为：

\[
L_{router}=\lambda_{bal}L_{balance}+\lambda_zL_z
\]

当前默认：

```yaml
load_balance_weight: 0.01
router_z_loss_weight: 0.001
```

还计算归一化路由熵，但训练 loop 当前只自动记录 `train/moe_aux`，未自动持久化逐层负载、熵和 overflow。

## 4. Backbone-MoE

### 4.1 Dense 基线 Block

Dense block 为：

```text
x
 ├─ RMSNorm → Causal Self-Attention → residual add
 └─ RMSNorm → Dense SwiGLU FFN       → residual add
```

Backbone-MoE 只替换指定层的 FFN，Attention、RoPE、RMSNorm 和残差路径不变。

### 4.2 稀疏 FFN

每个 MoE 层包含：

```text
hidden [B,L,D]
    ↓ attention_mask 过滤 PAD
active hidden [N,D]
    ├── shared SwiGLU expert（所有有效 token）
    └── Top-K Router
          ├── routed expert 0
          ├── routed expert 1
          ├── routed expert 2
          └── routed expert 3
    ↓ weighted index_add
shared(active) + routed(active)
    ↓ 恢复 [B,L,D]
Transformer residual add
```

单个 SwiGLU expert：

\[
E(x)=W_o\left(\operatorname{SiLU}(W_gx)\odot W_vx\right)
\]

`attention_mask` 之外的 padding 不进入 Router、专家容量和负载统计，输出保持 0。

### 4.3 容量与 overflow

对 \(N\) 个有效 token、Top-K 为 \(K\)、专家数为 \(E\)：

\[
C=\left\lceil c\frac{NK}{E}\right\rceil
\]

其中 `capacity_factor=c`，当前为 1.25。

- 训练模式下，单专家 assignment 超过容量时，只保留 router weight 最大的 `C` 个。
- 被裁剪 assignment 不进入 routed 输出，由 shared expert 提供保底路径。
- 评估/推理模式不裁剪，因此 `overflow_rate=0`，避免结果随 batch size 改变。

这意味着 train/eval dispatch 语义存在差异，训练必须监控 overflow，不能只看 validation loss。

### 4.4 当前冻结配置

```yaml
model:
  d_model: 1024
  n_layers: 18
  n_heads: 16
  ffn_hidden: 2816
  backbone_moe:
    enabled: true
    layer_indices: [14, 15, 16, 17]
    n_routed_experts: 4
    top_k: 1
    shared_expert_hidden: 1024
    routed_expert_hidden: 1792
    capacity_factor: 1.25
    load_balance_weight: 0.01
    router_z_loss_weight: 0.001
```

只有顶部 4 层使用 MOE，底部 14 层保留 Dense FFN。总参数量实测 297.59M；每个 token 只激活 shared expert 和一个 routed expert，但当前实现没有 expert-parallel all-to-all，不能把参数稀疏直接等同于更高吞吐。

### 4.5 与 OrderFlowFM 的集成

`OrderFlowFM` 在构造 block 时检查 `layer_indices`：

- 非 MOE 层构造 Dense `FeedForward`；
- MOE 层构造 `SparseMoEFeedForward`；
- forward 后缓存各层 router 输出、aux loss 和 overflow；
- `moe_auxiliary_loss()` 将各 MOE 层辅助损失求和；
- 训练总目标增加该辅助项。

```text
L_train = L_next_event + Σ_layer L_router(layer)
```

## 5. Temporal Regime-MoE

### 5.1 输入位置

`RegimeIntradayModel` 先调用 `IntradayAggregator` 得到多尺度摘要，再对 `full_day_summary` 做 regime 适配：

```text
chunk embeddings + timestamps + mask
    ↓ IntradayAggregator
full_day_summary
    ↓ TemporalRegimeMoE(hidden, regime_features)
regime_summary
```

### 5.2 专家结构与输出

每个 expert 为：

```text
LayerNorm → Linear(hidden→expert_hidden) → SiLU
→ Dropout → Linear(expert_hidden→hidden)
```

输出含显式残差和 shared expert：

\[
h'=h+E_{shared}(h)+\sum_{e\in TopK}w_eE_e(h)
\]

默认配置：4 experts、Top-2、expert hidden 256、router hidden 128、capacity factor 1.25。

### 5.3 Regime 特征契约

`RegimeFeatureSpec` 可记录字段名和 `availability_lag`；`RegimeFeatureNormalizer` 只用训练期拟合 mean/scale，并保存 `fit_end`。

推荐 Router 特征只能来自信号时点可用数据，例如：

- 当日截至当前时点的实现波动率；
- 市场宽度、成交额、价差和流动性分位；
- 指数/行业当日截至当前时点收益；
- 滞后一期的波动、成交额和横截面离散度。

禁止使用未来收益、收盘后才完整可知的当日统计、全样本标准化或 OOS 拟合参数。

重要限制：当前 API 只保存 `availability_lag` 和 `fit_end`，不会替调用方自动做 lag、日期过滤或 PIT as-of join；数据管道必须自行保证时点正确。

### 5.4 Artifact

`regime_moe_v1` 当前保存：

```text
model_state
moe_config
feature_normalizer
hidden_dim
regime_feature_dim
data_cutoff
base_model_sha256
```

`load_regime_moe_artifact()` 可由这些字段严格重建 Temporal 模块和 normalizer；artifact
仍未包含完整 aggregator、Ranker 和生产评分编排，因此不是一键生产 loader。

## 6. 训练目标与监控

### 6.1 必记指标

每层、每个 evaluation window 至少记录：

| 类别 | 指标 |
|---|---|
| 质量 | total loss、per-field CE、normalized NLL、ordinal loss |
| Router | expert fraction、normalized entropy、mean top-1 probability |
| 容量 | overflow rate、accepted assignments、capacity |
| 专家差异 | 输出余弦相似度、按专家梯度范数 |
| 性能 | non-pad tokens/s、dispatch/combine 时间、峰值显存 |
| 稳定性 | 按月份、波动桶、流动性桶的专家占比 |

当前 `run_judge` 研究训练 loop 已按 epoch 记录 expert fraction、entropy、top-1
probability、overflow 和 Router auxiliary loss；by-date/by-regime、专家输出相似度与梯度范数
仍未自动落盘。

### 6.2 告警建议

- 单一专家长期占比 >80%：疑似 collapse。
- 任一专家长期占比 <5%：疑似 dead expert。
- 归一化 entropy 长期接近 0：Router 过硬或塌缩。
- overflow 持续 >5%：提高容量、改善平衡或调整 batch/token 分布。
- Router auxiliary 长期常数：检查负载、梯度和 telemetry，不能仅以“不发散”判定健康。

阈值应先在 1k warmup 上校准，再冻结到正式实验；不能根据 test 结果回调阈值。

## 7. 当前 Backbone-MoE V1 训练状态

### 7.1 配置与数据

| 项 | 值 |
|---|---|
| schema | `cn_l2_v1` |
| 数据 | `cont60`，train 28 日、val 1 日、test 32 日 |
| 参数量 | 297.59M |
| 训练预算 | 50,000 optimizer updates |
| batch | 8 GPU × micro batch 2 × accum 8 = 128 sequences/update |
| context | 2048 |
| 盘口 | 无，`book.state_timing=none` |
| pooling | `flat_v1 / mean` |

### 7.2 现场状态

截至 2026-07-31 直接检查本地 final checkpoint：

- 训练已达到 update 50,000，共 12,875,096,787 non-pad tokens；
- 当前最佳为 update 50,000，`val_loss=5.81594505`；
- `best.pt`、`final.pt` 和 `final_resume.pt` 均存在；
- 因此状态是 **训练预算已完成，但路由/吞吐/OOS 尚未验收**。

同数据、同 token 预算、同 validation plan 的 Dense230M V1 在 update 50,000 达到
`val_loss=5.81305394`。MoE 高 `0.00289112`，约 0.05%；当前 next-event 指标没有显示
优势，但差值很小，而且 Router telemetry、吞吐和严格下游 OOS 缺失，不能据此做最终
投资效果结论。

## 8. 严格 V2 300 日方案

配置目标为：

```text
300 train days
60 validation days
100 frozen test days
FULL cn_l2_v2
顶部 4 层 Backbone-MoE
```

当前已有日期计划、配置和 storage-safe runbook，但不满足直接启动条件：

- 本地 V2 manifest/vocab/tokens 未完成；
- 现有 dataset 仍以本地 parquet 为主，未完成远端有界 pack cache；
- lazy window sampler、精确 sampler cursor resume 未完成；
- 磁盘 reservation、原子 checkpoint 轮转和硬停 guard 未完成；
- 训练控制入口 `moe_trainctl` 尚是目标接口。

因此 300 日配置属于 **计划就绪、工程前置未完成、训练未启动**。

## 9. 推荐实验与晋级顺序

### Stage A：Temporal Regime-MoE

1. 冻结同一个 Dense FM checkpoint 和 embedding contract。
2. 使用 3–5 个 Ranker seeds 比较无 MOE/Temporal MOE。
3. 固定 PIT regime features 和训练期 normalizer。
4. 通过 paired daily IC、Top-K 净收益和 router health 决定是否晋级。

建议门槛：平均 ΔRankIC ≥0.005、至少 3/5 seeds 为正、paired bootstrap 80% 下界 >0、推理额外开销 <5%。

### Stage B：25M/100M Backbone-MoE

- 固定 V2 Token、Loss、validation windows、有效 token 和 seeds；
- 与相同 active FLOPs 或相同墙钟预算的 Dense 对照；
- 先 25M 筛选，再 100M 三 seed 复验。

### Stage C：230M/300 日

只有 Dense V2 与小规模 MOE 都通过后，才训练大规模 Backbone-MoE。Validation 选择唯一候选，Test 只在冻结后执行一次。

## 10. 验收定义

一个 MOE 模型只有同时满足以下条件才能标记为“完成并可比较”：

1. 训练预算完成，存在 inference `best/final` 和 resumable `final_resume`。
2. 每层四个专家都有稳定负载，无 dead expert/collapse。
3. overflow、entropy、expert fraction 按日期与 regime 落盘。
4. 与 Dense 使用相同数据、Token、Loss、validation windows 和预算口径。
5. 预训练主要预测头无明显退化。
6. 多 seed 下游 paired OOS 稳定改善，而不是单日或单 seed 偶然结果。
7. 最终通过 untouched OOS；test 未参与训练、早停和参数选择。

## 11. 代码入口

| 功能 | 文件 |
|---|---|
| 配置对象 | `quant_fm/moe/config.py` |
| 公共 Router | `quant_fm/moe/router.py` |
| Backbone 稀疏 FFN | `quant_fm/moe/backbone.py` |
| Temporal Regime-MoE | `quant_fm/moe/temporal_moe.py` |
| Regime normalizer | `quant_fm/moe/regime_features.py` |
| Router telemetry | `quant_fm/moe/telemetry.py` |
| Regime artifact | `quant_fm/moe/artifact.py` |
| OrderFlowFM 集成 | `quant_fm/pretrain/model.py` |
| 训练目标集成 | `quant_fm/pretrain/train.py` |
| V1 MOE 实验配置 | `quant_fm/runs/backbone_moe_v1/config.yaml` |
| V2 300 日配置 | `quant_fm/pretrain/config_v2_backbone_moe_300d.yaml` |
| 训练计划 | `quant_fm/experiments/moe_300d_training_plan.yaml` |
