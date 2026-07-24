# 阶段 6：OrderFlow FM 预训练

> 当前状态（2026-07）：v1 训练路径继续可用；v2 数据集、字段融合、连续双通道、
> entropy-normalized/ordinal/masked loss、固定分层验证清单、诊断指标和严格 checkpoint
> metadata 已实现；micro/update/token 计数、shard-aware sampler、RoPE cache、causal
> fast path、显式 `ffn_hidden` 和推理/续训 checkpoint 分流也已落地。Temporal Regime-MoE
> 与顶部 Backbone-MoE 有代码和单测，但仍是待运行的研究组件。尚未完成真实 v2 token
> 训练、MoE 稳定性/吞吐测试、模型选择或 untouched OOS 验收。

## 模型与字段融合

核心实现：

```text
quant_fm/pretrain/model.py
quant_fm/pretrain/field_fusion.py
quant_fm/pretrain/heads.py
```

主干仍是 decoder-only causal Transformer：RoPE、pre-norm RMSNorm、因果 attention、
SwiGLU 和每目标独立分类 head。Dense FFN 可由 `model.ffn_hidden` 直接指定；未指定时
回退为 `int(d_model * ffn_mult)`。`OrderFlowFMConfig.field_fusion` 支持：

| 方法 | 行为 | 兼容性 |
|------|------|--------|
| `legacy_sum` | 每字段 `d_model` embedding 直接求和，不做额外 norm | v1 checkpoint 默认路径 |
| `scaled_sum` | 按有效字段权重平方和缩放，再做可选 RMSNorm | v2 消融候选 |
| `gated_sum` | 每字段可学习 sigmoid gate、field dropout、缩放与 RMSNorm | v2 配置默认候选 |
| `concat_mlp` | 每字段先降到 `field_dim`，拼接后两层 MLP 投影至 `d_model` | v2 消融候选 |

v2 数值双通道由 `OrderFlowFM.scalar_projections` 实现：`val_*` 经无 bias 线性层
投影后，加到对应 `tok_*_bin` embedding；纯连续字段也可作为 standalone scalar。

配置示例：

```yaml
model:
  d_model: 384
  n_layers: 10
  n_heads: 8
  ffn_hidden: 1056
  field_fusion:
    method: gated_sum
    categorical_dim: 32  # 当前 parser 同时接受 field_dim；统一作为字段维度
    field_dropout: 0.10
    input_norm: true
```

## v1 与 v2 数据加载

- v1：`quant_fm/pretrain/dataset.py::EventWindowDataset`，字段顺序沿用原固定 tuple；
- v2：`quant_fm/pretrain/dataset_v2.py::EventWindowDatasetV2`，调用
  `field_layout_from_vocab()` 从冻结 `FieldSpec` 生成 token、scalar、输入、目标和 mask；
- `collate_windows_v2()` 对 token 使用 v2 `PAD_ID`，对 scalar/布尔 mask 使用 0，
  并生成统一 `attention_mask`；
- v2 target 的 `mask_tok_*` 由 `NA_ID` 自动生成。

`train.py::_load_vocab()` 根据 artifact 中的显式 `vocab_version` 选择 loader，不会把
v2 文件静默当作 v1。v2 训练必须配置 `loss.targets`，否则立即失败。

## 训练计数、采样与注意力快路径

`TrainState` 当前明确区分：

| 字段 | 代码语义 |
|------|----------|
| `micro_step` | 每个 rank 读取并反向一个 micro-batch 后增加 1 |
| `update_step` | 每个梯度累积边界且参数更新成功后增加 1；LR、日志、验证和 checkpoint 均使用它 |
| `samples_seen` | update 时把本次累积窗口数跨 rank 求和后累计 |
| `non_pad_tokens_seen` | update 时把 `attention_mask.sum()` 跨 rank 求和后累计 |

`max_update_steps` 和 `max_train_tokens` 是 OR 停止条件，检查点位在完整 optimizer update
之后，因此 token 上限最多越过一个全局 update。学习率按固定 update horizon 调度；仅设置
token 上限时必须显式提供 `optim.lr_schedule_steps`，避免 horizon 随运行进度漂移。FP16 下
GradScaler 若因 overflow 跳过参数更新，`update_step/samples_seen/non_pad_tokens_seen` 都不会
推进，只有已经完成反向的 `micro_step` 保留。

`ShardAwareDistributedSampler` 的当前算法是：按 `window_shard_index()` 聚簇窗口，epoch
级打乱 shard 与 shard 内窗口，然后把等数量的连续索引片段交给各 rank。它减少 parquet
随机打开，但不是按 non-pad token/长度做负载均衡，也没有短尾长度 bucket；训练路径使用
`drop_last=True`，不能宣称每个 epoch 不遗漏尾部窗口。DataLoader 还支持
`persistent_workers`、`prefetch_factor`、pinned memory 和 non-blocking GPU copy。

`OrderFlowFM` 的 RoPE 表按 `(device.type, device.index, dtype)` 惰性缓存，至少分配到
`max(length, max_seq_len)`，不进入 `state_dict`。当一个 batch 的 `attention_mask` 全为真时，
attention 直接走 `scaled_dot_product_attention(attn_mask=None, is_causal=True)`；含 padding
时回退到显式 causal + key mask。两条路径已有 FP32 等价回归，但真实 GPU 吞吐仍需基准。

## v2 多任务损失

`quant_fm/pretrain/heads.py` 提供：

```python
TargetSpec
target_specs_from_config(...)
next_event_loss_v2(...)
```

每个目标支持：

- raw next-event cross entropy；
- 由训练 vocab occupancy 计算的 unigram entropy 归一化；
- 静态 task weight；
- `ordinal_ce` 的期望-bin smooth-L1 辅助项；
- 当前/下一位置 attention mask、PAD/NA、事件适用 id 和显式 `mask_field`；
- 全 NA/全不适用 batch 的零损失安全反向传播。

当前 v2 默认目标为 evt type、side、price、volume 和 delta time；session 只作输入，
不再是默认预测目标。实际配置：

```yaml
loss:
  normalize_by_train_entropy: true
  targets:
    tok_evt_type: {type: ce, weight: 1.0}
    tok_side: {type: ce, weight: 1.0}
    tok_price_bin: {type: ordinal_ce, weight: 1.0, ordinal_weight: 0.5}
    tok_volume_bin: {type: ordinal_ce, weight: 0.5, ordinal_weight: 0.25}
    tok_delta_t_bin: {type: ordinal_ce, weight: 0.5, ordinal_weight: 0.25}
```

没有 `loss.targets` 的 v1 配置继续使用原六字段等权 CE，不改变旧实验语义。

## Temporal Regime-MoE 与 Backbone-MoE

`quant_fm/moe/` 当前实现了：

- `TopKRouter`：FP32 softmax、归一化 Top-K 权重、load-balance/z-loss 和归一化 entropy；
- `TemporalRegimeMoE`：股日/时间聚合后的 residual + shared expert + capacity-limited
  routed experts；`RegimeIntradayModel` 可组合 `IntradayAggregator`；
- `SparseMoEFeedForward`：把配置指定的顶部 Transformer FFN 替换为 shared + routed
  SwiGLU experts，辅助路由损失由训练总损失自动相加；
- `summarize_moe()`：生成 expert fraction、entropy、mean top-1 probability 和 overflow；
- `save_regime_moe_artifact()`：保存模型状态、Regime 配置、feature normalizer、数据截止日
  和 base model SHA-256。

集成边界必须保留：Temporal 模块尚未接入此预训练 loop、embedding CLI 或生产 score；
Backbone 只记录合计 `train/moe_aux`，不会自动调用 telemetry，也不记录逐层 overflow。Regime
artifact loader 目前只校验版本并返回 payload，不能凭 artifact 单独重建 aggregator 和模型
维度。`RegimeFeatureSpec.availability_lag`/normalizer `fit_end` 是审计元数据，PIT join 和仅训练
期拟合仍由调用方负责。

训练模式发生 capacity overflow 时，当前实现会跨整个 batch 选择 router 权重最高的
assignment；Backbone 将全部有效 `[B,L]` token 展平后竞争容量，因此训练前向存在 batch
依赖和 train/eval dispatch 差异。评估/推理模式不做 capacity 裁剪，padding token 不参与
Backbone 路由，并已有低 capacity 下的 batch-size independence 回归。代码边界已明确，但
真实训练中的 overflow、路由稳定性、吞吐和 OOS 仍需实验，不能由单测推导生产可用。

## 固定验证窗口与诊断

`quant_fm/pretrain/validation_sampler.py` 会生成可复用 JSON 清单，近似平衡：

```text
date × exchange × board × liquidity_bucket × activity_bucket
```

liquidity metadata 可选；缺失时进入显式 `unknown` bucket。计划保存 manifest
fingerprint、window 参数、shard hash 和选择的 dataset index，换 manifest 或窗口设置时
fail fast。

可先独立创建计划：

```bash
uv run python -m quant_fm.pretrain.validation_sampler \
  --manifest quant_fm/runs/v2_shared/data/manifest.json \
  --split val --context 2048 --stride 2048 --min-len 16 \
  --seed 42 --max-windows 800 \
  --out quant_fm/runs/v2_shared/validation_windows.json
```

若配置中的计划不存在，`train.py` 也会由 rank 0 创建；后续架构应复用同一文件。

评估入口：

```bash
uv run python -m quant_fm.pretrain.eval \
  --checkpoint quant_fm/runs/v2_25m/run/best.pt \
  --config quant_fm/pretrain/config_v2_25m.yaml \
  --split val --max-batches 100 \
  --validation-plan quant_fm/runs/v2_shared/validation_windows.json \
  --out quant_fm/runs/v2_25m/run/eval_val.json \
  --device cpu
```

输出包含 per-field CE/perplexity/top-1/balanced accuracy、训练 unigram entropy、
CE/entropy、copy-previous baseline、gradient norm 和 total normalized CE。它们是模型
诊断，不替代冻结 embedding 后的 RankIC/组合回测。

## v2 配置与运行

已新增：

| 配置 | 角色 | 状态 |
|------|------|------|
| `quant_fm/pretrain/config_v2_25m.yaml` | Stage-1 小模型筛选，5000 optimizer updates | 配置已提交，待真实训练 |
| `quant_fm/pretrain/config_v2_100m.yaml` | Stage-2 winner 复验，20000 optimizer updates，FSDP | 配置已提交，待真实训练 |
| `quant_fm/pretrain/config_v2_230m.yaml` | Dense V2 候选，`ffn_hidden=2816`，50000 updates | 配置已提交，待真实训练 |
| `quant_fm/pretrain/config_v2_backbone_moe.yaml` | 顶部 4 层 shared + Top-1 routed expert | 实验配置；训练/吞吐/路由/OOS 未验证 |

四份配置都读取同一组基础 artifact：

```text
quant_fm/runs/v2_shared/data/manifest.json
quant_fm/runs/v2_shared/data/vocab_v2.json
quant_fm/runs/v2_shared/validation_windows.json
```

因此必须先完成 v2 raw replay、canonical event、vocab、token 和 manifest。单卡启动：

```bash
uv run python -m quant_fm.pretrain.train \
  --config quant_fm/pretrain/config_v2_25m.yaml
```

8 卡 Stage-2：

```bash
uv run python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m quant_fm.pretrain.train \
  --config quant_fm/pretrain/config_v2_100m.yaml
```

续训继续使用 `--resume <checkpoint>` 或 `--resume auto`。配置中的 `book.state_timing`
和 `pooling.version` 会写入 checkpoint metadata；`pooling.method/outputs` 不会在 FM
训练中执行，需在 embedding 阶段显式选择。

## Artifact 和加载约束

所有 v2 checkpoint 都保存模型/配置兼容元数据：

```text
fm_artifact_version=2.0
schema_version / vocab_version / vocab_sha256
field_specs（含顺序）
field_fusion / scalar_fields / continuous_normalizers
target_specs
book_state_timing / context_horizon / pooling_version
model / train_state
```

是否保存 optimizer/scaler 取决于文件角色：

| 文件 | optimizer/scaler | 用途 |
|------|------------------|------|
| `step<update>.pt` | 有 | 定期完整续训点 |
| `final_resume.pt` | 有 | 正常训练结束后的完整续训点 |
| `best.pt` | 无 | 固定验证损失最优的评估/推理权重 |
| `final.pt` | 无 | 最终评估/推理权重 |

加载 v2 checkpoint 必须传原 vocab：

```python
model = load_checkpoint(
    checkpoint_path,
    device,
    vocab_path=vocab_v2_path,
)
```

loader 会校验 artifact version、schema、vocab SHA-256、FieldSpec 和 input/target 顺序；
v2 resume 还逐项核对 field sizes、模型宽深/head/FFN/dropout/RoPE、fusion/scalar、
normalizer、book/context/pooling 和 Backbone-MoE 配置及 target specs。
旧 checkpoint 缺少融合字段时默认 `legacy_sum`，仍按 v1 路径加载。

训练入口会直接拒绝用 `best.pt/final.pt` resume。当前 `--resume auto` 只在编号最大的
`step*.pt` 和 `final_resume.pt` 中选择，并优先定期点；继续已正常结束的训练应显式传
`--resume <out_dir>/final_resume.pt`。checkpoint 仍不保存 DataLoader 精确游标和全部 RNG
状态，因此即便完整续训点也不保证逐字节复现。

## 输出与监控

```text
<out_dir>/
├── config.snapshot.yaml
├── validation_windows.json  # 配置未指定共享路径时
├── tb/
├── best.pt
├── final_resume.pt
├── final.pt
└── step*.pt
```

`best.pt` 按固定验证窗口上的配置损失选择。TensorBoard 记录 train loss/lr、raw 与
normalized/ordinal per-field loss、`train/moe_aux` 以及 val loss。当前不会自动记录 expert
fraction、router entropy/top-1 或 overflow；需要调用 `summarize_moe()` 并补集成。训练
结果仍需固定 seed、固定 Ranker 和同日 paired IC 做下游裁判。

`tests/test_train_smoke.py` 还会在 CPU 上真实执行一个 optimizer update，并核对
`micro_step/update_step/non_pad_tokens_seen` 以及 `final_resume.pt` 与 `final.pt` 的状态差异；
它验证训练循环可执行，不代表真实数据收敛或吞吐达标。

## 当前限制

- v2/MoE 基础设施通过全仓回归：`243 passed, 2 skipped, 1 xfailed`；尚未运行真实
  25M/100M/230M 或 MoE 训练，因此不能声称困惑度、IC、吞吐或收益已改善；
- raw replay 仍需显式 capture 盘口 transition，旧数据流水线不会自动产出 v2；
- checkpoint 未保存 DataLoader 精确游标和全部 RNG 状态，续训不是字节级复现；
- shard-aware sampler 还没有 token/长度均衡；token budget 最多越过一个成功的全局
  update，且需要调用方选择合理的 `lr_schedule_steps`；
- `IntradayAggregator` 与 `LinearCrossAssetModel` 已有独立实现和测试，但尚未接入该
  预训练循环、默认 embedding CLI 或生产评分；
- Temporal Regime-MoE 未接主流程，Backbone-MoE 尚未验证训练期 overflow 行为、路由
  遥测、FSDP 性能或真实 checkpoint 恢复；
- 2026 已分析的 60 日只能作为 architecture validation，不能作为最终 untouched OOS。
