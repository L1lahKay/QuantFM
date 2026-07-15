# 阶段 6：OrderFlow FM 预训练

## 目标

对 token 化订单流执行多字段 next-event 自监督学习，让模型学习事件类型、方向、时段、价格、量和时间节奏之间的序列关系。

## 模型

核心实现：`quant_fm/pretrain/model.py`。

- 各字段独立 embedding，逐位置求和；
- Decoder-only causal Transformer；
- RoPE 位置编码；
- pre-norm RMSNorm；
- SwiGLU 前馈层；
- 每个目标字段独立线性预测头。

默认预测字段：

```text
evt_type / side / session / price_bin / volume_bin / delta_t_bin
```

损失为六个下一事件交叉熵之和。位置 `t` 的 hidden state 预测位置 `t+1`，padding 边界不参与计算。

## 数据加载

`quant_fm/pretrain/dataset.py`：

- 从 manifest 选择 train/val shards；
- 每个股日按 `context`、`stride` 切窗口；
- DataLoader 动态 padding；
- 多卡使用 `DistributedSampler`。

## 训练能力

`quant_fm/pretrain/train.py` 提供：

- AdamW；
- warmup + cosine learning rate；
- bf16/fp16 autocast；
- 梯度累积与裁剪；
- 单卡和 FSDP；
- 定期验证、step checkpoint、`best.pt`、`final.pt`；
- TensorBoard；
- seed 与配置快照。

## 配置

| 配置 | 用途 |
|------|------|
| `config_pilot.yaml` | Pilot 单卡 |
| `config_pilot_8gpu.yaml` | Pilot 8 卡 FSDP |
| `config_medium_try_8gpu.yaml` | 5 日小规模真实试跑 |
| `config_medium_smoke_8gpu.yaml` | 60 日 × 少量股票 |
| `config_medium_8gpu.yaml` | 60 日 × 全市场 |

关键字段：

```yaml
data:
  manifest: ...
  vocab: ...
  context: 2048
model:
  d_model: 256
  n_layers: 6
  n_heads: 8
optim:
  lr: 5.0e-4
  max_steps: 20000
  precision: bf16
runtime:
  out_dir: ...
  eval_every: 500
  ckpt_every: 1000
  fsdp: true
```

## 运行

```bash
make train-pilot   # 单卡
make train-8gpu    # Pilot 8 卡
make train-medium-8gpu
```

## 输出

```text
<workdir>/run/
├── config.snapshot.yaml
├── tb/
├── best.pt
├── final.pt
└── step*.pt
```

`best.pt` 由最低验证损失选出；小数据实验中应优先用于下游，不应默认使用可能已过拟合的 `final.pt`。

## 监控

TensorBoard 指标：

- `train/loss`
- `train/lr`
- `train/ce_<field>`
- `val/loss`

健康训练通常表现为 train/val 同步下降；若 train 持续下降而 val 连续上升，则已过拟合。

## 当前限制

checkpoint 重点保存模型权重和模型配置。若要可靠支持中断后精确续训，还需同时持久化 optimizer、scaler、当前 step、随机数状态和 sampler epoch。
