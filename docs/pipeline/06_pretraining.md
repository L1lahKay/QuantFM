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
- seed 与配置快照；
- **断点续训**：checkpoint 同时保存模型权重、AdamW 优化器状态（含 FSDP full/scatter optim state）、GradScaler、以及 `train_state`（step / best_val / best_step）。

## 配置

| 配置 | 用途 |
|------|------|
| `config_pilot.yaml` | Pilot 单卡 |
| `config_pilot_8gpu.yaml` | Pilot 8 卡 FSDP |
| `config_medium_try_8gpu.yaml` | 5 日小规模真实试跑 |
| `config_medium_smoke_8gpu.yaml` | 60 日 × 少量股票 |
| `config_medium_8gpu.yaml` | 60 日 × 全市场 |
| `config_medium_try_300m_8gpu.yaml` | ~302M 小数据试跑 |
| `config_medium_300m_8gpu.yaml` | ~302M 正式（22 日全市场，Chinchilla 预算） |

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

# 300M：数据就绪后自动训练；也可单独续训
uv run python -m torch.distributed.run --standalone --nproc_per_node=8 \
  --master_port=29521 -m quant_fm.pretrain.train \
  --config quant_fm/pretrain/config_medium_300m_8gpu.yaml \
  --resume auto
```

`--resume` 取值：

| 值 | 行为 |
|----|------|
| 省略 | 从头训练 |
| 具体 `.pt` 路径 | 从该 checkpoint 恢复权重与优化器状态 |
| `auto` | 自动选 `out_dir` 下最新 `step*.pt`，否则 `final.pt` / `best.pt` |

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

- checkpoint 已覆盖模型 / optimizer / scaler / step，可支持训练中断后续训；
- 尚未持久化 DataLoader sampler 精确位置与全部 RNG 状态，因此续训后的 batch 顺序与「从未中断」不完全字节级一致，但不影响工程级恢复；
- 300M 配置约 `d_model=1024 × n_layers=18`，需匹配约 6B 训练事件（约 22 个全市场交易日）。
