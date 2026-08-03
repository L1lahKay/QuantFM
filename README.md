# QuantFM V2

> 当前分支：[`V2`](https://github.com/zeusamc/quant-fm/tree/V2)
>
> 最近核验：2026-08-03；全仓回归为 `545 passed, 2 skipped, 1 xfailed`。

面向 A 股 Level-2 订单流的端到端基础模型研发框架。项目将沪深逐笔委托与成交数据重建为统一市场事件，使用 Decoder-only Transformer 进行多字段 next-event 自监督预训练，并由冻结截面 Ranker 生成可交付的日频 `score` 信号。V2 分支在保持 `cn_l2_v1` 兼容性的同时，引入独立的 `cn_l2_v2` schema、词表、存储编码和 checkpoint artifact；两代产物必须显式匹配，不允许静默混用。

## 核心能力

- **订单簿重建**：支持沪深逐笔回放、集合竞价/连续竞价、撤单索引一致性，以及每个事件严格对齐的 pre/post 因果盘口状态。
- **双版本 Tokenizer**：v1 保持既有固定分箱；v2 使用冻结 `FieldSpec`、独立 `NA/UNK`、全训练流统计、确定性分层 reservoir，并为数值字段同时输出 bin token 与标准化 scalar。
- **OrderFlow FM v2**：在 RoPE + RMSNorm + SwiGLU 骨干上新增可配置字段融合、字段 dropout、熵归一化多任务损失、ordinal 距离约束和显式 `ffn_hidden`。
- **训练与推理性能语义**：严格区分 micro-batch、optimizer update 和全 rank non-pad token；支持 shard-local 采样、RoPE 缓存、无 padding causal SDPA 快路径，以及 resume/inference checkpoint 分离。
- **可选 MoE 研究组件**：已实现股日级 `TemporalRegimeMoE` 和顶部层 `BackboneMoE`，但尚未完成真实训练、路由稳定性、吞吐或生产验收，不是默认 score 路径。
- **分层表征**：支持跨 chunk 正确的 `mean/last/last-k`、交易阶段多尺度池化、因果日内聚合，以及同步 interval 上 O(T×N) 的市场/行业 leave-one-out 上下文。
- **Artifact 合约与血缘**：校验 manifest、vocab、token 物理编码、事件顺序、checkpoint、embedding 和预训练验收文件，严格入口默认 fail closed。
- **训练监控与验收**：提供实时训练看板、训练完成后的 val/test 串行评估、基线非劣门禁和严格 OOS 下游入口。
- **分布式训练**：支持单卡与 8 卡 FSDP、bf16、梯度累积、余弦学习率、TensorBoard、best/final checkpoint。
- **稳定信号接口**：生产侧仅交付 `date, symbol, score` 与版本清单；研究回测与交付链路隔离。
- **MinIO 读写分离**：从 `zeus-cn-quote` 读取原始 L2，将 tokens、词表和 manifest 备份到 `model-cache`。

## Pipeline

```text
MinIO L2
  │  trade / order parquet
  ▼
PyLOB 订单簿重建
  │  clean events
  ▼
cn_l2_v2（数据脚本默认）/ cn_l2_v1（显式兼容）规范事件流
  │  stock-day parquet
  ▼
字段级 Tokenizer ──► vocab.json / vocab_v2.json
  │  token parquet
  ▼
Manifest 与时间切分
  │  train / val / test + sha256
  ▼
V2 Artifact 合约审计
  │  schema / vocab / storage / ordering
  ▼
OrderFlow FM v1 / v2 预训练
  │  best.pt / final.pt
  ▼
股日 Embedding（内部）
  │
  ▼
冻结截面 Ranker
  │
  ▼
scores.parquet
```

逐阶段输入、输出、代码入口与验证条件见 [Pipeline 文档](docs/pipeline/README.md)。

## 5 分钟验证

要求 Python 3.12+，依赖由 [uv](https://docs.astral.sh/uv/) 管理。

```bash
cd QuantFM
uv sync --extra fm --group dev
uv run python -m quant_fm.scripts.smoke --workdir /tmp/quantfm-smoke
```

终端打印 `SMOKE OK: score signal generated` 即表示以下链路全部可用：

```text
合成事件 → 分箱/分词 → manifest → 微型预训练
→ embedding → 冻结 ranker → 无标签 score
```

运行完整测试：

```bash
uv run python -m pytest -q
```

当前 V2 分支基线（2026-08-03）：`545 passed, 2 skipped, 1 xfailed`。

## V2 推荐工作流

V2 不接受 V1 tokens 或只复制权重的半套产物。训练前至少需要同一 artifact 根目录下的 `data/manifest.json`、`data/vocab_v2.json` 和 manifest 引用的 token shards，并先完成契约审计：

```bash
uv run python -m quant_fm.scripts.audit_v2_artifacts \
  --root quant_fm/runs/v2_backbone_moe_300d \
  --sample-shards 12 \
  --full-path-check \
  --out quant_fm/runs/v2_backbone_moe_300d/artifact_audit.json
```

审计返回非零时不得进入训练。它验证 schema/vocab 版本、字段顺序、物理 dtype、Q16/legacy 存储契约、事件顺序、路径完整性和抽样 token 内容；真实订单簿 coverage 与快照一致性仍需单独验收。

当前训练配置分为三组：

| 组别 | 配置 | 用途 |
|------|------|------|
| Dense 消融 | `config_v2_25m.yaml`、`config_v2_100m.yaml`、`config_v2_230m.yaml` | 从小规模消融到 Dense 候选复验 |
| Backbone-MoE 严格基准 | `config_v2_backbone_moe_300d.yaml` | 300 train / 60 validation / 100 test 日的一次完整数据遍历 |
| 2026Q1 adaptation | `config_v2_backbone_moe_adapt_2026q1_dev.yaml`、`config_v2_backbone_moe_adapt_2026q1_refit.yaml` | 开发期选参和最终 refit 分离 |

以 8 卡 Backbone-MoE 严格基准为例：

```bash
uv run torchrun --standalone --nproc_per_node=8 \
  -m quant_fm.pretrain.train \
  --config quant_fm/pretrain/config_v2_backbone_moe_300d.yaml
```

对于带 `optim.max_update_steps` 的训练配置，可只读查看一次状态，或去掉 `--once` 持续刷新：

```bash
uv run python -m quant_fm.scripts.watch_training_live \
  --config quant_fm/pretrain/config_v2_230m.yaml \
  --once
```

`config_v2_backbone_moe_300d.yaml` 使用 `max_data_epochs: 1`，当前看板需要 update 上限，因此该配置应使用训练日志、checkpoint 和集群 Job 状态监控。

训练结束后先生成评估计划；只有显式添加 `--execute`，且训练进程已经退出、`final.pt` 与 `final_resume.pt` 均存在时，才会串行执行 validation 和 test：

```bash
uv run python -m quant_fm.scripts.posttrain_evaluation \
  --config quant_fm/pretrain/config_v2_backbone_moe_300d.yaml

# 确认计划后再执行
uv run python -m quant_fm.scripts.posttrain_evaluation \
  --config quant_fm/pretrain/config_v2_backbone_moe_300d.yaml \
  --baseline-checkpoint /path/to/frozen-baseline.pt \
  --execute
```

严格 OOS/交付入口还要求 `pretrain_acceptance.json` 为显式 PASS，并重新核对 checkpoint、manifest/vocab 与 embedding 血缘。完整入口见 [`run_dense230m_strict_oos.sh`](quant_fm/scripts/run_dense230m_strict_oos.sh)。

## 真实数据运行

### 1. 配置 MinIO

```bash
cp quant_fm/scripts/minio_env.example.sh ~/.minio_fm_env.sh
chmod 600 ~/.minio_fm_env.sh
# 编辑密钥后：
source ~/.minio_fm_env.sh
make check-minio
```

凭据文件不得提交到仓库。配置说明见 [MinIO 读写指南](docs/data/minio_setup.md)。

### 2. Pilot 数据准备与训练

```bash
make pilot          # 真实 L2 → clean → events → tokens → manifest
make train-8gpu     # 8 卡 FSDP 预训练
```

### 3. 完整 MinIO 流水线

```bash
make minio-full-pipeline       # 5 日 × 每市场 30 股：数据、上传、训练
make minio-full-pipeline-full  # 60 日 × 全市场：数据、上传、训练
```

`full` 在本项目中指 60 个均匀交易日 × 沪深全市场，约为完整历史数据的 1/10。

## 项目结构

```text
QuantFM/
├── quant_fm/                 # FM 主工程
│   ├── schema/               # cn_l2_v1 / cn_l2_v2 统一事件协议
│   ├── tokenizer/            # v1/v2 因果变换、分箱、字段级词表
│   ├── manifest/             # 分片哈希、时间切分与数据验证
│   ├── pretrain/             # Transformer、Dataset、训练与评估
│   ├── embedding/            # checkpoint → stock-day embedding
│   ├── moe/                  # temporal Regime-MoE 与可选顶部稀疏 FFN
│   ├── monitoring/           # 训练完成状态与非劣验收契约
│   ├── benchmark/            # non-pad token、延迟和显存指标
│   ├── experiments/          # 版本化消融实验登记
│   ├── cross_asset/          # PIT 对齐、同步上下文与线性复杂度跨股票模型
│   ├── signal/               # 冻结 Ranker artifact 与生产 score 接口
│   ├── downstream/           # Ranker 训练及 research-only 评估/回测
│   ├── scripts/              # Pilot、Medium、MinIO 与训练编排
│   └── data/                 # 版本化日期/标的清单
├── order_book/
│   └── pylob/                # 撮合、回放与 MinIO 清洗子项目
├── docs/
│   ├── pipeline/             # Pipeline 逐阶段文档
│   ├── architecture/         # 模型、Tokenizer 与 Loss 设计
│   ├── data/                 # MinIO 与数据加工流程
│   ├── evaluation/           # OOS、信号与回测契约
│   ├── operations/           # 集群、存储与训练运维
│   ├── project/              # 进展、分支说明与计划
│   ├── research/             # 调研和历史方案
│   └── README.md             # 全部文档索引
├── examples/                 # 数据清洗与 notebook 示例
├── tests/                    # 撮合、数据管线与 FM 回归测试
├── k8s/                      # GPU Job、PVC 与调度评估清单
├── Makefile                  # 常用操作入口
├── pyproject.toml            # QuantFM + uv workspace
└── uv.lock                   # 锁定依赖
```

`pylob` 是独立 workspace 子项目，但 Python 公共接口保持不变：

```python
from pylob import OrderBookSH, OrderBookSZ
from pylob.pipeline.workflow import build_clean_dataset
```

## 模型与训练

v1 Pilot 默认模型约 6.36M 参数：

- 多字段 embedding 求和；
- causal self-attention + RoPE；
- pre-norm RMSNorm；
- SwiGLU 前馈层；
- 6 个 next-event 预测头：事件类型、方向、时段、价格、量、时间间隔。

v2 保留相同 Transformer 骨干，但字段由 `vocab_v2.json` 内冻结的 `FieldSpec` 决定，并支持 `legacy_sum/scaled_sum/gated_sum/concat_mlp` 四种字段融合、连续双通道和显式多任务 Loss。当前配置为：

- [`config_v2_25m.yaml`](quant_fm/pretrain/config_v2_25m.yaml)：约 25M 的 Stage-1 消融配置，单进程优先；
- [`config_v2_100m.yaml`](quant_fm/pretrain/config_v2_100m.yaml)：约 100M 的 Stage-2 复验配置，面向 8 卡 FSDP；
- [`config_v2_230m.yaml`](quant_fm/pretrain/config_v2_230m.yaml)：显式 `ffn_hidden=2816` 的 Dense V2 候选；
- [`config_v2_backbone_moe.yaml`](quant_fm/pretrain/config_v2_backbone_moe.yaml)：仅顶部 4 层使用 shared + Top-1 routed expert 的实验配置，不是生产默认值。
- [`config_v2_backbone_moe_300d.yaml`](quant_fm/pretrain/config_v2_backbone_moe_300d.yaml)：严格日期计划下的 Backbone-MoE 基准配置；
- [`config_v2_backbone_moe_adapt_2026q1_dev.yaml`](quant_fm/pretrain/config_v2_backbone_moe_adapt_2026q1_dev.yaml) / [`config_v2_backbone_moe_adapt_2026q1_refit.yaml`](quant_fm/pretrain/config_v2_backbone_moe_adapt_2026q1_refit.yaml)：将 adaptation 选参与最终 refit 分离。

在 `quant_fm/runs/v2_shared/data/` 下准备好 v2 manifest 与 `vocab_v2.json` 后，可执行：

```bash
# 25M Stage-1
uv run python -m quant_fm.pretrain.train \
  --config quant_fm/pretrain/config_v2_25m.yaml

# 100M Stage-2（示例为单机 8 卡）
uv run torchrun --standalone --nproc_per_node=8 \
  -m quant_fm.pretrain.train \
  --config quant_fm/pretrain/config_v2_100m.yaml
```

这些 v2 配置复用 `quant_fm/runs/v2_shared/validation_windows.json`。首次训练会按日期、交易所、板块、流动性和活跃度分层生成固定验证窗口；之后配置、manifest 或窗口参数不匹配会直接报错。

v2 checkpoint 标记 `fm_artifact_version=2.0`，加载和续训时严格核对 `schema_version`、`vocab_version`、vocab SHA-256、`FieldSpec`、字段顺序和 loss target 声明。v1 checkpoint 则继续走原有 `legacy_sum` 与 v1 特殊 token 语义。

训练状态中的 `micro_step` 每个本 rank micro-batch 增加一次；`update_step` 只在参数更新成功后增加，并驱动学习率、日志、验证和定期存盘；`samples_seen/non_pad_tokens_seen` 在成功 update 后跨 rank 汇总，FP16 overflow 跳步不会计入。`max_update_steps` 与 `max_train_tokens` 任一达到即停止，token 预算最多越过一个全局 update；仅按 token 停止时必须显式给出 `optim.lr_schedule_steps`。当前 `ShardAwareDistributedSampler` 以 shard 聚簇并在 shard 内打乱窗口，再将等数量窗口切给各 rank；它优化文件局部性，但不是按 token 长度均衡的 sampler。

所有模型配置位于 `quant_fm/pretrain/config_*.yaml`。训练产物默认写入：

```text
quant_fm/runs/<experiment>/run/
├── config.snapshot.yaml
├── tb/
├── best.pt
├── final_resume.pt
├── final.pt
└── step*.pt
```

`step*.pt` 和 `final_resume.pt` 含 optimizer/scaler，可用于续训；`best.pt` 和 `final.pt` 不含 optimizer，面向评估/推理，训练入口会拒绝用它们 resume。`--resume auto` 只在 `step*.pt` 与 `final_resume.pt` 中选择，并优先编号最大的定期点；继续已正常结束的训练时应显式传 `--resume <.../final_resume.pt>`。

`quant_fm/runs/` 已被 `.gitignore` 排除，不应上传 checkpoint、tokens 或 TensorBoard 日志。

当前 V2 已具备数据合约审计、25M/100M/230M Dense 配置、Backbone-MoE 严格日期配置、adaptation dev/refit 配置、训练监控、预训练非劣门禁和严格 OOS 血缘检查。`make pilot`、`make medium` 与 `make minio-*-pipeline*` 现默认执行真实逐事件盘口回放，生成 `cn_l2_v2`、`vocab_v2.json`、窄 token/Q16 scalar、manifest 和 `artifact_audit.json`；旧实验只能通过底层入口显式传 `--data-version v1`。仓库中的配置、代码和回归测试仍不等同于真实全市场训练完成或收益验收；正式 V2 checkpoint、最终 V2 score、路由稳定性、吞吐和 OOS 结论必须以实际运行产物及 PASS 验收文件为准。

MoE 仍有待实证的训练边界：训练模式发生 capacity overflow 时，专家裁剪会在整个 batch（Backbone 中为全部有效 token）上竞争容量，形成 batch 依赖；评估/推理模式不执行容量裁剪，已覆盖 batch-size independence。该 train/eval 差异、路由健康度和真实吞吐尚未验证，因此仍只能作为研究候选。

## 文档入口

- [项目与阅读指南](docs/QuantFM.md)
- [Pipeline 逐阶段文档](docs/pipeline/README.md)
- [复现与验证指南](docs/REPRODUCIBILITY.md)
- [MinIO 读写配置](docs/data/minio_setup.md)
- [文档总索引](docs/README.md)
- [OrderFlow FM 包说明](quant_fm/README.md)
- [PyLOB 子项目说明](order_book/README.md)

## 验收边界

`make smoke` 证明系统能在推理日期没有 `fwd_ret`/`label` 时生成合法 score。QuantFM 的生产边界止于信号；股票池、持仓、成本、净值和风险指标由下游回测系统负责。
