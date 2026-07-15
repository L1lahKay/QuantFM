# QuantFM

面向 A 股 Level-2 订单流的端到端基础模型研发框架。项目将沪深逐笔委托与成交数据重建为统一市场事件，使用 Decoder-only Transformer 进行多字段 next-event 自监督预训练，并将冻结的股日表征交给下游截面排序与回测模块验收。

## 核心能力

- **订单簿重建**：支持沪深逐笔回放、集合竞价/连续竞价与结果一致性校验。
- **因果 Tokenizer**：对事件类型、方向、时段等类别字段独立编码；对相对价格、成交量和事件间隔进行训练窗分位数分箱。
- **OrderFlow FM**：RoPE + RMSNorm + SwiGLU 的 Decoder-only Transformer，以多头交叉熵预测下一市场事件。
- **分布式训练**：支持单卡与 8 卡 FSDP、bf16、梯度累积、余弦学习率、TensorBoard、best/final checkpoint。
- **可复现验收**：manifest 记录分片哈希与时间切分；提供无泄漏检查、RankIC/ICIR、CPCV、DSR 和 Top-K 回测。
- **MinIO 读写分离**：从 `zeus-cn-quote` 读取原始 L2，将 tokens、词表和 manifest 备份到 `model-cache`。

## Pipeline

```text
MinIO L2
  │  trade / order parquet
  ▼
PyLOB 订单簿重建
  │  clean events
  ▼
cn_l2_v1 规范事件流
  │  stock-day parquet
  ▼
字段级 Tokenizer ──► vocab.json
  │  token parquet
  ▼
Manifest 与时间切分
  │  train / val / test + sha256
  ▼
OrderFlow FM 预训练
  │  best.pt / final.pt
  ▼
股日 Embedding
  │  embeddings.parquet
  ▼
截面 Ranker 与回测验收
```

逐阶段输入、输出、代码入口与验证条件见 [Pipeline 文档](docs/pipeline/README.md)。

## 5 分钟验证

要求 Python 3.12+，依赖由 [uv](https://docs.astral.sh/uv/) 管理。

```bash
cd QuantFM
uv sync --extra fm --group dev
uv run python -m quant_fm.scripts.smoke --workdir /tmp/quantfm-smoke
```

终端打印 `SMOKE OK: all stages passed` 即表示以下链路全部可用：

```text
合成事件 → 分箱/分词 → manifest → 微型预训练
→ embedding → ranker → Top-K 回测
```

运行完整测试：

```bash
uv run python -m pytest -q
```

当前本地基线：`119 passed, 2 skipped, 1 xfailed`。

## 真实数据运行

### 1. 配置 MinIO

```bash
cp quant_fm/scripts/minio_env.example.sh ~/.minio_fm_env.sh
chmod 600 ~/.minio_fm_env.sh
# 编辑密钥后：
source ~/.minio_fm_env.sh
make check-minio
```

凭据文件不得提交到仓库。配置说明见 [MinIO 读写指南](docs/minio_setup.md)。

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
│   ├── schema/               # cn_l2_v1 统一事件协议
│   ├── tokenizer/            # 因果变换、分箱、字段级词表
│   ├── manifest/             # 分片哈希与时间切分
│   ├── pretrain/             # Transformer、Dataset、训练与评估
│   ├── embedding/            # checkpoint → stock-day embedding
│   ├── downstream/           # Ranker、RankIC、CPCV、DSR、回测
│   ├── scripts/              # Pilot、Medium、MinIO 与训练编排
│   └── data/                 # 版本化日期/标的清单
├── order_book/
│   └── pylob/                # 撮合、回放与 MinIO 清洗子项目
├── docs/
│   ├── pipeline/             # Pipeline 逐阶段文档
│   └── README.md             # 全部文档索引
├── examples/                 # 数据清洗与 notebook 示例
├── tests/                    # 撮合、数据管线与 FM 回归测试
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

Pilot 默认模型约 6.36M 参数：

- 多字段 embedding 求和；
- causal self-attention + RoPE；
- pre-norm RMSNorm；
- SwiGLU 前馈层；
- 6 个 next-event 预测头：事件类型、方向、时段、价格、量、时间间隔。

模型配置位于 `quant_fm/pretrain/config_*.yaml`。训练产物默认写入：

```text
quant_fm/runs/<experiment>/run/
├── config.snapshot.yaml
├── tb/
├── best.pt
├── final.pt
└── step*.pt
```

`quant_fm/runs/` 已被 `.gitignore` 排除，不应上传 checkpoint、tokens 或 TensorBoard 日志。

## 文档入口

- [项目与阅读指南](docs/QuantFM.md)
- [Pipeline 逐阶段文档](docs/pipeline/README.md)
- [复现与验证指南](docs/REPRODUCIBILITY.md)
- [MinIO 读写配置](docs/minio_setup.md)
- [文档总索引](docs/README.md)
- [OrderFlow FM 包说明](quant_fm/README.md)
- [PyLOB 子项目说明](order_book/README.md)

## 验收边界

`make smoke` 证明工程链路可运行，但其合成数据回测结果不代表真实策略收益。真实研究结论应以严格时间切分后的下游 RankIC、CPCV、Deflated Sharpe Ratio 与交易成本回测为准。
