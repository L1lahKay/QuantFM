# QuantFM Pipeline 文档

本目录按真实执行顺序拆解 QuantFM 的端到端流水线。每篇文档统一说明：目标、输入输出、核心代码、运行方式、验证条件和常见问题。

## 全链路

| 阶段 | 文档 | 核心产物 |
|------|------|----------|
| 1 | [MinIO 数据接入](01_minio_io.md) | 原始 trade/order DataFrame |
| 2 | [订单簿重建与清洗](02_order_book_rebuild.md) | `clean/<date>/<market>/<symbol>/events.parquet` |
| 3 | [cn_l2_v1 事件规范化](03_canonical_events.md) | `events/<market>/<symbol>/<date>.parquet` |
| 4 | [Tokenizer 与词表](04_tokenizer_vocab.md) | `vocab.json`、token parquet |
| 5 | [Manifest 与时间切分](05_manifest_splits.md) | `manifest.json` |
| 6 | [OrderFlow FM 预训练](06_pretraining.md) | `best.pt`、`final.pt`、TensorBoard 日志 |
| 7 | [股日 Embedding](07_embeddings.md) | `embeddings.parquet` |
| 8 | [下游 Ranker 与回测](08_downstream_evaluation.md) | RankIC、回测与 judge report |

## 三种运行路径

### 合成数据验收

```bash
uv run python -m quant_fm.scripts.smoke --workdir /tmp/quantfm-smoke
```

用于 CI 和重构回归，不依赖 MinIO。预期终态：`SMOKE OK: all stages passed`。

### 真实 Pilot

```bash
source ~/.minio_fm_env.sh
make pilot
make train-8gpu
```

默认处理 5 个交易日、3 只深圳股票，适合验证真实数据链路。

### Medium / 全市场编排

```bash
source ~/.minio_fm_env.sh
make minio-full-pipeline       # try
make minio-full-pipeline-full  # 60 日 × 全市场

# ~302M 正式：22 日全市场（Chinchilla）+ 断点续跑 + 并行清洗 + 自动续训
CLEAN_WORKERS=16 SKIP_UPLOAD=1 bash quant_fm/scripts/run_minio_300m_pipeline.sh
```

由 `quant_fm/scripts/run_minio_full_pipeline.sh` / `run_minio_300m_pipeline.sh` 统一编排：

1. 检查 MinIO；
2. 生成或恢复 tokens（按日 / 按标的断点续跑）；
3. 可选上传并校验远端副本；
4. 启动 8 卡训练（`--resume auto`）；
5. 可选删除本地 tokens。

常用环境变量：

| 变量 | 含义 |
|------|------|
| `CLEAN_WORKERS` | 并行洗股进程数（默认 `min(32, CPU/2)`） |
| `SKIP_DATA=1` | 本地 tokens/manifest 已就绪，直接训练 |
| `SKIP_TRAIN=1` | 只做数据 |
| `SKIP_UPLOAD=1` | 不上传 model-cache |

## 默认目录约定

```text
quant_fm/runs/<experiment>/
├── clean/          # PyLOB 中间产物，可删除
├── events/         # cn_l2_v1，可在 tokenize 后删除
├── tokens/         # 模型输入
├── data/
│   ├── vocab.json
│   ├── manifest.json
│   ├── .done/        # 日期级：canonicalize 完成
│   └── .clean_done/  # 日期级：当日 clean 完成（可复用）
├── run/            # checkpoint + TensorBoard
├── embeddings*.parquet
└── downstream/     # 下游裁判报告
```

该目录属于生成产物，已被 `.gitignore` 排除。断点续跑层级：日期 `.done` → clean 标的 `events.parquet` → events 股日 parquet → tokens 文件。

## 关键验证闸门

| 闸门 | 目的 | 实现 |
|------|------|------|
| 撮合/快照一致性 | 验证订单簿重建正确性 | `quant_fm/lob_rebuild/snapshot_check.py` |
| Token 覆盖率 | 监控极端 bin 与未知类别 | `coverage_report()` |
| 无泄漏断言 | 禁止 val/test 日期拟合词表 | `assert_no_leakage()` |
| 分片哈希 | 保证输入可审计、可复现 | manifest 中 `sha256` |
| 验证集最优模型 | 防止使用过拟合的最终权重 | `best.pt` |
| 下游统计检验 | 降低数据挖掘与偶然收益 | RankIC、CPCV、DSR |
