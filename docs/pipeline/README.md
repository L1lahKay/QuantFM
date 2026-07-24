# QuantFM Pipeline 文档

本目录按真实执行顺序拆解 QuantFM 的端到端流水线。每篇文档统一说明：目标、输入输出、核心代码、运行方式、验证条件和常见问题。

## 全链路

| 阶段 | 文档 | 核心产物 |
|------|------|----------|
| 1 | [MinIO 数据接入](01_minio_io.md) | 原始 trade/order DataFrame |
| 2 | [订单簿重建与清洗](02_order_book_rebuild.md) | `clean/<date>/<market>/<symbol>/events.parquet`；v2 可附逐事件 pre/post 盘口 transition |
| 3 | [cn_l2_v1/v2 事件规范化](03_canonical_events.md) | `events/<market>/<symbol>/<date>.parquet` |
| 4 | [Tokenizer 与词表](04_tokenizer_vocab.md) | `vocab.json` / `vocab_v2.json`、token/scalar parquet |
| 5 | [Manifest 与时间切分](05_manifest_splits.md) | `manifest.json` |
| 6 | [OrderFlow FM 预训练](06_pretraining.md) | 固定验证计划、`best.pt`、`final.pt`、TensorBoard 日志 |
| 7 | [股日 Embedding](07_embeddings.md) | 单尺度/多尺度 `embeddings.parquet`，可选 interval embedding |
| 8 | [Score 信号与研究验收](08_downstream_evaluation.md) | `scores.parquet`、`signal_manifest.json` |

## 三种运行路径

### 合成数据验收

```bash
uv run python -m quant_fm.scripts.smoke --workdir /tmp/quantfm-smoke
```

用于 CI 和重构回归，不依赖 MinIO。预期终态：`SMOKE OK: score signal generated`。

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
CLEAN_WORKERS=32 CANON_WORKERS=16 SKIP_UPLOAD=1 bash quant_fm/scripts/run_minio_300m_pipeline.sh

# 查询数据阶段进度（完成天数、当日洗股/规范化、ETA）
uv run python -m quant_fm.scripts.check_pipeline_progress
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
| `CANON_WORKERS` | 并行规范化进程数（默认 `min(16, CPU/4)`） |
| `SKIP_DATA=1` | 本地 tokens/manifest 已就绪，直接训练 |
| `SKIP_TRAIN=1` | 只做数据 |
| `SKIP_UPLOAD=1` | 不上传 model-cache |

进度查询：

```bash
uv run python -m quant_fm.scripts.check_pipeline_progress
uv run python -m quant_fm.scripts.check_pipeline_progress --workdir quant_fm/runs/medium_300m
```

读取 `data/.done/`、当日 `pipeline.log` 与 events/tokens/manifest 状态，输出完成比例与 ETA。

## 模型底层 v2 路径

v2 不改变 v1 已有产物。只有显式生成 `schema_version=cn_l2_v2` 的事件、`vocab_version=2.0` 的词表并使用 v2 token parquet 时，训练入口才切换到 `EventWindowDatasetV2`。

```text
按 exchange sequence 回放单个事件
  → BookStateTransition(pre, post)
  → cn_l2_v2（结构上逐行对齐；真实回放来源须另行验收）
  → FieldSpec + fit_vocab_v2（仅训练日期）
  → tokenize_events_v2（bin token + normalized scalar + NA mask）
  → manifest
  → 固定 validation_windows.json
  → 25M 消融 → 100M winner 复验
  → 通过前述闸门后才评估 230M dense / Backbone-MoE 候选
  → 多尺度股日表示
  → 可选 5 分钟 interval / PIT 行业 / leave-one-out 跨股票模型
```

关键入口：

| 子阶段 | 入口 |
|--------|------|
| 盘口转移 | `pylob.book_state.capture_book_transition` / `iter_book_state_transitions` |
| v2 schema | `quant_fm.schema.cn_l2_v2.events_to_canonical` |
| 词表拟合与 token 化 | `quant_fm.tokenizer.fit_bins_v2.fit_vocab_v2` / `tokenize_events_v2.tokenize_path_v2` |
| 数据与训练 | `quant_fm.pretrain.dataset_v2` / `quant_fm.pretrain.train` |
| 固定验证 | `quant_fm.pretrain.validation_sampler` / `quant_fm.pretrain.eval` |
| 多尺度与跨股票 | `quant_fm.embedding` / `quant_fm.cross_asset` |
| 性能与实验登记 | `quant_fm.benchmark` / `quant_fm.experiments.registry` |
| 实验性 MoE | `quant_fm.moe` / `quant_fm/pretrain/config_v2_backbone_moe.yaml` |

当前 MinIO/Pilot/Medium 一键脚本仍产出 v1 数据。v2 的上述数据阶段已经有库 API 和测试，但尚未封装成一条新的 MinIO CLI；因此下面命令的前提是 v2 manifest、token shards 和 `vocab_v2.json` 已准备完毕。

```bash
# 可选：提前冻结 800 个验证窗口；训练首次运行也会自动创建
uv run python -m quant_fm.pretrain.validation_sampler \
  --manifest quant_fm/runs/v2_shared/data/manifest.json \
  --split val --context 2048 --stride 2048 --min-len 16 \
  --seed 42 --max-windows 800 \
  --out quant_fm/runs/v2_shared/validation_windows.json

# Stage-1
uv run python -m quant_fm.pretrain.train \
  --config quant_fm/pretrain/config_v2_25m.yaml

# Stage-2，仅复验 Stage-1 winner
uv run torchrun --standalone --nproc_per_node=8 \
  -m quant_fm.pretrain.train \
  --config quant_fm/pretrain/config_v2_100m.yaml

# 使用同一固定计划输出字段诊断
uv run python -m quant_fm.pretrain.eval \
  --checkpoint quant_fm/runs/v2_25m/run/best.pt \
  --config quant_fm/pretrain/config_v2_25m.yaml \
  --validation-plan quant_fm/runs/v2_shared/validation_windows.json \
  --device cpu --out quant_fm/runs/v2_25m/run/val_diagnostics.json
```

v2 artifact 应作为一个整体管理：`events/tokens + manifest + vocab_v2.json + validation_windows.json + checkpoint`。checkpoint 会记录 vocab hash、完整有序 FieldSpec、连续 normalizer、输入/目标字段、盘口状态时序、context/pooling 版本和 loss targets。当前推理加载会核对 artifact/schema、vocab hash、FieldSpec 和输入/目标字段顺序；续训还会逐项核对模型宽深、FFN/RoPE、fusion/scalar/MoE、盘口时序、context/pooling 和 loss targets。manifest 自身的 schema/vocab 路径、每个 shard 的行数/哈希与 checkpoint 之间尚未全部自动交叉验证，仍需冻结配置、validation-plan fingerprint 和实验登记共同审计。不要用文件名相同或字段数相同来代替现有 hash 与元数据校验。

## 默认目录约定

```text
quant_fm/runs/<experiment>/
├── clean/          # PyLOB 中间产物，可删除
├── events/         # cn_l2_v1 / cn_l2_v2，可在 tokenize 后删除
├── tokens/         # v1 token 或 v2 token+scalar 模型输入
├── data/
│   ├── vocab.json / vocab_v2.json
│   ├── manifest.json
│   ├── .done/        # 日期级：canonicalize 完成
│   └── .clean_done/  # 日期级：当日 clean 完成（可复用）
├── validation_windows.json  # v2 固定验证计划；配置也可指向共享目录
├── run/            # checkpoint + TensorBoard
├── embeddings*.parquet  # 内部中间产物
├── signal_artifact/     # 冻结 Ranker
└── delivery/            # scores.parquet + signal_manifest.json
```

该目录属于生成产物，已被 `.gitignore` 排除。断点续跑层级：日期 `.done` → clean 标的 `events.parquet` → events 股日 parquet → tokens 文件。

## 关键验证闸门

| 闸门 | 目的 | 实现 |
|------|------|------|
| 撮合/快照一致性 | 验证订单簿重建正确性 | `quant_fm/lob_rebuild/snapshot_check.py` |
| 因果盘口一致性 | 当前事件 post-state 只来自当前及历史事件；撤单同步更新簿与索引 | `pylob.book_state` + `test_book_state_causality.py` |
| Token 覆盖率 | 监控极端 bin、实际 bin 数、NA/UNK 与 occupancy | `coverage_report()` / `coverage_report_v2()` |
| 无泄漏断言 | 禁止 val/test 日期拟合词表或 continuous normalizer | `assert_no_leakage()` / `assert_no_leakage_v2()` |
| 分片哈希 | 保证输入可审计、可复现 | manifest 中 `sha256` |
| 固定验证窗口 | 禁止候选顺序和前 N batch 改变模型比较 | `validation_windows.json` + manifest fingerprint |
| v2 artifact 兼容 | 推理核对 schema/vocab/FieldSpec/字段顺序；续训另核对 loss targets | `fm_artifact_version=2.0` + vocab SHA-256 |
| 验证集最优模型 | 防止使用过拟合的最终权重 | `best.pt` |
| 无标签信号生成 | 防止生产推理读取未来收益 | `quant_fm.signal.generate` |

## 当前验证结果与边界

```bash
uv run python -m pytest -q
# 243 passed, 2 skipped, 1 xfailed（2026-07-24）
```

该结果覆盖 v1 兼容、因果盘口、Tokenizer v2、字段融合、Loss、固定验证、跨 chunk 池化、日内聚合和跨股票因果性。它证明接口和不变量成立，不代表已经完成正式 v2 训练，也不代表收益改善。25M/100M checkpoint、真实数据 tokenizer coverage、盘口快照逐档一致率、新 untouched OOS、跨股票模块增益和交易成本后收益仍是下一阶段验收项。

训练计数/吞吐、attention fast path、RoPE cache、推理 checkpoint、实验登记以及
Temporal/Backbone-MoE 也已有代码级回归。`config_v2_230m.yaml` 和
`config_v2_backbone_moe.yaml` 仍只是候选配置；没有真实训练、消融和 untouched OOS 时，
不得把它们写成效果结论或默认生产路径。
