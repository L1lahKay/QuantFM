# quant_fm — 端到端 A股订单流决策 FM 预训练

> **新手请先读**：[docs/QuantFM.md](../docs/QuantFM.md)（含阅读顺序、Python 概念、代码内 `# [导读]` 注释索引）  
> **全部文档索引**：[docs/README.md](../docs/README.md)（含 MinIO、流水线、调研与阶段汇报）

在 `order_book/pylob/` 沪深订单簿子项目之上，构建一套**可复现、可验证**的订单流基础模型流水线。Python 导入名仍为 `pylob`。FM embedding 是内部中间产物，冻结 Ranker 的最终生产输出仅为日频 `score`。

## 数据流

```
【读】MinIO :9000  zeus-cn-quote  原始 L2
  → pylob 沪深订单簿重建 (Polars 流式读)
  → cn_l2_v1 标准事件流         quant_fm/schema
  → 全局字段级 tokenizer/词表    quant_fm/tokenizer
  → 分片 manifest + 时间切分     quant_fm/manifest
【写】MinIO :9100  model-cache   tokens/ + vocab + manifest
  → （可选）decoder-only 预训练  quant_fm/pretrain
  → 冻结 stock-day embedding     quant_fm/embedding（内部）
  → 冻结截面 ranker              quant_fm/downstream
  → date/symbol/score            quant_fm/signal（唯一交付）
```

MinIO **读写**详见 [docs/minio_setup.md](../docs/minio_setup.md)。

## MinIO 完整流水线（读 → tokens → 写 → 训练）

```bash
source ~/.minio_fm_env.sh   # 读写密钥（见 minio_env.example.sh）
make check-minio

# 【推荐】一键：读 zeus-cn-quote → 洗成 tokens → 写 model-cache → 8 卡训练
make minio-full-pipeline           # 试跑：5日×30股/市场
make minio-full-pipeline-full      # 60日×全市场（≈总量 1/10）+ 训练

# 只要数据（上传后删本地 tokens，不训练）
make minio-pipeline                # 试跑
make minio-pipeline-full           # 60日×全市场

# 已上传过、本机无 tokens 时：从 model-cache 拉回再训
make download-medium
make train-medium-8gpu
```

## 安装

```bash
uv sync --extra fm     # 或  pip install -e ".[fm]"
```

`schema` / `tokenizer` / `manifest` 不依赖 torch；`pretrain` / `embedding` / `downstream` 需要 `fm` 额外依赖。

## 快速验证（合成数据，CPU，无需 MinIO）

```bash
make smoke
```

`smoke` 会跑通全部环节：合成事件 → tokenize → 微型预训练 → embedding → 离线冻结 Ranker → 在没有未来标签的日期生成 score。结尾打印 `SMOKE OK: score signal generated` 即表示生产链路可用。

正式交付目录只包含：

```text
delivery/
├── scores.parquet
└── signal_manifest.json
```

`score(T)` 仅在 T 日收盘后可用，且只保证同日横截面可比。组合构建、交易成本与回测不属于生产信号链路。

## 真实 Pilot

先配置凭据（endpoint 已内置，见 [docs/minio_setup.md](../docs/minio_setup.md)）：

```bash
cp quant_fm/scripts/minio_env.example.sh ~/.minio_fm_env.sh
vim ~/.minio_fm_env.sh    # 只填 key；或直接用已有 mc alias myminio
source ~/.minio_fm_env.sh
make check-minio          # 读 9000 / 写 9100 自检
```

```bash
make pilot          # 读 zeus-cn-quote @ :9000
make upload-pilot   # 写 model-cache @ :9100
make train-pilot
```

或手动分步：

```bash
python -m quant_fm.scripts.run_pilot \
  --dates 2026-02-02,2026-02-03,2026-02-04,2026-02-05,2026-02-06 \
  --symbols 000001,000002,300750 --market SZ \
  --train-end 2026-02-04 --val-end 2026-02-05 --n-bins 32
python -m quant_fm.pretrain.train --config quant_fm/pretrain/config.yaml
python -m quant_fm.embedding.extract_hidden \
  --checkpoint quant_fm/runs/pilot/run/final.pt \
  --manifest  quant_fm/runs/pilot/data/manifest.json \
  --split test --out quant_fm/runs/pilot/embeddings.parquet
```

## 词表（两级）

- **一级（已实现）**：字段级全局固定分箱。`price_rel`（相对 EW-VWAP 中间价的 log）、`log_volume`、`log_delta_t` 各 32 bin；`evt_type/side/session/board/order_type/event_source` 为固定类别。各字段独立 id 空间，PAD=0。**分箱边界只在训练窗口拟合并冻结**（`vocab.json`），val/test 复用，`assert_no_leakage` 强制校验。
- **二级（Phase 2 可选）**：在事件 embedding 上训 VQ/BSQ 码本，监控码本利用率防坍缩。

不使用巨大 composite vocab；模型对每个字段分别 embedding，并用**多头**分别预测 `event_type/side/session/price_bin/volume_bin/delta_t_bin`，损失 = 各字段下一 token 交叉熵之和。

## 三道验证闸门

1. **重建正确性**（`lob_rebuild/snapshot_check.py`）：重建盘口 vs 3 秒快照逐档一致率。
2. **tokenizer**（`tokenizer/tokenize_events.py`）：`coverage_report` 极端 bin / UNK 率 + `assert_no_leakage`。
3. **下游**（`downstream/evaluate.py`）：`cpcv_splits`（purge+embargo）、`deflated_sharpe_ratio`、`correlation_gate`、`rank_ic`/`rank_icir`、`group_monotonicity`。

## 模型规模

`config.yaml` 默认 pilot（d_model=256, 6 层 ≈ 5–15M）。全市场用 `d_model=512, n_layers=10`（≈ 80–120M）。context 默认 2048（可到 4096）。

## 复现要点

- 固定 `seed`（python/numpy/torch）。
- 每次训练把 `config.yaml` 快照为 `config.snapshot.yaml` 存在 checkpoint 旁。
- manifest 记录每个分片的 `sha256` 与 split。
- 依赖版本在 `pyproject.toml` 的 `fm` extra 中固定下限。
