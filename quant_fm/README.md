# quant_fm — 端到端 A股订单流决策 FM 预训练

> **新手请先读**：[docs/QuantFM.md](../docs/QuantFM.md)（含阅读顺序、Python 概念、代码内 `# [导读]` 注释索引）  
> **全部文档索引**：[docs/README.md](../docs/README.md)（含 MinIO、流水线、调研与阶段汇报）

在 `order_book/pylob/` 沪深订单簿子项目之上，构建一套**可复现、可验证**的订单流基础模型流水线。Python 导入名仍为 `pylob`。FM embedding 是内部中间产物，冻结 Ranker 的最终生产输出仅为日频 `score`。`cn_l2_v1`/Tokenizer v1 保持兼容；v2 通过显式 artifact 版本启用，禁止与 v1 词表或 checkpoint 静默混用。

## 数据流

```
【读】MinIO :9000  zeus-cn-quote  原始 L2
  → pylob 沪深订单簿重建 (Polars 流式读)
  → cn_l2_v1（稳定）或 cn_l2_v2（显式启用）  quant_fm/schema
  → v1 vocab.json / v2 vocab_v2.json        quant_fm/tokenizer
  → 分片 manifest + 时间切分     quant_fm/manifest
【写】MinIO :9100  model-cache   tokens/ + vocab + manifest
  → （可选）decoder-only 预训练  quant_fm/pretrain
  → 冻结 stock-day / interval embedding     quant_fm/embedding（内部）
  → 可选日内聚合 + Regime-MoE               quant_fm/moe（研究阶段）
  → 可选同步跨股票上下文                     quant_fm/cross_asset（研究阶段）
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

## Tokenizer v1 / v2

- **v1 稳定路径**：字段级全局固定分箱。`price_rel`、`log_volume`、`log_delta_t` 各 32 bin；类别字段使用固定映射。每个字段独立 id 空间，PAD=0；边界仅在训练窗口拟合并冻结到 `vocab.json`。
- **v2 显式路径**：[`field_spec.py`](tokenizer/field_spec.py) 冻结字段来源、语义、bin 数、适用事件和输入/目标角色；[`vocab_v2.py`](tokenizer/vocab_v2.py) 使用独立的 `PAD/UNK/NA/BOS/EOS/SESSION_BREAK` id 空间，并记录全流 occupancy、缺失率、训练熵、分层采样参数和连续 normalizer。
- v2 数值字段同时生成 `tok_*_bin` 与 `val_*`，模型把标准化 scalar 投影后加到对应 token 表示；缺失 scalar 为 0，但由独立 `NA` token 区分，不与真实 0 混淆。
- [`fit_bins_v2.py`](tokenizer/fit_bins_v2.py) 使用确定性、路径顺序无关的分层 priority reservoir，边界只从训练 shard 拟合；[`tokenize_events_v2.py`](tokenizer/tokenize_events_v2.py) 负责 v2 token/scalar 输出和覆盖率检查。

两版均不使用巨大 composite vocab。v1 默认等权 CE；v2 由 `loss.targets` 显式声明主目标，支持训练熵归一化、任务权重、`ordinal_ce` 距离约束和字段适用性 mask。

## 模型底层 v2

v2 的主要代码入口如下：

| 能力 | 入口 |
|------|------|
| 因果盘口状态与 v2 schema | [`pylob/book_state.py`](../order_book/pylob/book_state.py)、[`schema/cn_l2_v2.py`](schema/cn_l2_v2.py) |
| FieldSpec、词表拟合与 token 化 | [`tokenizer/field_spec.py`](tokenizer/field_spec.py)、[`tokenizer/fit_bins_v2.py`](tokenizer/fit_bins_v2.py)、[`tokenizer/tokenize_events_v2.py`](tokenizer/tokenize_events_v2.py) |
| v2 Dataset 与字段融合 | [`pretrain/dataset_v2.py`](pretrain/dataset_v2.py)、[`pretrain/field_fusion.py`](pretrain/field_fusion.py) |
| 多任务 Loss 与训练 | [`pretrain/heads.py`](pretrain/heads.py)、[`pretrain/train.py`](pretrain/train.py) |
| 训练计数、shard locality 与性能基准 | [`pretrain/sampler.py`](pretrain/sampler.py)、[`benchmark/`](benchmark/) |
| 固定验证和字段诊断 | [`pretrain/validation_sampler.py`](pretrain/validation_sampler.py)、[`pretrain/eval.py`](pretrain/eval.py) |
| 多尺度股日表示 | [`embedding/pool_stock_day.py`](embedding/pool_stock_day.py)、[`embedding/intraday_aggregator.py`](embedding/intraday_aggregator.py) |
| Regime/Backbone MoE | [`moe/`](moe/)、[`pretrain/config_v2_backbone_moe.yaml`](pretrain/config_v2_backbone_moe.yaml) |
| 同步跨股票上下文 | [`cross_asset/`](cross_asset/) |

`OrderFlowFM` 支持四种事件内融合：`legacy_sum`、`scaled_sum`、`gated_sum` 和 `concat_mlp`。v2 配置默认使用 `gated_sum`、字段 dropout 和输入 RMSNorm；v1 旧 checkpoint 缺少这些元数据时仍按 `legacy_sum` 加载。

主干还支持显式 `ffn_hidden`、按 device/dtype 复用且不写入 checkpoint 的 RoPE cache，以及无 padding batch 的 `scaled_dot_product_attention(is_causal=True)` 快路径；含 padding 时仍使用显式 causal + key mask。训练状态把本 rank micro-batch 次数、optimizer update 次数和跨 rank 汇总的有效 token 数分别记录，学习率/日志/验证/存盘均按 update 计数。

`quant_fm.moe` 提供 `TemporalRegimeMoE`（股日/时间聚合层）、`SparseMoEFeedForward`（可选顶部 Transformer FFN）、基础路由遥测和 Regime artifact 序列化。Temporal 模块尚未接入默认训练/embedding/score；Backbone 模块可由 YAML 配置接入 `OrderFlowFM`，训练当前只记录合计 `train/moe_aux`，没有自动记录 expert fraction、entropy 或 overflow。两者都属于待实验候选。

跨股票模块只消费已经按 5 分钟交易时钟同步的 interval embedding，使用市场均值、行业 leave-one-out 和逐股因果 GRU，复杂度为 O(T×N×D)。行业历史采用严格 PIT as-of join；该模块目前是研究组件，尚未接入默认 Ranker/score 生产路径。

## v2 配置与运行

新增配置：

- [`pretrain/config_v2_25m.yaml`](pretrain/config_v2_25m.yaml)：Stage-1 约 25M 消融；
- [`pretrain/config_v2_100m.yaml`](pretrain/config_v2_100m.yaml)：仅让 Stage-1 winner 晋级的约 100M、8 卡 FSDP 复验；
- [`pretrain/config_v2_230m.yaml`](pretrain/config_v2_230m.yaml)：约 230M 的 Dense V2 候选；
- [`pretrain/config_v2_backbone_moe.yaml`](pretrain/config_v2_backbone_moe.yaml)：顶部 4 层 shared + Top-1 routed expert 候选，禁止未经消融直接作为生产默认值。

两者预期共享以下冻结产物：

```text
quant_fm/runs/v2_shared/
├── data/
│   ├── manifest.json
│   └── vocab_v2.json
└── validation_windows.json
```

准备好 v2 events/tokens/manifest 后运行：

```bash
uv run python -m quant_fm.pretrain.train \
  --config quant_fm/pretrain/config_v2_25m.yaml

uv run torchrun --standalone --nproc_per_node=8 \
  -m quant_fm.pretrain.train \
  --config quant_fm/pretrain/config_v2_100m.yaml
```

固定验证计划也可提前生成：

```bash
uv run python -m quant_fm.pretrain.validation_sampler \
  --manifest quant_fm/runs/v2_shared/data/manifest.json \
  --split val --context 2048 --stride 2048 --min-len 16 \
  --seed 42 --max-windows 800 \
  --out quant_fm/runs/v2_shared/validation_windows.json
```

每个 v2 checkpoint 固化 `fm_artifact_version=2.0`、schema/vocab 版本、vocab SHA-256、有序 FieldSpec、输入/目标字段、连续 normalizer、盘口时序、context horizon、pooling 版本和 loss target 声明。`load_checkpoint()` 和 `--resume` 任一检查不一致都会 fail-fast；加载 v2 checkpoint 必须提供其原始 vocab 路径。

checkpoint 分两类：定期 `step*.pt` 与 `final_resume.pt` 含 optimizer/scaler/train state，可恢复优化器；`best.pt` 与 `final.pt` 不含 optimizer/scaler，仅用于评估和推理，训练入口会拒绝用它们 resume。`--resume auto` 只查找 `step*.pt` 与 `final_resume.pt`，并优先编号最大的定期点；继续一个已经正常结束的 run 时应显式指定 `final_resume.pt`。

## 四道验证闸门

1. **重建正确性**（`lob_rebuild/snapshot_check.py` + v2 因果回放测试）：重建盘口 vs 3 秒快照逐档一致率，并验证事件 t 的 post-state 只来自不晚于 t 的事件。
2. **tokenizer**（`tokenize_events.py` / `tokenize_events_v2.py`）：极端 bin、NA/UNK、实际 bin 数、occupancy 和训练日期无泄漏。
3. **模型诊断**（`pretrain/eval.py`）：固定分层窗口上的 per-field CE、归一化 CE、top-k、copy/unigram baseline、预测熵和梯度范数。
4. **下游**（`downstream/evaluate.py`）：`cpcv_splits`、DSR、RankIC/ICIR、分组单调性及严格 OOS 研究回测。

## 模型规模

`config.yaml` 默认 v1 pilot（d_model=256, 6 层，约 5–15M）。v2 提供约 25M、100M 和 230M Dense 档，以及一个顶部 Backbone-MoE 候选；context 为 2048。配置名表示实验档位，真实参数量以训练日志 `model parameters` 为准。

## 复现要点

- 固定 `seed`（python/numpy/torch）。
- 每次训练把 `config.yaml` 快照为 `config.snapshot.yaml` 存在 checkpoint 旁。
- manifest 记录每个分片的 `sha256` 与 split。
- v2 的 25M/100M 比较必须复用同一个 `validation_windows.json`；计划带 manifest fingerprint 和窗口参数，不匹配时拒绝运行。
- v2 词表、token parquet 与 checkpoint 必须作为一套 artifact 保存，不能只复制模型权重。
- `micro_step` 是每 rank 的 DataLoader batch 计数；`update_step` 只在参数更新成功后增加；`non_pad_tokens_seen` 在成功 update 后跨 rank 汇总，FP16 overflow 跳步不计入。`max_update_steps`/`max_train_tokens` 是 OR 停止条件，LR 按 update 调度；token-budget-only 配置必须给出 `optim.lr_schedule_steps`。
- shard-aware sampler 只保证 shard 聚簇、确定性 epoch shuffle 和各 rank 等窗口数；`drop_last=true` 会丢弃尾部窗口，它尚未按 token 长度做负载均衡。
- 依赖版本在 `pyproject.toml` 的 `fm` extra 中固定下限。

## 当前边界

- 代码与回归测试已完成（当前 `243 passed, 2 skipped, 1 xfailed`），但尚未生成正式全市场 v2 artifact，也未完成 25M/100M/230M 重训、MoE 消融或新的 untouched OOS；因此不能由本次改造直接推导收益提升。
- 现有 `make pilot`、`make minio-full-pipeline*` 默认仍走 v1 数据路径；v2 数据准备目前是库 API，不是新的 Make 一键入口。
- `pooling.method: multi_scale` 是配置/产物契约；实际抽取需在 `extract_hidden` 明确传 `--pooling multi_scale --vocab <vocab_v2.json>`。
- `cross_asset` 与 `IntradayAggregator` 已具备因果单测，但尚未纳入默认预训练或 score 交付链路。
- `RegimeFeatureNormalizer.fit_end` 与 `availability_lag` 当前是审计元数据，不会替调用方执行训练期过滤或 as-of lag；Regime 输入必须在外部先做 PIT 对齐。
- 训练模式发生 capacity overflow 时，当前 Temporal/Backbone MoE 会在 batch 或全部有效 token 范围内按 router 权重裁剪，存在 batch 依赖和 train/eval dispatch 差异；评估/推理模式不裁剪，已有 batch-size independence 测试。真实路由稳定性、overflow、吞吐和 OOS 仍未验证。
- `save_regime_moe_artifact()`/`load_regime_moe_metadata()` 已实现版本化保存与 metadata 校验，但尚无完整的模型重建 loader；调用方仍需保存并重建 aggregator/输入维度等结构。
