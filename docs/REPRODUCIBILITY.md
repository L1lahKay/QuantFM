# QuantFM 复现与验证指南

本文汇总项目的推荐阅读路径、复现命令和实验验证边界。

## 10 分钟阅读路径

1. 阅读 [根 README](../README.md) 的 Pipeline 与项目结构；
2. 阅读 [Pipeline 总览](pipeline/README.md)；
3. 查看以下核心文件：
   - `order_book/pylob/pipeline/workflow.py`
   - `quant_fm/tokenizer/transforms.py`
   - `quant_fm/tokenizer/fit_bins.py`
   - `quant_fm/pretrain/model.py`
   - `quant_fm/pretrain/train.py`
   - `quant_fm/downstream/run_judge.py`
4. 运行 smoke 和测试；
5. 若需真实数据，再配置 MinIO。

若复现模型底层 v2，还应按顺序查看：

1. `order_book/pylob/book_state.py` 与 `quant_fm/schema/cn_l2_v2.py`；
2. `quant_fm/tokenizer/field_spec.py`、`fit_bins_v2.py`、`tokenize_events_v2.py`；
3. `quant_fm/pretrain/dataset_v2.py`、`field_fusion.py`、`heads.py`；
4. `quant_fm/pretrain/train.py`、`sampler.py`、`validation_sampler.py` 与 `eval.py`；
5. `quant_fm/benchmark/` 与 `quant_fm/experiments/registry.py`；
6. `quant_fm/embedding/`、`quant_fm/cross_asset/` 与实验性的 `quant_fm/moe/`。

## 无外部依赖复现

```bash
cd QuantFM
uv sync --extra fm --group dev
uv run python -m pytest -q
uv run python -m quant_fm.scripts.smoke --workdir /tmp/quantfm-smoke
```

预期：

- 测试通过；
- 日志出现 `training complete`；
- 生成 embedding；
- 日志最终出现 `SMOKE OK: score signal generated`。

Smoke 使用合成数据和微型 CPU 模型，只验证工程链路，不代表真实投资收益。

当前全仓回归基线（分支 `MOE`，2026-07-24）：

```text
243 passed, 2 skipped, 1 xfailed
```

其中 skip 为依赖本机真实数据/环境的测试，xfail 是已显式登记的预期行为；不要为了得到相同计数而关闭本机可用的真实数据测试。

模型底层 v2 的快速专项回归：

```bash
uv run python -m pytest -q \
  tests/test_book_state_causality.py \
  tests/test_order_book_cancel_consistency.py \
  tests/test_tokenizer_v2.py \
  tests/test_fit_bins_stratified.py \
  tests/test_field_fusion.py \
  tests/test_multitask_loss_v2.py \
  tests/test_pretrain_v2_integration.py \
  tests/test_validation_sampler.py \
  tests/test_pretrain_eval.py \
  tests/test_hierarchical_pooling.py \
  tests/test_intraday_aggregator.py \
  tests/test_cross_asset_causality.py \
  tests/test_cross_asset_dataset_model.py \
  tests/test_training_schedule.py \
  tests/test_shard_aware_sampler.py \
  tests/test_attention_fast_path.py \
  tests/test_rope_cache.py \
  tests/test_inference_checkpoint.py \
  tests/test_experiment_registry.py \
  tests/test_moe_router.py \
  tests/test_moe_causality.py \
  tests/test_backbone_moe.py
```

## 真实数据验证

```bash
source ~/.minio_fm_env.sh
make check-minio
make pilot
```

检查：

```text
quant_fm/runs/pilot/
├── events/
├── tokens/
└── data/
    ├── vocab.json
    └── manifest.json
```

真实预训练：

```bash
make train-8gpu
```

重点观察：

- `train/loss` 是否下降；
- `val/loss` 是否先下降并形成稳定最低点；
- `best.pt` 是否生成；
- per-field CE 是否均优于随机基线；
- 后期 train/val gap 是否显示过拟合。

## 设计验证要点

### 数据正确性

- 上海/深圳交易所差异是否正确收敛到 `OrderBookSH` / `OrderBookSZ`；
- canonical schema 是否稳定；
- 事件顺序是否保持因果；
- manifest 是否按日期切分而非随机切分。

### 防泄漏

- 连续分箱边界只在训练日期拟合；
- val/test 仅复用冻结词表；
- 下游标签不进入预训练输入；
- checkpoint 选择不使用 test 指标。

### 模型与训练

- 多字段 embedding 与独立预测头是否符合任务；
- padding mask 和 next-event shift 是否正确；
- FSDP、AMP、梯度累积和有效 batch 是否符合配置；
- `best.pt` 与 `final.pt` 的语义是否清晰。

### V2 artifact 兼容性

v2 不把“列看起来一样”视为兼容。下列内容必须作为一套实验 artifact 冻结：

| 产物/元数据 | 必须满足的约束 |
|-------------|----------------|
| `vocab_v2.json` | `vocab_version=2.0`；六个特殊 token id 固定；含完整有序 FieldSpec、fit dates、occupancy、normalizer 与采样参数 |
| token shards + manifest | token/scalar 列与 FieldSpec 一致；manifest shard SHA-256、日期 split 不变 |
| `validation_windows.json` | canonical SHA-256 覆盖 context/stride/min_len、seed、精确窗口上限、分层输入与完整有序窗口记录 |
| v2 checkpoint | `fm_artifact_version=2.0`；记录 schema/vocab 版本、有序字段、loss targets、盘口时序、context/pooling，以及 `pretrain_data_contract_v3` |

`load_checkpoint()` 加载 v2 权重时必须传入原始 vocab 路径。当前推理加载会严格核对
artifact 版本、schema、vocab SHA-256、完整有序 FieldSpec，并确认 checkpoint 中选择的
输入/目标字段是 vocab 字段的合法保序子序列；续训还会把 schema、vocab hash、
FieldSpec、输入/目标字段、loss targets、模型/fusion/scalar/book/context/pooling/MoE 配置与
当前配置逐项比较。`pretrain_data_contract_v3` 保留原始 manifest 字节 SHA 作为来源证据，
并以跨路径的 `core_generation_id`、保留 shard 顺序的 `manifest_semantic_sha256` 和 v2
`coverage_sha256` 约束安全存储 rebase；绝对路径可变，但 shard 顺序、内容/split、coverage、
vocab 和日期语义不可变。旧 v2 data contract 在严格 resume 路径 fail closed。不得通过修改
checkpoint 字典或放宽现有检查来“修复”不兼容。

v1 继续使用原有 `PAD=0, N_SPECIAL=1` 与 `legacy_sum`。v2 的 `PAD/UNK/NA/BOS/EOS/SESSION_BREAK` 是独立 id 空间，禁止把 v2 常量回写到 v1 artifact。

### 固定验证计划

首次 v2 训练可自动生成配置指定的 `validation_windows.json`，也可提前创建：

```bash
uv run python -m quant_fm.pretrain.validation_sampler \
  --manifest quant_fm/runs/v2_shared/data/manifest.json \
  --split val --context 2048 --stride 2048 --min-len 16 \
  --seed 42 --max-windows 800 \
  --out quant_fm/runs/v2_shared/validation_windows.json
```

25M 与 100M 实验必须使用同一份计划。计划按日期、交易所、板块、流动性和活跃度尽可能平衡；没有 PIT 流动性输入时使用显式 `unknown` bucket。加载时会重算 canonical SHA、manifest/分层输入指纹和确定性选择结果。训练配置的 `data.validation_windows=N` 是精确数量契约：新建候选不足时不落盘，已有计划数量不等也会被拒绝。

v2 训练与诊断入口：

```bash
uv run python -m quant_fm.pretrain.train \
  --config quant_fm/pretrain/config_v2_25m.yaml

uv run python -m quant_fm.pretrain.eval \
  --checkpoint quant_fm/runs/v2_25m/run/best.pt \
  --config quant_fm/pretrain/config_v2_25m.yaml \
  --validation-plan quant_fm/runs/v2_shared/validation_windows.json \
  --validation-windows 800 \
  --train-unigram-plan quant_fm/runs/v2_shared/train_unigram_windows.json \
  --unigram-windows 800 \
  --device cpu --out quant_fm/runs/v2_25m/run/val_diagnostics.json
```

仓库还提供 `config_v2_230m.yaml` 与 `config_v2_backbone_moe.yaml` 作为后续 dense/
sparse 对照候选。配置文件存在不代表相应模型已经训练，也不应跳过 25M/100M 闸门直接
宣称放大或 MoE 有效。

诊断始终完整消费冻结 validation plan，并记录实际窗口数和逐字段预测数；显式
`--validation-windows N` 与 `--unigram-windows N` 都要求恰好选中 N 个窗口，候选不足时
fail closed，旧 `--max-batches` 只保留为新建 validation plan 的 cap。训练 unigram 使用
独立且不依赖 device batch size 的窗口计划。`train_unigram_normalization_v3` 保存 canonical
counts 原像、counts SHA、实际消费窗口数、逐字段 prediction counts 与重算所得 entropy；
checkpoint 的有序 target fields 是唯一评估字段真值。acceptance v8 要求 candidate 与
baseline 的 validation/train-unigram plan canonical SHA、完整消费计数、counts/entropy
原像和字段集合一致，重算所有 normalization 数学关系并实时核对 checkpoint/plan。由于
counts 目前不能由 acceptance 独立证明来自 live train-plan 数据，PASS 只允许使用 raw
`total_ce`；归一化 CE 仅作诊断、不得参与加权或晋级。旧 v7/v6/v2 artifact 在严格路径
fail closed。诊断至少比较 per-field CE、
train-unigram CE、copy baseline、训练熵归一化 CE、top-k、预测熵和字段梯度范数；不能只看
总 loss。

验收门槛由 validator 独立提供，默认 `expected tolerance=0.01`。artifact 中的 tolerance
必须与它精确相等，不能通过改 artifact 自行授权更宽门槛；非默认值必须在生成、独立验收和
lineage 复核三个入口分别显式传入同一 `--tolerance` / `--expected-tolerance`。

### 研究有效性

- 预训练 loss 仅是过程指标；
- embedding 必须通过严格时间切分的下游任务验证；
- 回测包含交易成本；
- RankIC、CPCV 和 DSR 共同用于降低偶然性。

## 不应提交到 GitHub

- `quant_fm/runs/`
- `data/clean/`
- parquet、checkpoint、TensorBoard 日志
- `.venv/`
- `~/.minio_fm_env.sh` 或任何真实 Access Key / Secret Key

上述内容均应由 `.gitignore` 排除。`uv.lock`、日期清单和标的清单应保留，以保证复现。

## 已知边界

- PyLOB 是离线研究引擎，不是生产低延迟撮合系统；
- checkpoint 分为两类：`step*.pt` / `final_resume.pt` 保存模型、optimizer、scaler 与
  `TrainState`，用于续训；`best.pt` / `final.pt` 不含 optimizer/scaler，面向评估和推理。
  后两者显式传给 `--resume` 会直接报错；`--resume auto` 优先选择编号最大的
  `step*.pt`，仅在不存在定期 checkpoint 时尝试 `final_resume.pt`，二者都不存在则从头
  训练。可续训 checkpoint 还保存并恢复逐 rank Python/NumPy/Torch CPU/CUDA RNG、sampler
  epoch 与 epoch 内消费位置，以及 config/validation-plan/stop-budget/拓扑身份；已有带
  dropout 的单 rank CPU 连续训练与分段 resume bitwise 等价测试，真实多 rank/FSDP 故障
  恢复仍需演练；
- 数据阶段支持日期级 / 标的级断点续跑（`.done`、`.clean_done`、`skip_existing`）、`CLEAN_WORKERS` 多进程清洗与 `CANON_WORKERS` 并行规范化；可用 `check_pipeline_progress` 查询进度；
- manifest 中仍包含绝对路径，跨机器迁移时需要受控改写；v3 data contract 用跨路径的
  core/semantic/coverage 身份判断安全 rebase，不再以原始 manifest 字节 SHA 单独决定兼容；
- 合成 smoke 的回测指标没有经济含义；
- MinIO/Pilot/Medium 一键脚本现默认生成带真实逐事件盘口的 V2 events、`vocab_v2.json`、Q16 token/scalar、manifest 与审计报告；V1 只允许通过 `--data-version v1` 显式复现；
- v2 实现已经通过单元/集成回归，但尚未生成正式 25M/100M checkpoint，未做新 untouched OOS，也未证明 score 或交易成本后收益改善；
- 多尺度池化、`IntradayAggregator` 和 `cross_asset` 已有因果测试，但尚未接入默认生产 score 路径；
- Temporal Regime-MoE 与顶部 Backbone-MoE 已有模块、配置和代码级测试，但尚无正式
  真实数据训练、消融或 OOS 结论，也未接入默认 score 路径；
- 全市场效果需要更长时间跨度与独立 out-of-sample 数据验证。
