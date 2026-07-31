# `MOE` 分支工作说明

本文档根据当前 `MOE` 工作分支与 `main` 的代码差异整理。该分支以
`feat/score-signal-only` 的生产 score 契约为基础，继续加入模型底层 v2、训练性能语义、
Regime/Backbone MoE 候选和严格 OOS 研究组件。

## 分支定位

本分支在 `main` 的端到端订单流基础模型流水线上，补齐了大规模并行处理、样本外（OOS）运行和下游评估能力，并建立独立的生产信号边界：生产侧只接收冻结的股日 embedding 和冻结 Ranker，最终只交付 `date, symbol, score` 及可追溯清单。本次又加入模型底层 v2、训练/推理性能修复和 MoE 研究组件；v2 与现有 v1 artifact 显式隔离，当前处于“代码与回归完成、正式重训/MoE/OOS 待执行”的阶段。

核心原则如下：

- 训练需要的未来收益标签与生产打分彻底分离；
- embedding 是内部中间产物，`score` 是唯一对外交付；
- Ranker 权重、特征顺序和训练截止日均随 artifact 固化；
- 信号日期默认必须晚于 Ranker 的训练截止日；
- 回测、持仓、成本和风险指标继续留在 research-only 下游模块；
- v2 schema、vocab、FieldSpec、字段顺序和 checkpoint 必须严格匹配，禁止用 v1 artifact 静默降级或迁移。
- Temporal/Backbone MoE 均是实验候选；未通过因果、路由健康度、吞吐和新 OOS 门槛前不进入生产 score。

## 相对 `main` 的主要工作

### 1. 生产级 score 信号接口

新增 `quant_fm/signal/` 包：

- `train.py`：从带标签训练样本拟合 Ranker，并保存冻结 artifact；
- `artifact.py`：保存和加载 Ranker 权重、结构配置、特征列顺序、训练截止日及来源信息；
- `generate.py`：使用无标签 embedding 生成 `scores.parquet` 和 `signal_manifest.json`；
- `schema.py`：校验输出列、主键唯一性、数值有效性和同日横截面 score；
- `__init__.py`：提供轻量、稳定的 `generate_scores` 与 `validate_scores` 公共接口。

`quant_fm/downstream/make_features.py` 将特征构建显式拆分为训练和打分两条路径。打分路径拒绝 `label`、`fwd_ret`、`xs_ret` 等未来信息，并检查缺失值、NaN、Inf、股票代码格式和最小横截面规模。

`quant_fm/downstream/train_ranker.py` 增加无标签预测路径，严格校验推理特征列与 artifact 中记录的列及顺序一致，输出固定为：

```text
date | symbol | score
```

### 2. 可复现的信号交付

信号生成会同时写出交付清单，记录 Ranker、embedding、可选 FM checkpoint 和 vocab 的哈希及生成时间等来源信息。写入过程采用临时文件替换，避免中断后留下半成品。

根 `Makefile` 新增：

```bash
make signal        # 使用冻结 Ranker 生成生产 score
make signal-smoke  # 在合成数据上验证无标签打分链路
```

默认输入和输出可通过 `SIGNAL_EMBEDDINGS`、`SIGNAL_ARTIFACT`、`SIGNAL_OUT` 覆盖。

### 3. 大规模数据与并行运行

本分支同时包含面向 300M 级训练和 2026 OOS 数据的工程增强：

- 加速沪深订单簿重建、事件导出和 MinIO I/O；
- 支持按交易日并行清洗、断点续跑、复用词表及 token 化目录；
- 支持按 shard 负载均衡的多 GPU embedding 抽取；
- 增加连续 60 日及 OOS 2026 的固定日期、沪深股票池清单；
- 增加 OOS 增量编排、watchdog、状态监控和安全 token 清理回执；
- 增强 302M 模型训练配置、评估、训练监控和完成后自动 judge。

主要入口位于 `quant_fm/scripts/`，包括：

- `run_medium_parallel_days.sh`
- `extract_embeddings_parallel.sh`
- `run_oos2026_incremental.sh`
- `watchdog_oos2026_parallel.sh`
- `run_judge_300m.sh`
- `watch_300m_train.sh`

### 4. 下游研究与交付链路

下游模块新增或增强了：

- 日频 panel 构建及传统因子基线；
- RankIC/ICIR、Top-K 和 Ranker 训练评估；
- 显式 entry/exit 日期与 forward-return horizon，拒绝不完整的未来交易日历；
- 固定 top-name 成交语义、换手/冲击/佣金成本、Newey-West 统计、因子暴露残差化和风险归因；
- 连续样本与 OOS 样本的增量交付；
- Ranker 缓存、并发锁、增量合并和失败后的锁清理；
- 研究评估与生产 score 接口的文档化边界。

`make judge-300m`、`make watch-300m` 和 `make baselines-300m` 属于研究评估入口，不属于线上 score 推理的必要依赖。

### 5. 模型底层 v2

本分支新增独立的 `cn_l2_v2`/Tokenizer v2/训练 artifact 路径，同时保持 v1 旧 checkpoint 可加载。

| 方向 | 主要改动 |
|------|----------|
| 盘口 | `pylob.book_state` 捕获逐事件 pre/post 状态；撤单同步删除价格档和活动索引，保持 FIFO/盘口一致 |
| Schema/Tokenizer | `cn_l2_v2` 要求逐行结构对齐的盘口字段；真实 replay provenance 另由回放、coverage、prefix-causality 与快照验收；FieldSpec 冻结字段；独立 NA/UNK；分层 reservoir；全流统计；bin+scalar 双通道 |
| 模型/Loss | 四种字段融合、字段 dropout、连续投影、训练熵归一化、ordinal 距离损失和 applicability mask |
| 训练/评估 | v1/v2 loader 分流；固定分层验证窗口；字段 CE、unigram/copy baseline、熵、top-k 和梯度范数诊断 |
| 性能语义 | micro/update/global token 独立计数；shard 聚簇 sampler；RoPE cache；无 padding causal fast path；显式 `ffn_hidden`；续训/推理 checkpoint 分流 |
| 表征 | 修复跨 chunk `last/last-k`；增加交易阶段多尺度池化和严格因果日内聚合器 |
| 跨股票 | 5 分钟同步、PIT 行业映射、行业 leave-one-out 和 O(T×N×D) 的轻量跨股票模型 |
| MoE | 股日级 `TemporalRegimeMoE`、顶部 `SparseMoEFeedForward`、基础 telemetry 和 Regime artifact serializer；均为未验证研究组件 |

训练配置：

- `quant_fm/pretrain/config_v2_25m.yaml`：Stage-1 约 25M 消融；
- `quant_fm/pretrain/config_v2_100m.yaml`：Stage-1 winner 的约 100M、8 卡 FSDP 复验；
- `quant_fm/pretrain/config_v2_230m.yaml`：显式 `ffn_hidden=2816` 的 Dense V2 候选；
- `quant_fm/pretrain/config_v2_backbone_moe.yaml`：顶部 4 层 shared + Top-1 routed expert 实验候选。

v2 checkpoint 使用 `fm_artifact_version=2.0`，固化并校验 schema/vocab 版本、vocab SHA-256、完整有序 FieldSpec、输入/目标字段、loss targets、连续 normalizer、盘口时序、context 和 pooling 版本。各 v2 配置共享 `quant_fm/runs/v2_shared/validation_windows.json`；计划含 manifest fingerprint，数据或窗口参数改变时拒绝复用。

训练状态现在分别记录每 rank `micro_step`、梯度累积边界 `update_step` 和跨 rank 汇总的
`samples_seen/non_pad_tokens_seen`；只有成功参数更新才推进 update/sample/token，FP16
overflow 跳步不计入，token-budget-only 要求显式 `lr_schedule_steps`。LR、日志、验证和
存盘按 update 驱动。`step*.pt`、
`final_resume.pt` 含 optimizer/scaler；`best.pt`、`final.pt` 不含 optimizer，面向评估/推理。
训练入口拒绝用推理文件 resume；`--resume auto` 只找定期 `step*.pt`/`final_resume.pt` 并
优先定期点，继续已完成 run 应显式指定
`final_resume.pt`。

MoE 集成仍有硬边界：Temporal 模块尚未接默认训练/embedding/score；Backbone 训练只接入
合计 auxiliary loss，没有自动记录 expert fraction/entropy/overflow。训练模式发生
capacity overflow 时仍有 batch 容量竞争；评估/推理已禁用容量裁剪、排除 padding token，
并通过低 capacity 的 batch-size independence 测试。训练期 dispatch、路由健康度、吞吐与
收益仍未验证。

现有 `make pilot` 与 MinIO/Medium/300M 脚本仍默认走 v1。v2 的数据生成接口已经实现，但尚未封装为 MinIO 一键任务，也尚未生成正式 25M/100M/230M 或 MoE checkpoint，更没有新的 untouched OOS 结果。

### 6. 测试与文档

新增的重点回归测试覆盖：

- 无未来标签的 score 生成；
- 输出 schema、主键和数值合法性；
- artifact 元数据与权重一致性；
- 训练期/信号期边界和特征顺序校验；
- OOS 增量合并、Ranker 缓存、shard 均衡及安全清理。
- 因果盘口与撤单后订单簿/索引一致性；
- Tokenizer v2、FieldSpec、分层分箱和 v1/v2 artifact 隔离；
- 字段融合、多任务 Loss、固定验证和字段诊断；
- 跨 chunk/多尺度池化、日内聚合和跨股票 PIT/因果性。
- micro/update/token 状态恢复、shard-aware sampling、RoPE cache、causal fast path 和
  inference checkpoint；
- Temporal/Backbone MoE 的基础路由、梯度、padding 排除和评估模式 batch-size independence；
- 单步 CPU 训练 smoke、成功 update 计数、resume/inference checkpoint 分流。

相关测试文件为：

```text
tests/test_signal_generation.py
tests/test_signal_schema.py
tests/test_oos_incremental.py
tests/test_book_state_causality.py
tests/test_tokenizer_v2.py
tests/test_pretrain_v2_integration.py
tests/test_hierarchical_pooling.py
tests/test_cross_asset_dataset_model.py
tests/test_training_schedule.py
tests/test_shard_aware_sampler.py
tests/test_attention_fast_path.py
tests/test_rope_cache.py
tests/test_inference_checkpoint.py
tests/test_train_smoke.py
tests/test_moe_router.py
tests/test_moe_causality.py
tests/test_backbone_moe.py
```

## 建议验证方式

```bash
# 快速验证完整的无标签信号路径
make signal-smoke

# 运行信号与 OOS 专项测试
uv run pytest -q \
  tests/test_signal_generation.py \
  tests/test_signal_schema.py \
  tests/test_oos_incremental.py

# 运行全量回归
make test
```

当前全仓基线（2026-07-24）：`243 passed, 2 skipped, 1 xfailed`。该结果只证明已覆盖的工程接口与因果/兼容不变量；MoE 评估模式已覆盖低 capacity 的 batch-size independence，但训练期 overflow 尚未做真实负载验证。正式 v2 数据 coverage、25M/100M/230M 与 MoE 训练、盘口快照逐档一致率、路由/吞吐、下游增益和新 OOS 仍需单独验收。

## 与其他分支的区别

- 相对 `main`：增加并行/OOS/300M 工程能力、独立 score-only 生产接口、模型底层 v2 和 MoE 研究组件。
- 相对 `feat/score-signal-only`：保留相同生产信号契约，新增训练计数/性能快路径、Dense 230M 候选、Temporal Regime-MoE 与顶部 Backbone-MoE；这些新增项尚未获得收益或生产证据。
- 相对 `ParallelVersion`：保留并强化 `quant_fm/signal/`、无标签打分、冻结 Ranker artifact 和严格交付 schema；`ParallelVersion` 的最终代码树刻意不包含这些生产信号模块。
