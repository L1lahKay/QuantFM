# `feat/score-signal-only` 分支工作说明

本文档根据本分支与 `main` 分支的代码差异整理，说明该分支的目标、主要改动、使用入口和验收边界。

## 分支定位

本分支在 `main` 的端到端订单流基础模型流水线上，补齐了大规模并行处理、样本外（OOS）运行和下游评估能力，并进一步建立了独立的生产信号边界：生产侧只接收冻结的股日 embedding 和冻结 Ranker，最终只交付 `date, symbol, score` 及可追溯清单。

核心原则如下：

- 训练需要的未来收益标签与生产打分彻底分离；
- embedding 是内部中间产物，`score` 是唯一对外交付；
- Ranker 权重、特征顺序和训练截止日均随 artifact 固化；
- 信号日期默认必须晚于 Ranker 的训练截止日；
- 回测、持仓、成本和风险指标继续留在 research-only 下游模块。

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
- 连续样本与 OOS 样本的增量交付；
- Ranker 缓存、并发锁、增量合并和失败后的锁清理；
- 研究评估与生产 score 接口的文档化边界。

`make judge-300m`、`make watch-300m` 和 `make baselines-300m` 属于研究评估入口，不属于线上 score 推理的必要依赖。

### 5. 测试与文档

新增的重点回归测试覆盖：

- 无未来标签的 score 生成；
- 输出 schema、主键和数值合法性；
- artifact 元数据与权重一致性；
- 训练期/信号期边界和特征顺序校验；
- OOS 增量合并、Ranker 缓存、shard 均衡及安全清理。

相关测试文件为：

```text
tests/test_signal_generation.py
tests/test_signal_schema.py
tests/test_oos_incremental.py
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

## 与其他分支的区别

- 相对 `main`：增加并行/OOS/300M 工程能力和独立的 score-only 生产接口。
- 相对 `ParallelVersion`：保留并强化 `quant_fm/signal/`、无标签打分、冻结 Ranker artifact 和严格交付 schema；`ParallelVersion` 的最终代码树刻意不包含这些生产信号模块。
