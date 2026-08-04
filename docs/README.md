# QuantFM 文档索引

## 阅读入口

第一次阅读建议按以下顺序：

1. [根 README](../README.md)：项目目标、架构、快速验证和目录结构；
2. [复现与验证指南](REPRODUCIBILITY.md)：建议阅读路径、复现命令和实验边界；
3. [项目与代码阅读指南](QuantFM.md)：面向新读者的模块与概念说明；
4. [阶段进展](project/阶段进展.md)：当前实验与工程进展。

## 目录结构

| 目录 | 内容 |
|------|------|
| [`pipeline/`](pipeline/README.md) | 从 MinIO 数据接入到下游验收的逐阶段文档 |
| [`architecture/`](architecture/) | V2、MoE、Tokenizer、Loss 与模型底层设计 |
| [`data/`](data/) | MinIO、原始 L2、events 和 tokens 数据工作流 |
| [`evaluation/`](evaluation/) | OOS、信号交付、回测契约与联调 |
| [`operations/`](operations/) | CPU/GPU Kubernetes、PVC、调度评估和训练 runbook |
| [`project/`](project/) | 分支说明、阶段进展和执行计划 |
| [`research/`](research/) | 调研材料与历史方案 |
| `assets/` | 文档图片和评估证据；由对应正文引用，不作为独立阅读入口 |
| `handout-khalil-gpu/` | GPU 集群接入材料，包含访问配置和证书，仅限授权人员使用 |

## Pipeline 逐阶段文档

| 阶段 | 文档 | 主要内容 |
|------|------|----------|
| 总览 | [pipeline/README.md](pipeline/README.md) | 全链路、运行模式、目录与闸门 |
| 1 | [MinIO 数据接入](pipeline/01_minio_io.md) | 读写分离、对象布局、权限验证 |
| 2 | [订单簿重建](pipeline/02_order_book_rebuild.md) | 沪深撮合、回放与正确性检查 |
| 3 | [事件规范化](pipeline/03_canonical_events.md) | `cn_l2_v1/v2` schema、因果盘口字段与股日分片 |
| 4 | [Tokenizer 与词表](pipeline/04_tokenizer_vocab.md) | 因果特征、分位数分箱、防泄漏 |
| 5 | [Manifest 与时间切分](pipeline/05_manifest_splits.md) | 分片哈希、train/val/test |
| 6 | [OrderFlow FM 预训练](pipeline/06_pretraining.md) | 模型、FSDP、checkpoint 续训与监控 |
| 7 | [股日 Embedding](pipeline/07_embeddings.md) | 冻结模型、分块编码与池化 |
| 8 | [下游验收](pipeline/08_downstream_evaluation.md) | Ranker、RankIC、CPCV、DSR、严格执行面板与研究回测 |
| 补充 | [OOS 加速与增量交付](pipeline/08_oos_acceleration.md) | 增量执行、缓存、并行与研究隔离 |

以上阶段文档同时标注 V1 兼容路径与 V2 默认路径。MinIO/Pilot/Medium 一键编排现已接入真实盘口 V2 数据生成与合约审计；正式全市场 artifact、checkpoint 与 OOS score 仍待真实运行和验收。

## 数据与存储

| 文档 | 说明 |
|------|------|
| [MinIO 读写指南](data/minio_setup.md) | endpoint、bucket、凭据和排错 |
| [MinIO 数据工作流](data/minio_data_workflow.md) | 清洗、上传、恢复和磁盘策略 |
| [原始 L2 到 events/tokens](data/raw_to_events_tokens.md) | 字段级转换细节 |
| 进度查询 | `uv run python -m quant_fm.scripts.check_pipeline_progress` |

## 评估与回测

| 文档 | 说明 |
|------|------|
| [严格 OOS 研究回测](evaluation/严格OOS研究回测.md) | ReturnSpec、未来日历、冻结 score 评估与结果边界 |
| [信号回测对接](evaluation/信号回测对接文档.md) | 生产 `date/symbol/score` 交付契约 |
| [下游回测对接](evaluation/下游回测对接说明.md) | execution panel、成本与回测输入输出 |
| [回测接口契约与小样本联调](evaluation/回测接口契约与小样本联调.md) | 最小交付样本、接口约束与联调验收 |

## 模型架构设计

| 文档 | 说明 |
|------|------|
| [V1 / V2 完整差异总览](architecture/V1-V2完整差异总览.md) | V1/V2 全链路对照，重点展开 Token、Loss、两类 MoE 与当前实证边界 |
| [V2 性能与 Regime-MoE 代码改造方案](architecture/QuantFM-V2-性能与Regime-MoE代码改造方案.md) | 后续性能、Dense V2/Regime-MoE 路线；其中部分阶段仍是规划，不等同于当前实现 |
| [模型底层 V2 代码改造指导](architecture/模型底层v2代码改造指导.md) | 本次已实现的盘口、Tokenizer、字段融合、Loss、池化与跨股票上下文设计依据 |
| [MoE 架构完整结构](architecture/MOE架构完整结构.md) | Backbone-MoE 组件、路由和张量流 |
| [Token 修改完整方案](architecture/Token修改完整方案.md) | V1/V2 隔离、字段编码和 artifact 约束 |
| [Loss 函数改进与已训练模型](architecture/Loss函数完整改进与已训练模型.md) | Loss 演进、已有实验和模型说明 |
| [Loss 总结](architecture/Loss总结.md) | 排序、辅助与多期限 Loss 的代码口径 |
| [Top 选股 Loss 最终方案](architecture/Top选股Loss最终方案.md) | Top-K 排序目标、数据血统和验收方案 |

## V2 代码入口

| 模块 | 主要入口 |
|------|----------|
| 盘口与 schema | [`order_book/pylob/book_state.py`](../order_book/pylob/book_state.py)、[`quant_fm/schema/cn_l2_v2.py`](../quant_fm/schema/cn_l2_v2.py) |
| Tokenizer v2 | [`field_spec.py`](../quant_fm/tokenizer/field_spec.py)、[`fit_bins_v2.py`](../quant_fm/tokenizer/fit_bins_v2.py)、[`tokenize_events_v2.py`](../quant_fm/tokenizer/tokenize_events_v2.py) |
| 模型与 Loss | [`field_fusion.py`](../quant_fm/pretrain/field_fusion.py)、[`heads.py`](../quant_fm/pretrain/heads.py)、[`train.py`](../quant_fm/pretrain/train.py) |
| 固定验证 | [`validation_sampler.py`](../quant_fm/pretrain/validation_sampler.py)、[`eval.py`](../quant_fm/pretrain/eval.py) |
| 股日与跨股票表示 | [`embedding/`](../quant_fm/embedding/)、[`cross_asset/`](../quant_fm/cross_asset/) |
| 训练配置 | [`config_v2_25m.yaml`](../quant_fm/pretrain/config_v2_25m.yaml)、[`config_v2_100m.yaml`](../quant_fm/pretrain/config_v2_100m.yaml)、[`config_v2_230m.yaml`](../quant_fm/pretrain/config_v2_230m.yaml)、[`config_v2_backbone_moe.yaml`](../quant_fm/pretrain/config_v2_backbone_moe.yaml) |

v2 checkpoint 加载必须携带原始 `vocab_v2.json`，并核对 schema、vocab SHA-256、有序 FieldSpec、输入/目标字段和 loss 声明。25M/100M 比较应复用同一份带 manifest fingerprint 的 `validation_windows.json`。

当前全仓回归基线（2026-08-03）：

```bash
uv run python -m pytest -q
# 545 passed, 2 skipped, 1 xfailed
```

## 包级说明

| 文档 | 说明 |
|------|------|
| [OrderFlow FM](../quant_fm/README.md) | 模型与常用 Make 入口 |
| [PyLOB](../order_book/README.md) | 撮合引擎子项目与公共 API |

## 运维与集群

| 文档 | 说明 |
|------|------|
| [GPU 集群训练后端选型报告（最终版）](GPU集群训练后端执行选型评估报告.md) | 当前决策入口；包含 Native K8s、Kueue、Volcano 实测结果以及 LGB、NN、Transformer 的后端选择 |
| [GPU 调度功能实测记录（2026-07-31）](operations/GPU调度功能实测记录-2026-07-31.md) | ResourceQuota、Kueue 队列与抢占、Volcano Queue/Gang 的详细执行记录 |
| [GPU 调度与存储早期调研](operations/GPU-K8S-Kueue-Volcano-调度与存储评估报告.md) | 历史材料，保留早期权限和组件状态；当前结论以最终版报告为准 |
| [CPU K8s 使用手册](operations/CPU-K8S-完整使用手册-khalil.md) | CPU 集群接入、Job、日志和排障 |
| [GPU K8s 使用与测试手册](operations/GPU-K8S-集群使用与测试手册-khalil.md) | GPU Job 规范、验证和故障定位 |
| [GPU PVC 申请与迁移](operations/GPU-PVC-申请与迁移-khalil.md) | 存储申请、确认与迁移清单 |
| [300 日 MoE 磁盘安全训练手册](operations/CODEX_MOE_300D_STORAGE_SAFE_RUNBOOK.md) | 容量闸门、checkpoint 轮转和恢复流程 |
| [GPU 集群群公告文案](operations/GPU-K8S-群公告文案.md) | 面向使用者的简明规则 |

## 项目记录与研究材料

这些文档提供背景，不作为当前代码接口的唯一事实来源：

- [当前分支工作说明](project/BRANCH_WORK.md)
- [项目进展与规划](project/项目进展与规划.md)
- [Dense230M 训练期间并行工作计划](project/Dense230M训练期间并行工作计划.md)
- [项目阶段进展](project/阶段进展.md)
- [基于 LLM 的端到端量化策略调研](research/基于LLM的端对端量化策略调研_7.6Update.md)

当历史文档与代码不一致时，以根 README、`docs/pipeline/` 和当前配置文件为准。
