# QuantFM 文档索引

## 阅读入口

第一次阅读建议按以下顺序：

1. [根 README](../README.md)：项目目标、架构、快速验证和目录结构；
2. [复现与验证指南](REPRODUCIBILITY.md)：建议阅读路径、复现命令和实验边界；
3. [项目与代码阅读指南](QuantFM.md)：面向新读者的模块与概念说明；
4. [阶段进展](阶段进展.md)：当前实验与工程进展。

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

以上阶段文档同时标注 v1 稳定路径与 v2 新路径。现有 MinIO/Pilot 一键编排默认仍走 v1；v2 的 schema、词表、训练和表征代码已经落地，但正式 v2 数据 artifact、checkpoint 与 OOS 结果仍待生成。

## 工程操作

| 文档 | 说明 |
|------|------|
| [MinIO 读写指南](minio_setup.md) | endpoint、bucket、凭据和排错 |
| [MinIO 数据工作流](minio_data_workflow.md) | 清洗、上传、恢复和磁盘策略 |
| [原始 L2 到 events/tokens](raw_to_events_tokens.md) | 字段级转换细节 |
| [严格 OOS 研究回测](严格OOS研究回测.md) | ReturnSpec、未来日历、冻结 score 评估与结果边界 |
| [信号回测对接](信号回测对接文档.md) | 生产 `date/symbol/score` 交付契约 |
| [下游回测对接](下游回测对接说明.md) | execution panel、成本与回测输入输出 |
| 进度查询 | `uv run python -m quant_fm.scripts.check_pipeline_progress` |

## 模型架构设计

| 文档 | 说明 |
|------|------|
| [V2 性能与 Regime-MoE 代码改造方案](QuantFM-V2-性能与Regime-MoE代码改造方案.md) | 后续性能、Dense V2/Regime-MoE 路线；其中部分阶段仍是规划，不等同于当前实现 |
| [模型底层 V2 代码改造指导](模型底层v2代码改造指导.md) | 本次已实现的盘口、Tokenizer、字段融合、Loss、池化与跨股票上下文设计依据 |

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

当前全仓回归基线（2026-07-24）：

```bash
uv run python -m pytest -q
# 243 passed, 2 skipped, 1 xfailed
```

## 包级说明

| 文档 | 说明 |
|------|------|
| [OrderFlow FM](../quant_fm/README.md) | 模型与常用 Make 入口 |
| [PyLOB](../order_book/README.md) | 撮合引擎子项目与公共 API |

## 研究材料与历史记录

这些文档提供背景，不作为当前代码接口的唯一事实来源：

- [基于 LLM 的端到端量化策略调研](基于LLM的端对端量化策略调研_7.6Update.md)
- [项目阶段进展](阶段进展.md)

当历史文档与代码不一致时，以根 README、`docs/pipeline/` 和当前配置文件为准。
