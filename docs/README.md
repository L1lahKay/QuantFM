# QuantFM 文档索引

## 审阅入口

第一次阅读建议按以下顺序：

1. [根 README](../README.md)：项目目标、架构、快速验证和目录结构；
2. [Reviewer 审阅指南](REVIEW.md)：建议阅读路径、复现命令和验收边界；
3. [项目与代码阅读指南](QuantFM.md)：面向新读者的模块与概念说明；
4. [阶段进展汇报](阶段进展汇报_MinIO与模型跑通.md)：当前实验与工程进展。

## Pipeline 逐阶段文档

| 阶段 | 文档 | 主要内容 |
|------|------|----------|
| 总览 | [pipeline/README.md](pipeline/README.md) | 全链路、运行模式、目录与闸门 |
| 1 | [MinIO 数据接入](pipeline/01_minio_io.md) | 读写分离、对象布局、权限验证 |
| 2 | [订单簿重建](pipeline/02_order_book_rebuild.md) | 沪深撮合、回放与正确性检查 |
| 3 | [事件规范化](pipeline/03_canonical_events.md) | `cn_l2_v1` schema 与股日分片 |
| 4 | [Tokenizer 与词表](pipeline/04_tokenizer_vocab.md) | 因果特征、分位数分箱、防泄漏 |
| 5 | [Manifest 与时间切分](pipeline/05_manifest_splits.md) | 分片哈希、train/val/test |
| 6 | [OrderFlow FM 预训练](pipeline/06_pretraining.md) | 模型、FSDP、checkpoint 与监控 |
| 7 | [股日 Embedding](pipeline/07_embeddings.md) | 冻结模型、分块编码与池化 |
| 8 | [下游验收](pipeline/08_downstream_evaluation.md) | Ranker、RankIC、CPCV、DSR、回测 |

## 工程操作

| 文档 | 说明 |
|------|------|
| [MinIO 读写指南](minio_setup.md) | endpoint、bucket、凭据和排错 |
| [MinIO 数据工作流](minio_data_workflow.md) | 清洗、上传、恢复和磁盘策略 |
| [原始 L2 到 events/tokens](raw_to_events_tokens.md) | 字段级转换细节 |

## 包级说明

| 文档 | 说明 |
|------|------|
| [OrderFlow FM](../quant_fm/README.md) | 模型与常用 Make 入口 |
| [PyLOB](../order_book/README.md) | 撮合引擎子项目与公共 API |

## 研究材料与历史记录

这些文档提供背景，不作为当前代码接口的唯一事实来源：

- [基于 LLM 的端到端量化策略调研](基于LLM的端对端量化策略调研_7.6Update.md)
- [MinIO 与模型跑通阶段汇报](阶段进展汇报_MinIO与模型跑通.md)

当历史文档与代码不一致时，以根 README、`docs/pipeline/` 和当前配置文件为准。
