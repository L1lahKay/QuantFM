# OOS 流水线加速与流式交付

## 目标

在不停止 MinIO → token 主生产进程的情况下，降低 CPU、内存和磁盘峰值，并让
OOS score 真正按新增日期增量更新。

## 已实施优化

1. **Ranker checkpoint 缓存**：2025 训练期不变时只训练一次；后续加载
   `delivery_oos/ranker_checkpoint.pt`。
2. **增量 score**：`score_state.json` 记录已经处理的 embedding 日期，只预测新日期；
   输入未变化时整次调用为 O(1) no-op。
3. **流式 token 释放**：存在 `data/.prune_embedded_tokens` 时，只有在 embedding、score
   和状态文件均成功落盘后，才删除对应日期 tokens；每个日期保留审计 receipt。
4. **日期 shard index**：tokenize 完成时生成 `data/shard_index/<date>.json`，增量
   manifest 不再反复遍历整个目录树。
5. **低内存自动降卡**：主机 `MemFree < 32 GiB` 时，embedding 从 8 进程降为 4 进程，
   避免 OOM killer。
6. **按 token 行数均衡 GPU 分片**：替代按文件数 stride，减少活跃股票造成的尾部等待。
7. **融合 canonicalize + tokenize**：冻结 vocab 的 OOS 路径直接从 clean events 写 tokens，
   不再落中间 canonical parquet。
8. **原生 symbol predicate**：源字段为 String 时不再执行 `cast().zfill()`，允许 parquet
   reader 尽可能进行 predicate pushdown。
9. **失败显式化**：失败标的自动重试一次；仍失败则写 `data/.failed/<date>.json`，日期
   marker 标为 `tokenized_with_gaps`，不再静默丢失。

## 运行时文件

```text
delivery_oos/
  ranker_checkpoint.pt
  ranker_metadata.json
  score_state.json
  scores/oos.parquet

data/
  .prune_embedded_tokens
  shard_index/<date>.json
  pruned_token_receipts/<date>.json
  .failed/<date>.json
```

## 安全语义

- token 仅在对应 embedding 已进入 `oos_all.parquet`、score 写入成功后清理；
- receipt 记录删除前的文件数、总字节数和 embedding 文件指纹；
- Ranker 缓存键包含训练 embedding、训练 panel 及训练参数；任一变化都会自动重训；
- OOS panel 或 Ranker 变化会使增量 score 状态失效并全量重算；
- 流式清理模式不生成不完整的最终 token manifest。

## 后续高成本方向

若还需要数量级提升，应把 Python 逐事件撮合核心迁移到 Rust/PyO3、Cython 或 Numba，
并推动 MinIO 原始数据按 symbol 分区或按 symbol 聚簇 row group。两项都需要独立的逐事件
等价性回归和性能基准，不应在正在运行的 OOS 作业中热切换。
