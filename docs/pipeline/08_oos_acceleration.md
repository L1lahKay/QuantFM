# OOS 流水线加速、增量交付与研究隔离

## 1. 目标

在不停止 MinIO → event → token 主生产进程的前提下：

- 对已完成的新交易日增量抽取 embedding；
- 冻结历史期 Ranker，对 OOS embedding 生成无标签 score；
- 降低 CPU、主存、GPU 和磁盘峰值；
- 保持生产三列 score 与研究 execution panel 完全隔离。

## 2. 生产入口与增量运维入口

正式 Top-K 重训/评估只使用 `run_dense230m_strict_oos.sh`。该入口先跑
预训练验收及 `validate_pretrain_lineage`，把 candidate/baseline 报告、FM checkpoint、
manifest/vocab 和两期 embedding 串成同一条可哈希血缘；随后再跑
`preflight_topk_ranker`，强制校验新版因果 embedding、训练与评分期 PIT 股票池、
完整 T+2 交易日历及它们的契约指纹。下面两个入口用于通过严格预检后的缓存/增量运维，
不会把旧 V1 产物升级成 V2。

### 2.1 缓存化 OOS delivery builder

`quant_fm.scripts.build_oos_delivery` 在一个输出目录内管理 Ranker 缓存、已打分日期和
三列交付：

```bash
uv run python -m quant_fm.scripts.build_oos_delivery \
  --train-emb-dir /new/v2/train_embeddings \
  --train-panel /new/v2/train_execution_panel.parquet \
  --train-calendar /data/train_calendar_through_t2.txt \
  --train-universe /data/train_pit_universe.parquet \
  --test-emb /new/v2/oos_embeddings/all.parquet \
  --test-universe /data/oos_pit_universe.parquet \
  --pretrain-acceptance /new/v2/pretrain/acceptance.json \
  --out-dir /new/v2/delivery_oos \
  --epochs 30 \
  --seed 42 \
  --device cuda:0 \
  --min-names-per-day 350
```

`--test-panel` 是已弃用的兼容参数；生产打分不读 OOS 未来标签。
正式模式不再把人工填写的 FM 截止日当作证明，而是从已验收 checkpoint 的
manifest 训练日期和 vocab 拟合日期派生较晚值。`--fm-training-end-date` 仅保留为可选的
人工断言，若与派生值不一致会直接失败。acceptance v2 同时绑定 candidate/baseline
评估报告的完整 SHA-256，来源文件变化后原 PASS 自动失效。

缓存键包含预训练血缘、训练 embedding/panel/PIT universe/交易日历指纹、表示契约、
损失配置、epochs、seed 和 `min_names_per_day`。这些输入不变时复用
`ranker_checkpoint.pt`；OOS embedding
增加新日期时，只对 `score_state.json` 未记录的日期出分，再与旧 score 去重合并。
完全无变化时是快速 no-op。

### 2.2 按完成日轮询的编排器

```bash
nohup bash quant_fm/scripts/run_oos2026_incremental.sh \
  > quant_fm/runs/oos2026/incremental.log 2>&1 &
```

该脚本以 `data/.done/<date>` 内的 `tokenized` 标记为整日完成条件，仅抽取
`embedded_dates.txt` 中没有的新日期，累积到 `embeddings/incr/oos_all.parquet`。
历史期 `signal.train` 只在 artifact 不存在时运行；每轮用 `signal.generate` 原子更新
`delivery_oos/scores.parquet` 和 `signal_manifest.json`。

这条 shell 编排路径和上节的 `build_oos_delivery` 可二选一。不要假设前者会生成
`score_state.json`；该文件属于后者的内部增量状态。

## 3. 已实施的加速和安全措施

1. **Ranker 缓存**：历史训练输入和参数不变时不重训。
2. **增量 score**：缓存化 builder 只预测新 embedding 日期，并按
   `(date, symbol)` 去重、稳定排序。
3. **原子交付**：先写临时 parquet/JSON，成功后替换正式文件。
4. **流式 token 释放**：缓存化 builder 发现 `data/.prune_embedded_tokens` 时，
   只有在 embedding、score 和状态持久化后才删除已消费日期的 token，
   并写删除 receipt。
5. **日期 shard index**：`data/shard_index/<date>.json` 让 adhoc manifest 优先按日期查找，
   避免每轮全树扫描；索引不存在时仍有目录扫描回退。
6. **低内存降卡**：`extract_embeddings_parallel.sh` 在 `MemFree < 32 GiB` 且请求进程过多时，
   将 `NPROC` 降到 `LOW_MEM_NPROC=4`。
7. **GPU 互斥**：`.score.lock` 避免 GPU 0 上的 Ranker 打分与多卡 FM 抽取争抢显存。
8. **按 token 行数均衡分片**：减少高活跃股票导致的 GPU 尾部等待。
9. **canonicalize + tokenize 融合**：冻结 vocab 的 OOS 路径可直接从 clean event
   写 token，不保留不必要的中间 parquet。
10. **失败显式化**：失败标的重试后写 `data/.failed/<date>.json`，整日 marker 可标记为
    `tokenized_with_gaps`；后续审计必须将该日视为带明示缺口的覆盖。
11. **订单簿热路径降复杂度**：连续竞价只读取当前最优价，不再为每笔委托复制全部价格档；
    集合竞价使用累计量一次扫描候选价，不再对每个价格重复扫描整本簿。
12. **Embedding fused fast path**：相同长度 chunk 组 batch，避免 padding 让 SDPA 退出 fused
    causal 路径；pooling 和 stride 默认从 checkpoint 冻结契约读取。V2 Top-K
    使用 `hierarchical_selected_v2` 与因果重叠窗口，旧 checkpoint 仍按自身的 mean/独立分块语义复现。
13. **Embedding 内容缓存**：part sidecar 记录 checkpoint SHA-256、token shard 指纹和全部推理
    参数；完全匹配时在加载模型前直接复用，参数或 checkpoint 改变时自动失效。
14. **候选模型快速横截面**：`MAX_DATES=N` 会在开发 split 等距选择 N 个完整交易日，
    保留每日全股票横截面。此时 `run_judge_300m_fast.sh` 默认 `DEV_ONLY=1`，只读
    train/val、不碰 test，并把 embedding 与报告写入独立 quick 目录，不会覆盖全量结果。

模型迭代应复用与参数无关的冻结 token/manifest，不要在每个模型目录重复运行 MinIO → LOB
replay。示例：

```bash
DATA_WORKDIR=quant_fm/runs/cont60 \
WORKDIR=quant_fm/runs/candidate_01 \
CKPT=quant_fm/runs/candidate_01/run/best.pt \
PANEL=quant_fm/runs/cont60/panel/daily_panel_cont60.parquet \
MAX_DATES=8 EPOCHS=10 NPROC=8 BATCH=16 \
bash quant_fm/scripts/run_judge_300m_fast.sh
```

同一 checkpoint 重跑会验证并复用 embedding part。候选胜出后将 `MAX_DATES=0` 跑一次正式
全量验收，此时才读取 test；快速横截面只用于筛选，不能替代最终 test 结论。

## 4. 文件契约

```text
delivery_oos/
├── scores.parquet              # 唯一主数据交付；严格 date,symbol,score
├── signal_manifest.json       # 交付元数据
├── ranker_checkpoint.pt       # build_oos_delivery 内部缓存
├── ranker_metadata.json       # build_oos_delivery 内部缓存元数据
├── score_state.json           # build_oos_delivery 内部增量状态
└── .score.lock                # 短命锁，异常路径也会释放

embeddings/incr/
├── embedded_dates.txt         # shell 轮询路径的已抽日期
└── oos_all.parquet            # shell 轮询路径的累计 embedding

data/
├── .done/<date>
├── .failed/<date>.json
├── .prune_embedded_tokens
├── shard_index/<date>.json
└── pruned_token_receipts/<date>.json
```

对外交付时仍只发布 `scores.parquet` 和 `signal_manifest.json`。Ranker 缓存、增量状态、
embedding、token receipt 和日志都是内部运维产物。

## 5. 不变式与失效规则

- OOS score 生成不读取 OOS panel、`fwd_ret` 或任何后验可交易标志。
- `scores.parquet` 必须通过 `validate_scores()`：三列、无 null/Inf/NaN、主键唯一。
- Ranker 缓存键变化时全量重训 Ranker，并不复用不兼容的已有 score 状态。
- OOS embedding 文件指纹不变时可 no-op；新日期只追加新 score。
- token receipt 记录删除前的 shard 数、字节数和 embedding 文件指纹。
- 流式清理会使全量 token 树不完整，因此编排器在该模式下不构建伪完整 manifest。

## 6. 生产完成后的研究评估

研究面板必须在生产 score 冻结后独立构建，不反向改写 score 或 `score_state.json`：

```bash
CALENDAR=/path/to/calendar_with_future_days.txt \
RETURN_SPEC=vwap_t1_vwap_t2 \
make research-oos2026
```

2026-60d 的产物如今已被用于架构验证，可继续用来对账加速前后的确定性和回归，
但不能重新命名为 v2 的 untouched OOS。

## 7. 后续高成本方向

若还需数量级提升，可将 Python 逐事件订单簿重建迁移到 Rust/PyO3、Cython 或 Numba，
并推动 MinIO 原始数据按 symbol 分区或按 symbol 聚簇 row group。两项都必须先通过
逐事件等价性回归和性能基准，不在正在运行的 OOS 作业中热切换。
