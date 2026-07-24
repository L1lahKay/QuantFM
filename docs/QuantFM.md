# QuantFM 

> 完整工程步骤见 [Pipeline 逐阶段文档](pipeline/README.md)；复现方法与验证边界见 [复现与验证指南](REPRODUCIBILITY.md)。

---

## 0. 这个项目在做什么？

一句话：**把 A 股 Level-2 订单流变成整数序列，用 Transformer 做「预测下一个市场事件」的预训练，再抽出 embedding 给下游选股模型用。**

当前同时保留两条路径：

- **V1 稳定主链**：`cn_l2_v1` + `vocab.json` + 直接字段 embedding 求和，现有 MinIO / smoke / 302M 产物继续使用该路径。
- **V2 研究链**：加入因果盘口状态、版本化 `FieldSpec`、token+scalar、可配置字段融合、entropy-normalized/ordinal loss、多尺度聚合和股票间上下文模块。底层代码、训练性能改造、25M/100M/230M dense 配置，以及实验性的 Regime/Backbone-MoE 模块与配置已经入库；这些只代表工程候选已就绪，尚未完成多 seed 训练和新的 untouched OOS 验收。

详细的实施边界见 [模型底层 V2 代码改造指导](./模型底层v2代码改造指导.md)；严格研究评估见 [严格 OOS 研究回测](./严格OOS研究回测.md)。

项目由两个 workspace 子项目组成：


| 目录                  | 作用                         | 是否需要懂深度学习       |
| ------------------- | -------------------------- | --------------- |
| `quant_fm/`         | 词表、训练、embedding、回测         | 需要一点 PyTorch 概念 |
| `order_book/pylob/` | 沪深订单簿重建（撮合引擎，包名仍为 `pylob`） | 否，先当黑盒          |


---

## 1. 先搞懂 5 个 Python 概念

读代码时你会反复遇到这些：

### 1.1 `from xxx import yyy`

从别的文件「借」函数/类。例如：

```python
from quant_fm.pretrain.train import train
```

表示：去 `quant_fm/pretrain/train.py` 里找名叫 `train` 的函数。

### 1.2 `def 函数名(...):`

定义一段可重复调用的逻辑。函数名后面带 `_` 开头（如 `_stage_events`）通常表示「内部用，别在外部直接调」。

### 1.3 `class 类名:`

把数据和操作打包在一起。例如 `EventWindowDataset` 是一个「数据集」类，PyTorch 会调用它的 `__getitem__` 取一条训练样本。

### 1.4 `Path` 与 parquet

- `Path("a/b/c")`：跨平台的路径对象，`/` 拼接文件夹。
- `.parquet`：列式表格文件，类似 Excel 但适合大数据；用 `polars.read_parquet` 读取。



### 1.5 PyTorch 张量 `Tensor`

可以想成「带 shape 的多维数组」：

- `[B, L]`：B 条序列，每条长 L
- `[B, L, d_model]`：每条序列每个位置有一个 d_model 维向量

---



## 2. 数据流水线（最重要的一张图）

```
【读】MinIO :9000  zeus-cn-quote  原始 L2
    ↓  pylob 清洗（Polars 流式读，examples/run_zeus_clean.py）
events.parquet → cn_l2_v1 规范 events
    ↓  fit_bins（仅训练日）→ tokenize → manifest
【写】MinIO :9100  model-cache  tokens/ + vocab.json + manifest.json
    ↓  （可选）预训练 quant_fm/pretrain/
模型训练 → final.pt
    ↓  quant_fm/embedding/
每股每日 embedding
    ↓  quant_fm/downstream/
截面排序 + 回测
```

MinIO 读写配置与命令详见 **[minio_setup.md](./minio_setup.md)**。  
一键读→处理→写：`make minio-pipeline`（无训练）。  
完整读→tokens→写→训练：`make minio-full-pipeline` / `make minio-full-pipeline-full`。

**新手第一遍阅读顺序**：`smoke.py` → `dataset.py` → `model.py` → `heads.py` → `train.py`。  
真实 MinIO 数据如何生成 events / tokens，见 **[raw_to_events_tokens.md](./raw_to_events_tokens.md)**。  
MinIO **读写**配置与命令，见 **[minio_setup.md](./minio_setup.md)**。

运行一遍（无需 MinIO）：

```bash
source .venv/bin/activate
python -m quant_fm.scripts.smoke --workdir quant_fm/runs/my_smoke
```

看到最后一行 `SMOKE OK` 即表示全流程打通。

---



## 3. 核心文件速查表



### 3.1 入口脚本


| 文件                                            | 干什么                              |
| --------------------------------------------- | -------------------------------- |
| `quant_fm/scripts/smoke.py`                   | 合成数据跑通全流程（**最佳入门入口**）            |
| `quant_fm/scripts/run_pilot.py`               | 真实 MinIO 数据试点（读 :9000）           |
| `quant_fm/scripts/run_minio_data_pipeline.sh` | **MinIO 读→写**（无训练）               |
| `quant_fm/scripts/run_minio_full_pipeline.sh` | **MinIO 读→tokens→写→8卡训练**（完整流水线） |
| `quant_fm/scripts/run_minio_300m_pipeline.sh` | **~302M**：22 日全市场 + 并行清洗 + 断点续跑 + `--resume auto` |
| `quant_fm/scripts/download_from_minio.py`     | 从 model-cache 拉回 tokens          |
| `quant_fm/scripts/minio_config.py`            | 读写 endpoint / bucket 默认值         |
| `quant_fm/scripts/upload_to_minio.py`         | 上传 tokens 到 model-cache（写 :9100） |
| `docs/minio_setup.md`                         | **MinIO 读写完整文档**                 |
| `quant_fm/pretrain/train.py`                  | 预训练（含 FSDP、checkpoint 续训）        |
| `examples/run_zeus_clean.py`                  | 只负责 pylob 清洗                     |
| `quant_fm/scripts/run_oos2026_research.sh`    | **RESEARCH ONLY**：构建 execution panel 并评估冻结 score |




### 3.2 数据与词表


| 文件                                      | 干什么            |
| --------------------------------------- | -------------- |
| `quant_fm/schema/cn_l2_v1.py`           | 事件长什么样（22 列）   |
| `quant_fm/tokenizer/vocab.py`           | 词表结构、PAD=0     |
| `quant_fm/tokenizer/fit_bins.py`        | **只在训练集**拟合分箱  |
| `quant_fm/tokenizer/tokenize_events.py` | 事件 → tok_* 整数列 |
| `quant_fm/manifest/build_manifest.py`   | 训练文件清单 + 时间切分  |
| `order_book/pylob/book_state.py`        | 事件前/后因果盘口快照       |
| `quant_fm/schema/cn_l2_v2.py`           | 带紧凑盘口特征的 V2 schema    |
| `quant_fm/tokenizer/field_spec.py`      | V2 字段顺序、语义与用途契约    |
| `quant_fm/tokenizer/vocab_v2.py`        | V2 特殊 ID、分箱与标准化 artifact |
| `quant_fm/tokenizer/tokenize_events_v2.py` | V2 事件 → token + scalar 列    |




### 3.3 模型与训练


| 文件                              | 干什么                |
| ------------------------------- | ------------------ |
| `quant_fm/pretrain/dataset.py`  | 把一天的事件切成 2048 长度窗口 |
| `quant_fm/pretrain/model.py`    | Transformer 本体     |
| `quant_fm/pretrain/heads.py`    | 损失函数（预测下一事件）       |
| `quant_fm/pretrain/train.py`    | 训练循环（优化器、存盘、`--resume`） |
| `quant_fm/pretrain/config.yaml` | 超参数配置              |
| `quant_fm/pretrain/config_medium_300m_8gpu.yaml` | ~302M 正式 8 卡配置 |
| `quant_fm/pretrain/dataset_v2.py` | 从 V2 artifact 派生 token/scalar/mask 批次 |
| `quant_fm/pretrain/field_fusion.py` | legacy/scaled/gated/concat 字段融合 |
| `quant_fm/pretrain/validation_sampler.py` | 冻结、分层的验证窗口清单 |
| `quant_fm/pretrain/sampler.py` | shard-aware 分布式训练采样 |
| `quant_fm/pretrain/config_v2_25m.yaml` | V2 Stage-1 小模型消融配置 |
| `quant_fm/pretrain/config_v2_100m.yaml` | V2 Stage-2 复验配置 |
| `quant_fm/pretrain/config_v2_230m.yaml` | V2 dense 放大候选配置（未正式训练验收） |
| `quant_fm/pretrain/config_v2_backbone_moe.yaml` | 顶部 Backbone-MoE 实验配置（未正式训练验收） |
| `quant_fm/benchmark/` | 模型与 embedding 性能基准库 |
| `quant_fm/experiments/registry.py` | 实验配置、环境和结果登记 |
| `quant_fm/moe/` | Temporal Regime-MoE 与 Backbone-MoE 研究实现 |




### 3.4 下游


| 文件                                     | 干什么              |
| -------------------------------------- | ---------------- |
| `quant_fm/embedding/extract_hidden.py` | 冻结模型，抽 embedding |
| `quant_fm/downstream/train_ranker.py`  | 截面排序模型           |
| `quant_fm/downstream/backtest_topk.py` | Top-K 回测         |
| `quant_fm/embedding/pool_stock_day.py` | 跨 chunk 正确池化与多尺度统计 |
| `quant_fm/embedding/intraday_aggregator.py` | 可训练的日内时间聚合器（研究模块） |
| `quant_fm/cross_asset/` | 5 分钟对齐、PIT 行业上下文与 O(N) 股票间模型 |
| `quant_fm/downstream/run_score_evaluation.py` | 冻结 `date,symbol,score` 的独立研究评估 |


---



## 4. 读懂「一个事件」和「一个 token」



### 4.1 一个事件（清洗后）

来自订单簿的一条记录，例如：

- 有人挂单（ADD）
- 有人撤单（CANCEL）
- 有人成交（EXEC / TRADE）

**不**把股票代码、绝对价格原样当 token（会过拟合身份）。

### 4.2 分词后（模型吃的）

每个事件变成多列整数，例如：

```
tok_evt_type=2    # 可能是 EXEC
tok_side=1        # 可能是 B（买）
tok_price_bin=15  # 价格在第 15 个箱子里
...
```

V1 对**每个字段**分别 embedding，再直接相加。V2 仍保留该模式用于兼容，同时可选择 `scaled_sum`、`gated_sum` 或 `concat_mlp`，并将标准化连续 scalar 投影到对应字段。实现见 `pretrain/field_fusion.py` 和 `model.py` 的 `encode()`。

### 4.3 训练任务：next-event

给定前 t 个事件，预测第 t+1 个事件的各字段 —— 类似 GPT 预测下一个词。V1 使用 6 个头的等权 CE；V2 由 `loss.targets` 显式冻结目标、权重、适用性 mask、训练集熵归一化和 ordinal 辅助损失。

---



## 5. 读懂 `train.py` 主循环（7 步）

打开 `quant_fm/pretrain/train.py`，`train()` 函数逻辑：

1. 读 `config.yaml`
2. `Manifest.load` → 知道读哪些 parquet
3. `Vocab.load` → 知道每个字段有多少个 id
4. V1 用 `EventWindowDataset`，V2 用 `EventWindowDatasetV2`，再由 `DataLoader` 按批取数据
5. `OrderFlowFM` → 建模型；若指定 `--resume`/`auto`，只接受含 optimizer state 的
   `step*.pt` / `final_resume.pt` 并恢复训练状态
6. 循环：`logits = model(batch)` → `loss = next_event_loss(...)` → `backward` → `optimizer.step`
7. 定期 `evaluate`，保存两类 checkpoint：`step*.pt` / `final_resume.pt` 含
   optimizer、scaler 与训练计数，用于续训；`best.pt` / `final.pt` 只保留模型、配置、
   元数据与训练计数，面向评估和推理

`best.pt` / `final.pt` 是 inference-only；显式传给 `--resume` 会直接报错，不能用于续训。
`--resume auto` 会先选择编号最大的 `step*.pt`，没有定期 checkpoint 时再尝试
`final_resume.pt`；二者都不存在便记录日志并从头训练。已正常结束的 run 若要续训，建议
显式指定 `final_resume.pt`。

**有效 batch 大小** = `batch_size × grad_accum × world_size`（多卡时）。

---



## 6. 读懂 `dataset.py`：一条训练样本

- **shard** = 一只股票 + 一个交易日 的 token 文件
- **window** = 从 shard 里切出连续 `context`（默认 2048）个事件
- `collate_windows` 把多条 window 拼成一批，短的用 `PAD_ID=0` 补齐

`attention_mask`：True 表示真实事件，False 表示 padding，模型和 loss 都会忽略 padding。

V2 额外读取 `val_*` 连续列，并为每个目标生成 `mask_tok_*`；`PAD_ID=0` 与 `NA_ID=2` 含义不同，不得混用。

---



## 7. 读懂 `model.py`：数据怎么流过网络

```
batch["tok_evt_type"] 等整数（V2 可同时含 val_* scalar）
    → 每字段 Embedding + 可选 scalar projection
    → legacy/scaled/gated/concat 字段融合
    → N 层 Transformer（只能看过去，不能看未来 = 因果）
    → 每个预测字段一个 Linear 头
    → logits["tok_evt_type"]: [B, L, 词表大小]
```

`encode()` 只算隐状态；`forward()` = `encode` + 分类头。

---



## 8. pylob 层（可选深入）

若你想理解「events.parquet 怎么来的」：


| 文件                                         | 作用                          |
| ------------------------------------------ | --------------------------- |
| `order_book/pylob/orderbook_builder_sz.py` | 深圳规则                        |
| `order_book/pylob/orderbook_builder_sh.py` | 上海规则                        |
| `order_book/pylob/pipeline/workflow.py`    | `build_clean_dataset()` 总入口 |
| `order_book/pylob/pipeline/events.py`      | 合并行 → ADD/CANCEL/TRADE 事件流  |


新手阶段：**知道 pylob 输出** `events.parquet` **即可**，不必先读撮合细节。

---



## 9. 配置与目录约定

一次完整试点运行后，目录大致是：

```
quant_fm/runs/pilot/
  clean/          # pylob 原始清洗输出
  events/         # cn_l2_v1 规范事件
  tokens/         # 分词后整数列
  data/
    vocab.json
    manifest.json
  run/
    final.pt      # 训练好的模型
    config.snapshot.yaml
```

V2 必须使用独立目录，例如 `quant_fm/runs/v2_shared/data/` 与 `quant_fm/runs/v2_25m/run/`。不能用 V2 tokenizer 覆盖 V1 `vocab.json`/tokens，也不能用新字段顺序静默加载 V1 checkpoint。

---



## 10. 常见问题

**Q：为什么要 manifest，不能直接扫文件夹？**  
A：需要记录 sha256、train/val/test 切分、CPCV 参数，保证可复现。

**Q：为什么 fit_bins 只用训练日期？**  
A：用全样本分位数会「看见未来」，验证集指标虚高（数据泄漏）。

**Q：loss 很大正常吗？**  
A：预训练看相对下降；真正验收是下游 RankIC 是否提升。

**Q：我没有 GPU 能跑吗？**  
A：`make smoke` 可在 CPU 上跑；真实训练建议 GPU，`config.yaml` 里 `device: auto`。

---



## 11. 推荐学习路径（约 3 天）


| 天     | 任务                                                                  |
| ----- | ------------------------------------------------------------------- |
| Day 1 | 跑通 `make smoke`；读 `smoke.py` + `cn_l2_v1.py` + `tokenize_events.py` |
| Day 2 | 读 `dataset.py` → `model.py` → `heads.py`；对照本指南第 5–7 节               |
| Day 3 | 读 `train.py`；改 `config.yaml` 里 `max_steps` 做小规模实验                   |
| Day 4 | 对照 V2 指导读 `field_spec.py` → `dataset_v2.py` → `field_fusion.py` → `heads.py` |


---



## 12. 代码里 `# [导读]` 注释在哪？

已在以下文件加入面向新手的行内备注（搜索 `[导读]` 可跳转）：

- `quant_fm/scripts/smoke.py`
- `quant_fm/schema/cn_l2_v1.py`
- `quant_fm/tokenizer/tokenize_events.py`
- `quant_fm/manifest/build_manifest.py`
- `quant_fm/pretrain/dataset.py`
- `quant_fm/pretrain/heads.py`
- `quant_fm/pretrain/model.py`
- `quant_fm/pretrain/train.py`

更详细的 API 说明仍以各文件顶部 docstring 为准。
