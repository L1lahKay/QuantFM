# 阶段 8：生产 Score 信号与研究验收

## 目标

生产目标是从内部 embedding 和冻结 Ranker 生成 `date, symbol, score`。生产推理不得读取 panel、`fwd_ret` 或 `label`。RankIC、CPCV 与 Top-K 回测保留为 research-only 工具，不属于交付链路。

## 核心代码

| 文件 | 责任 |
|------|------|
| `quant_fm/downstream/make_features.py` | embedding 与日频面板拼接、过滤 |
| `quant_fm/downstream/train_ranker.py` | 截面 Ranker 训练与预测 |
| `quant_fm/signal/train.py` | 用历史标签离线训练并冻结 Ranker |
| `quant_fm/signal/generate.py` | 无标签生成生产 score |
| `quant_fm/signal/schema.py` | 三列信号契约与校验 |
| `quant_fm/downstream/evaluate.py` | RankIC/ICIR、分组单调性、CPCV、DSR |
| `quant_fm/downstream/backtest_topk.py` | Top-K 多空/多头回测 |
| `quant_fm/downstream/run_judge.py` | 完整下游验收与报告持久化 |
| `quant_fm/downstream/build_panel_from_minio.py` | 从 MinIO 快照构造日频标签面板 |

## 输入

- train/val/test 股日 embedding；
- 日频面板与未来收益 `fwd_ret`；
- 股票状态过滤字段，如 ST、停牌、新股、涨跌停锁定；
- checkpoint 元数据。

## 生产流程

```text
历史 embedding + 历史 panel → signal.train → ranker.pt
信号日 embedding + ranker.pt → signal.generate
                              → scores.parquet + signal_manifest.json
```

最新信号日即使尚无未来收益，也必须能够正常出分。对外交付不包含 embedding、panel、checkpoint、持仓或回测指标。

## Research-only 流程

```text
embedding + panel
  → 特征对齐与股票池过滤
  → 仅用 train split 训练 Ranker
  → val/test 预测
  → RankIC / ICIR / 分组单调性
  → Top-K 多头与多空回测（含成本）
  → Deflated Sharpe Ratio
  → CPCV（purge + embargo，按时间块重训 Ranker）
  → 持久化 judge report
```

## 生产运行

```bash
uv run python -m quant_fm.signal.train \
  --embeddings runs/history/embeddings/train.parquet \
  --panel runs/history/panel/daily_panel.parquet \
  --out-dir runs/history/signal_artifact

uv run python -m quant_fm.signal.generate \
  --embeddings runs/oos/embeddings.parquet \
  --ranker runs/history/signal_artifact/ranker.pt \
  --ranker-metadata runs/history/signal_artifact/ranker_metadata.json \
  --out-dir runs/oos/delivery
```

## Research-only 运行

```bash
# 1) 抽股日 embedding（按 split）
uv run python -m quant_fm.embedding.extract_hidden \
  --checkpoint quant_fm/runs/medium_try/run/best.pt \
  --manifest quant_fm/runs/medium_try/data/manifest.json \
  --split train \
  --out quant_fm/runs/medium_try/embeddings/train.parquet \
  --device cuda:0
# 对 val / test 重复上述命令

# 2) 下游验收（需 panel/daily_panel.parquet）
uv run python -m quant_fm.downstream.run_judge \
  --workdir quant_fm/runs/medium_try \
  --checkpoint quant_fm/runs/medium_try/run/best.pt
```

Make 入口：

```bash
make judge-medium-try
```

300M 训完后只需把 `--checkpoint` / `--workdir` 换成 `medium_300m` 对应路径，并先抽好 `embeddings/{train,val,test}.parquet`。
## 输出

```text
<workdir>/downstream/
├── runs/<timestamp>_<checkpoint>.json
├── latest.json
└── history.jsonl
```

报告包含 checkpoint 路径/哈希、数据规模、训练历史、各 split 指标与回测结果。

## 关键指标

- **RankIC**：每日预测分数与未来收益的 Spearman 秩相关；
- **ICIR**：RankIC 均值相对波动；
- **分组单调性**：预测分位组收益是否有序；
- **Top-K 回测**：含交易成本的多头或多空组合；
- **CPCV**：purge + embargo 的组合式时间交叉验证；
- **DSR**：对多次试验中最佳 Sharpe 的运气成分折扣。

## 验收原则

1. 标签必须严格来自未来区间，且不进入预训练输入；
2. Ranker 仅使用训练日期拟合；
3. checkpoint 选择只能依赖验证集，不应用 test 反复调参；
4. 回测必须包含合理交易成本；
5. smoke 的合成标签结果只证明代码可运行，不代表真实收益；
6. 若 test RankIC、单调性和 DSR 不稳健，应扩大数据、检查标签或调整建模，而不是只追求训练 loss 更低。
