# 阶段 8：生产 Score 与严格研究评估

## 1. 目标与边界

本阶段分成两条不可混用的链路：

| 链路 | 目的 | 允许读取 | 对外交付 |
|---|---|---|---|
| 生产信号 | 用冻结 Ranker 对新 embedding 出分 | 当日及历史信息；不允许 panel/label | `scores.parquet` + `signal_manifest.json` |
| 研究评估 | 评估冻结 score 的 IC、组合与风险 | 独立 execution panel 和可选 PIT 因子 | 内部 metrics、账本、manifest |

生产输出的列名和顺序必须恰好为 `date, symbol, score`。
`score(T)` 仅在 T 日收盘后可用，是同日截面排序分数，不是预测收益率或目标仓位。

2026 年 60 个信号日的结果已被用于问题诊断和 v2 架构设计，因此现在是
**架构验证/研究窗口**，不再是 v2 的 untouched OOS。

## 2. 核心代码

| 文件 | 责任 |
|---|---|
| `quant_fm/signal/schema.py` | 严格三列 score 契约、类型、主键和有限值校验 |
| `quant_fm/signal/train.py` | 用历史 embedding/panel 训练并冻结 Ranker artifact |
| `quant_fm/signal/generate.py` | 不读标签的生产 score 生成与 manifest 写入 |
| `quant_fm/downstream/train_ranker.py` | 截面 Ranker；`fit_ranker()` 支持验证 IC early stop 和恢复最佳 epoch |
| `quant_fm/downstream/return_spec.py` | 信号日到建仓/退出日和价格字段的显式定义 |
| `quant_fm/downstream/build_panel_from_minio.py` | EOD 价格和严格 execution panel 构建 |
| `quant_fm/downstream/evaluate.py` | RankIC、ICIR、HAC t、block bootstrap、分组、CPCV 和 DSR |
| `quant_fm/downstream/portfolio_simulator.py` | 持仓缓冲、拒单、现金和显式成本的研究组合 |
| `quant_fm/downstream/risk_attribution.py` | 组合因子暴露和横截面收益残差化 |
| `quant_fm/downstream/run_score_evaluation.py` | 对冻结 score 运行可复现的独立研究评估 |
| `quant_fm/downstream/run_judge.py` | embedding + panel 的历史一体化 judge/CPCV 链路 |

## 3. 生产链路

```text
历史 embedding + 历史 panel
  → signal.train
  → ranker.pt + ranker_metadata.json

信号日 embedding + 冻结 Ranker
  → signal.generate（无标签）
  → scores.parquet + signal_manifest.json
```

```bash
uv run python -m quant_fm.signal.train \
  --embeddings runs/history/embeddings/all.parquet \
  --panel runs/history/panel/daily_panel.parquet \
  --out-dir runs/history/signal_artifact \
  --device cuda:0

uv run python -m quant_fm.signal.generate \
  --embeddings runs/oos/embeddings/all.parquet \
  --ranker runs/history/signal_artifact/ranker.pt \
  --ranker-metadata runs/history/signal_artifact/ranker_metadata.json \
  --fm-checkpoint runs/history/run/best.pt \
  --vocab runs/history/data/vocab.json \
  --out-dir runs/oos/delivery \
  --device cuda:0
```

`signal.generate` 默认要求信号日严格晚于 Ranker 元数据里的
`training_end_date`。`--allow-in-sample` 只可用于 smoke/研究，不得出现在生产交付中。

## 4. 研究收益与 execution panel

`ReturnSpec` 将时点写成可审计配置：

| 名称 | 收益区间 | 用途 |
|---|---|---|
| `close_t_close_t1` | T close → T+1 close | 不可执行的研究上限/旧口径 |
| `vwap_t_vwap_t1` | T VWAP → T+1 VWAP | 不可执行的研究上限/旧口径 |
| `open_t1_close_t1` | T+1 open → T+1 close | 可交易日内近似 |
| `vwap_t1_vwap_t2` | T+1 VWAP → T+2 VWAP | 当前严格默认 |

```bash
uv run python -m quant_fm.downstream.build_panel_from_minio \
  --from-embeddings runs/oos/delivery/scores.parquet \
  --calendar-file data/oos_calendar_with_future_days.txt \
  --return-spec vwap_t1_vwap_t2 \
  --out runs/oos/research/execution_panel.parquet
```

execution panel 将信号日可见的 `eligible_at_signal` 与之后的
`entry_fillable/exit_fillable` 分开。选股不得使用后两者；执行失败不得事后补选。
当前组合模拟在 interval 的 `entry_date` 调仓，买卖均使用 `entry_fillable`；未来
`exit_fillable` 不参与当前调仓或本期持仓收益。

## 5. 独立 score 研究评估

```bash
uv run python -m quant_fm.downstream.run_score_evaluation \
  --scores runs/oos/delivery/scores.parquet \
  --panel runs/oos/research/execution_panel.parquet \
  --out-dir runs/oos/research/evaluation \
  --quantile-groups 10 \
  --top-k-grid 20,50,100,150,200 \
  --rebalance-grid 1,5 \
  --smoothing-grid 1,3 \
  --cost-bps-grid 0,15,30
```

Make 入口：

```bash
make research-score \
  SCORES=/path/to/scores.parquet \
  PANEL=/path/to/execution_panel.parquet \
  OUT_DIR=/path/to/evaluation

CALENDAR=/path/to/full_calendar.txt make research-oos2026
```

默认输出：

```text
evaluation/
├── metrics.json
├── evaluation_manifest.json
├── daily_ic.parquet
├── quantile_returns.parquet
├── topk_grid.parquet
├── portfolio_daily.parquet
├── holdings.parquet
└── trades.parquet
```

传入包含 `factor_*` 列的 `--factors` 时，额外输出日度因子暴露和中性化 IC。
正式评估不得使用 `--allow-incomplete-horizon`。默认严格模式要求 score 键全部命中、
eligible 标签覆盖率为 100%、每日至少 3 个有效标的且至少一期有限 IC；指标同时报告
`join_coverage` 与 `label_coverage`，JSON 中的非有限统计量写为 `null`。

## 6. 评估内容

- RankIC 仅使用 score 和收益的有限值；面板包含 `eligible_at_signal` 时自动过滤。
- `ic_statistics()` 返回均值、标准差、ICIR、正 IC 比例、朴素 t 和 Newey-West t。
- `block_bootstrap_mean_ci()` 默认用 5 日移动块、2000 次重采样给出 95% 均值区间。
- 分位组结果是逐日逐组收益，`group=0` 为最低分组。
- 主组合采用历史 score 平滑、入/出排名缓冲、成交拒绝、未投资现金和买卖显式成本。
- 等权可投资股票的 `fwd_ret` 均值作为基准，用于 active return 和 information ratio。
- `portfolio_exposures()` 按持仓权重汇总 `factor_*`；`residualize_returns()` 逐日 OLS 返回收益残差。

这些是研究级实现，不含完整交易所规则、排队、冲击或官方 ST/新股日历，
不能替代生产回测引擎。

## 7. Paired OOS 验收

`run_score_evaluation` 没有 paired CLI 参数，一次只评估一个模型。严格 A/B 比较流程是：

1. 锁定同一 execution panel、return spec、股票池和参数网格；
2. 校验 A/B score 的 `(date, symbol)` 键集合一致；
3. 分别运行评估，保留两份 manifest；
4. 按日期连接两份 `daily_ic.parquet`，对 `candidate_ic - baseline_ic`
   运行 `ic_statistics()` 与 `block_bootstrap_mean_ci()`；
5. 一并报告换手、成本后收益、回撤和因子暴露，不只看均值 IC。

2026-60d 上的 paired 结果只能支持开发决策。对 v2 的最终验收必须换用新的
untouched OOS 区间。

## 8. 历史 Judge 链路

`run_judge` 仍用于 embedding 层的研究验收：

```bash
uv run python -m quant_fm.downstream.run_judge \
  --workdir quant_fm/runs/medium_try \
  --checkpoint quant_fm/runs/medium_try/run/best.pt \
  --panel quant_fm/runs/medium_try/panel/daily_panel.parquet \
  --emb-dir quant_fm/runs/medium_try/embeddings \
  --epochs 30 \
  --device cuda:0
```

它使用 train 拟合 Ranker，用 val IC 选最佳 epoch，再输出 train/val/test、随机对照和
CPCV 报告。这条链路会读标签，必须标记为 research-only；冻结 score 的独立验收优先使用
`run_score_evaluation`。
