# 严格 OOS 研究回测

> 版本：2026-07-24
>
> 边界：本文描述的是 research-only 评估。生产信号仍只交付
> `scores.parquet` 和 `signal_manifest.json`，不读取未来行情。

## 1. 当前样本的正确定位

2026 年 60 个信号日最初是严格晚于 2025 年 FM/Ranker 训练期的时序 OOS。
但该区间的 score、IC、净值和回撤已经被查看，并用来诊断与设计模型底层 v2
改造，因此从现在开始它的统计身份是：

- **架构验证/研究窗口**；
- 可用于复现问题、调试评估链路和比较方案；
- **不再是 untouched OOS**，不得用于对 v2 做最终泛化能力宣称。

v2 的最终验收需预先冻结模型、Ranker、收益口径、股票池、组合规则与成本网格，
再使用没有参与任何选型的更晚时间段。

## 2. 生产与研究的数据边界

| 链路 | 输入 | 输出 | 是否允许未来行情 |
|---|---|---|---|
| 生产 score | 信号日 embedding + 冻结 Ranker | 严格三列 `date, symbol, score` | 不允许 |
| 研究 execution panel | 信号日 + 完整交易日历 + PIT 行情 | 建仓/退出日、价格、可成交性、`fwd_ret` | 允许，但只能位于 research 侧 |
| 研究评估 | 冻结 score + execution panel | IC、分组、组合、风险归因和 manifest | 允许，不反向流入生产推理 |

`quant_fm.signal.schema.validate_scores()` 对生产表的约束是**列名和顺序必须恰好为**
`date, symbol, score`，`(date, symbol)` 唯一，`symbol` 为 6 位数字，`score`
必须有限。`score(T)` 在 T 日收盘后才可用，仅保证同日截面可比。

## 3. 显式收益口径

`quant_fm.downstream.return_spec` 定义了四个稳定名称：

| `return_spec` | 建仓 | 退出 | 定位 |
|---|---|---|---|
| `close_t_close_t1` | T 收盘 | T+1 收盘 | 研究上限/旧口径对照；对收盘后 score 不可执行 |
| `vwap_t_vwap_t1` | T VWAP | T+1 VWAP | 研究上限/旧口径对照；不可作为实盘结论 |
| `open_t1_close_t1` | T+1 开盘 | T+1 收盘 | 可交易的日内近似 |
| `vwap_t1_vwap_t2` | T+1 VWAP | T+2 VWAP | 当前严格研究默认 |

默认的 `vwap_t1_vwap_t2` 要求交易日历至少覆盖最后一个信号日之后的第二个
交易日。默认情况下，日历不足或任一 score 日没有有效 `fwd_ret` 会直接失败，
避免把 60 个信号日静默评成 59 期。

## 4. Execution panel 契约

严格面板由 `build_execution_panel()` 构造，以信号日 `date` 为键，主要字段为：

- 时点：`date, entry_date, exit_date`；
- 价格与口径：`entry_px`、`exit_px`、`fwd_ret`、`return_spec`、
  `entry_price_field`、`exit_price_field`；
- 信号日状态：`is_st_at_signal`、`is_new_at_signal`、`is_halt_at_signal`、
  `limit_locked_at_signal`、`eligible_at_signal`；
- 执行状态：`is_halt_entry`、`limit_locked_entry`、`entry_fillable`、
  `is_halt_exit`、`limit_locked_exit`、`exit_fillable`。

目标股只能根据信号日已知的 `eligible_at_signal` 和 score 决定。
`entry_fillable`/`exit_fillable` 是之后才能观察到的执行结果，只能用于成交或拒单；
买入失败不得用执行日信息补选更低排名的股票。

当前日频组合状态在每个 interval 的 `entry_date` 调仓，因此买单和卖单都只读取该行的
`entry_fillable`；`exit_fillable` 用于退出日诊断，不得提前影响本期
`entry_date → exit_date` 的持仓收益。这样修改未来退出日状态不会改变更早的调仓结果。

当前面板中 `is_st`/`is_new` 由 L2 快照构造时默认为 `False`，不是券商级
PIT 状态。正式研究应用官方 ST、新股、停复牌和价格限制数据覆盖。

## 5. 构建面板

`--calendar-file` 必须是覆盖完整 entry/exit horizon 的连续交易日历。
`--from-embeddings` 参数名为历史兼容名，实际只用输入文件的 `date/symbol`，
因此可以直接传三列 score 文件：

```bash
uv run python -m quant_fm.downstream.build_panel_from_minio \
  --from-embeddings quant_fm/runs/oos2026/delivery_oos/scores.parquet \
  --calendar-file /path/to/calendar_with_two_future_days.txt \
  --return-spec vwap_t1_vwap_t2 \
  --out quant_fm/runs/oos2026/research/execution_panel.parquet
```

仓库内的 `quant_fm/data/oos2026_dates.txt` 只为旧链路保留 60 个信号日
+ 1 个末日，不足以覆盖 `vwap_t1_vwap_t2` 的最后 T+2，不能作为该命令的默认日历。
`run_oos2026_research.sh` 因此要求显式传入 `CALENDAR`。

也可用 `--signal-dates-file` 单独给信号日。`--return-spec` 必须与
`--calendar-file` 同时提供。`--allow-incomplete-horizon` 仅用于排查，不能出现在
验收报告中。

## 6. 运行严格 score 评估

### 6.1 一键 2026 研究窗口

```bash
CALENDAR=/path/to/calendar_with_two_future_days.txt \
SCORES=/path/to/frozen_scores.parquet \
RETURN_SPEC=vwap_t1_vwap_t2 \
RESEARCH_DIR=/path/to/research \
make research-oos2026
```

`quant_fm/scripts/run_oos2026_research.sh` 会先构建 execution panel，再执行
`run_score_evaluation`。可覆盖的环境变量为 `OOS_WORKDIR`、`SCORES`、
`CALENDAR`、`RETURN_SPEC`、`RESEARCH_DIR`、`PANEL`。

### 6.2 已有 execution panel

```bash
make research-score \
  SCORES=/path/to/scores.parquet \
  PANEL=/path/to/execution_panel.parquet \
  OUT_DIR=/path/to/research/evaluation
```

等价的完整 CLI：

```bash
uv run python -m quant_fm.downstream.run_score_evaluation \
  --scores /path/to/scores.parquet \
  --panel /path/to/execution_panel.parquet \
  --out-dir /path/to/evaluation \
  --factors /path/to/pit_factors.parquet \
  --quantile-groups 10 \
  --top-k-grid 20,50,100,150,200 \
  --rebalance-grid 1,5 \
  --smoothing-grid 1,3 \
  --cost-bps-grid 0,15,30
```

`--factors` 可选，表中因子列必须以 `factor_` 开头。
默认严格模式要求 score 键全部命中 panel、每个 `eligible_at_signal=True` 的 score 都有
有限 `fwd_ret`、每个信号日至少有 3 个有效横截面，并且至少产生一期有限 IC；score
本身的 null/NaN/Inf 也会直接失败。`--allow-incomplete-horizon` 只放宽标签完整性和
逐日最小截面，仍不允许零 IC 期或非法 score，只能用于排查。

## 7. 评估内容与默认组合

`run_score_evaluation` 会计算：

- 日度 Spearman RankIC、ICIR、正 IC 比例、朴素 t 值和 Newey-West/HAC t 值；
- 日度 IC 移动块 bootstrap 95% 均值区间；
- 日度分位组收益；
- Top-K × 调仓间隔 × score 历史平滑 × 单边成本网格；
- 默认带持仓缓冲组合、等权基准超额和 information ratio；
- 可选的因子暴露和横截面中性化后 IC。

主组合默认值是：

```text
candidate_top_k=150, target_holdings=100
entry_rank=80, exit_rank=180
rebalance_interval=1, score_smoothing_days=3, max_turnover=1.0
buy_bps=15, sell_bps=15, stamp_duty_bps_sell=0
```

成交失败会保留现金，不会事后补选。这是可审计的日频研究模拟器，尚未完整实现
交易所 T+1 卖出规则、冲击函数、分档佣金、印花税时变和排队成交，因此不等价于
生产交易引擎。

## 8. 输出和可复现性

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

传入 `--factors` 且存在 `factor_*` 列时，还会写出
`exposure_daily.parquet` 和 `daily_neutralized_ic.parquet`。

`evaluation_manifest.json` 记录 score、panel 和可选 factors 的绝对路径与 SHA-256，
并保留 panel 中的 return spec 元数据。`metrics.json` 将键连接覆盖
`join_coverage` 与 eligible 标签覆盖 `label_coverage` 分开记录，同时包含缺失/不足
日期、IC、组合和主动收益摘要；非有限统计量写为 JSON `null`，不会写出非标准 `NaN`。

## 9. Paired OOS 比较

当前 CLI **一次只评估一份 score**，没有 `--baseline-scores` 或 `--paired`
参数。模型 A/B 必须：

1. 冻结同一份 execution panel 和所有网格参数；
2. 先校验两份 score 的 `(date, symbol)` 键集合完全相同；
3. 分别运行 `run_score_evaluation`；
4. 将两份 `daily_ic.parquet` 按 `date` 内连接，以
   `delta_ic = ic_candidate - ic_baseline` 作为配对序列；
5. 对 `delta_ic` 计算 `ic_statistics()` 和 `block_bootstrap_mean_ci()`，
   同时比较同口径的换手、成本后收益和回撤。

```python
import polars as pl

from quant_fm.downstream.evaluate import (
    block_bootstrap_mean_ci,
    ic_statistics,
)

base = pl.read_parquet("baseline_eval/daily_ic.parquet").rename(
    {"ic": "ic_baseline"}
)
candidate = pl.read_parquet("candidate_eval/daily_ic.parquet").rename(
    {"ic": "ic_candidate"}
)
paired = base.join(candidate, on="date", how="inner").with_columns(
    (pl.col("ic_candidate") - pl.col("ic_baseline")).alias("delta_ic")
)
delta = paired.select(pl.col("delta_ic").alias("ic"))
stats = ic_statistics(delta)
ci95 = block_bootstrap_mean_ci(delta["ic"].to_numpy())
```

仅在入参哈希、return spec、股票池、日期和超参网格都锁定时，
`delta_ic` 才能解读为模型变化的配对增量。因 2026-60d 已参与架构选型，它上的
paired 结果只是研究证据；最终结论必须来自新的 untouched OOS。
