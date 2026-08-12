# Regime 特征生产流水线

`regime_features_l2_v1` 是 Temporal Regime-MoE 的纯 Level-2、收盘后特征契约。
它不会把成交额活跃度称为换手率，也不会把交易板块称为行业。

## 三层产物

1. 清洗 worker 在盘口 transitions 仍在内存时写
   `clean/<date>/<market>/<symbol>/regime_atomic.parquet`；
2. `run_medium --build-regime-data` 在删除当日 clean 目录前写
   `data/regime/stock_day_atomic/<date>.parquet` 及 manifest；
3. 所有并行日期完成后，用连续交易日 EOD 和 PIT universe 统一生成
   `data/regime/features/regime_features_l2_v1.parquet` 及 manifest。

逐股 atomic 包含时间加权 spread、L5 数量深度和由相邻 L1 pre/post 状态计算的
Cont OFI。市场滚动字段只在全局阶段计算，窗口严格按传入的连续交易日历索引。

## v1 固定口径

所有市场/板块横截面都只使用当日 PIT universe 内、EOD 可用的股票，并采用等权，
不引入指数成分、自由流通股本或行业分类数据。

| 字段 | 固定定义 |
| --- | --- |
| `market_return_5d` | 最近 5 个交易日“股票日收益等权均值”的复合收益 |
| `market_return_20d` | 最近 20 个交易日“股票日收益等权均值”的复合收益 |
| `market_realized_vol_20d` | 上述市场日收益最近 20 日样本标准差乘 `sqrt(252)` |
| `market_breadth` | 当日 `mean(sign(close/pre_close-1))`，范围 `[-1,1]` |
| `market_amount_ratio_20d` | 当日全市场累计成交额 / 最近 20 日全市场累计成交额均值 |
| `board_relative_strength_20d` | 股票所属交易板块的 20 日等权复合收益减市场 20 日复合收益 |
| `stock_spread_bps` | 连续竞价期有效双边盘口的 `10000*(ask1-bid1)/mid` 时间加权均值 |
| `stock_depth_l5_log` | 连续竞价期 `log1p(bid_depth_l5+ask_depth_l5)` 时间加权均值 |
| `stock_ofi_l1` | 连续竞价事件 Cont L1 OFI 之和 / OFI 绝对值之和 |

“板块”由交易所和证券代码确定为 MAIN、SME、CHINEXT、STAR 等，因此本版不输出
`industry_relative_strength_20d`。若以后接入可审计的 PIT 行业成员表，应新增 artifact/
formula 版本，不能在本版本内静默替换含义。同理，本版没有自由流通股本，所以输出
`market_amount_ratio_20d`，不输出 `market_turnover`。

盘口时间加权使用事件后的状态，持续到下一事件或同一连续竞价时段结束；午间不跨段
携带状态。OFI 是事件量，不受相邻事件时间差为零影响。特征截止时点固定为信号日
`15:00:00+08:00`，因此只适用于收盘后训练/打分；盘中策略不得把 `availability_lag=0`
解释为盘中已知。

## 完整性门禁

- 日级 archive 必须与 V2 clean coverage receipt 的 `(market,symbol)` 集合完全一致；
- EOD 默认要求每个交易日收益与成交额覆盖率均不低于 98%；
- 每个 atomic 键必须在目标日 PIT universe 中，且 market/board 与 EOD 推导一致；
- 第一条信号日前必须有 19 个 warm-up 交易日，20 日窗口不允许按并行日期组计算；
- 最终九个字段必须全部有限，任一缺失、NaN 或 Inf 都拒绝发布；
- 每日 parquet、coverage receipt、EOD、universe、calendar 和最终 parquet 均写入
  manifest 哈希，resume 时验证而不是仅检查文件存在。

## 单独 finalize

```bash
uv run python -m quant_fm.regime.cli finalize \
  --atomic-dir <workdir>/data/regime/stock_day_atomic \
  --eod /path/to/continuous_eod.parquet \
  --universe /path/to/continuous_pit_universe.parquet \
  --calendar /path/to/trading_calendar.txt \
  --signal-dates-file /path/to/requested_signal_dates.txt \
  --out <workdir>/data/regime/features/regime_features_l2_v1.parquet
```

`--signal-dates-file` 推荐显式传入，避免复用 workdir 时把目录内其他合法 atomic 日期
一起发布；`run_v2_parallel_data --build-regime-data` 会自动传入本次 dates 文件。

EOD 至少需要 `date,symbol,market,close,pre_close,total_notional`。PIT universe
沿用严格契约：`date,symbol,asof_date,universe_policy`。第一条信号日期之前必须包含
至少 19 个连续交易日的 EOD 和 universe warm-up。

生产上建议显式设置 `--min-book-valid-ratio`（例如先用真实数据分布决定 0.8 或其他
阈值）；默认值 0 只保留“字段必须可计算”的硬门禁，方便先做覆盖率审计。

最终字段与 `quant_fm/moe/config_regime_l2_v1.yaml` 一一对应。训练和推理继续通过
现有 `--regime-features` 参数读取最终 parquet。
