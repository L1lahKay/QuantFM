# Top 选股 Loss 最终方案

## 1. 最终决策

正式 Ranker 使用以下日度联合目标：

\[
L = 1.00L_{\text{Multi-}K\ \mathrm{LambdaNDCG}}
  + 0.30(1-\mathrm{IC})
  + 0.05L_{\mathrm{Huber}}
\]

其中：

- `K=(50, 300, 350)`，权重为 `(0.20, 0.60, 0.20)`；
- 300 对齐正式目标持仓数，350 对齐退出/候选缓冲边界，50 负责约束最头部质量；
- IC 在完整横截面上稳定整体排序；
- Huber 只训练独立辅助收益头，共享主干但不直接作为最终 score；
- 每个交易日分别计算 loss，再对交易日等权，避免大横截面日期自动获得更高权重。

本阶段不加入 CCC、方向分类、主预测 L2、bottom-tail loss、换手 loss 或多期限 loss。
CCC 与 IC 高度重叠；方向分类不直接对应 Top-K；主 score 已逐日标准化；bottom-tail
与当前 long-only 目标不一致；换手和成本先作为验证与上线门禁，避免在第一版引入不稳定的
可微组合近似。

## 2. 标签与 gain

信号 `score(T)` 在 T 日收盘后可用，因此严格收益标签为：

\[
r_i = \frac{VWAP_i(T+2)}{VWAP_i(T+1)}-1
\]

每天先计算横截面超额收益：

\[
x_i=r_i-\operatorname{mean}_d(r)
\]

主排序标签是 `[0,1]` 百分位：

\[
u_i=\frac{\operatorname{rank}_{avg}(x_i)-1}{N_d-1}
\]

LambdaNDCG 的连续头部 gain 为：

\[
g_i=\left(\frac{\max(u_i-0.5,0)}{0.5}\right)^2
\]

这比二元 `is_top_k` 更稳健：头部内部仍有强弱梯度，边界附近不会因微小收益扰动而发生
完全离散的标签翻转。

辅助收益目标按日使用 median/MAD 标准化：

\[
z_i=\operatorname{clip}\left(
\frac{x_i-\operatorname{median}_d(x)}
{1.4826\operatorname{MAD}_d(x)+\epsilon},-3,3\right)
\]

MAD 退化时使用当日标准差，再退化时使用 1。辅助头采用
`SmoothL1(beta=0.5)`，不直接参与生产 score。

## 3. Multi-K sampled LambdaNDCG

主 score 每日标准化：

\[
\hat s_i=\frac{s_i-\bar s}{\sqrt{\operatorname{mean}(s-\bar s)^2}+\epsilon}
\]

对真实标签较高的股票 `i` 与较低的股票 `j`，使用：

\[
\ell_{ij}=\operatorname{softplus}
\left(-\frac{\hat s_i-\hat s_j}{\tau}\right)
\]

pair 权重为交换两只股票对 `NDCG@K` 的绝对影响，并在当日采样 pair 内归一化。
每天最多采样 8192 对：75% 来自真实/预测头部及 50、300、350 边界，25% 来自全截面
随机 pair；忽略标签百分位差小于 0.02 的近似平局。采样 seed 由全局 seed、epoch 和日期
稳定生成。小横截面自动将 K 和 pair budget 截断，但严格生产训练要求每日不少于 350 只。

## 4. 数据清洗契约

训练样本只按信号时点已知的 `eligible_at_signal` 过滤。`entry_fillable` 和
`exit_fillable` 是未来执行结果，不得用于删除训练行；它们只供回测执行器拒单。

清洗过程还执行：

- `fwd_ret` 宽松转数值并删除 null、NaN、正负 Inf；
- embeddings、panel、factors、universe 的 `(date,symbol)` 必须非空且唯一；
- 所有 `emb_*`、`factor_*` 必须是有限数值；
- 评分入口拒绝 `label`、`fwd_ret`、`xs_ret`、`target_return`、`aux_target`、
  `head_gain` 及所有 `target_*`，防止未来目标泄漏；
- 训练与评分必须显式传入逐日 PIT
  `(date,symbol,asof_date,universe_policy)` 股票池。不能用未来固定成分股回填历史。

项目现有 Dense230M 训练 embedding 的每日中位股票数约 5093，而当前 OOS 约 998。
若不做股票池对齐，Top-300 在训练期约是前 6%，在生产期却约是前 30%，目标语义不同。
因此严格流程要求训练/OOS 两侧同 policy 的 PIT 股票池，并把其文件指纹和逐日宽度写入
cache 与交付元数据。
固定 K 模式下，训练与评分股票池的每日中位宽度比还必须位于 `[0.75, 1.25]`，并且两端
每个交易日都不得少于 350 只；否则直接失败。

## 5. 训练、验证与产物规则

- 使用时间尾部验证集，默认 10 天；训练与验证之间 purge 2 个交易日；
- purge 不得小于执行标签的 `exit_day_lag`；
- early stopping 指标为 exact weighted Multi-K NDCG 加 `0.30 × IC`；
- Top-300 实现收益只记录为报告指标，不直接参与 checkpoint 选择；
- checkpoint/cache 代际为 `multi_lambda_ndcg_v1`，目标或股票池任一字段变化都会重训；
- Ranker artifact 为 v2，必须严格包含辅助头、objective 和训练契约；v1 仅允许显式
  inference-only 迁移，不能续训；
- 生产信号日期必须严格晚于 `label_end_date`，而不只是晚于最后一个训练信号日。

## 6. 上线前实验

必须在相同时间切分、股票池、seed 和执行收益上做四组消融：

1. 原 RankIC baseline；
2. 仅 Multi-K LambdaNDCG；
3. Multi-K LambdaNDCG + 0.30 IC；
4. Multi-K LambdaNDCG + 0.30 IC + 0.05 Huber。

主判据依次为：验证集 `NDCG@300/350`、Top-300 相对股票池收益、15 bps 成本后的缓冲
组合 active return、换手和跨 seed 稳定性；全截面 IC 是必要的稳定性指标，但不再作为唯一
模型选择标准。当前 2026 区间已经用于多轮架构研究，最终结论应再保留一段从未参与调参的
前向区间。

## 7. 与 token / FM 设计的兼容性

本 Loss 只用于冻结股日 embedding 之后的横截面 Ranker，不替换 FM 的 next-event
预训练 Loss，也不把未来收益梯度回传到 tokenizer 或 FM。两层职责为：

```text
T 日单股事件 token
  -> 因果 FM（冻结）
  -> 一股一日 embedding（T 日收盘后可用）
  -> 当日完整横截面 Ranker
  -> score(T)
  -> T+1 建仓、T+2 退出标签
```

因此 Multi-K LambdaNDCG、IC 和辅助 Huber 与离散 token 字段、next-event 分类头没有
目标冲突，`K` 和损失权重也不需要因 token 设计调整。必须保持两阶段训练边界：Top-K
Loss 不能直接替换 token 预训练 CE，也不能在没有严格时间隔离的情况下联合微调 FM。

当前实际 Dense230M 产物是 `cn_l2_v1` 兼容基线，契约为
`book_state_timing=none`、`context=2048`、`flat_v1 + mean pooling`；它不是 V2
多尺度/严格因果产物。本轮结论只能表述为“新 Top-K Loss + Dense230M V1 mean
embedding”。训练与评分必须逐项锁定同一 FM checkpoint、vocab、context、pooling、
`last_k`、embedding 有序列与收盘后可用时点；同为 1024 维的 `mean`、`last`、
`lastk_mean` 不能只靠列数判断兼容。

上述代码质量债现已修复，但修复不会追溯改变旧 artifact：

- SH/SZ 新清洗默认按交易所 `(int_time, serial, input_order)` 稳定排序，不再按
  `local_time` 接收时钟排序；
- EW-VWAP 在首个当前/历史有效价格出现前保持 NaN，不再从未来首个价格回填；
- vocab、token shard sidecar、manifest、FM checkpoint 和 embedding contract 均记录
  排序与变换语义；vocab 拟合日期必须全部属于 manifest 训练切分；
- 新 token sidecar 还绑定完整 vocab SHA-256。预训练、评估和 embedding
  抽取会在读取/cache hit 前重新校验实际 shard 字节、sidecar、manifest
  记录和 vocab，不再只信 manifest 顶层声明；
- 新 FM checkpoint 保存并侧载 `pretrain_data_contract`，绑定 manifest/vocab
  指纹、训练/vocab-fit/validation 选择日期边界和 token 语义；有效截止日保守取三者
  最晚值。严格 OOS 截止日必须从该契约派生，
  不接受人工填一个字符串作为证明；
- 预训练非劣验收使用 acceptance v2，除重新计算相同 validation plan 上的 CE 门槛外，
  还绑定 candidate/baseline 报告的完整 SHA-256。严格入口会继续复核报告所指向的
  checkpoint/config、checkpoint sidecar、manifest/vocab 与训练/OOS embedding 身份；
- 新 V2 embedding 使用 `context=2048, stride=512` 的因果重叠窗口。历史前缀进入
  attention，但每个事件只进入股日 pooling 一次；
- `hierarchical_selected_v2` 严格按配置输出
  `mean_all/last_256/continuous_pm/close_30m`，宽度固定为 `4*d_model`，不再静默输出
  历史 `8*d_model+1`；
- strict Top-K 默认拒绝旧排序、未来回填、独立 chunk、旧 pooling、缺 sidecar 或
  训练/评分 contract 不一致。legacy override 只允许诊断，不得标记为正式结果。
- 正式 Ranker artifact 必须内嵌已重验的 T+1/T+2 execution contract、
  PIT universe、时间留出/purge、V2 表示契约和完整冻结 Loss 参数；
  缺任一项的 V2 权重也不能通过生产 `signal.generate`。

只读抽样审计 `medium_300m` manifest 的 500 个历史 shard，发现 2 个 shard、2 次
`int_time` 倒序；全部旧 shard 都没有语义 sidecar，并被识别为
`local_time_v1 + ew_vwap_future_backfill_v1`。因此旧 Dense230 Token/checkpoint/embedding
不能通过补写 metadata 升级，必须在新输出根按“clean → vocab/token → manifest → FM
重训 → embedding 重抽 → Ranker 重训”完整重建。resume 遇到语义不匹配会停止，不会
覆盖现有约 98GB 历史产物。

## 8. 代码位置

- 标签与训练股票池清洗：`quant_fm/downstream/make_features.py`
- Loss、采样、辅助头和 early stopping：`quant_fm/downstream/train_ranker.py`
- FM/vocab/pooling 与 Ranker 的表征契约：`quant_fm/embedding/contract.py`
- V2 pooling 布局：`quant_fm/embedding/pooling_spec.py`
- Token 时间顺序与 artifact 契约：`order_book/pylob/event_ordering.py`、
  `quant_fm/tokenizer/artifact_contract.py`
- Token shard/manifest 运行时字节验证：`quant_fm/manifest/validation.py`
- FM 训练日期与 checkpoint 血统：`quant_fm/pretrain/data_contract.py`
- 预训练验收与端到端血缘复核：`quant_fm/monitoring/acceptance.py`、
  `quant_fm/scripts/validate_pretrain_lineage.py`
- 历史 Token 只读审计：`quant_fm/scripts/audit_token_ordering.py`
- 严格执行收益：`quant_fm/downstream/build_panel_from_minio.py`
- PIT 股票池契约：`quant_fm/downstream/universe.py`
- 正式重训预检：`quant_fm/scripts/preflight_topk_ranker.py`
- OOS 训练、缓存和交付契约：`quant_fm/scripts/build_oos_delivery.py`
- Ranker v2 artifact：`quant_fm/signal/artifact.py`
- 独立评估：`quant_fm/downstream/run_score_evaluation.py`

严格重训还必须由外部提供训练期和 OOS 两套完整 T+2 交易日历，以及两套逐日 PIT
股票池。PIT 文件必须包含 `date,symbol,asof_date,universe_policy`，满足
`asof_date <= date`，且训练/评分使用相同 policy、相近的实际截面宽度。代码不会根据
今天的成分股伪造历史 PIT 数据。
代码能校验格式、as-of 时点、策略标识、宽度和日历映射，但不能单凭文件
内容证明它们确实来自官方交易日历或真实 PIT 快照；这一部分仍是外部数据治理门禁。

在昂贵重建/训练前先运行：

```bash
python -m quant_fm.scripts.preflight_topk_ranker \
  --train-embeddings TRAIN.parquet --oos-embeddings OOS.parquet \
  --train-calendar TRAIN_DATES.txt --oos-calendar OOS_DATES.txt \
  --train-universe TRAIN_PIT.parquet --oos-universe OOS_PIT.parquet \
  --min-names-per-day 350 --out preflight.json
```

任一真实输入或新因果 embedding 未到位时，正式入口应失败，而不是退回旧标签、固定
未来股票池或无版本 embedding。
