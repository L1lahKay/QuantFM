# QuantFM Loss 函数完整改进与已训练模型

> 文档基线：2026-07-30。本文把事件级基础模型预训练 Loss、MOE Router Loss 和股日横截面 Ranker Loss 分开说明，并在末尾盘点本地真实 checkpoint。

## 1. 最终方案概览

QuantFM 存在三个训练层级，不能用一个 Loss 互相替代：

```text
事件 Token
  └─ FM next-event Loss
       ├─ Dense / Backbone-MoE
       └─ + Router auxiliary Loss（仅 MOE）
              ↓ 冻结 FM，提取 embedding
股日横截面
  └─ Ranker Multi-K LambdaNDCG + global IC + auxiliary Huber
              ↓
date, symbol, score
```

最终建议：

- V1 checkpoint 保持原始六路等权 CE，保证可复现。
- V2 FM 使用显式 TargetSpec、有效性 mask、训练熵归一化和 ordinal 辅助项。
- Backbone/Temporal MOE 在主任务 Loss 外增加负载均衡与 Router z-loss。
- 下游 Ranker 使用日等权 Multi-K sampled LambdaNDCG，并用全截面 IC 稳定整体排序、独立辅助头 Huber 学习稳健超额收益。
- FM 预训练 Loss 与 Ranker Top-K Loss 保持两阶段边界，未来收益梯度不得回传到 tokenizer 或 FM。

## 2. FM V1：六路等权 Next-Event CE

### 2.1 目标

用位置 \(t\) 的 hidden state 预测下一事件 \(t+1\) 的六个字段：

```text
tok_evt_type
tok_side
tok_session
tok_price_bin
tok_volume_bin
tok_delta_t_bin
```

单字段：

\[
L_f=\operatorname{CE}(\hat y^f_{t+1},y^f_{t+1})
\]

总 Loss：

\[
L_{V1}=\sum_{f=1}^{6}L_f
\]

当前位或下一位为 padding 时忽略，`PAD_ID=0` 不进入 CE。

### 2.2 优点与问题

优点是简单、稳定、已完成多档训练。主要问题：

- session 等低熵字段和 volume 等高熵字段直接相加，难度口径不一致；
- V1 没有独立 NA，缺失/不适用目标不能精确屏蔽；
- price/volume/time 是有序桶，但普通 CE 把相邻桶和远距离桶视为同等错误；
- total CE 下降可能主要来自容易字段，掩盖重要字段退化。

现有 V1 模型必须继续按此 Loss 解释，不能把它们描述成 V2 normalized/ordinal 模型。

## 3. FM V2：显式多任务 Loss

### 3.1 TargetSpec

每个目标显式冻结：

```text
name
loss_type: ce | ordinal_ce
weight
train entropy
ordinal_weight / ordinal_start_id
applicable_event_ids
ignore_ids
mask_field
```

V2 若没有 `loss.targets` 会直接拒绝训练，避免静默回退到 V1 Loss。

### 3.2 有效目标 Mask

字段 \(f\) 的有效位置为：

\[
M_f=M_{t}\land M_{t+1}\land \neg Ignore_f
\land Applicable_f\land ExplicitMask_f
\]

默认忽略 `PAD=0` 和 `NA=2`。如果字段只对某类事件有效，还需匹配 `applicable_event_ids`。全 NA 的 task 返回保留计算图的 0，不产生 NaN，也不阻断 backward。

### 3.3 训练熵归一化

Vocab V2 从完整训练流的 occupancy 计算字段熵：

\[
H_f=-\sum_c p_{f,c}\log p_{f,c}
\]

训练时：

\[
L^{norm}_f=\frac{L^{CE}_f}{\max(H_f,10^{-6})}
\]

这样各字段更接近“相对自身无条件不确定性的改善”，避免高熵字段仅因类别多而天然支配总 Loss。熵只能来自训练集，必须随 vocab/checkpoint 固化。

### 3.4 Ordinal 辅助项

对 price/volume/delta_t 的普通 ID 去掉前 6 个特殊 token 后，计算预测期望桶：

\[
\hat b=\sum_{j=0}^{B-1}P(j)j
\]

再按词表宽度归一化，并使用 Smooth L1：

\[
L^{ord}_f=\operatorname{SmoothL1}
\left(\frac{\hat b}{B-1},\frac{b}{B-1}\right)
\]

它保留 CE 的完整分布监督，同时让“错一个相邻桶”比“错到远端桶”代价更小。

### 3.5 V2 总目标

\[
L_{V2}=\sum_f w_f\left(
\frac{L^{CE}_f}{H_f}+\alpha_fL^{ord}_f
\right)
\]

当前配置：

| 字段 | 类型 | 主权重 \(w_f\) | ordinal 权重 \(\alpha_f\) |
|---|---|---:|---:|
| `tok_evt_type` | CE | 1.00 | 0 |
| `tok_side` | CE | 1.00 | 0 |
| `tok_price_bin` | ordinal CE | 1.00 | 0.50 |
| `tok_volume_bin` | ordinal CE | 0.50 | 0.25 |
| `tok_delta_t_bin` | ordinal CE | 0.50 | 0.25 |
| `tok_session` | 仅输入，不建预测头 | — | — |

## 4. MOE Router Loss

对每个 MOE 层：

\[
L_{router}=0.01L_{balance}+0.001L_z
\]

其中：

\[
L_{balance}=E\sum_e I_eF_e,\qquad
L_z=\mathbb E\left[\log\sum_e\exp(z_e)\right]^2
\]

最终 FM 训练目标：

\[
L_{FM+MOE}=L_{next-event}+\sum_{l\in MOE\ layers}L_{router}^{(l)}
\]

Router Loss 只用于防止路由塌缩和 logits 失控，不能替代主任务 Loss。必须同时记录 expert fraction、entropy 和 overflow；仅看到辅助 Loss 稳定不代表专家健康。

## 5. 下游 Ranker：标签与输出

### 5.1 两阶段边界

```text
T 日因果事件 → 冻结 FM embedding
T 日收盘后可用特征 → Ranker
T+2 执行口径未来收益 → 仅用于 Ranker 训练标签
```

Top-K Loss 不直接替换 FM CE，也不能把未来收益梯度传回 tokenizer/FM。

### 5.2 日内标签

对每个交易日先计算去均值收益：

\[
r_i^{xs}=r_i-\bar r_d
\]

主排序标签为日内收益百分位：

\[
y_i=\frac{rank(r_i^{xs})-1}{N_d-1}\in[0,1]
\]

头部 gain 只强调上半区：

\[
g_i=\left[\max\left(\frac{y_i-0.5}{0.5},0\right)\right]^2
\]

辅助回归目标使用稳健尺度：

\[
a_i=\operatorname{clip}\left(
\frac{r_i^{xs}-median_d}{1.4826\,MAD_d},-3,3
\right)
\]

MAD 退化时回退到日内标准差，再退化时使用 1。

## 6. Multi-K Sampled LambdaNDCG

### 6.1 为什么替换旧 ApproxNDCG

旧全量 ApproxNDCG 需要日内 \(N\times N\) 矩阵，且在输出被 `tanh` 压到窄范围时，近似 rank 容易集中在横截面中部，使 Top-K mask 饱和、梯度变弱。

新方案使用 sampled LambdaNDCG：

- 复杂度由全量 \(O(N^2)\) 降为有界 pair sampling；
- loss 前先对每日分数标准化，避免原始输出尺度限制 pairwise 梯度；
- 同时优化多个 K，降低只针对单一持仓数过拟合；
- pair 权重直接使用交换两只股票造成的 ΔNDCG。

### 6.2 每日分数标准化

\[
s_i=\frac{p_i-\bar p_d}
{\sqrt{\frac{1}{N_d}\sum_j(p_j-\bar p_d)^2}+\epsilon}
\]

### 6.3 Pair 采样

默认每个交易日最多 8192 对：

- 75% hard pairs：真实头部、预测头部和各 K 边界附近；
- 25% global pairs：全横截面随机；
- 真实标签百分位差小于 0.02 的 pair 丢弃；
- 采样 seed 由全局 seed、epoch 和 date 稳定生成；
- canonical order 由数值决定，保持输入行置换不变性。

### 6.4 Multi-K 与权重

默认：

```text
K = (50, 300, 350)
weight = (0.20, 0.60, 0.20)
```

对一个 pair \((i,j)\)，在当前预测排名下计算交换两者造成的：

\[
\Delta NDCG_K=
\frac{|(g_i-g_j)(D(r_i)-D(r_j))|}{IDCG_K}
\]

多个 K 加权后得到 \(\lambda_{ij}\)。Pairwise logistic Loss：

\[
L_{pair}=\frac{\sum_{(i,j)}\lambda_{ij}
\operatorname{softplus}\left(-\frac{s_i-s_j}{\tau}\right)}
{\sum_{(i,j)}\lambda_{ij}}
\]

默认 `score_temperature=1.0`。

## 7. 全局 IC 与辅助 Huber

### 7.1 全局 IC 稳定项

\[
L_{IC}=1-\operatorname{Pearson}(s,y)
\]

它覆盖整个横截面，防止模型只修正 K 边界而破坏整体排序。代码函数名保留 `rank_ic`，但当前可微实现是 Pearson 对百分位标签的相关性，不是对预测再做不可微 hard rank。

### 7.2 独立辅助头

Ranker 有两个输出头：

```text
共享横截面主干
  ├─ out：最终 score
  └─ aux_out：只用于训练的稳健收益预测
```

辅助项：

\[
L_{aux}=\operatorname{SmoothL1}(\hat a,a;\beta=0.5)
\]

它不直接作为交付 score，只通过共享主干提供收益幅度监督。

## 8. Ranker 最终总 Loss 与模型选择

训练总目标：

\[
L_{ranker}=1.0L_{pair}+0.30L_{IC}+0.05L_{aux}
\]

所有 loss 每个交易日独立计算，训练循环逐日更新，因此日期等权，不让大横截面日期自动获得更多权重。

验证阶段计算 exact multi-K NDCG 与 IC，选择分数：

\[
S_{select}=1.0\,NDCG_{multi-K}+0.30\,IC
\]

使用尾部时间验证集，默认 purge 2 个交易日、patience 8。Top-300 实现收益只作为报告指标，不参与 checkpoint 选择，避免把噪声较大的短期收益直接当 early-stop 目标。

## 9. 本轮明确不加入的 Loss

为了保持单因素可解释性，当前正式方案不加入：

- CCC；
- 方向 BCE；
- 主输出 L2；
- bottom-tail loss；
- 换手/交易成本的可微 surrogate；
- 多期限联合目标；
- 未来收益直接微调 FM。

这些项不是永久否定，但必须在现方案形成稳定基线后逐项消融，不能一次堆叠。

## 10. 评估与验收

### 10.1 FM 预训练

至少记录：

- raw CE、normalized CE、ordinal loss、valid count；
- per-field accuracy、NLL、perplexity；
- copy/unigram baseline；
- 主要预测头梯度范数；
- 固定 validation windows 上的结果。

晋级要求不能只看 total loss。V2 主要字段 normalized NLL 不应比对应基线退化超过 1%。

### 10.2 Ranker

至少报告：

- exact NDCG@50/300/350；
- 日均 RankIC、ICIR、Newey-West t；
- paired daily IC bootstrap；
- Top-K 扣成本收益、换手、MDD、相对全市场超额；
- 3–5 seeds 稳定性；
- 按月份、波动和流动性分桶表现。

### 10.3 数据门

- 标签、calendar、PIT universe 和 embedding contract 必须版本化；
- T+2 标签边界 purge=2 个交易日；
- Validation 选模型，Test/OOS 不参与调参；
- 训练和评分使用相同 universe policy；
- 历史 Token 语义不合格时必须重建 embedding，不能仅补 metadata。

## 11. 当前已训练模型盘点

### 11.1 基础模型 Checkpoint

状态定义：

- **完成**：达到配置训练终点并存在 final checkpoint；
- **阶段性**：存在可用 best/step checkpoint，但未达到终点或没有 final；
- **Smoke**：只验证工程链路，不构成正式实验。

| 模型 | 参数/结构 | Token/Loss | 数据与预算 | 当前产物与结果 | 状态 |
|---|---|---|---|---|---|
| Pilot V1 | 6.36M；256×6 | V1，六路等权 CE | 5 日×3 股；20k legacy steps | `pilot/run/best.pt`、`final.pt` | 完成，小样本工程验证 |
| Medium Try V1 | 42.09M；512×10 | V1，六路等权 CE | 5 日×60 股；10k legacy steps | `medium_try/run/best.pt`、`final.pt`；已跑 tiny downstream judge | 完成，但下游样本不足 |
| Medium Try 302M | 约 302.3M；1024×18 | V1，六路等权 CE | 复用 5 日×60 股；计划 10k | 只有 `best.pt`；历史记录约 step 1000、val loss 6.15 | 阶段性，已停止 |
| Medium 302M V1 | 约 302.3M；1024×18 | V1，六路等权 CE | 22 日全市场；40k 旧 micro-step 口径 | `best.pt`、`final.pt`；best val loss 5.3288 | 完成；不能表述为 40k optimizer updates |
| Dense230M V1 | 231.52M；1024×18，FFN 2816 | V1 Token；无显式 V2 Loss；gated fusion | cont60；50k optimizer updates；12.875B non-pad tokens | `best.pt`、`final.pt`、`final_resume.pt`；best val loss 5.8131 | 完成；V1 compatibility baseline |
| V2 25M Smoke | 实际 18.15M；384×10 | V2 normalized/ordinal Loss | 真实 V1 canonical 转 V2 base；5 updates | `best.pt`、`final.pt`、`final_resume.pt`；val loss 4.1333 | Smoke，不是正式 25M |
| Backbone-MoE V1 | 297.59M；顶部4层、4专家Top-1 | V1 next-event CE + Router aux | cont60；计划 50k updates | 最新 38,375；best@38k val loss 5.8386；无 final | 阶段性，当前停止 |

路径均位于 `quant_fm/runs/<name>/run/`。这里的 val loss 不能跨 V1 旧训练循环、Dense230 新训练循环和不同验证采样直接横比；最有效的当前对照是同为 cont60 的 Dense230M V1 与 Backbone-MoE V1，但后者尚未完成。

### 11.2 下游 Ranker

本地存在两份约 1.6MB 的 Ranker checkpoint：

```text
quant_fm/runs/oos2026/delivery_oos/ranker_checkpoint.pt
quant_fm/runs/oos2026_dense230/delivery_oos/ranker_checkpoint.pt
```

两者均为 `in_dim=1024, hidden=128, depth=2, n_heads=4`，训练 history 是 30 个 float IC 值；checkpoint 不含 `objective` 和 `artifact_version`。因此它们是 **legacy Ranker 产物**，不能证明采用了本文的 Multi-K LambdaNDCG 联合目标，也不能作为新的严格 Ranker v2 artifact。

新的 Top-K Loss 代码、标签构造、早停和严格 artifact 契约已经实现，但截至本文基线没有发现一份带冻结 objective/training contract 的正式新 Ranker checkpoint。

### 11.3 尚未训练的正式候选

以下只有配置或计划，不应列入“已训练模型”：

| 候选 | 状态 |
|---|---|
| V2 25M 多 seed | 配置就绪，正式数据/训练未完成 |
| V2 100M | 配置就绪，未训练 |
| V2 Dense230M | 配置就绪，未训练 |
| V2 Backbone-MoE | 配置就绪，未训练 |
| V2 Backbone-MoE 300/60/100 日 | 日期计划与 runbook 就绪，安全数据面未完成，未启动 |
| Temporal Regime-MoE | 模块已实现，无正式 checkpoint |
| 新 Multi-K Ranker v2 | 代码已实现，无可审计正式 checkpoint |

## 12. 推荐下一轮实验

1. 完成 FULL V2 五日全市场数据 Pilot，冻结 vocab、manifest 和 validation windows。
2. 以 25M、2 seeds 做单变量 Loss 消融：
   - V1 等权 CE；
   - entropy normalized CE；
   - normalized CE + ordinal。
3. winner 晋级 100M、3 seeds；同时保持 Dense，不启用 Backbone-MoE。
4. 冻结胜出 FM 与 embedding contract，正式训练 Multi-K Ranker v2。
5. 在 Ranker 基线稳定后测试 Temporal Regime-MoE。
6. 最后比较相同 Token/Loss/预算下的 Dense V2 与 Backbone-MoE V2。

## 13. 代码入口

| 功能 | 文件 |
|---|---|
| FM V1/V2 Loss | `quant_fm/pretrain/heads.py` |
| Loss 配置接入 | `quant_fm/pretrain/train.py` |
| 预训练诊断 | `quant_fm/pretrain/eval.py` |
| Ranker 标签 | `quant_fm/downstream/make_features.py` |
| Multi-K Ranker Loss | `quant_fm/downstream/train_ranker.py` |
| 严格 Ranker 训练 | `quant_fm/signal/train.py` |
| Ranker artifact | `quant_fm/signal/artifact.py` |
| Top-K preflight | `quant_fm/scripts/preflight_topk_ranker.py` |
| 独立 OOS 评估 | `quant_fm/downstream/run_score_evaluation.py` |
