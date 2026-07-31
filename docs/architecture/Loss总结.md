# QuantFM Loss 总结

当前代码并不是同时使用所有“排序loss”，而是从三种主排序loss中选择一种，再叠加若干辅助loss。

默认配置可以写成：

\[ \begin{aligned} L_{\text{total}} =\;& L_{\text{ApproxNDCG@500}}\\ &+0.1L_{\text{CCC}}\\ &+0.05L_{\text{SmoothL1}}\\ &+0.02L_{\text{Direction}}\\ &+0.01L_{\text{Pred-L2}} \end{aligned} \]

LambdaRankIC默认关闭，多期限loss默认关闭。

对应实现位于 [Baseline.py (line 1686)](/root/model_signal/Baseline.py:1686)。

## 一、总体结构

模型有三个输出头：

```
共享模型主干
    ├── 主预测头 pred
    │      用于股票排序、最终signal
    │
    ├── 辅助回归头 aux_pred
    │      用于直接拟合未来收益
    │
    └── 方向分类头 direction_logit
           用于判断未来涨跌
```

各loss作用于不同输出：

|Loss|使用的输出|默认权重|默认状态|
|---|---|---|---|
|ApproxNDCG / ListNet / Pearson IC|主预测头|1.00|三选一|
|CCC|主预测头|0.10|开启|
|Prediction L2|主预测头|0.01|开启|
|LambdaRankIC|主预测头|0.00|关闭|
|Smooth L1|辅助回归头|0.05|开启|
|Direction BCE|方向分类头|0.02|开启|
|多期限排序loss|额外预测头|默认0.3/期限|关闭|

因此辅助回归头和方向头不会直接作为最终signal，但它们会更新共享主干参数，从而间接帮助主排序头。

---

# 二、主排序Loss

通过参数选择：

```
--ic-loss-type approx_ndcg
--ic-loss-type listnet
--ic-loss-type pearson
```

三者互斥，不会同时生效。当前默认是：

```
approx_ndcg
```

## 1. Pearson IC Loss

实现位置：[Baseline.py (line 1147)](/root/model_signal/Baseline.py:1147)

### 计算方法

对每个交易日 \(d\)，计算当天所有股票预测值和真实收益的Pearson相关系数：

\[ IC_d = \frac{ \operatorname{Cov}(p_d,y_d) }{ \sqrt{\operatorname{Var}(p_d)\operatorname{Var}(y_d)} } \]

代码返回负相关系数：

\[ L_{\text{IC}}=-\frac{1}{D}\sum_d IC_d \]

因为优化器执行最小化，所以：

```
IC越高
→ -IC越低
→ loss越好
```

如果当天IC为0.05，对应loss约为：

```
-0.05
```

因此训练日志中出现负loss是正常的。

### 按日计算的重要性

代码不是把所有年份、所有股票混在一起计算相关性，而是：

```
每天分别计算IC
       ↓
对有效交易日等权平均
```

这与横截面选股任务一致。股票较多的日期不会因为行数更多而自动获得更大权重。

每天至少需要5个有效股票，否则该日跳过。

### 优点

- 计算简单；
- 梯度相对稳定；
- 直接优化横截面线性相关性；
- 对预测的整体放大、缩小和平移不敏感；
- 适合作为基础排序loss。

### 局限

Pearson IC不是Spearman Rank IC。

例如：

```
真实收益排名：A > B > C
预测值：      A > B > C
```

只要数值关系近似线性，Pearson会很高；但它没有直接优化股票名次。

另外，Pearson IC平等关注整个横截面：

```
第10名和第20名排错
第2000名和第2010名排错
```

两者都会影响loss。对于只持有Top-K的策略，它没有特别强调头部。

### 适用场景

- 建立最简单、稳定的基线；
- 观察模型是否具备整体横截面预测能力；
- 与NDCG结果对照；
- NDCG训练不稳定时作为替代。

---

## 2. ListNet Loss

实现位置：[Baseline.py (line 1272)](/root/model_signal/Baseline.py:1272)

ListNet把一天的全部股票看作一个完整列表。

### 第一步：把真实收益转换成目标概率

对当天股票真实收益 \(y_i\) 做softmax：

\[ q_i = \frac{\exp(y_i/\tau)} {\sum_j \exp(y_j/\tau)} \]

其中：

```
τ = listnet_temperature
默认值 = 0.1
```

收益越高的股票，目标概率越大。

### 第二步：把模型预测转换成预测概率

\[ \hat q_i = \frac{\exp(p_i)} {\sum_j \exp(p_j)} \]

### 第三步：计算交叉熵

\[ L_{\text{ListNet}} = -\sum_i q_i\log \hat q_i \]

然后对不同交易日等权平均。

### Temperature的作用

`true_temperature`只作用于真实收益分布：

```
temperature较小
→ 目标概率集中在少数高收益股票
→ 更重视头部

temperature较大
→ 目标概率更均匀
→ 更重视整体列表
```

例如：

```
--listnet-temperature 0.05
```

会比默认0.1更关注高收益股票。

### 优点

- 整个列表共同参与训练；
- 比直接回归收益更关注相对排序；
- 对单只股票极端收益的敏感度通常低于MSE；
- 全程可微；
- 不需要计算硬排序。

### 局限

它没有明确的Top-K截断。所有股票都会参与softmax，只是高收益股票权重更大。

当前主预测被限制在：

```
[-0.1, 0.1]
```

因此预测softmax中任意两只股票最大概率比约为：

\[ \frac{e^{0.1}}{e^{-0.1}}=e^{0.2}\approx1.22 \]

也就是说，主预测softmax很难变得非常集中。当前输出范围对ListNet的表达能力存在一定限制。

如果使用ListNet，更合理的方式是：

- 使用未经过 `tanh` 限制的raw logit计算loss；
- 最终导出signal时再做有界变换；
- 或者给预测softmax也设置温度参数。

---

## 3. ApproxNDCG@K Loss

实现位置：[Baseline.py (line 1336)](/root/model_signal/Baseline.py:1336)

这是当前默认的主排序loss。

NDCG原本用于信息检索，目标是让真正“相关性高”的对象出现在排名前面。对应到股票任务：

```
一个交易日 = 一个查询列表
当天股票 = 待排序对象
未来收益排名 = relevance
模型signal = 排序分数
```

### 第一步：将真实收益转换成relevance

代码先按当天真实收益降序排序，然后赋予线性相关度：

```
真实收益最高 → relevance = 1.0
真实收益最低 → relevance = 0.0
中间股票     → 在0～1之间线性分布
```

它只使用真实收益名次，不使用收益幅度。

因此：

```
第一名收益10%
第二名收益2%
```

和：

```
第一名收益2.1%
第二名收益2.0%
```

在relevance名次上的差异相同。

这个设计可以降低极端收益值对loss的直接影响，但也丢失了收益差距信息。

### 第二步：近似模型预测名次

真实排序操作不可微，所以代码使用成对sigmoid近似一只股票前面有多少股票：

\[ \widetilde{rank}_i = 0.5+\sum_j \sigma\left( -s(p_i-p_j) \right) \]

其中：

- \(p_i\) 是股票i的预测分数；
- \(s\) 是 `ndcg_sigma`，默认1；
- 如果 \(p_j > p_i\)，股票j会增加股票i的近似名次；
- 预测最高的股票应接近rank 1；
- 预测最低的股票应接近rank N。

### 第三步：排名折扣

\[ discount_i = \frac{1}{\log_2(\widetilde{rank}_i+1)} \]

排名越靠后，贡献越小。

### 第四步：Soft Top-K Mask

代码没有直接使用不可微的“rank是否小于K”，而是使用：

\[ mask_i = \sigma\left[ s(K+0.5-\widetilde{rank}_i) \right] \]

效果是：

```
预测名次明显小于K → mask接近1
预测名次明显大于K → mask接近0
预测名次接近K     → mask处于0～1
```

### 第五步：计算DCG和NDCG

\[ DCG = \sum_i relevance_i \times discount_i \times mask_i \]

再除以理想排序下的最大DCG：

\[ NDCG=\frac{DCG}{IDCG} \]

最终loss为：

\[ L_{\text{NDCG}}=-NDCG \]

NDCG越高，loss越低。

### 优点

- 明确关注Top-K；
- relevance来自收益名次，对极端收益更稳健；
- 与Top-K选股逻辑比Pearson更接近；
- 中间和尾部股票排错的影响较小。

### 计算成本

代码会为每天构造：

```
股票数 × 股票数
```

的预测差值矩阵，因此复杂度约为：

\[ O(N^2) \]

如果一天有4000只股票，就会构造约1600万个成对元素。`dates-per-batch`较大时，计算和显存压力会明显增加。

### 当前配置存在的重要问题

当前同时使用：

```
output_scale = 0.1
ndcg_sigma = 1.0
K = 500
训练股票池 = 全A股，每天约2000～4800只
```

主预测被限制在 `[-0.1,0.1]`，所以任意预测差最大只有0.2：

\[ \sigma(p_j-p_i) \in \sigma([-0.2,0.2]) \approx[0.45,0.55] \]

这意味着近似名次会被压缩在横截面中间附近。例如一天4000只股票，很多近似名次可能集中在大约：

```
1800～2200附近
```

而不是覆盖1～4000。

此时 `K=500` 的Top-K mask可能接近：

\[ \sigma(500.5-1800)\approx0 \]

即大量股票的Top-K mask饱和到0，导致ApproxNDCG梯度非常弱。

这意味着：**当前默认ApproxNDCG可能没有代码注释所描述的那么有效，实际模型能力可能主要来自CCC、Smooth L1和方向辅助loss。**

这个问题尤其值得优先检查。

### 建议修正方式

更合理的选择包括：

1. 使用未经过 `tanh` 限制的raw prediction计算NDCG；
2. 仅在导出signal时使用有界输出；
3. 对每天预测先做截面标准化，再计算近似排名；
4. 显著提高 `ndcg_sigma`，但需要系统调参；
5. 将主loss限制到当日中证1000，减少横截面规模并与目标股票池对齐；
6. 记录每项loss和梯度范数，确认NDCG是否真的产生有效梯度。

不建议仅凭总loss变化判断NDCG有效，因为总loss还包含其他辅助项。

---

# 三、辅助Loss

## 4. CCC Loss

实现位置：[Baseline.py (line 1195)](/root/model_signal/Baseline.py:1195)

CCC是Concordance Correlation Coefficient，一致性相关系数。

对每个交易日：

\[ CCC_d = \frac{ 2\operatorname{Cov}(p,y) }{ \operatorname{Var}(p) +\operatorname{Var}(y) +(\mu_p-\mu_y)^2 } \]

对应loss：

\[ L_{\text{CCC}} = -\frac{1}{D}\sum_d CCC_d \]

默认权重：

```
0.1
```

### 与Pearson IC的区别

Pearson相关系数：

\[ \rho = \frac{Cov(p,y)} {\sqrt{Var(p)Var(y)}} \]

只关心线性相关方向，不关心预测均值和尺度。

例如：

```
prediction = 100 × true_return
```

Pearson仍可能接近1。

CCC还会惩罚：

- 预测方差与真实收益方差不同；
- 预测均值与真实收益均值不同。

因此CCC同时要求：

```
走势相似
均值接近
尺度接近
```

### 它对当前模型的作用

主排序loss主要学习相对名次，CCC给主预测头补充收益尺度信息，使主预测不只是任意尺度的排序分数。

但最终signal主要用于排名，所以CCC只是辅助项，权重0.1相对合理。

### 潜在冲突

如果最终只关心排序，强行要求signal均值和收益尺度一致未必必要。特别是模型输出被限制在 `[-0.1,0.1]`，CCC会推动主signal同时承担“排序分数”和“收益估计”两个角色。

建议测试：

```
CCC weight = 0
CCC weight = 0.05
CCC weight = 0.1
```

观察Top-K收益，而不只观察验证loss。

---

## 5. Smooth L1辅助回归Loss

实现位置：[Baseline.py (line 1734)](/root/model_signal/Baseline.py:1734)

Smooth L1作用于独立的辅助回归头：

```
aux_pred
```

而不是最终主signal。

误差：

\[ e=aux\_pred-y \]

当前设置：

```
beta = 0.01
weight = 0.05
```

Smooth L1公式：

\[ L(e)= \begin{cases} \frac{e^2}{2\beta}, & |e|<\beta\\ |e|-\frac{\beta}{2}, & |e|\ge\beta \end{cases} \]

### 含义

当误差小于1%时，近似二次损失：

```
鼓励精细拟合
梯度平滑
```

当误差超过1%时，转成近似绝对误差：

```
降低极端涨跌股票的影响
避免MSE被少数异常收益主导
```

### 为什么需要辅助回归头

排序loss只告诉模型股票之间谁应该更高，不一定能学到收益幅度和方向。

Smooth L1提供逐股票监督：

```
这一只股票未来收益大约是多少
```

它可以让共享主干学习更丰富的收益结构。

### 与主signal的关系

最终预测使用主头，不使用 `aux_pred`。Smooth L1的作用路径是：

```
Smooth L1
    ↓
更新辅助头和共享主干
    ↓
共享主干变好
    ↓
间接影响主排序头
```

### 加权方式

它是逐股票计算后求平均，不是先按日平均。

因此股票数量较多的日期会贡献更多样本。这个口径与按日等权的NDCG、Pearson、CCC不同。

---

## 6. Direction BCE Loss

实现位置：[Baseline.py (line 1744)](/root/model_signal/Baseline.py:1744)

方向头预测股票下一日是否上涨：

\[ target_i = \begin{cases} 1,&y_i>0\\ 0,&y_i\le0 \end{cases} \]

模型输出未经sigmoid的logit：

```
direction_logit
```

使用二元交叉熵：

\[ L_{\text{dir}} = -[z\log\sigma(a)+(1-z)\log(1-\sigma(a))] \]

默认权重：

```
0.02
```

### 作用

它给共享主干一个更简单的辅助任务：

```
不要求预测精确收益
只判断涨跌方向
```

在收益回归噪声较大时，方向分类可能提供相对稳定的监督。

### 局限

中证1000增强是横截面相对收益问题，而Direction BCE预测的是绝对涨跌：

```
某股票上涨1%，指数上涨2%
```

方向标签为正，但这只股票实际跑输指数。

因此方向loss不完全对应超额收益目标。未来若目标明确是指数增强，可以比较：

```
原始方向：
股票收益 > 0

超额方向：
股票收益 - 中证1000收益 > 0

风险残差方向：
residual_return > 0
```

后两种与超额目标更一致。

Direction BCE同样作用于独立方向头，只通过共享主干间接影响主signal。

---

## 7. Prediction L2 Loss

实现位置：[Baseline.py (line 1714)](/root/model_signal/Baseline.py:1714)

对主预测分数计算：

\[ L_{\text{pred-L2}} = \frac{1}{N}\sum_i p_i^2 \]

默认权重：

```
0.01
```

### 作用

- 抑制过大的预测值；
- 减少signal极端化；
- 在相关性loss梯度不稳定时提供约束；
- 提高不同日期预测分布的稳定性。

### 当前实际影响较小

主预测已经通过：

\[ 0.1\tanh(raw/0.1) \]

限制在 `[-0.1,0.1]`。

因此：

\[ p^2\le0.01 \]

再乘权重0.01，最大量级约为：

\[ 10^{-4} \]

所以当前Prediction L2大概率只是非常弱的正则项。

不要把它与AdamW的weight decay混淆：

- Prediction L2惩罚模型输出；
- AdamW weight decay惩罚模型参数。

---

## 8. Sampled LambdaRankIC Loss

实现位置：[Baseline.py (line 1429)](/root/model_signal/Baseline.py:1429)

默认：

```
weight = 0
```

因此当前没有启用。

它直接针对股票成对排序和Spearman Rank IC设计。

### 第一步：每天随机采样股票对

默认每个日期最多：

```
4096对
```

而不是计算全部 \(N(N-1)/2\) 对。

### 第二步：确定正确顺序

如果真实收益：

\[ y_i>y_j \]

模型应该满足：

\[ p_i>p_j \]

### 第三步：成对Logistic Loss

\[ L_{ij} = \log\left( 1+\exp[-s(p_i-p_j)] \right) \]

也就是代码中的：

```
softplus(-sigma * (p_i - p_j))
```

如果顺序正确且差距足够大，loss较小；顺序相反，loss较大。

### 第四步：按Rank IC交换影响加权

代码计算如果交换股票i和j的预测名次，对Spearman Rank IC会造成多大影响：

\[ \Delta_{ij} = \frac{ 12 |r^p_j-r^p_i| |r^y_i-r^y_j| }{ N(N^2-1) } \]

名次差距越大、真实排名差距越大的股票对，权重越高。

### 优点

- 比Pearson更直接针对排序；
- 比全量两两比较便宜；
- 重点修复会显著影响Rank IC的错误顺序；
- 可以作为NDCG之外的排序辅助项。

### 局限

- 随机采样会增加训练噪声；
- 当前预测rank和pair权重是detach的，只在分数差上反向传播；
- 它优化整体Spearman交换影响，不专门针对Top-K；
- 默认主预测范围较窄，pairwise分数差也被限制在0.2以内。

适合先用较小权重测试：

```
0.01、0.05、0.1
```

并记录训练稳定性和Top-K结果。

---

# 四、多期限Loss

当前默认关闭。通过：

```
--multi-horizons 5,10
--multi-horizon-weights 0.3,0.2
```

可以增加额外预测头。

假设启用t+5和t+10：

\[ L = L_{t+1} +0.3L_{t+5} +0.2L_{t+10} +\text{t+1辅助项} \]

每个额外期限使用与主头相同类型的排序loss：

- ApproxNDCG；
- ListNet；
- 或Pearson IC。

但CCC、Smooth L1、方向分类、Prediction L2和LambdaRankIC只作用于主t+1头，不作用于额外期限头。

### 多期限标签的实际定义

代码不是精确复利收益，而是把未来每日简单收益相加：

\[ y_{t,h} = r_{t,t+1} +r_{t+1,t+2} +\cdots +r_{t+h-1,t+h} \]

这是小收益情况下对累计收益的近似。严格累计收益应为：

\[ \prod_{j=1}^{h}(1+r_j)-1 \]

如果未来正式使用多期限，建议考虑改成复利标签。

---

# 五、默认总Loss的实际解释

当前默认：

```
主排序：ApproxNDCG@500
辅助一致性：CCC
辅助收益预测：Smooth L1
辅助涨跌预测：Direction BCE
输出正则：Prediction L2
```

可以理解为：

```
ApproxNDCG：
把真正高收益股票排到前500

CCC：
让主signal和收益在相关性、均值、尺度上更接近

Smooth L1：
让共享主干具备逐股票收益预测能力

Direction BCE：
让共享主干能识别绝对涨跌

Prediction L2：
防止主signal过于极端
```

其中：

- NDCG、Pearson、CCC按交易日计算后等权平均；
- Smooth L1、Direction BCE和Prediction L2按股票样本平均；
- 当前没有MetaWeightNet，所以训练时 `weights=None`，所有股票样本统一权重。

---

# 六、Loss与验证指标并不完全一致

训练默认优化ApproxNDCG@500，但early stopping默认依据验证集Spearman Rank IC：

```
训练目标：ApproxNDCG@500 + 辅助loss
模型选择：验证集Rank IC
最终用途：中证1000 Top500收益
```

这里存在三层不完全一致：

1. 全A NDCG@500不等于中证1000 NDCG@500；
2. NDCG@500不等于全截面Spearman Rank IC；
3. Rank IC不等于扣费后的Top500超额收益。

因此未来更合理的模型选择方式是同时记录：

```
中证1000 NDCG@K
中证1000 Rank IC
Top50/100/200/500收益
Top500 - Bottom500
换手和扣费后超额
```

不能只用一个总loss判断模型优劣。

# 七、当前最应该优先检查的事项

在继续调loss权重前，我建议先完成以下检查：

1. 分别记录每个loss的原始值和加权后值。
2. 分别计算每个loss对主干参数的梯度范数。
3. 检查默认ApproxNDCG的Top-K mask分布。
4. 检查预测近似rank是否集中在横截面中间。
5. 比较关闭NDCG后，仅用CCC、Smooth L1和方向loss的结果。
6. 比较Pearson、ListNet和修正后的NDCG。
7. 将主排序loss和验证指标限制在历史中证1000范围内。

尤其是第3和第4项。根据当前 `output_scale=0.1`、`ndcg_sigma=1` 和全A股横截面规模，默认ApproxNDCG存在明显的梯度饱和风险。在确认这一点之前，继续微调 `CCC=0.1` 或 `SmoothL1=0.05` 的收益可能有限。
