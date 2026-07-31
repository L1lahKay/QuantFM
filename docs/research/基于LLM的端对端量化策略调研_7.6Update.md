# 基于LLM的端对端量化策略调研Report   7\.6Update

> 文档定位：本文保留 7.6 时点的调研背景与方案演进，其中 `cn_l2_v1` 示例不是当前 V2 产物契约。截至 2026-07-24，代码已增加因果盘口、Tokenizer V2、可配置字段融合/损失、多尺度聚合和 PIT 股票间上下文模块；实施真值以 [模型底层 V2 代码改造指导](../architecture/模型底层v2代码改造指导.md) 和 [严格 OOS 研究回测](../evaluation/严格OOS研究回测.md) 为准。V2 尚未完成正式多 seed 训练或新 untouched OOS 验收。

# 核心定义

**浓缩版更新：**本项目的核心不是让大模型直接预测价格或直接吐出 Top\-K 股票，而是把 TradeFM\-like 模型定位为 A 股 Level\-2 订单流的自监督预训练模块。它负责从 order / trade / snapshot 重建出的标准事件流中学习可迁移的微观结构 embedding；最终选股仍由下游截面 ranker、meta\-labeling 和风控模块完成。

- 目标：用订单流预训练 embedding 为 T\+1 截面 Top\-K 选股提供增量信息。

- 原则：不预测绝对价格，优先学习相对强弱、交易可靠性和可执行收益。

- 验收：看 embedding 是否提升 RankIC、ICIR、分组单调性和扣成本后收益。



> 原始调研中的图片/PDF 附件未随代码仓库分发。本文保留文字结论，附件名称仅作来源记录。

原附件：`image.png`

Φ：原始数据\|\|特征因子

Y：预测目标

η：从Φ到预测目标的解释效率

I：Φ 对目标 Y 提供了多少有效信息

H：目标 Y 本身的不确定性

我们的目标是拉高η，要么让预测目标Y相对简单 拥有可解释性，要么增强Φ的信息密度

# Ref

### TradeFM: A Generative Foundation Model for Trade\-flow and Market Microstructure

原附件：`TradeFM.pdf`

#### 把异构的 high\-frequency order flow 变成统一的离散序列，让 decoder\-only Transformer 像预测下一个词一样预测下一个市场事件。单个事件是一个多字段 tuple，包括事件间隔时间、价格深度、成交/订单量、动作类型、买卖方向等。

```Plain Text
event = {
  interarrival_time,  // 距离上一事件的时间间隔
  price_depth,        // 订单价格相对 mid-price 的深度
  volume,             // 订单/成交量
  action,             // add 或 cancel
  side                // buy 或 sell
}
```

#### 尺度不变表示和词表处理

把预训练模型和确定性的市场模拟器相结合

成交量和到达时间间隔需要做对数处理

#### 中间价估计

对于真实市场中间价的正确估计非常重要，这有助于对与价格相关的特征进行标准化处理。在我们这种部分信息的情况下，中间价是不可观测的；我们只能看到有噪声的执行价格。在基于经典体积加权平均价的方法基础上，我们引入了指数加权平均价这一时间感知的估计方法，它能够同时考虑交易量和最新交易情况： p^tEW\-VWAP=EMA\(ptexec⋅vt\)/EMA\(vt\) 。平滑因子 α 由基于时间的半衰期决定，这样就能让较大和较新的交易得到更多的权重，从而提供一个稳定且具有代表性的价格基准。与简单的滚动平均值不同，指数加权平均价能够在流动性差异很大的不同资产之间进行比较

#### 预训练工作

基于Llama

在配备 3 个 Nvidia A100 GPU 的 AWS 实例上训练模型，每个 GPU 拥有 80GB 的 RAM。所有训练操作都采用 fp16 半精度格式进行。为了达到有效的批量大小为 4,032，我们采用了每设备 24 个样本的批量大小策略，并且梯度累积采用 56 个步骤的方式。为了进一步优化内存使用和加速训练过程，我们使用了 Accelerate 库。模型训练时使用了 AdamW 优化器，具有线性的学习率调度机制，学习率为 5×10−5 ，并且包含了 500 个 warmup 步骤。根据针对大型数据集训练的建议（Muennighoff 等人，2023 年），我们总共进行了 4 个训练周期。

### Generative AI for End\-to\-End Limit Order Book Modelling: A Token\-Level Autoregressive Generative Model of Message Flow Using a Deep State Space Network

原附件：`LOB Modelling.pdf`

### TSFM调研

原附件：`从零构建词表并对金融订单簿与市场交易数据进行端到端重分词预训练.pdf`

Latest LLM Pretrain

原附件：`端到端金融订单簿与逐笔交易预训练框架研究报告.pdf`

# Benchmark

LOB\-Bench: Benchmarking Generative AI for Finance – an Application to Limit Order Book Data

# End2End

**建议主链路：**

```text
A 股 Level-2 order / trade / snapshot
→ 沪深规则适配与订单簿重建
→ 标准化事件流
→ 字段级 tokenizer
→ next-event 自监督预训练
→ stock-window / stock-day embedding
→ 截面排序模型
→ Top-K 候选
→ meta-labeling / 风控 / 组合管理
```

这里 TradeFM\-like 模型只负责“学市场语言”，不直接负责下单或输出最终股票名单。

### 输入

A股SH SZ的快照 订单流 事件流，后者能承载快照丢失的微结构信息

### 输出

分析订单簿（Order Book, L2 行情）的微观不平衡（Imbalance），预测未来的微观价格趋势。有时序性，也是我们的主要生成的策略。

# trick

- **Tokenizer：**避免直接 token 化股票代码、绝对价格、原始 order\_id；优先使用相对价档、log volume、delta\_t、session\_phase、limit\_status、exchange、liquidity\_bucket 等字段。

- **词表：**不建议使用巨大 composite vocab，更推荐字段级 embedding 加多头预测，分别预测 event\_type、side、price\_bin、volume\_bin、delta\_t\_bin 等。

- **标签：**下游 ranker 建议使用 T\+1 可成交 VWAP 口径的截面超额收益 rank，而不是单股票绝对涨跌。

- **风控：**Top\-K 之后再用 meta\-labeling 估计 P\(correct\)，过滤低置信度交易并控制仓位。

### 1、不预测价格本身，而是判断该不该交易

预测价格本质上是在要求模型恢复大量不可预测的信息，价格是市场状态高度压缩后的结果，历史价格里确实有一点点可交易信息，但绝大多数是不可恢复噪声。

最好去预测方向、排序、触发止盈/止损、相对强弱、风险状态、可执行收益。

因此要多做打标，增加信息熵，**压低标签里的无意义噪声。**

### 2、利用深度强化学习进行一阶优化（梯度反向传播）

把写一个因子看成写一个句子，一个因子表达式就是一串token，让自回归Transformer把因子一个一个吐出来。

不进行预训练，直接强化学习，用回测当奖励，用策略梯度去更新网络。整个系统里没有一个标准答案，只有市场给的分数。把一个长在离散符号空间里、目标不可微的搜索问题搬进了 Transformer 那个连续可微的参数空间，用上了一阶优化的全套机器（实践中是 PPO 或 GRPO 这类带方差削减的进阶版策略梯度，加 baseline、加 clip，来解决方差大问题）

缺点是回测分数极其稀疏、极其嘈杂、还极其昂贵：网络写一个因子，要跑一整趟历史回测才拿得到一个标量，无法做credit assignment

### 3、截面采用去掉RoPE的Transformer

把当天 500 只股票作为一个集合\(set\)整个喂进去,每只票是一个 token,token 的特征就是它的 OHLCV \+ 上一节 stacking 出来的因子。注意力机制让每只票在被打分时,都能看到当天所有其他票,并自动学会我该和谁比、和谁的相对关系重要，这恰恰就是截面相对比较的本相。

去掉位置编码后的置换等变性\(permutation equivariance\)。Transformer 本来是为有顺序的序列\(语言\)设计的,所以配了位置编码去告诉它”谁在第几个”。可截面问题里,股票没有顺序——你把 500 只票的输入顺序打乱,这件事在金融上没有任何意义,输出理应原样跟着打乱、每只票的预测值分毫不变。

为防止过拟合 transfomer满足以下特点

1. 层数浅\(2–4 层足够,不堆几十层\)、宽度窄、注意力头不多

2. dropout 开得比 NLP 里重得多

3. rmsnorm\+low rank decay

### 4、市场和交易分开拟合

金融数据信噪比非常低，如果采用大参数模型的话拟合的都是市场中的噪音，因此选取小参数模型。而小参数模型的泛化能力不足，难以说端到端的既预测市场走向，又调整持仓仓位。因此将两个任务分给两个模型来完成，第一个模型输入结构化数据，输出hidden layer，与验证有效的因子拼接，共同输入给第二个模型。

### 5、自动保持时效性

根据每天传入的L2数据自动清洗，自动完成LoRA微调，以天为单位重新部署

### 6、重做词表和不重做两套方案一起训

# Milestones

1. 数据重建：清洗沪深 Level\-2 order / trade / snapshot，完成事件排序、订单簿重建和快照对账。

2. Tokenizer v1：构造字段级 token 序列，先覆盖 ADD / CANCEL / TRADE、side、price\_depth、volume、delta\_t、交易阶段和涨跌停状态。

3. 预训练：训练 50M–100M 级 OrderFlow FM，主任务为 next\-event field prediction，产出 stock\-day embedding。

4. 下游验证：冻结 embedding，和传统因子拼接训练截面 ranker，对比 baseline 与 embedding 增量。

5. 交易过滤：加入 meta\-labeling、不可交易过滤、成本、滑点、容量和回撤控制，再考虑低学习率联合微调。

### 确定可选方案 多线并进

做截面top\-k，把单股低 IC放大成可观 IR，顺带对冲掉市场 beta。 **不赌大盘方向，赌相对强弱排序**。

### 打标\&\&数据清洗

#### 三重屏障

对持有约 1 日用首达时间替代固定终点，滤掉终点噪声：

- 上屏障（止盈）= \+u·σ\_t，下屏障（止损）= −d·σ\_t，垂直屏障 = 持有上限 h。

- **屏障按当时已实现波动 σ\_t 缩放**（高/低波动期标签可比，本质是标签层的单位无关化）。

- 按哪道屏障**先被触到**打标 → 标签更贴近真实可实现盈亏，抬高 I\_V，同时仍是低 H 的离散标签。

在日频持有下，用日内路径判定首达；这把T\+1终点恰好回调造成的伪负样本滤掉。

#### Meta\-labeling（方向与力度分离）

把交易决策显式分解：F\(x\) = s\(x\)·m\(x\)，方向 s∈\{±1\}，力度 m∈\[0,1\]。

- **主模型**（可以是 LLM/因子给的截面排序，甚至朴素规则）只定方向/选 top\-k，**刻意高召回，防止假阴性**

- **次级模型**（小而强正则的二分类）只学 **P（主模型这次判断的可信度），**用它做二次判断与仓位控制。

top\-k 选股给方向，次级模型按 P\(correct\) 决定每只的权重与是否放弃。

#### 数据清洗

- 沪深双订单簿重建（深市撤单在 trade、channel\+ApplSeqNum 排序、buy\_id/sell\_id 关联；沪市撤单在 order、乱序与即时全成交特例），**用 3 秒快照对账校验到逐档一致**。

- Point\-in\-time、复权对齐、剔除幸存者偏差（含退市/停牌标的）。

- 因果归一化（rolling/expanding，**绝不全样本**）；价格 ÷10000 还原。

- 列存（DolphinDB 做重建\+实时、ClickHouse/Parquet 做训练样本仓）；预处理产物落盘。

**喂给计算的数据，只读、对齐、不包含未来信息**。

### 搭Pipeline

原附件：`v3_data_flow_with_validation_gate.png`

#### 核验

**样本外留出** \+ 与已有因子的相关性闸门（每个因子单独闯关）。

**组合净化交叉验证 CPCV**：purge（删训练集中标签区间与测试集重叠的样本，三重屏障跨多日必须删）\+ embargo（测试段后再禁用一小段，掐断自相关渗漏），产出分布而非单点。

**Deflated Sharpe Ratio（DSR）**：拿候选夏普去和「N 个噪声策略的最大夏普」比，扣掉「搜出来的运气」；试得越多、越分散，门槛越高。

**先有逻辑，后有因子**：经济学先验在搜索前就砍小假设空间 → 降低 N → 降低 √\(2lnN\) 。讲不出道理的高 IC 表达式，默认是噪声极值，除非跨品种/跨时段反复复现

#### 持仓组合

ERC或者可运用强化学习优化？待施工

### 模型训练

原附件：`embedding_to_feature_matrix_to_models.png`

#### 模型1 订单流 LLM（市场表征学习）

- **范式**：自监督，**不碰交易标签**。这是它和其余模型最大的不同——它在海量 intraday 事件流上学"市场语言"，不需要人工标注。

- **数据**：三年全市场、重建并 token 化后的事件序列（数十亿 token 量级），按窗口切片（如 2048–4096 事件/窗）。

- **两阶段训练**：先训 tokenizer（VQ/BSQ，损失 = 重构 \+ 量化项，盯码本利用率防坍缩）→ 再训主干（decoder\-only，损失 = 各字段**下一 token 交叉熵之和**：事件类型/方向/价档/量桶/时间桶各算一个 CE）。

- **架构与正则**：小（\~50–100M）、做弱（强 dropout、weight decay）；分层（窗口编码 → 日级注意力池化）；bf16 \+ 梯度裁剪 \+ FSDP/DeepSpeed \+ AdamW \+ warmup/cosine；课程式（embedding 先行 → 全量）。

- **产物**：每股每日 embedding（池化），之后**冻结**或低学习率微调。

- Result：预训练看 perplexity，**真正的验收是下游**——它的 embedding 能不能让主模型的 RankIC 显著高于不含它时。若兼作模拟器，还要看它能否复现尖峰厚尾、波动聚集等 stylized facts。

#### 模型2 方向模型

**目标**：吃特征矩阵 Φ（LLM embedding ⊕ 手工因子 ⊕ 多周期因子），给全市场打分排序、选 top\-k。**只定方向，刻意高召回**（宽松取前 10%\~15%）。

**标签**：T\+1 截面超额的**秩**（可成交 VWAP 口径、截面去均值），或三重屏障的方向。

**损失**：截面 **RankIC / IC loss**（`L = -corr(预测, 标签)`，按"每天一个截面"算）

**架构**：在 LLM embedding 上接一个轻量头（MLP 或带截面注意力的浅 transformer，可与 LLM 端到端低 LR 微调0

**验收**：RankIC/ICIR、分组单调性。

#### 模型3 力度模型

让训好的主模型在历史上跑一遍，得到它每天挑的名单，再对照后来的真实结果，给每只候选打一个"主模型这次对没对"的标签（0 或 1），用这堆标签训一个很小的二分类模型，学"在什么情况下主模型靠谱"。

### 回测上线监控

- 严格按时间切分回测，扣成本 \+ 涨跌停不可成交 \+ T\+1 \+ 停牌/ST/次新剔除；最近 3–6 个月 holdout。

- 仿真盘：实时特征流水线复用离线同一套重建\+归一化代码，逐日对账防 train\-serve skew。

- 小资金灰度 → 放量；程序化交易报备。

- 监控闭环：因子滚动表现监控、衰减降权淘汰；e\-过程/检验鞅 kill\-switch 探测信号死亡。

# 目前进度

**建议 MVP 聚焦一个可验证假设：**A 股 Level\-2 订单流预训练 embedding 是否能为下游截面选股提供稳定增量。

- 数据范围：先用最近 1–3 年、连续竞价阶段样本，过滤 ST、停牌、次新、涨跌停不可成交样本。

- 模型范围：context length 先取 2048，embedding 先取 128 / 256 维，不急于端到端大模型化。

- 评估范围：以 RankIC、ICIR、Top\-K 扣成本收益、换手率、最大回撤和样本外稳定性为主。

### 词表构建

词表构建的目标，是把 A 股 Level\-2 的 order / trade / snapshot 统一转成可训练的离散事件表示。第一版不建议直接 token 化股票代码、绝对价格或原始订单号，而应围绕“盘口行为”构建词表。

先设计原始的一级词表 再训练浓缩 得到后面的二级词表

**1\. 事件输入字段**

- 连续字段：相对价格或价档、盘口深度、成交量、事件时间间隔。

- 类别字段：交易动作、买卖方向、参与方或主动方、交易所、交易阶段、涨跌停状态。

- 订单编号类字段只用于订单簿重建和引用关系生成，不作为语义 token。

**2\. 连续字段分箱**

- 对 price / depth / volume / time 等连续字段先清理 NaN 和无穷值。

- 对双侧字段使用 1% / 99% 分位数截尾；单侧字段使用 99% 上界截尾。

- 分箱方式可选等频分箱或直方图分箱，第一版建议每个核心字段取 32 或 64 个 bin。

- volume 建议先做 `log1p(volume / 100)`，time 建议使用 `log1p(delta_t)`，price 建议使用相对 mid price 或 tick distance。

**3\. 类别字段映射**

- `action`: ADD / CANCEL / TRADE / SNAPSHOT\_SYNC / STATUS

- `side`: BUY / SELL / UNKNOWN / BUY\_AGGRESSOR / SELL\_AGGRESSOR

- `exchange`: SH / SZ

- `session_phase`: OPEN\_AUCTION / CONTINUOUS\_AM / CONTINUOUS\_PM / CLOSE\_AUCTION 等

- `limit_status`: NORMAL / NEAR\_LIMIT\_UP / AT\_LIMIT\_UP / NEAR\_LIMIT\_DOWN / AT\_LIMIT\_DOWN

**4\. Composite token 公式**

如果沿用 TradeFM 风格的单 token 方案，可将多字段笛卡尔积编码成一个整数：

```text
T =
I_action × N_side × N_depth × N_volume × N_time
+ I_side × N_depth × N_volume × N_time
+ I_depth × N_volume × N_time
+ I_volume × N_time
+ I_time
```

这个方案实现简单，但字段一多词表会快速膨胀，长尾 token 很难学好。

**5\. 推荐方案：字段级 tokenizer**

更推荐把每个事件表示为多个字段 token：

```text
[event_type, side, price_bin, depth_bin, volume_bin, delta_t_bin, session_phase, limit_status, exchange]
```

模型侧对每个字段分别做 embedding，再加和或拼接：

```text
event_embedding =
Embed(event_type)
+ Embed(side)
+ Embed(price_bin)
+ Embed(volume_bin)
+ Embed(delta_t_bin)
+ ...
```

训练时使用多头预测，分别预测下一事件的 event\_type、side、price\_bin、volume\_bin、delta\_t\_bin 等字段。这样既保留 TradeFM 的离散事件建模思想，又避免巨大 composite vocab 带来的稀疏问题。

# 阶段性成果

### 7\.3

调研TSFM方案 近期大厂对于模型预训练新理解

读出SH SZ的快照 订单 交易数据，并撮合成订单簿。调研测试词表大小、 模型参数量合理区间。日期 股票号作为索引 不加入词表。

数据整理按照每只股票每天分类，单独重建订单簿（数量级太大 只取了sample出来）

项目架构be like：

```Plain Text
quant_fm/
  data/
    raw/
    cleaned/
    events/
    tokens/
    embeddings/

  lob_rebuild/
    adapters/
      sh_adapter.py
      sz_adapter.py
    replay.py
    snapshot_check.py

  tokenizer/
    fit_bins.py
    tokenize_events.py
    vocab.py
    transforms.py

  pretrain/
    dataset.py
    model.py
    heads.py
    train.py
    config.yaml

  embedding/
    extract_hidden.py
    pool_stock_day.py

  downstream/
    make_features.py
    train_ranker.py
    backtest_topk.py
    evaluate.py
```

### 7\.6

设计数据schema be like,把不同市场映射到同一套抽象空间

设计一级原始词表



```JSON
{
  "schema_version": "cn_l2_v1",
  "date": "2024-07-03",
  "ts_event_ns": 1719981234567890000,
  "ts_recv_ns": 1719981234567891000,
  "exchange": "XSHE",
  "market": "A_SHARE",
  "symbol": "000001.SZ",
  "security_id": "000001",
  "board": "MAIN | STAR | CHINEXT | SME | BSE | UNKNOWN",
  "session": "PRE_OPEN | OPEN_CALL | COOLING | CONT_AM | LUNCH | CONT_PM | CLOSE_CALL | AFTER_CLOSE | HALT | UNKNOWN",
  "event_source": "ORDER | TRADE | SNAPSHOT | QUEUE | DERIVED",
  "evt_type": "ADD | CANCEL | MODIFY | EXEC | SNAP | STATUS | UNKNOWN",
  "side": "B | S | N",
  "price": 10.25,
  "qty": 500,
  "amount": 5125.0,
  "order_type": "LIMIT | MARKET | BEST | CANCEL | UNKNOWN",
  "exec_type": "TRADE | CANCEL | UNKNOWN",
  "level": 1,
  "order_id_hash": "optional",
  "trade_id_hash": "optional",
  "buy_order_id_hash": "optional",
  "sell_order_id_hash": "optional",
  "source_seqnum": 123456789,
  "raw_msg_type": "vendor_original_type",
  "quality_flag": 0
}
```



> (注：内容由 AI 生成，请谨慎参考）
