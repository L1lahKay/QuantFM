# QuantFM Token 修改完整方案

> 文档基线：2026-07-30。本文以当前仓库代码和本地产物为准，区分“已经实现”“已经 smoke”“正式训练完成”和“仍待接通”。

## 1. 最终结论

Token 改造采用 **V1 兼容保留、V2 新根目录重建**，不原地修改 V1 词表、token shard 或 checkpoint。

- V1 继续用于复现现有 Pilot、Medium、302M、Dense230M 和 Backbone-MoE V1 产物。
- V2 使用 `cn_l2_v2 + vocab_v2.json + token/scalar 双通道 + sidecar contract`。
- V2 的字段声明、特殊 token、确定性拟合、窄整数存储、dataset、字段融合和兼容校验已经实现。
- 当前 V2 只有 5-update smoke。正式 V2 训练语料尚未通过 MinIO 一键流水线完整生产。
- 现有 V1 历史 shard 被审计出旧排序/旧参考价语义，不能靠补 metadata 升级；必须从新输出根重新执行 `clean → causal replay → vocab → tokens → manifest`。

建议冻结以下版本：

| 对象 | 版本 |
|---|---|
| 事件 schema | `cn_l2_v2` |
| vocab | `2.0` |
| token sidecar | `2.0` |
| 存储编码 | `token_uint_scalar_q16_v1` |
| 盘口时点 | `post_event`；事件价格距离使用 `pre_event` |

## 2. 为什么必须改 Token

### 2.1 V1 的有效部分

V1 已跑通完整训练链路，其结构是字段级 token，而不是把一行事件组合成一个巨型词表：

| 类型 | 字段 |
|---|---|
| 类别输入 | `evt_type`、`side`、`session`、`board`、`order_type`、`event_source` |
| 连续分箱 | `price_rel`、`log_volume`、`log_delta_t` |
| next-event 目标 | `evt_type`、`side`、`session`、`price_bin`、`volume_bin`、`delta_t_bin` |

它保留了“一个事件位置、多个字段并行编码”的正确方向，也实现了只用训练日期拟合连续分箱。

### 2.2 V1 的主要问题

1. 只有 `PAD=0`，没有独立的 `UNK/NA/BOS/EOS/SESSION_BREAK`。
2. 连续缺失值可能经 `nan_to_num` 落到数值 0，缺失与真实 0 不可区分。
3. 字段顺序、输入/目标角色散落在全局 tuple 和默认逻辑中，artifact 不够自描述。
4. 连续字段只保留离散桶，精细的桶内数值信息丢失。
5. `board/event_source/order_type` 中部分字段信息量有限，`order_type` 还是规则推断值。
6. 没有逐事件因果盘口状态，模型无法直接看到价差、深度与不平衡等微观结构状态。
7. 历史 V1 数据中存在 `local_time_v1 + ew_vwap_future_backfill_v1` 语义，不能作为严格因果 V2 基线。

## 3. V2 总体数据流

```text
原始逐笔委托/成交
    ↓ 交易所序号与确定性 tie-break 排序
PyLOB 因果回放
    ↓ 每个事件捕获 BookStateTransition(pre, post)
cn_l2_v2 canonical event
    ↓ 只扫描 train dates，三遍确定性拟合
vocab_v2.json
    ↓ 冻结字段、边界、normalizer、特殊 ID
tokenize_events_v2
    ├── tok_*：uint8 / uint16
    ├── val_*：对称 Q16 int16
    └── *.parquet.contract.json
    ↓
manifest.json + validation_windows.json
    ↓ 契约校验
EventWindowDatasetV2
    ↓
EventFieldFusion（gated_sum）→ OrderFlowFM
```

硬约束：事件顺序、参考价初始化、盘口捕获时点、vocab hash 和存储解码尺度必须同时匹配，任一不匹配都应 fail-fast。

## 4. 冻结字段设计

### 4.1 基础字段

`DEFAULT_FIELD_SPECS_V2` 定义 6 个基础字段：

| 逻辑字段 | 源列 | 类型 | 输入 | 目标 | 输出 |
|---|---|---|---:|---:|---|
| `evt_type` | `evt_type` | categorical | 是 | 是 | `tok_evt_type` |
| `side` | `side` | categorical | 是 | 是 | `tok_side` |
| `session` | `session` | categorical | 是 | 否 | `tok_session` |
| `price` | `price_rel` | ordinal, 32 bins | 是 | 是 | `tok_price_bin + val_price` |
| `volume` | `log_volume` | ordinal, 32 bins | 是 | 是 | `tok_volume_bin + val_volume` |
| `delta_t` | `log_delta_t` | ordinal, 32 bins | 是 | 是 | `tok_delta_t_bin + val_delta_t` |

相对 V1，`board/event_source/推断 order_type` 不再作为第一阶段默认输入。若后续证明有增量，应通过新的 `FieldSpec` 和新 vocab 版本加入，不能静默改变现有 V2。

### 4.2 因果盘口字段

`BOOK_FIELD_SPECS_V2` 追加 9 个字段：

| 字段 | 时点 | bins | 含义 |
|---|---|---:|---|
| `book_valid_post` | post | categorical | 当前盘口是否有效 |
| `spread_ticks_post` | post | 16 | 买一卖一价差 |
| `microprice_delta_ticks_post` | post | 21 | microprice 相对中间价偏移 |
| `imbalance_l1_post` | post | 21 | 一档不平衡 |
| `imbalance_l5_post` | post | 21 | 五档不平衡 |
| `imbalance_l10_post` | post | 21 | 十档不平衡 |
| `log_bid_depth_l5_post` | post | 32 | 买五档深度 |
| `log_ask_depth_l5_post` | post | 32 | 卖五档深度 |
| `event_price_distance_ticks_pre` | pre | 32 | 事件价格相对事件前盘口的距离 |

`FULL_FIELD_SPECS_V2` 是 6 个基础字段与 9 个盘口字段的并集。盘口列不作为当前默认 next-event 目标，只作为输入上下文。

### 4.3 FieldSpec 契约

每个字段必须显式保存：

```text
name / source / kind / n_bins / applicable_events
is_input / is_target / missing_token
```

由 `FieldSpec` 派生 `tok_*` 与 `val_*` 列名，并校验逻辑名、token 列和 scalar 列均唯一。训练和推理均从 vocab artifact 读取字段顺序，不允许依赖调用方默认顺序。

## 5. 特殊 Token 与 ID 空间

V2 每个字段使用独立 ID 空间，但共享特殊 ID 约定：

| Token | ID | 用途 |
|---|---:|---|
| `PAD` | 0 | batch padding，不进入 loss |
| `UNK` | 1 | 有值但不在冻结类别表中 |
| `NA` | 2 | 字段缺失或对该事件不适用，不等于数值 0 |
| `BOS` | 3 | 序列起点预留 |
| `EOS` | 4 | 序列终点预留 |
| `SESSION_BREAK` | 5 | 交易阶段断点预留 |
| 首个普通值 | 6 | 类别或数值 bin 起点 |

当前 tokenizer 已冻结这 6 个 ID；BOS/EOS/SESSION_BREAK 是否实际插入序列仍需由数据编排显式实现和消融验证，不能仅因 ID 已预留就声称已启用。

## 6. 连续字段双通道

对每个 ordinal/continuous 字段同时生成：

```text
离散通道：tok_<field>[_bin]
连续通道：val_<field>
```

离散通道用于稳定类别建模和 next-event 分类；连续通道保留桶内信息，并通过训练期冻结的 normalizer 编码：

\[
v_{norm}=\operatorname{clip}\left(\frac{v-\mu_{train}}{\sigma_{train}},-c,c\right)
\]

缺失位置的 scalar 写 0，但对应 token 必须是 `NA_ID=2`，模型因此能区分“真实标准化 0”和“缺失占位 0”。

## 7. Vocab 拟合方案

### 7.1 数据边界

- 只允许读取训练日期。
- `fit_dates` 必须与实际读到的 `observed_dates` 完全相等，而不只是“不包含 val/test”。
- vocab 冻结后才能处理 validation、test 和 OOS。
- 分层维度默认为 `date × exchange × board × evt_type`。

### 7.2 三遍拟合

1. **Pass 1：全流统计**
   - 统计 count、missing、mean、population std、min/max；
   - 统计每个 stratum 的完整样本量；
   - 类别字段统计 occupancy、unknown 和 missing。
2. **Pass 2：确定性分层 reservoir**
   - 先保证每个非空 stratum 至少一个配额，再按样本量比例分配余量；
   - 使用 `shard identity + event index + field + seed` 产生稳定 priority；
   - 每个 stratum 保留 priority 最小的 bottom-k；
   - 结果不依赖输入 path 顺序。
3. **Pass 3：冻结边界后的精确统计**
   - 在完整训练流上计算实际 bin occupancy；
   - 记录实际 bin 数、缺失率、训练熵和 normalizer；
   - 重复分位点合并，实际 bin 数允许小于请求值。

### 7.3 分位数与缩尾

- 双边字段（价格偏移、价差距离、microprice、不平衡、signed OFI）使用 1%–99% 缩尾。
- 单边字段下界从样本最小值开始，上界使用 99% 分位数。
- 分位边界必须有限且严格递增；重复边界合并。
- 超出训练范围的推理值落入边缘桶，同时纳入 edge occupancy 告警。

## 8. Token shard 与存储编码

### 8.1 窄整数

- 字段 vocab size `≤256`：token 保存为 `uint8`。
- 字段 vocab size `≤65536`：token 保存为 `uint16`。
- scalar 使用对称 `int16` Q16，解码为 `float32`。

Q16 编码尺度由 normalizer 的 clip 冻结：

\[
q=\operatorname{round}\left(\frac{v_{norm}}{c}\times32767\right)
\]

### 8.2 Sidecar

每个 parquet 同目录写入：

```text
<shard>.parquet.contract.json
```

至少绑定：

```text
artifact_version
schema_version
vocab_sha256
event_ordering_version
feature_transform_version
reference_price_initialization
storage encoding version / dtype / scale / metadata_sha256
```

manifest、预训练、评估和 embedding 提取均应在读取前验证 sidecar。无 sidecar 的历史 shard 只能按 legacy 语义读取，不能与显式 V2 vocab 混用。

## 9. 模型输入与字段融合

`EventWindowDatasetV2` 负责：

- 按 vocab 冻结顺序读取 token 和 scalar；
- 解码 Q16 scalar；
- 生成 `attention_mask`、字段适用 mask 和目标 mask；
- 拒绝 schema/vocab/storage contract 不一致的 shard。

推荐融合为 `gated_sum`：

1. 每个 token 字段独立 embedding；
2. scalar 经小型 projection 映射到 `d_model`；
3. 每个字段独立归一化；
4. 学习字段 gate 后求和；
5. 训练时使用小比例 field dropout，降低单字段依赖。

V1 checkpoint 加载时保持 `legacy_sum`，不得用新融合层解释旧权重。

## 10. Artifact 与目录隔离

推荐目录：

```text
quant_fm/runs/v1_*/                    # 只读复现
quant_fm/runs/v2_shared/
  data/vocab_v2.json
  data/manifest.json
  validation_windows.json
  tokens/{market}/{symbol}/{date}.parquet
  tokens/{market}/{symbol}/{date}.parquet.contract.json
quant_fm/runs/v2_25m/run/
quant_fm/runs/v2_100m/run/
quant_fm/runs/v2_dense_230m/run/
quant_fm/runs/v2_backbone_moe/run/
```

checkpoint 至少冻结：schema、vocab hash、FieldSpec、TargetSpec、normalizer、字段融合、盘口时点、上下文长度、pooling 版本、事件排序和特征变换版本。

## 11. 实施顺序

### Stage 0：补齐真实 V2 回放编排

1. 在 PyLOB 单事件撮合循环中输出 `BookStateTransition(pre, post)`。
2. 用 `transitions_to_feature_frame()` 生成行对齐盘口特征。
3. 强制 `cn_l2_v2.events_to_canonical()` 接收真实 `book_features`，禁止占位列。
4. 给 `run_pilot.py/run_medium.py` 增加独立 V2 模式和独立输出根。

### Stage 1：五日全市场数据 Pilot

- 选择时间跨度、市场状态和活跃度不同的 5 个训练日；
- 验证逐事件因果性、字段覆盖、NA/UNK、边缘桶、存储压缩率；
- 任一盘口一致性或 sidecar 校验失败都不得进入训练。

### Stage 2：25M 单因素消融

固定数据、有效 token、validation windows 和 seeds，分别比较：

```text
V1 fields vs V2 base fields
离散-only vs token+scalar
base V2 vs FULL book V2
gated_sum off/on
```

现有 `v2_25m_smoke` 只证明代码可运行，不属于本阶段正式结果。

### Stage 3：100M 复验

只让 Stage 2 winner 晋级，至少 3 seeds。检查预训练 normalized NLL、下游 paired daily IC、Top-K 收益、吞吐和存储成本。

### Stage 4：230M 与 MoE

在 Dense V2 完成且优于 V1 后，才启动 Backbone-MoE。不要同时改变 Token、Loss、MoE 和股票池定义。

## 12. 验收门槛

### 数据门

- 修改未来事件不改变当前及过去 token/盘口特征；
- `fit_dates == observed_train_dates`；
- val/test/OOS 不进入 vocab、normalizer 或分箱拟合；
- token parquet、sidecar、manifest、vocab hash 全部一致；
- `NA != 0`，未知类别进入 `UNK`，不适用字段进入 `NA`；
- FULL V2 九个盘口字段均来自真实回放。

### 统计门

- 每个有效数值字段至少 2 个实际 bin；
- NA、UNK、首尾桶占比和熵均写入报告；
- 不允许大面积常量字段或单桶塌缩；
- train/val 分布漂移按字段报告，不以改动 val 分箱来“修复”。

### 模型门

- 相同预算下 normalized NLL 不退化超过 1%；
- 至少两个 25M seeds 下游 paired IC 为正；
- 100M 阶段至少 2/3 seeds 为正，平均 ΔRankIC 目标不低于 0.005；
- 最终必须在 untouched OOS 上复验。

## 13. 当前状态

| 项目 | 状态 |
|---|---|
| `FieldSpec` 与 15 字段 FULL V2 | 已实现 |
| 六个特殊 ID | 已实现 |
| 确定性分层三遍拟合 | 已实现 |
| token + scalar 双通道 | 已实现 |
| uint8/uint16 + Q16 存储 | 已实现 |
| token sidecar 与 vocab hash | 已实现 |
| V2 dataset / gated fusion / checkpoint 校验 | 已实现 |
| 5-update V2 smoke | 已完成，仅工程 smoke |
| MinIO 全量真实盘口 V2 编排 | 未接通 |
| 正式 25M 多 seed | 未完成 |
| 正式 100M / 230M V2 | 未完成 |

## 14. 代码入口

| 功能 | 文件 |
|---|---|
| 字段声明 | `quant_fm/tokenizer/field_spec.py` |
| V2 vocab | `quant_fm/tokenizer/vocab_v2.py` |
| 确定性拟合 | `quant_fm/tokenizer/fit_bins_v2.py` |
| Token 化 | `quant_fm/tokenizer/tokenize_events_v2.py` |
| 存储编码 | `quant_fm/tokenizer/storage_encoding_v2.py` |
| Sidecar 契约 | `quant_fm/tokenizer/artifact_contract.py` |
| V2 schema | `quant_fm/schema/cn_l2_v2.py` |
| 盘口转换 | `quant_fm/tokenizer/lob_transforms.py` |
| V2 dataset | `quant_fm/pretrain/dataset_v2.py` |
| 字段融合 | `quant_fm/pretrain/field_fusion.py` |
| 历史语义审计 | `quant_fm/scripts/audit_token_ordering.py` |
