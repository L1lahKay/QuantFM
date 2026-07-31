# Dense230M 训练期间并行工作计划

版本：1.0  
制定日期：2026-07-24  
适用分支：`MOE`  
当前训练：`Dense230M-V1-compat`

## 1. 计划摘要

当前 Dense230M 已使用 8 卡 FSDP 启动。训练期间的目标不是继续改变正在运行的模型，
而是提前完成监控、验收、下游评估、V2 数据准备和后续实验设计，使训练完成后可以直接回答：

1. 训练是否稳定、可恢复、可复现；
2. Dense230M 是否比旧模型更好或更高效；
3. 表征在 RankIC、组合回测和不同市场状态下是否有效；
4. 是否值得推进正式 V2 Dense 或 Backbone/Regime-MoE；
5. 下一次训练需要使用什么数据、配置和验收门槛。

本计划以 optimizer update 为主时间轴，墙钟时间仅作为参考。预计训练总时长约 36～45 小时，
实际时间会受验证、checkpoint 写盘和数据加载影响。

## 2. 当前训练基线

| 项目 | 当前值 |
|---|---|
| 运行标识 | `Dense230M-V1-compat` |
| 配置 | `quant_fm/runs/dense_230m_v1/config.yaml` |
| 日志 | `quant_fm/runs/dense_230m_v1/train.log` |
| 输出 | `quant_fm/runs/dense_230m_v1/run/` |
| 持久会话 | `tmux: quantfm_dense230m_v1` |
| 模型 | 231.52M 参数，18 层，`d_model=1024`，`ffn_hidden=2816` |
| 架构 | Dense；`backbone_moe.enabled=false` |
| 数据 | `cont60` V1，训练集约 79.7 亿事件、396 万窗口 |
| 并行 | 8 卡 FSDP，BF16 |
| 有效 batch | 128 个序列/update，约 262,144 token/update |
| 预算 | 50,000 updates，约 131 亿调度 token |
| 内置验证 | 每 1,000 updates，最多 200 batches |
| 检查点 | 每 2,000 updates |

重要边界：本次训练是完整 V1 数据上的 Dense230M 兼容基线，不是正式 V2 结果。
正式 V2 仍需要真实 LOB replay、post-event 盘口特征、`vocab_v2.json`、V2 tokens 和 manifest。

### 2.1 已落地的低干扰工具

一次性采集训练状态：

```bash
.venv/bin/python -m quant_fm.scripts.monitor_training \
  --config quant_fm/runs/dense_230m_v1/config.yaml \
  --log quant_fm/runs/dense_230m_v1/train.log \
  --tmux-session quantfm_dense230m_v1 \
  --world-size 8
```

持续监控时增加 `--interval 300`。该工具只读日志和系统状态，不杀进程、不自动恢复，
并在 run 目录维护：

- `training_status.json`：当前 update、loss、验证、GPU、磁盘和告警；
- `checkpoint_registry.json`：checkpoint 类型、大小、mtime 和连续观测稳定性；
- `run_metadata.json`：Git、配置、manifest、vocab、验证窗口 hash 和训练预算；
- `training_report.md`：对应本计划决策门的轻量运行报告。

训练完成后先生成串行评估计划：

```bash
.venv/bin/python -m quant_fm.scripts.posttrain_evaluation \
  --config quant_fm/runs/dense_230m_v1/config.yaml \
  --baseline-checkpoint quant_fm/runs/medium_300m/run/best.pt
```

只有 `final.pt`、`final_resume.pt` 均存在且训练进程退出后，才允许显式增加 `--execute`。
存在基线时，评估顺序为候选 validation→旧 300M validation→1% 非劣门槛→候选 test；
非劣失败会在 test 前停止。它不会在训练中抢占 GPU。外部评估同时兼容旧
`optim.batch_size` 和新 `optim.micro_batch_size` 配置，并强制候选与基线使用相同的
validation-plan fingerprint 和窗口数。

V2 artifact 生产契约可以在不读取完整数据内容的情况下先做轻量审计：

```bash
.venv/bin/python -m quant_fm.scripts.audit_v2_artifacts \
  --root quant_fm/runs/v2_shared \
  --sample-shards 12 \
  --out quant_fm/runs/v2_shared_readiness.json
```

审计检查 VocabV2/manifest schema、split 日期重叠、vocab 拟合泄漏、完整盘口 FieldSpec、
token parquet 列与行数。它只证明 artifact 契约就绪；正式开训前仍必须补充 coverage、
NA/UNK/edge-bin 和真实 LOB 因果回放验收。

完整训练结束后的一键串行验收入口为：

```bash
bash quant_fm/scripts/run_dense230m_posttrain.sh
```

脚本在缺少 `final.pt`/`final_resume.pt` 或训练进程仍存活时直接拒绝执行。通过预训练非劣
门槛后，它才依次抽取 train/val/test embedding，并运行现有 RankIC、CPCV 和成本后组合
回测。所有阶段串行使用 GPU，不会与当前训练并发。

## 3. 总体目标与非目标

### 3.1 目标

- 当前训练不中断、异常可发现、checkpoint 可恢复；
- 在结果产生前冻结评价口径，降低事后选择指标的风险；
- 准备固定窗口上的预训练评估和旧模型对照；
- 准备 embedding、RankIC、组合回测和 regime 分层评估；
- 完成正式 V2 数据生产链的设计审计和小样本验收方案；
- 给 Dense、Backbone-MoE、Regime-MoE 建立统一实验矩阵。

### 3.2 非目标

- 不在本次运行中热修改模型、loss、学习率或数据；
- 不将 V1 训练结果描述为 V2 结果；
- 不在同一台机器上并行启动重型 GPU 实验；
- 不在训练期间全速执行大规模 MinIO 下载、LOB replay 或 Parquet 重写；
- 不因为单次 validation loss 更低就直接判定可上线。

## 4. 执行原则

1. **训练优先**：8 张 GPU 全部归当前 FSDP 作业使用。
2. **低干扰并行**：训练期间仅执行轻量日志、配置、文档和元数据工作。
3. **固定口径**：比较必须使用相同数据切分、窗口、目标、checkpoint 选择规则和交易成本。
4. **先 Dense 后 MoE**：Dense230M 是后续 MoE 的必要对照组。
5. **训练/测试隔离**：test split 只在候选模型冻结后使用，不参与 checkpoint 选择。
6. **版本可追踪**：配置、代码 commit、词表 hash、manifest fingerprint 和 checkpoint 一起归档。

## 5. 阶段计划

| 阶段 | 触发条件 | 主要工作 | 主要交付物 |
|---|---|---|---|
| A：运行护航 | 现在至 update 1,000 | 监控、告警规则、评价口径冻结 | 监控清单、验收指标定义 |
| B：首轮验证 | update 1,000～2,000 | 检查内置验证、首个 checkpoint、磁盘增长 | 首轮健康报告、checkpoint 登记 |
| C：中期准备 | update 2,000～20,000 | 下游评估配置、旧模型对照、V2 数据审计 | 评估配置、对照矩阵、V2 数据清单 |
| D：趋势确认 | update 20,000～40,000 | 学习曲线诊断、过拟合/欠拟合判断、候选 checkpoint 规则 | 中期报告、候选规则 |
| E：收尾验收 | update 40,000～50,000 | 最终 checkpoint、正式评估、embedding 与回测 | 训练报告、模型卡、决策结论 |

## 6. 工作流一：训练监控与恢复

### M-01 轻量运行监控

监控频率建议为每 5 分钟一次，仅采集：

- tmux 会话和 8 个 rank 是否存活；
- GPU 利用率、显存、温度和异常进程；
- 最新 update、loss、learning rate、token 数；
- 日志是否出现 `Traceback`、`NaN`、`OOM`、NCCL timeout；
- checkpoint 目录与磁盘剩余空间；
- 日志在 15 分钟内是否继续更新。

告警分级：

| 级别 | 条件 | 处理 |
|---|---|---|
| P0 | tmux/rank 退出、NaN、OOM、NCCL error | 停止自动操作，保存日志并分析；有可用 checkpoint 后再恢复 |
| P1 | 日志 15 分钟无更新、GPU 长时间低利用率 | 检查数据加载、I/O、验证或 checkpoint 写盘状态 |
| P1 | 磁盘可用空间低于 20% | 暂停新增重型任务，制定 checkpoint 归档方案 |
| P2 | 单点 loss 波动 | 记录并观察趋势，不因单个 batch 干预训练 |

### M-02 Checkpoint 登记

每个 `step*.pt` 记录：

- update、文件大小、生成时间；
- 对应代码 commit 和配置快照；
- validation loss 与当时的训练 loss；
- 是否完成写盘并在两个观测周期内保持大小不变；
- 是否为当前 best。

训练期间不对每个大 checkpoint 反复计算 SHA-256，以避免持续占用磁盘 I/O；最终候选和归档版本再计算 hash。

### M-03 恢复策略

- update 2,000 前若失败：没有正式恢复点，应保留日志并从头启动；
- update 2,000 后若失败：使用相同代码、环境和配置执行 `--resume auto`；
- 不使用 `best.pt`/`final.pt` 恢复 optimizer；应使用 `step*.pt` 或 `final_resume.pt`；
- 恢复后核对 update、micro step、累计 token、optimizer 和学习率连续性；
- 不通过杀死当前健康进程来做恢复演练。

## 7. 工作流二：预训练验收标准

### E-01 训练稳定性指标

- 无 NaN、Inf、OOM 和不可恢复 NCCL 错误；
- loss 在 warmup 后呈总体下降趋势，允许批次级波动；
- 8 卡有效 token/update 与配置一致；
- 验证阶段能够完成，且不会改变训练数据状态；
- checkpoint 写入成功，训练能够继续前进。

### E-02 预训练质量指标

在固定 validation windows 上记录：

- total validation loss；
- 每个预测头的 CE 和 perplexity；
- top-1 accuracy 和 balanced accuracy；
- 与 copy-previous、unigram baseline 的差值；
- `tok_evt_type`、`tok_side`、`tok_session`、价格、成交量和时间间隔各头表现；
- 按市场、日期、板块和活跃度分层后的稳定性。

checkpoint 选择只使用 validation，不查看 test。建议保留：

1. `best validation`；
2. `final`；
3. 一个中期 checkpoint；
4. 用于恢复的 `final_resume`。

### E-03 公平对照

Dense230M 至少与以下对象比较：

- 已完成的 V1 300M 模型；
- copy-previous/unigram baseline；
- 如果有可比的小模型，再增加参数效率曲线。

比较时固定：

- 同一 V1 vocab 和 manifest；
- 同一 validation window 清单；
- 同一 context 和评估 batch 规则；
- 同一指标实现；
- 同一精度和推理设备。

建议的非劣门槛：Dense230M 的主 validation loss 不应比旧 300M 基线恶化超过 1%；
如果质量相当，则以参数量、吞吐、显存或下游 RankIC 的优势决定是否晋级。

## 8. 工作流三：Embedding 与下游评估

### D-01 Embedding 产出

对候选 checkpoint 使用冻结模型导出 embedding，并固化：

- checkpoint hash；
- vocab/manifest hash；
- pooling 方式；
- embedding 维度；
- 数据截止日；
- 是否使用盘中/收盘可得信息。

V1 基线不得标注为 `post_event` V2 盘口表征。

### D-02 因子评估

至少报告：

- 日度与周度 RankIC；
- RankIC 均值、中位数、标准差和 ICIR；
- 多空分组收益与单调性；
- 换手率、容量代理和交易成本敏感性；
- 按年份、市场、板块、流动性分层的稳定性；
- 相对旧 300M embedding 的配对差异。

### D-03 组合回测

回测必须明确：

- 标签区间和持有期；
- 调仓时点与信息可得性；
- 停牌、涨跌停和不可交易处理；
- 手续费、滑点和冲击成本；
- 股票池、行业/市值中性化；
- 最大回撤、收益波动、Sharpe、换手和容量。

test 只用于最终冻结模型的一次性报告。若根据 test 结果继续调参，则该 test 不再是最终 OOS。

## 9. 工作流四：正式 V2 数据准备

这是训练期间最重要的下一阶段准备，但重型生产不能与当前训练在同机全速并行。

### V-01 数据源审计

确认：

- 原始逐笔委托/成交是否包含稳定的交易所序号和订单标识；
- 是否能够确定性执行 LOB replay；
- pre-event/post-event 状态是否行对齐；
- 训练、验证、测试日期和 purge/embargo 是否冻结；
- 预计 events、tokens 和临时文件的磁盘峰值；
- MinIO 读取带宽和失败恢复策略。

### V-02 小规模真实 V2 验收

使用少量真实股票和多个交易日完成端到端：

```text
raw order/trade
  -> LOB replay
  -> pre/post-event book features
  -> cn_l2_v2 canonical events
  -> fit_vocab_v2（仅训练日）
  -> tokenize_path_v2
  -> V2 manifest
  -> 25M smoke
```

验收：

- 行数和事件顺序不变；
- 必需盘口字段不存在时 fail fast；
- vocab 无验证/测试泄漏；
- token 特殊 ID、字段顺序和 normalizer 正确；
- coverage、NA、UNK、edge-bin 比例合理；
- `book.state_timing=post_event` 与数据契约一致。

### V-03 全量生产决策

只有在小规模 V2 验收通过后，才估算并启动全量生产。若本机磁盘或 I/O 不足，应选择：

- 独立机器生成；
- 按日生成、tokenize、上传、删除中间文件；
- 远端对象存储直接承载 V2 tokens；
- 限制 CPU/Polars worker，避免影响当前训练。

## 10. 工作流五：Dense 与 MoE 实验矩阵

MoE 实验必须与 Dense230M 使用相同数据和 token 预算。

| 实验 | 数据 | 模型 | 目的 |
|---|---|---|---|
| B0 | V1 cont60 | Dense230M | 当前兼容基线 |
| B1 | V2 shared | Dense25M | 验证 V2 数据和 loss |
| B2 | V2 shared | Dense100M | 规模复验 |
| B3 | V2 shared | Dense230M | 正式 Dense V2 基线 |
| M1 | V2 shared | Backbone-MoE | 检查稀疏专家的质量/吞吐 |
| M2 | V2 shared + regime features | Regime-MoE | 检查行情专家化收益 |

MoE 晋级除预测指标外还必须检查：

- expert utilization 和负载均衡；
- router entropy、top-1 probability、overflow；
- 不同行情阶段的专家分工是否稳定；
- 参数量、激活显存、训练/推理吞吐；
- 相同 token 预算下相对 Dense 的 OOS 增益。

如果 MoE 只降低训练 loss，却没有稳定的 OOS/RankIC 收益，或者吞吐明显退化，则不晋级。

## 11. 资源安排

### 当前训练期间允许

- 编辑文档、配置草案和报告模板；
- 轻量读取日志、`nvidia-smi` 和目录元数据；
- 对 manifest/vocab 做不扫描大文件内容的审计；
- 编写但不启动重型评估或数据生产命令；
- 在独立机器上进行 V2 数据准备。

### 当前训练期间禁止或需审批

- 使用任意一张训练 GPU；
- 重建 `.venv` 或升级 PyTorch/CUDA；
- 修改、移动或覆盖当前 token、manifest、vocab；
- 同机全速 MinIO 下载、LOB replay、全量 tokenization；
- 删除旧 checkpoint 或大文件以腾空间；
- 修改运行中的配置快照或训练输出。

## 12. 决策门

### Gate 1：运行健康

触发：首个 checkpoint 完成。  
通过条件：进程稳定、验证成功、checkpoint 写盘完整、磁盘预算安全。

### Gate 2：预训练质量

触发：至少三个 validation 点。  
通过条件：validation 总体改善，无明显持续过拟合；相对旧基线达到预先定义的非劣标准。

### Gate 3：下游价值

触发：候选 checkpoint embedding 和回测完成。  
通过条件：RankIC/ICIR、成本后组合表现和 regime 稳定性至少一项有实质改善，且其他核心指标不显著恶化。

### Gate 4：V2/MoE 晋级

触发：真实 V2 小规模验收通过。  
通过条件：数据契约、无泄漏、覆盖率、训练稳定性全部通过；之后才允许启动 V2 Dense，并以 Dense 为 MoE 对照。

## 13. 角色分工

### 业务/研究负责人

- 确认最终标签、持有期、股票池和交易成本；
- 确认 RankIC、回撤、换手和容量的业务门槛；
- 决定 Dense 与 MoE 的优先级和算力预算；
- 批准 test split 的最终启封时间。

### 工程执行

- 训练监控、日志汇总和 checkpoint 登记；
- 固定评估配置和可复现命令；
- embedding、评估、报告和 artifact 归档；
- V2 数据小样本与批量生产编排。

### 训练主机

- 当前阶段只承载 Dense230M FSDP 和轻量监控；
- 不承担并行重型数据生产或第二个 GPU 作业。

## 14. 最终交付物

训练完成后应形成：

1. Dense230M 训练健康报告；
2. 固定窗口预训练评估报告；
3. Dense230M 与旧 300M 对照报告；
4. embedding 元数据和 RankIC 报告；
5. 成本后组合回测报告；
6. 分市场、流动性和 regime 稳定性报告；
7. 候选 checkpoint、hash、配置和代码 commit；
8. 正式 V2 数据生产审计与容量估算；
9. Dense/Backbone-MoE/Regime-MoE 下一阶段决策记录。

## 15. 执行检查表

### 立即执行

- [ ] 冻结本计划和评价口径；
- [x] 建立轻量 watchdog 与告警规则；
- [x] 提供当前 commit、配置、manifest 和 vocab 的冻结工具；
- [x] 建立 validation/checkpoint 登记工具；
- [ ] 确认磁盘告警线和 checkpoint 空间预算。

### 首个 checkpoint 后

- [ ] 确认 `step2000.pt` 写盘完成；
- [ ] 记录首轮 validation 结果；
- [ ] 验证 resume 所需状态字段存在；
- [ ] 更新预计完成时间和磁盘增长速度；
- [ ] 不停止健康训练做恢复演练。

### 训练中期

- [x] 完成旧 300M 固定窗口公平对照与 1% 非劣门槛工具；
- [x] 完成训练结束后的 embedding 与下游串行评估入口；
- [x] 完成 V2 artifact 轻量契约审计工具；
- [ ] 完成 V2 原始数据和磁盘容量实测；
- [ ] 冻结最终 checkpoint 选择规则；
- [x] 准备训练完成后的串行评估队列。

### 训练完成后

- [ ] 校验 `best.pt`、`final.pt`、`final_resume.pt`；
- [ ] 对候选 artifact 计算 SHA-256；
- [ ] 运行 validation 和一次性 test；
- [ ] 导出 embedding 并完成 RankIC/回测；
- [ ] 形成模型卡和最终晋级决策；
- [ ] 决定是否启动正式 V2 Dense230M。
