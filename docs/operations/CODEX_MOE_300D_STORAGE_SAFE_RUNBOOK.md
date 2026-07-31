# Codex 管理的 300 日 MoE 磁盘安全训练手册

## 结论

Codex 可以接手实现、试跑并最终启动这次训练，但不能直接运行现有
`config_v2_backbone_moe_300d.yaml`。现有配置仍假定本地 manifest/tokens、一次全事件
epoch 约 330k updates，并且没有磁盘硬闸门、checkpoint 轮转、远端有界缓存或精确的
sampler 游标恢复。

本作业的唯一机器合同是：

- `quant_fm/experiments/moe_300d_storage_safe_codex_job.yaml`

初始状态必须保持 `IMPLEMENTATION_REQUIRED`。Codex 先补齐安全控制面和流式数据面，
所有 gate 通过后才能把运行状态推进到正式训练。

## 不可突破的边界

1. 本任务新增本地占用不超过 80 GiB。
2. 开始重型阶段前可用空间至少 150 GiB。
3. 任意时刻最多生产一个交易日。
4. 本任务只能清理自身 `owner_run_root` 下明确列出的 stage/cache/run 路径。
5. 不得删除、覆盖或搬动既有实验目录。
6. `/tmp` 与工作区同盘，不能作为额外容量。
7. vocab 只允许读取精确的 300 个训练日期。
8. 60 日 validation 只用于选择；100 日 test 在候选冻结后只运行一次。
9. MinIO 远端空间无法证明充足时，必须停止，不能假定无限容量。
10. 任何 gate 失败都不得通过提高磁盘、缓存或自动重试上限绕过。

## 为什么当前不能直接启动

- 首次 vocab 路径会先积累全部 events，见
  [`run_medium.py`](../../quant_fm/scripts/run_medium.py)。
- 当前上传在全量产物完成后递归复制，只按文件数验证，见
  [`upload_to_minio.py`](../../quant_fm/scripts/upload_to_minio.py)。
- manifest 保存本机绝对路径，上传后不能直接远程训练。
- [`dataset_v2.py`](../../quant_fm/pretrain/dataset_v2.py) 直接读取本地 Parquet，并为全部
  窗口创建 Python 对象。
- [`sampler.py`](../../quant_fm/pretrain/sampler.py) 在每个 rank 再构造完整窗口索引。
- [`monitor_training.py`](../../quant_fm/scripts/monitor_training.py) 只告警，不会暂停写入。
- [`train.py`](../../quant_fm/pretrain/train.py) 永久保留周期 checkpoint，保存也不是
  `.partial → fsync → atomic replace`。

因此，磁盘 guard 必须接入每一个重型写入点；外部监控不能代替写入前预留。

## Codex 必须实现的控制入口

目标入口为：

```bash
uv run python -m quant_fm.scripts.moe_trainctl <command> \
  --job quant_fm/experiments/moe_300d_storage_safe_codex_job.yaml
```

它目前是目标接口，Codex 必须先实现并测试，不能假装命令已经存在。支持的阶段命令：

```text
preflight
pilot
resume-drill
fit-vocab
build-corpus
validate-corpus
warmup
launch
status
resume
evaluate
```

默认行为必须是只读 `preflight`。只有显式执行参数、最新 PASS 报告与当前配置/代码/
日期/vocab/manifest hash 完全一致时，控制器才可以调用 `torchrun`。

## 阶段 0：安全控制面

先实现以下能力及测试：

### 磁盘 guard

- 任务目录总增量、stage、cache、checkpoint、metadata 分别计费。
- `.partial` 和所有并发 reservation 必须计费。
- 多进程通过共享 ledger 和文件锁进行原子预留。
- 每次开始日期、下载、写 pack、写 checkpoint 前都要预留。
- free 低于 90 GiB 或任务用量超过 70 GiB：暂停 producer、停止预取、驱逐未
  pin 的 LRU cache。
- free 恢复到 100 GiB 且任务用量低于 60 GiB：才允许恢复 producer。
- free 低于 70 GiB 或任务达到 80 GiB：有序停机。
- free 低于 60 GiB：禁止所有新写入，使用最近稳定 checkpoint 退出。

### 原子 checkpoint 和轮转

```text
stepN.pt.partial
→ fsync
→ 轻量格式/元数据校验
→ os.replace(stepN.pt)
→ FSDP barrier
→ 新点稳定后删除最旧周期点
```

只保留最近两个 resumable 周期点、一个 best、final 和 final_resume。checkpoint 目录
连同临时文件不得超过 15 GiB。训练开始前在该预算内预留 4 GiB 的应急空间。

### 必需测试

```text
tests/test_storage_guard.py
tests/test_checkpoint_retention.py
tests/test_moe_training_preflight.py
tests/test_moe_pilot_gate.py
tests/test_remote_pack_cache.py
tests/test_streaming_vocab_v2.py
tests/test_lazy_window_sampler.py
tests/test_v2_minio_commit.py
```

至少要覆盖跨进程并发预留、外部任务抢占磁盘、模拟 ENOSPC、部分下载、错误 SHA、
进程中断、reader pin、8 rank 同时请求同一个 pack，以及恢复后 sampler 连续性。

## 阶段 1：只读 Preflight

Codex 生成原子的 `preflight_report.json`。检查：

- 日期文件精确为 300/60/100，严格有序且两两不重合；
- `vocab.fit_dates` 必须与 train 日期完全相等，不能只是“不含未来”；
- FULL V2 的九个盘口字段、post-event 时序、loss 和normalizer契约；
- 本地 free ≥ 150 GiB、inode 充足、各目录配额可建立；
- 8 张 GPU、bf16、端口和已有训练进程；
- MinIO 源端读和目标隔离前缀写/读/hash测试；
- 远端是否为独立容量池；
- 当前 Git commit、dirty diff hash及所有输入文件SHA；
- 不存在另一份同job的controller锁。

任何条件失败，状态进入 `BLOCKED`，不得启动重型任务。

## 阶段 2：五日全市场 FULL V2 Pilot

从300个训练日中选择覆盖时间范围和活跃度的五日；至少包含开头、中间、末尾日期。
每次只处理一日，必要时按市场和128只股票左右的批次拆分。

记录：

- raw、clean、events、tokens/pack各阶段峰值；
- 最低磁盘free、inode、bytes/row、bytes/股票日和处理时间；
- FULL V2 schema和字段覆盖；
- stage≤40 GiB、cache≤20 GiB、任务总增量≤80 GiB；
- 本地pack与远端对象的schema、rows、size、SHA及Parquet footer一致；
- 远端`COMMITTED.json`回读成功后才删除本地文件；
- 中断恢复没有重复计数、误删或遗留partial。

用五日实测值计算远端投影并乘1.25安全系数。没有容量证明时停在
`AWAITING_REMOTE_CAPACITY_ATTESTATION`。

## 阶段 3：流式 vocab 与语料

### Pass A

只扫描300个train日期。因果FULL V2仅在单日/股票批内存在；更新可合并统计、类别
计数和确定性bottom-k reservoir。每日summary提交后删除中间数据。

### Pass B

使用冻结encoding边界按日生成pack，同时累计精确occupancy/NA/UNK/类别计数。300个
train日全部commit后才生成最终`vocab_v2.json`。若encoding hash发生变化或统计总数与
Pass A不一致，立即失败。

val/test只在最终vocab冻结后处理。所有pack为不可变内容寻址对象；小型根manifest引用
Parquet索引，索引只保存逻辑ID、远端key、size、SHA、row group，不允许绝对本机路径。

## 阶段 4：窗口与训练预算

默认每个有效股票日平均四个连续2048事件窗口：

1. 开盘或连续竞价初段；
2. 正常时段动态窗口；
3. 高波动、大单或高成交窗口；
4. 收盘、盘口失衡、价差异常或其他稀有状态窗口。

低活跃股票使用全部不重复窗口；任一非空股票日至少一个、最多八个窗口。窗口不得跨
午休或交易阶段。每个epoch以`dataset_id/seed/epoch/date/market/symbol/stratum`确定性
换样本。

按约5000股票估算：

| 指标 | 每个epoch |
|---|---:|
| 股票日 | 约150万 |
| 窗口 | 约600万 |
| 事件位置 | 122.88亿 |
| updates | 约46,875 |

正式步数必须在语料commit后用真实有效股票日重新计算。先运行两个epoch，约93,750
updates；最多三个epoch，约140,625 updates。LR schedule在第一个update前按真实的三轮
上限冻结，避免续跑第三轮时改变调度定义。

## 阶段 5：恢复演练和1000步Warmup

先在pilot语料上训练100–200 updates并人为中断。显式从稳定resume点恢复，核对：

- model、optimizer、scaler、LR与update连续；
- epoch、sampler cursor和所有RNG连续；
- 没有跳过窗口，也没有从epoch开头大量重放；
- cache、pack和checkpoint轮转仍在硬预算内。

然后运行8 GPU、1000 updates warmup。必须无OOM、NaN/Inf、NCCL错误；四个专家均有
负载，overflow健康；固定validation可运行；原子checkpoint可恢复。

## 阶段 6：正式训练与第三轮决策

- 每50 updates记录loss与资源；
- 每5000 updates运行固定validation并保存周期resume点；
- 每个epoch末运行完整60日流式validation；
- producer受cache背压，不能无限领先；
- 正式训练期间禁止读取100日test。

第二轮后，只有validation改善超过`max(0.2%, 2×按日期bootstrap标准误)`、主要预测头
没有超过1%退化、最近三次validation没有持续变差且路由健康时，才显式从第二轮
`final_resume.pt`进入第三轮。不得使用`resume=auto`猜测恢复点。

## 阶段 7：冻结与一次性 Test

使用validation选择唯一候选并记录SHA-256。候选冻结后只运行一次100日test；test结果
不得反向修改模型、采样、loss或超参数。随后输出RankIC、Top-K成本后收益、换手、MDD、
相对全市场超额和按月/波动/流动性稳定性。

## 持久化状态和故障处理

控制器状态机：

```text
P0_IMPLEMENT_SAFETY
→ P1_PREFLIGHT
→ P2_FULL_V2_PILOT_5D
→ P3_RESUME_DRILL
→ P4_VOCAB_AND_CORPUS
→ P5_WARMUP_1K
→ P6_EPOCHS_1_AND_2
→ P7_EPOCH_3_DECISION
→ P8_FINAL_VALIDATION_AND_TEST
→ DONE / BLOCKED
```

每阶段只在证据文件写入并原子commit后推进。自动恢复最多两次，且仅限已识别的瞬态
错误。磁盘不足、OOM、数据损坏、schema/vocab不一致和重复NCCL失败禁止盲目重启。

## 运行证据

至少保留：

```text
controller_state.json
preflight_report.json
pilot_storage_report.json
remote_capacity_projection.json
resume_drill_report.json
vocab_audit.json
corpus_audit.json
config.snapshot.yaml
run_metadata.json
disk_guard.jsonl
resource_usage.jsonl
producer.jsonl
metrics.jsonl
checkpoint_registry.json
training_status.json
training_report.md
frozen_candidate.sha256
final_test_report.json
```

Codex每推进一个阶段都应汇报当前阶段、gate结果、任务峰值、本地free、远端容量投影和
下一步。长期命令应运行在一个持久控制会话中；只读监控可以复用现有
`quant_fm.scripts.monitor_training`，但硬停必须由新controller和写入路径内的guard执行。

## 启动 Codex

将下面文件作为Codex的主任务提示词：

- `quant_fm/experiments/CODEX_MOE_300D_STORAGE_SAFE_PROMPT.md`

Codex第一步只能做现状审计和P0实现，不能直接开始300日数据生产或`torchrun`。
