# Codex 主任务：实现并运行磁盘安全的 300 日 Backbone-MoE 训练

请在 `/home/khalil/DataCleaning7.3/QuantFM` 中持续接手以下作业：

- 机器合同：`quant_fm/experiments/moe_300d_storage_safe_codex_job.yaml`
- 运行手册：`docs/operations/CODEX_MOE_300D_STORAGE_SAFE_RUNBOOK.md`
- 科学日期合同：`quant_fm/experiments/moe_300d_training_plan.yaml`

最终目标是完成 FULL `cn_l2_v2` Backbone-MoE 的 300日Train、60日Validation、
100日一次性Test。默认每股票日平均4个连续2048事件窗口，8 GPU，先2个动态采样
epoch，只有固定validation通过晋级规则时才运行第3个epoch。

执行要求：

1. 完整阅读机器合同和运行手册，并以机器合同为唯一安全边界。
2. 先检查Git状态、当前磁盘/inode、日期合同、MinIO读写端和现有测试；保护所有用户
   未提交修改，不得重置或覆盖。
3. 当前作业状态是`IMPLEMENTATION_REQUIRED`。不要直接运行现有
   `config_v2_backbone_moe_300d.yaml`，也不要直接开始300日生产。
4. 先实现并测试：跨进程磁盘reservation guard、原子checkpoint和最近2点轮转、
   fail-closed preflight、持久化控制器和任务独占锁。
5. 再实现并测试：全市场因果FULL V2流式生产、流式vocab summary与精确occupancy、
   不可变MinIO pack/day commit、远端manifest、共享字节上限cache、lazy窗口索引、
   pack-aware分布式sampler、精确sampler cursor恢复、MoE负载和overflow遥测。
6. 本任务新增本地占用不得超过80 GiB。free低于90 GiB或任务超过70 GiB时暂停
   producer并驱逐cache；free低于70 GiB或任务达到80 GiB时有序停机；free低于
   60 GiB时禁止新写入。不得自行放宽这些阈值。
7. 任意时刻最多生产一个交易日。只能清理机器合同列出的本任务stage/cache/run路径；
   不得删除、移动或覆盖其他`quant_fm/runs`内容。
8. 每个pack必须校验schema、rows、size、SHA-256和Parquet footer。远端不可变对象及
   当日`COMMITTED.json`回读通过后，才允许删除本地唯一副本。
9. vocab只能读取精确的300个train日期；validation/test有任何重合立即失败。
10. 先完成五日全市场pilot、远端容量投影、8 GPU恢复演练和1000 updates warmup。
    所有gate未PASS前禁止正式launch。
11. checkpoint只保留最近2个稳定resumable及best/final；保存必须是
    `.partial → fsync → validate → atomic replace`，新点稳定前不能删旧点。
12. 训练过程中只用validation做选择，候选冻结后才允许查看100日test一次。
13. 每完成一个阶段，原子更新`controller_state.json`和证据报告，并向用户汇报：
    当前阶段、gate结果、任务磁盘峰值、当前free、远端容量投影、训练进度和下一步。
14. 长任务必须使用持久执行会话并持续监控。用户询问状态时先报告证据，再继续任务。
15. 自动恢复最多2次，仅限已识别瞬态故障。磁盘、OOM、数据损坏、schema/vocab错误、
    恢复连续性失败或重复NCCL错误必须停止并报告，禁止盲目重试。
16. 不得在日志、命令行、报告或配置中输出MinIO密钥。

建议按以下状态机推进：

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

目标控制入口为：

```bash
uv run python -m quant_fm.scripts.moe_trainctl <command> \
  --job quant_fm/experiments/moe_300d_storage_safe_codex_job.yaml
```

该入口当前尚未实现。先通过测试驱动方式实现`preflight`和P0安全能力；默认命令必须
只读、fail closed，并且只有显式执行标志加所有最新gate PASS后才可调用`torchrun`。

现在从只读现状审计开始，给出P0具体修改计划并直接实施安全控制面；不要直接启动
300日数据生产或正式训练。
