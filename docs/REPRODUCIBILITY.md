# QuantFM 复现与验证指南

本文汇总项目的推荐阅读路径、复现命令和实验验证边界。

## 10 分钟阅读路径

1. 阅读 [根 README](../README.md) 的 Pipeline 与项目结构；
2. 阅读 [Pipeline 总览](pipeline/README.md)；
3. 查看以下核心文件：
   - `order_book/pylob/pipeline/workflow.py`
   - `quant_fm/tokenizer/transforms.py`
   - `quant_fm/tokenizer/fit_bins.py`
   - `quant_fm/pretrain/model.py`
   - `quant_fm/pretrain/train.py`
   - `quant_fm/downstream/run_judge.py`
4. 运行 smoke 和测试；
5. 若需真实数据，再配置 MinIO。

## 无外部依赖复现

```bash
cd QuantFM
uv sync --extra fm --group dev
uv run python -m pytest -q
uv run python -m quant_fm.scripts.smoke --workdir /tmp/quantfm-smoke
```

预期：

- 测试通过；
- 日志出现 `training complete`；
- 生成 embedding；
- 日志最终出现 `SMOKE OK: all stages passed`。

Smoke 使用合成数据和微型 CPU 模型，只验证工程链路，不代表真实投资收益。

## 真实数据验证

```bash
source ~/.minio_fm_env.sh
make check-minio
make pilot
```

检查：

```text
quant_fm/runs/pilot/
├── events/
├── tokens/
└── data/
    ├── vocab.json
    └── manifest.json
```

真实预训练：

```bash
make train-8gpu
```

重点观察：

- `train/loss` 是否下降；
- `val/loss` 是否先下降并形成稳定最低点；
- `best.pt` 是否生成；
- per-field CE 是否均优于随机基线；
- 后期 train/val gap 是否显示过拟合。

## 设计验证要点

### 数据正确性

- 上海/深圳交易所差异是否正确收敛到 `OrderBookSH` / `OrderBookSZ`；
- canonical schema 是否稳定；
- 事件顺序是否保持因果；
- manifest 是否按日期切分而非随机切分。

### 防泄漏

- 连续分箱边界只在训练日期拟合；
- val/test 仅复用冻结词表；
- 下游标签不进入预训练输入；
- checkpoint 选择不使用 test 指标。

### 模型与训练

- 多字段 embedding 与独立预测头是否符合任务；
- padding mask 和 next-event shift 是否正确；
- FSDP、AMP、梯度累积和有效 batch 是否符合配置；
- `best.pt` 与 `final.pt` 的语义是否清晰。

### 研究有效性

- 预训练 loss 仅是过程指标；
- embedding 必须通过严格时间切分的下游任务验证；
- 回测包含交易成本；
- RankIC、CPCV 和 DSR 共同用于降低偶然性。

## 不应提交到 GitHub

- `quant_fm/runs/`
- `data/clean/`
- parquet、checkpoint、TensorBoard 日志
- `.venv/`
- `~/.minio_fm_env.sh` 或任何真实 Access Key / Secret Key

上述内容均应由 `.gitignore` 排除。`uv.lock`、日期清单和标的清单应保留，以保证复现。

## 已知边界

- PyLOB 是离线研究引擎，不是生产低延迟撮合系统；
- 训练 checkpoint 已保存模型、optimizer、scaler 与 step，支持 `--resume` / `--resume auto` 断点续训；尚未持久化 sampler 精确位置与全部 RNG 状态；
- 数据阶段支持日期级 / 标的级断点续跑（`.done`、`.clean_done`、`skip_existing`）、`CLEAN_WORKERS` 多进程清洗与 `CANON_WORKERS` 并行规范化；可用 `check_pipeline_progress` 查询进度；
- manifest 中包含绝对路径，跨机器迁移时应重建或改写；
- 合成 smoke 的回测指标没有经济含义；
- 全市场效果需要更长时间跨度与独立 out-of-sample 数据验证。
