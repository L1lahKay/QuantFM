# 阶段 7：股日 Embedding

## 目标

冻结预训练 FM，将每个 `(date, symbol)` 的事件序列编码为固定维度 stock-day embedding，供下游截面模型使用。

## 核心代码

| 文件 | 责任 |
|------|------|
| `quant_fm/embedding/extract_hidden.py` | 加载 checkpoint、分块编码、写 parquet |
| `quant_fm/embedding/pool_stock_day.py` | hidden state 池化 |
| `quant_fm/pretrain/train.py::load_checkpoint` | 按 checkpoint 配置重建模型 |

## 处理流程

1. 加载 `best.pt` 并切换为 eval；
2. 从 manifest 选择指定 split；
3. 每个股日按 `context` 分块，避免一次性加载完整事件序列到 GPU；
4. 对每块执行 `model.encode()`；
5. 池化块内 hidden state；
6. 按块事件数加权汇总为单个股日向量；
7. 写入 parquet。

默认池化方式为 mean。

## 运行

```bash
uv run python -m quant_fm.embedding.extract_hidden \
  --checkpoint quant_fm/runs/pilot/run/best.pt \
  --manifest quant_fm/runs/pilot/data/manifest.json \
  --split test \
  --out quant_fm/runs/pilot/embeddings.parquet \
  --context 2048 \
  --device auto
```

## 输入与输出

输入：

- 训练 checkpoint；
- 与训练一致的 manifest；
- split、context、pooling、device。

输出：

```text
date | symbol | market | split | emb_0 | emb_1 | ... | emb_<d-1>
```

每行对应一个股日。向量维度等于模型 `d_model`。

## 验证条件

- 输出行数等于所选 split 的非空股日分片数；
- embedding 列数等于 checkpoint 的 `d_model`；
- 不包含 NaN/Inf；
- 相同 checkpoint、输入与池化配置产生稳定结果；
- train/val/test 的 embedding 不混用标签信息。

## 注意事项

- 使用 `best.pt` 通常比 `final.pt` 更稳健；
- context 应与模型 `max_seq_len` 兼容；
- embedding 本身不是收益预测，需要在严格时间切分的下游任务中验收；
- `embeddings.parquet` 属于生成产物，不应提交到 GitHub。
