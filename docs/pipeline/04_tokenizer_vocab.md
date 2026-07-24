# 阶段 4：Tokenizer 与字段级词表

> 当前状态（2026-07）：v1 Tokenizer 保持稳定；v2 的 `FieldSpec`、独立特殊 token、
> 全流分层 priority reservoir、字段级词表、连续双通道、coverage/leakage 检查均已
> 实现并有单元测试。v2 尚未接入现有 medium/300M 数据生产 CLI，真实训练集 vocab
> 和 token parquet 仍需生成。

## v1 稳定路径

v1 实现位于：

```text
quant_fm/tokenizer/vocab.py
quant_fm/tokenizer/fit_bins.py
quant_fm/tokenizer/tokenize_events.py
```

它使用九个并行 token 通道：

```text
tok_evt_type / tok_side / tok_session / tok_board / tok_order_type
tok_event_source / tok_price_bin / tok_volume_bin / tok_delta_t_bin
```

v1 每字段有独立 id 空间，只有 `PAD=0`，连续字段默认使用 32 个全局训练期分位
bin。现有 v1 vocab、token 和 checkpoint 不迁移、不原地改写。

## v2 模块和字段契约

| 文件 | 责任 |
|------|------|
| `quant_fm/tokenizer/field_spec.py` | 冻结字段来源、类型、bin 数、输入/目标用途与适用事件 |
| `quant_fm/tokenizer/vocab_v2.py` | `VocabV2`、字段级词表、normalizer 和严格 artifact loader |
| `quant_fm/tokenizer/fit_bins_v2.py` | 完整训练流三遍扫描和确定性分层 priority reservoir |
| `quant_fm/tokenizer/tokenize_events_v2.py` | 生成 token/scalar 双通道及 coverage/leakage 检查 |
| `quant_fm/tokenizer/lob_transforms.py` | 因果盘口 pre/post 特征转换 |

`FieldSpec` 是 v2 的唯一字段顺序来源：

```python
from quant_fm.tokenizer import FULL_FIELD_SPECS_V2, FieldSpec

FieldSpec(
    name="imbalance_l5_post",
    source="imbalance_l5_post",
    kind="ordinal",
    n_bins=21,
    is_input=True,
    is_target=False,
)
```

标准集合包括：

- `DEFAULT_FIELD_SPECS_V2`：evt type、side、session、price、volume、delta time；
- `BOOK_FIELD_SPECS_V2`：book valid、spread、microprice、L1/L5/L10 imbalance、
  bid/ask L5 depth 和 pre-event price distance；
- `FULL_FIELD_SPECS_V2`：上述两组按冻结顺序拼接。

v2 第一阶段有意不包含 v1 的确定性 `event_source`、伪 `order_type` 和逐事件
`board`。恢复真实交易所 order type 前，不应重新加入该字段。

## v2 特殊 token 和双通道

v2 特殊 id 固定为：

```text
PAD=0, UNK=1, NA=2, BOS=3, EOS=4, SESSION_BREAK=5, N_SPECIAL=6
```

这些值只存在于 `vocab_v2.py`；v1 的 `PAD=0, N_SPECIAL=1` 不变。类别词表外值
映射到 `UNK`，真实缺失和字段不适用映射到 `NA`。

数值字段同时输出：

```text
tok_<field>_bin  # 有序 bin，缺失为 NA_ID
val_<field>      # 训练期 mean/std 标准化并 clip；缺失位置为 0
```

标量的 0 不承担缺失语义，模型依据独立 `NA` token 区分缺失。在模型侧，
`val_*` 经无 bias scalar projection 后加到对应 ordinal token 表示。

## 拟合 v2 vocab

目前 v2 拟合/分词提供 Python API，尚无独立批量 CLI。最小调用如下：

```python
from pathlib import Path

from quant_fm.tokenizer.field_spec import FULL_FIELD_SPECS_V2
from quant_fm.tokenizer.fit_bins_v2 import fit_vocab_v2
from quant_fm.tokenizer.tokenize_events_v2 import (
    assert_no_leakage_v2,
    coverage_report_v2,
    tokenize_path_v2,
)

train_paths = [Path(path) for path in training_event_shards]
vocab = fit_vocab_v2(
    train_paths,
    field_specs=FULL_FIELD_SPECS_V2,
    max_samples_per_field=5_000_000,
    fit_dates=train_dates,
    seed=42,
    schema_version="cn_l2_v2",
)
assert_no_leakage_v2(vocab, val_dates, test_dates)
vocab.save(Path("quant_fm/runs/v2_shared/data/vocab_v2.json"))

tokenize_path_v2(source_event_path, destination_token_path, vocab)
```

拟合过程按 `date × exchange × board × evt_type` 分层：

1. 第一遍遍历所有训练 shard，统计有限值、缺失、类别和各层样本量；
2. 第二遍对所有 shard 运行稳定 priority reservoir，不会在预算填满后提前停止；
3. 合并重复分位边界，实际 bin 数允许小于请求数；
4. 第三遍在冻结边界上统计完整训练流的精确 occupancy；
5. 保存 min/max、缺失率、训练 entropy、normalizer、采样 seed/方法和 fit dates。

路径输入顺序不会改变 artifact；但输入文件内容、FieldSpec、seed 或训练日期变化都会
生成不同 artifact，必须触发重新 token 化。

## 输出与验证闸门

建议 v2 产物使用独立目录：

```text
quant_fm/runs/v2_shared/data/vocab_v2.json
quant_fm/runs/v2_shared/data/tokens/<market>/<symbol>/<date>.parquet
```

关键检查：

- `VocabV2.load()` 拒绝 v1 artifact、错误 `vocab_version` 和特殊 id 漂移；
- `fit_vocab_v2()` 要求输入存在非空 `date`，并验证显式 `fit_dates` 与实际参与拟合
  shard 中的日期完全一致；artifact 始终写入实际观测日期，不能由调用方伪报；
- `assert_no_leakage_v2()` 再拒绝 val/test 日期出现在冻结后的 `fit_dates`；
- `coverage_report_v2()` 报告 edge-bin、NA、UNK 和实际 bin 数；
- token parquet 字段顺序来自 artifact 中的 `field_specs`；
- 同一事件 parquet、vocab 和代码版本应产生相同 token。

截至本次改造，全仓测试结果为 `243 passed, 2 skipped, 1 xfailed`，其中包含
`test_tokenizer_v2.py` 和 `test_fit_bins_stratified.py`。这仅说明实现和回归通过；真实
数据的 bin occupancy、缺失率、吞吐和模型收益仍需跑数验收。

## 版本隔离

- v1：`Vocab`、`vocab.json`、原 token schema、legacy checkpoint；
- v2：`VocabV2`、`vocab_version=2.0`、`schema_version=cn_l2_v2`、新 token 目录；
- v1 checkpoint 只能配 v1 vocab；v2 checkpoint 加载时必须提供原始
  `vocab_v2.json` 以校验 SHA-256、schema 和字段顺序；
- 不允许把 v2 特殊 id 写回 `quant_fm/tokenizer/vocab.py`。
