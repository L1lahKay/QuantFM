# 阶段 5：Manifest 与时间切分

## 目标

为训练器生成唯一、可审计的数据目录，明确每个 token 分片的路径、大小、内容哈希和 train/val/test 归属。

## 核心代码

`quant_fm/manifest/build_manifest.py`

Manifest 的 shard 结构被 V1/V2 共用。`build_manifest()` 会从显式传入的 vocab artifact 加载版本并写入 `schema_version`、vocab hash、事件排序和特征变换契约；只有调用方不传 vocab 时才保留 `cn_l2_v1` 兼容默认值。当前 V2 一键脚本始终传入同批 `vocab_v2.json`，并在发布前运行 artifact 审计。

目录扫描约定：

```text
tokens/<market>/<symbol>/<date>.parquet
```

每个 `ShardEntry` 记录：

- `market`
- `symbol`
- `date`
- `path`
- `rows`
- `sha256`
- `split`

## 时间切分

```text
date <= train_end                 → train
train_end < date <= val_end       → val
date > val_end                    → test
```

Pilot 显式指定边界；Medium 未显式指定时按日期数量进行 70% / 15% / 15% 切分。

严禁随机打散事件或股日后再切分，否则未来市场状态可能泄漏到训练集。

## 输入与输出

输入：

- `tokens_dir`
- `train_end`
- `val_end`
- market 列表
- `vocab_path`

输出：

```text
<workdir>/data/manifest.json
```

结构示例：

```json
{
  "schema_version": "cn_l2_v1",
  "train_end": "2026-02-04",
  "val_end": "2026-02-05",
  "vocab_path": "quant_fm/runs/pilot/data/vocab.json",
  "shards": [
    {
      "market": "SZ",
      "symbol": "000001",
      "date": "2026-02-02",
      "path": ".../tokens/SZ/000001/2026-02-02.parquet",
      "rows": 201014,
      "sha256": "...",
      "split": "train"
    }
  ]
}
```

## 运行

该阶段通常由 `run_pilot.py` 或 `run_medium.py` 自动完成：

```python
manifest = build_manifest(
    tokens_dir,
    train_end=train_end,
    val_end=val_end,
    markets=("SZ", "SH"),
    vocab_path=str(vocab_path),
)
# 仅 V2 产物链需要；V1 保持默认值
# manifest.schema_version = vocab.schema_version
manifest.save(data_dir / "manifest.json")
```

## 验证条件

- train/val/test 日期严格按时间分离；
- manifest 中所有文件存在；
- `rows` 与 parquet metadata 一致；
- 重新计算的 SHA-256 与记录一致；
- `vocab_path` 指向生成这些 token 的冻结词表。
- V2 manifest 的 `schema_version` 与 `vocab_v2.json` 一致，且分片实际包含冻结 `FieldSpec` 派生的全部 `tok_*` / `val_*` 列。

V2 checkpoint 会固化并在加载时校验 `schema_version`、vocab SHA-256、字段顺序和 target specs；但这不替代生成 manifest 时的数据列审计。

## 可迁移性注意

当前 shard path 为绝对路径，仓库或数据目录迁移后应重建 manifest 或替换为新根路径。上传到 MinIO 后再下载的脚本负责恢复本地目录结构。
