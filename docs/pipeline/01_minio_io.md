# 阶段 1：MinIO 数据接入

## 目标

从只读 bucket 获取沪深 Level-2 原始逐笔数据，并将可复用训练产物写入独立的模型缓存 bucket。

| 方向 | Endpoint | Bucket | 内容 |
|------|----------|--------|------|
| 读 | `192.168.2.11:9000` | `zeus-cn-quote` | trade/order 原始 parquet |
| 写 | `192.168.2.11:9100` | `model-cache` | tokens、vocab、manifest |

## 配置

```bash
cp quant_fm/scripts/minio_env.example.sh ~/.minio_fm_env.sh
chmod 600 ~/.minio_fm_env.sh
# 编辑凭据
source ~/.minio_fm_env.sh
make check-minio
```

代码优先读取环境变量；未配置时回退到 `~/.mc/config.json` 的 `myminio` alias。推荐显式加载环境文件，因为读写端可能使用不同密钥。

完整变量说明见 [MinIO 读写指南](../minio_setup.md)。

## 对象布局

真实流水线采用 `zeus_default`：

```text
HDS/SOURCE=zeus/DOMAIN=quote/DATASET=china_stock/
  YYYY.MM/YYYY.MM.DD/default/2/all.parquet  # trade
  YYYY.MM/YYYY.MM.DD/default/3/all.parquet  # order
```

路径由以下函数生成：

- `pylob.pipeline.paths.zeus_default_trade_key()`
- `pylob.pipeline.paths.zeus_default_order_key()`

当前账号可能没有 `ListBucket` 权限，因此流水线按确定的 object key 直接读取，不依赖列举 bucket。

## 核心代码

| 文件 | 责任 |
|------|------|
| `quant_fm/scripts/minio_config.py` | 读写 endpoint、bucket、凭据与 storage options |
| `order_book/pylob/pipeline/paths.py` | 构造 trade/order object key |
| `order_book/pylob/pipeline/s3_io.py` | Polars 流式读取 S3 parquet |
| `quant_fm/scripts/upload_to_minio.py` | 上传 tokens、vocab、manifest |
| `quant_fm/scripts/download_from_minio.py` | 从模型缓存恢复训练数据 |

## 输入与输出

输入：

- 日期；
- 市场与标的列表；
- MinIO 读凭据。

输出：

- Polars `DataFrame` 形式的原始 trade/order；
- 或远端 `s3://model-cache/fm-pretrain/<user>/<tag>/` 训练数据副本。

## 验证

健康检查只验证服务存活，不能证明权限正确：

```bash
make check-minio
```

真实权限应通过直接读取已知 key、上传探测文件并删除来验证。不得将密钥或用户环境文件提交到 GitHub。

## 常见问题

- `AccessDenied`：凭据缺少读取权限，或 object key 缺少 `HDS/...` 前缀。
- `SignatureDoesNotMatch`：写端使用了错误密钥，确认已 `source ~/.minio_fm_env.sh`。
- 读取很慢：单日合并文件可达数亿行，属预期行为；下游通过 symbol filter 降低进入撮合阶段的数据量。
