#!/usr/bin/env bash
# 已合并到 run_minio_full_pipeline.sh（读→tokens→写→训练）。
# 保留本文件以免旧文档链接失效。
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_minio_full_pipeline.sh" "$@"
