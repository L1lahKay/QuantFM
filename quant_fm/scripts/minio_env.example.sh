#!/usr/bin/env bash
# 一次性 MinIO 凭据配置（endpoint / bucket 已在代码里写死，通常不用改）。
#
# 用法：
#   cp quant_fm/scripts/minio_env.example.sh ~/.minio_fm_env.sh
#   vim ~/.minio_fm_env.sh
#   source ~/.minio_fm_env.sh

# ── 读端凭据（:9000 / zeus-cn-quote）──
export MINIO_ACCESS_KEY="REPLACE_ME"
export MINIO_SECRET_KEY="REPLACE_ME"
export MINIO_READ_ENDPOINT=192.168.2.11:9000
export MINIO_BUCKET=zeus-cn-quote

# ── 写端凭据（:9100 / model-cache；可与读端不同）──
export MINIO_WRITE_ACCESS_KEY="REPLACE_ME"
export MINIO_WRITE_SECRET_KEY="REPLACE_ME"
export MINIO_WRITE_ENDPOINT=192.168.2.11:9100
export MINIO_OUTPUT_BUCKET=model-cache
export MINIO_OUTPUT_PREFIX="fm-pretrain/${USER:-user}"

# export MINIO_SECURE=false
