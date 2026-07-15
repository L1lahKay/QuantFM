.PHONY: help install install-fm smoke pilot medium medium-try medium-pipeline medium-estimate medium-smoke judge-medium-try tensorboard-medium train train-pilot train-8gpu train-medium-8gpu check-minio upload-pilot upload-medium download-medium minio-pipeline minio-pipeline-full minio-full-pipeline minio-full-pipeline-full test lint

PY ?= python
WORKDIR ?= quant_fm/runs/pilot
MEDIUM_WORKDIR ?= quant_fm/runs/medium

help:
	@echo "QuantFM 目标："
	@echo "  make install-fm   安装 FM 训练额外依赖（torch、pyyaml、tensorboard）"
	@echo "  make smoke        在合成数据上跑通全流程（CPU，无需 MinIO）"
	@echo "  make pilot        清洗并 tokenize（读 MinIO :9000，凭据见 minio_setup.md）"
	@echo "  make upload-pilot 上传 tokens 到 model-cache（写 MinIO :9100）"
	@echo "  make check-minio  检查 MinIO 读写 endpoint 连通性"
	@echo "  make medium       全量 medium：60 日 × 全市场（仅本地数据，不上传/训练）"
	@echo "  make minio-pipeline       MinIO读→tokens→写（试跑，无训练）"
	@echo "  make minio-pipeline-full  同上，60日×全市场（无训练）"
	@echo "  make minio-full-pipeline       【推荐】读→tokens→写→8卡训练（试跑）"
	@echo "  make minio-full-pipeline-full  同上，60日×全市场 + 训练"
	@echo "  make medium-estimate  仅估算事件量与推荐模型规模"
	@echo "  make train        从 quant_fm/pretrain/config.yaml 开始预训练"
	@echo "  make train-pilot  单卡 pilot 预训练（需先 make pilot）"
	@echo "  make train-8gpu   8 卡 pilot 预训练（torchrun + FSDP）"
	@echo "  make train-medium-8gpu  8 卡中等规模预训练（需先有 medium tokens）"
	@echo "  make test         运行 pytest"
	@echo "  make lint         运行 ruff"

install:
	uv sync

install-fm:
	uv sync --extra fm

# 在合成数据上做端到端验证，需先安装 fm 额外依赖
smoke:
	$(PY) -m quant_fm.scripts.smoke --workdir quant_fm/runs/smoke

# 真实试点：先设置 MINIO_* 环境变量，下方为示例日期/股票
pilot:
	$(PY) -m quant_fm.scripts.run_pilot \
		--dates 2026-02-02,2026-02-03,2026-02-04,2026-02-05,2026-02-06 \
		--symbols 000001,000002,300750 \
		--market SZ \
		--workdir $(WORKDIR) \
		--train-end 2026-02-04 --val-end 2026-02-05 --n-bins 32

# 中等规模：2025 均匀 60 交易日 × 沪深全市场；删除中间产物以省磁盘
medium-estimate:
	$(PY) -m quant_fm.scripts.run_medium --estimate-only --workdir $(MEDIUM_WORKDIR)

medium:
	$(PY) -m quant_fm.scripts.run_medium \
		--workdir $(MEDIUM_WORKDIR) \
		--drop-clean --drop-events --resume

# 试跑：每市场 50 只股票 × 60 天（验证流水线，磁盘约数 GB）
medium-smoke:
	$(PY) -m quant_fm.scripts.run_medium \
		--workdir $(MEDIUM_WORKDIR)_smoke \
		--max-symbols-per-market 50 \
		--drop-clean --drop-events

# MinIO 数据流水线（读 9000 → 写 9100，不含训练）
minio-pipeline:
	MODE=try bash quant_fm/scripts/run_minio_data_pipeline.sh

minio-pipeline-full:
	MODE=full bash quant_fm/scripts/run_minio_data_pipeline.sh

# 完整流水线：读 MinIO → tokens → 写 MinIO → 8 卡训练（本地保留 tokens）
minio-full-pipeline:
	MODE=try bash quant_fm/scripts/run_minio_full_pipeline.sh

minio-full-pipeline-full:
	MODE=full bash quant_fm/scripts/run_minio_full_pipeline.sh

medium-try:
	MODE=try bash quant_fm/scripts/run_minio_data_pipeline.sh

medium-pipeline:
	@echo "DEPRECATED: use make minio-full-pipeline" && MODE=try bash quant_fm/scripts/run_minio_full_pipeline.sh

tensorboard-medium:
	bash quant_fm/scripts/start_tensorboard_medium.sh

train:
	$(PY) -m quant_fm.pretrain.train --config quant_fm/pretrain/config.yaml

train-pilot:
	$(PY) -m quant_fm.pretrain.train --config quant_fm/pretrain/config_pilot.yaml

train-8gpu:
	bash quant_fm/scripts/train_pilot_8gpu.sh

train-medium-8gpu:
	bash quant_fm/scripts/train_medium_8gpu.sh

upload-pilot:
	$(PY) -m quant_fm.scripts.upload_to_minio --workdir $(WORKDIR) --tag pilot

upload-medium:
	$(PY) -m quant_fm.scripts.upload_to_minio --workdir $(MEDIUM_WORKDIR) --tag medium

download-medium:
	$(PY) -m quant_fm.scripts.download_from_minio --workdir $(MEDIUM_WORKDIR) --tag medium

# 下游裁判（默认用 best.pt；结果写入 workdir/downstream/runs/ + history.jsonl）
judge-medium-try:
	$(PY) -m quant_fm.downstream.run_judge --workdir quant_fm/runs/medium_try --checkpoint quant_fm/runs/medium_try/run/best.pt

check-minio:
	$(PY) -m quant_fm.scripts.check_minio

test:
	uv run python -m pytest -q

lint:
	uv run ruff check quant_fm order_book/pylob tests
