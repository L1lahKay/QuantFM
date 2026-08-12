.PHONY: help install install-fm smoke signal signal-smoke backtest-contract-fixture dense230-finalize pilot medium medium-try medium-pipeline medium-estimate medium-smoke judge-medium-try judge-300m research-score research-oos2026 watch-300m baselines-300m tensorboard-medium train train-pilot train-8gpu train-medium-8gpu check-minio upload-pilot upload-medium download-medium minio-pipeline minio-pipeline-full minio-full-pipeline minio-full-pipeline-full test lint

PY ?= python
WORKDIR ?= quant_fm/runs/v2_pilot
MEDIUM_WORKDIR ?= quant_fm/runs/v2_shared
SIGNAL_EMBEDDINGS ?= quant_fm/runs/oos2026/embeddings/all.parquet
SIGNAL_TRAIN_WORKDIR ?= quant_fm/runs/medium_300m
SIGNAL_ARTIFACT ?= $(SIGNAL_TRAIN_WORKDIR)/signal_artifact
SIGNAL_FM_CHECKPOINT ?= $(SIGNAL_TRAIN_WORKDIR)/run/best.pt
SIGNAL_VOCAB ?= $(SIGNAL_TRAIN_WORKDIR)/data/vocab_v2.json
SIGNAL_UNIVERSE ?=
SIGNAL_REGIME_FEATURES ?=
SIGNAL_OUT ?= quant_fm/runs/oos2026/delivery

help:
	@echo "QuantFM 目标："
	@echo "  make install-fm   安装 FM 训练额外依赖（torch、pyyaml、tensorboard）"
	@echo "  make smoke        在合成数据上跑通全流程（CPU，无需 MinIO）"
	@echo "  make signal       用冻结 Ranker 生成 date/symbol/score 生产信号"
	@echo "  make signal-smoke 验证无标签 score 生成链路"
	@echo "  make backtest-contract-fixture 生成隔离的合成回测联调包"
	@echo "  make dense230-finalize 等待真实信号并自动审计、门禁和打包"
	@echo "  make pilot        真实盘口回放并生成 V2 events/tokens/manifest"
	@echo "  make upload-pilot 上传 tokens 到 model-cache（写 MinIO :9100）"
	@echo "  make check-minio  检查 MinIO 读写 endpoint 连通性"
	@echo "  make medium       V2 medium：60 日 × 全市场（本地生成并完整审计）"
	@echo "  make minio-pipeline       MinIO读→tokens→写（试跑，无训练）"
	@echo "  make minio-pipeline-full  同上，60日×全市场（无训练）"
	@echo "  make minio-full-pipeline       【推荐】读→tokens→写→8卡训练（试跑）"
	@echo "  make minio-full-pipeline-full  同上，60日×全市场 + 训练"
	@echo "  make medium-estimate  仅估算事件量与推荐模型规模"
	@echo "  make train        从 quant_fm/pretrain/config.yaml 开始预训练"
	@echo "  make train-pilot  单卡 pilot 预训练（需先 make pilot）"
	@echo "  make train-8gpu   8 卡 pilot 预训练（torchrun + FSDP）"
	@echo "  make train-medium-8gpu  8 卡中等规模预训练（需先有 medium tokens）"
	@echo "  make judge-300m   [research-only] embedding → panel → 下游 judge"
	@echo "  make watch-300m   监控 302M 训练（AUTO_JUDGE=1 训完自动 judge）"
	@echo "  make baselines-300m  [research-only] 生成传统因子基线"
	@echo "  make test         运行 pytest"
	@echo "  make lint         运行 ruff"

install:
	uv sync

install-fm:
	uv sync --extra fm

# 在合成数据上做端到端验证，需先安装 fm 额外依赖
smoke:
	$(PY) -m quant_fm.scripts.smoke --workdir quant_fm/runs/smoke

signal-smoke: smoke

signal:
	@test -f "$(SIGNAL_EMBEDDINGS)" || { echo "missing SIGNAL_EMBEDDINGS=$(SIGNAL_EMBEDDINGS)" >&2; exit 2; }
	@test -f "$(SIGNAL_ARTIFACT)/ranker.pt" || { echo "missing ranker artifact under $(SIGNAL_ARTIFACT)" >&2; exit 2; }
	@test -f "$(SIGNAL_ARTIFACT)/ranker_metadata.json" || { echo "missing ranker metadata under $(SIGNAL_ARTIFACT)" >&2; exit 2; }
	@test -f "$(SIGNAL_FM_CHECKPOINT)" || { echo "missing SIGNAL_FM_CHECKPOINT=$(SIGNAL_FM_CHECKPOINT)" >&2; exit 2; }
	@test -f "$(SIGNAL_VOCAB)" || { echo "missing SIGNAL_VOCAB=$(SIGNAL_VOCAB)" >&2; exit 2; }
	@test -n "$(SIGNAL_UNIVERSE)" || { echo "SIGNAL_UNIVERSE must point to the daily PIT scoring universe" >&2; exit 2; }
	@test -f "$(SIGNAL_UNIVERSE)" || { echo "missing SIGNAL_UNIVERSE=$(SIGNAL_UNIVERSE)" >&2; exit 2; }
	$(PY) -m quant_fm.signal.generate \
		--embeddings "$(SIGNAL_EMBEDDINGS)" \
		--ranker "$(SIGNAL_ARTIFACT)/ranker.pt" \
		--ranker-metadata "$(SIGNAL_ARTIFACT)/ranker_metadata.json" \
		--fm-checkpoint "$(SIGNAL_FM_CHECKPOINT)" \
		--vocab "$(SIGNAL_VOCAB)" \
		--universe "$(SIGNAL_UNIVERSE)" \
		$(if $(strip $(SIGNAL_REGIME_FEATURES)),--regime-features "$(SIGNAL_REGIME_FEATURES)" ,)--out-dir "$(SIGNAL_OUT)"

backtest-contract-fixture:
	$(PY) -m quant_fm.scripts.build_backtest_contract_fixture \
		--out-root "$${OUT_ROOT:-quant_fm/runs/backtest_contract_fixture_$$(date +%Y%m%dT%H%M%S)}"

dense230-finalize:
	WAIT="$${WAIT:-1}" bash quant_fm/scripts/finalize_dense230m_delivery.sh

# 真实试点：先设置 MINIO_* 环境变量，下方为示例日期/股票
pilot:
	$(PY) -m quant_fm.scripts.run_pilot \
		--dates 2026-02-02,2026-02-03,2026-02-04,2026-02-05,2026-02-06 \
		--symbols 000001,000002,300750 \
		--market SZ \
		--workdir $(WORKDIR) \
		--data-version v2 \
		--train-end 2026-02-04 --val-end 2026-02-05 --n-bins 32

# 中等规模：2025 均匀 60 交易日 × 沪深全市场；删除中间产物以省磁盘
medium-estimate:
	$(PY) -m quant_fm.scripts.run_medium --estimate-only --workdir $(MEDIUM_WORKDIR)

medium:
	$(PY) -m quant_fm.scripts.run_v2_parallel_data \
		--workdir $(MEDIUM_WORKDIR) \
		--groups $${NGROUPS:-2} \
		--clean-workers $${CLEAN_WORKERS:-30} \
		--canon-workers $${CANON_WORKERS:-8} \
		--tokenize-workers $${TOKENIZE_WORKERS:-16}

# 试跑：每市场 50 只股票 × 60 天（验证流水线，磁盘约数 GB）
medium-smoke:
	$(PY) -m quant_fm.scripts.run_medium \
		--workdir $(MEDIUM_WORKDIR)_smoke \
		--data-version v2 \
		--fast-clean \
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
	CONFIG=quant_fm/pretrain/config_v2_25m.yaml MEDIUM_WORKDIR=$(WORKDIR) NPROC=1 \
		bash quant_fm/scripts/train_medium_8gpu.sh

train-8gpu:
	CONFIG=quant_fm/pretrain/config_v2_25m.yaml MEDIUM_WORKDIR=$(WORKDIR) NPROC=8 \
		bash quant_fm/scripts/train_medium_8gpu.sh

train-medium-8gpu:
	CONFIG=quant_fm/pretrain/config_v2_230m.yaml MEDIUM_WORKDIR=$(MEDIUM_WORKDIR) \
		bash quant_fm/scripts/train_medium_8gpu.sh

upload-pilot:
	$(PY) -m quant_fm.scripts.upload_to_minio --workdir $(WORKDIR) --tag v2_pilot

upload-medium:
	$(PY) -m quant_fm.scripts.upload_to_minio --workdir $(MEDIUM_WORKDIR) --tag v2_shared

download-medium:
	$(PY) -m quant_fm.scripts.download_from_minio --workdir $(MEDIUM_WORKDIR) --tag v2_shared --data-version v2

# RESEARCH ONLY：下游裁判不属于 score 生产链路
judge-medium-try:
	$(PY) -m quant_fm.downstream.run_judge --workdir quant_fm/runs/medium_try --checkpoint quant_fm/runs/medium_try/run/best.pt

# 302M：抽 embedding → panel → judge（训练完成后）
judge-300m:
	bash quant_fm/scripts/run_judge_300m.sh

# 严格 OOS score 研究评估（调用方提供 execution panel）
research-score:
	@test -n "$(SCORES)" || (echo "need SCORES=/path/to/scores.parquet" && exit 2)
	@test -n "$(PANEL)" || (echo "need PANEL=/path/to/execution_panel.parquet" && exit 2)
	$(PY) -m quant_fm.downstream.run_score_evaluation \
		--scores "$(SCORES)" \
		--panel "$(PANEL)" \
		--out-dir "$${OUT_DIR:-quant_fm/runs/research_score}"

research-oos2026:
	@test -n "$(CALENDAR)" || (echo "need CALENDAR=/path/to/calendar_with_two_future_days.txt" && exit 2)
	bash quant_fm/scripts/run_oos2026_research.sh

# 监控 302M 训练；AUTO_JUDGE=1 训完自动跑下游
watch-300m:
	AUTO_JUDGE=$${AUTO_JUDGE:-1} bash quant_fm/scripts/watch_300m_train.sh

# 传统因子基线（默认跳过全量 OFI；加 MAX_OFI_SHARDS=3000 可扫部分 tokens）
baselines-300m:
	$(PY) -m quant_fm.downstream.baselines \
		--panel quant_fm/runs/medium_300m/panel/daily_panel.parquet \
		--out quant_fm/runs/medium_300m/panel/factors.parquet \
		--skip-ofi

check-minio:
	$(PY) -m quant_fm.scripts.check_minio

test:
	uv run python -m pytest -q

lint:
	uv run ruff check quant_fm order_book/pylob tests
