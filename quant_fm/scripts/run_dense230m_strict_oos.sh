#!/usr/bin/env bash
# 用全新目录重建 Dense230M 的严格可交易 Ranker 与 2026 OOS 评估。
# score(T) 在 T 收盘后才可用，训练标签和 OOS 收益统一使用 T+1 VWAP 到 T+2 VWAP。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
WORKDIR="${WORKDIR:-quant_fm/runs/oos2026_dense230}"
# 不给 legacy Dense230M 路径设默认值：strict gate 会拒绝它，调用方必须显式指向
# 由 exchange_time_sequence_v2 + causal transform + overlap/selected-v2 生成的新产物。
TRAIN_EMB_DIR="${TRAIN_EMB_DIR:-}"
TRAIN_EMBEDDINGS="${TRAIN_EMBEDDINGS:-${TRAIN_EMB_DIR:+$TRAIN_EMB_DIR/all.parquet}}"
OOS_EMBEDDINGS="${OOS_EMBEDDINGS:-}"
TRAIN_CALENDAR="${TRAIN_CALENDAR:-}"
OOS_CALENDAR="${OOS_CALENDAR:-}"
RETURN_SPEC="${RETURN_SPEC:-vwap_t1_vwap_t2}"
STRICT_DIR="${STRICT_DIR:-$WORKDIR/strict_v2}"
TRAIN_PANEL="${TRAIN_PANEL:-$STRICT_DIR/train_execution_panel.parquet}"
TRAIN_UNIVERSE="${TRAIN_UNIVERSE:-}"
OOS_UNIVERSE="${OOS_UNIVERSE:-}"
OOS_PANEL="${OOS_PANEL:-$STRICT_DIR/oos_execution_panel.parquet}"
DELIVERY_DIR="${DELIVERY_DIR:-$STRICT_DIR/delivery_oos}"
EVALUATION_DIR="${EVALUATION_DIR:-$STRICT_DIR/evaluation}"
MARKET_BENCHMARK_PANEL="${MARKET_BENCHMARK_PANEL:-}"
PRETRAIN_ACCEPTANCE="${PRETRAIN_ACCEPTANCE:-}"
EPOCHS="${EPOCHS:-30}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
MIN_NAMES_PER_DAY="${MIN_NAMES_PER_DAY:-350}"
RANKER_VAL_DAYS="${RANKER_VAL_DAYS:-10}"
RANKER_PURGE_DAYS="${RANKER_PURGE_DAYS:-2}"
RANKER_PATIENCE="${RANKER_PATIENCE:-8}"
# 可选人工断言；真实截止日由验收 checkpoint 的 manifest/vocab 血缘派生。
FM_TRAINING_END_DATE="${FM_TRAINING_END_DATE:-}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ -x "$PYTHON" ]] || fail "Python 不可执行：$PYTHON"
[[ "$RETURN_SPEC" == "vwap_t1_vwap_t2" ]] || \
  fail "严格入口只允许 RETURN_SPEC=vwap_t1_vwap_t2，收到：$RETURN_SPEC"
[[ -n "$TRAIN_EMB_DIR" ]] || fail "必须设置新版因果 TRAIN_EMB_DIR"
[[ -d "$TRAIN_EMB_DIR" ]] || fail "训练 embedding 目录不存在：$TRAIN_EMB_DIR"
[[ -n "$TRAIN_EMBEDDINGS" ]] || fail "必须设置新版因果 TRAIN_EMBEDDINGS"
[[ -n "$OOS_EMBEDDINGS" ]] || fail "必须设置新版因果 OOS_EMBEDDINGS"
[[ -f "$TRAIN_EMBEDDINGS" ]] || fail "缺少训练 embedding：$TRAIN_EMBEDDINGS"
[[ -f "$TRAIN_EMB_DIR/all.parquet" ]] || \
  fail "build_oos_delivery 实际训练文件不存在：$TRAIN_EMB_DIR/all.parquet"
[[ -f "$OOS_EMBEDDINGS" ]] || fail "缺少 OOS embedding：$OOS_EMBEDDINGS"
TRAIN_EMBEDDINGS_RESOLVED="$(realpath -e -- "$TRAIN_EMBEDDINGS")" || \
  fail "无法解析 TRAIN_EMBEDDINGS：$TRAIN_EMBEDDINGS"
TRAIN_EMB_ALL_RESOLVED="$(realpath -e -- "$TRAIN_EMB_DIR/all.parquet")" || \
  fail "无法解析 TRAIN_EMB_DIR/all.parquet：$TRAIN_EMB_DIR/all.parquet"
[[ "$TRAIN_EMBEDDINGS_RESOLVED" == "$TRAIN_EMB_ALL_RESOLVED" ]] || \
  fail "TRAIN_EMBEDDINGS 必须解析为 TRAIN_EMB_DIR/all.parquet；预检与实际训练不得读取不同文件"
[[ -n "$TRAIN_CALENDAR" ]] || fail "必须设置 TRAIN_CALENDAR（需覆盖最后训练信号的 T+2）"
[[ -n "$OOS_CALENDAR" ]] || fail "必须设置 OOS_CALENDAR（需覆盖最后 OOS 信号的 T+2）"
[[ -f "$TRAIN_CALENDAR" ]] || fail "训练日历不存在：$TRAIN_CALENDAR"
[[ -f "$OOS_CALENDAR" ]] || fail "OOS 日历不存在：$OOS_CALENDAR"
[[ -n "$TRAIN_UNIVERSE" ]] || fail "必须设置逐日 PIT TRAIN_UNIVERSE（date,symbol,asof_date,universe_policy）"
[[ -f "$TRAIN_UNIVERSE" ]] || fail "训练 PIT 股票池不存在：$TRAIN_UNIVERSE"
[[ -n "$OOS_UNIVERSE" ]] || fail "必须设置逐日 PIT OOS_UNIVERSE（不能把 embedding 当股票池）"
[[ -f "$OOS_UNIVERSE" ]] || fail "OOS PIT 股票池不存在：$OOS_UNIVERSE"
[[ -n "$PRETRAIN_ACCEPTANCE" ]] || fail "严格入口必须设置新版 FM 的 PRETRAIN_ACCEPTANCE"
[[ -f "$PRETRAIN_ACCEPTANCE" ]] || fail "预训练非劣验收尚未通过/生成：$PRETRAIN_ACCEPTANCE"

mkdir -p "$STRICT_DIR"

"$PYTHON" -m quant_fm.scripts.validate_pretrain_acceptance \
  --path "$PRETRAIN_ACCEPTANCE"

lineage_args=(
  -m quant_fm.scripts.validate_pretrain_lineage
  --acceptance "$PRETRAIN_ACCEPTANCE"
  --train-embeddings "$TRAIN_EMBEDDINGS"
  --oos-embeddings "$OOS_EMBEDDINGS"
  --out "$STRICT_DIR/pretrain_lineage.json"
)
if [[ -n "$FM_TRAINING_END_DATE" ]]; then
  lineage_args+=(--expected-training-end "$FM_TRAINING_END_DATE")
fi
echo "==> 复核 FM 验收、checkpoint、manifest/vocab 与 embedding 全链路"
"$PYTHON" "${lineage_args[@]}"

if [[ -n "$MARKET_BENCHMARK_PANEL" ]]; then
  [[ -f "$MARKET_BENCHMARK_PANEL" ]] || fail "全市场基准 panel 不存在：$MARKET_BENCHMARK_PANEL"
fi

echo "==> 严格输入预检（PIT/as-of/policy/宽度/T+1-T+2 日历）"
"$PYTHON" -m quant_fm.scripts.preflight_topk_ranker \
  --train-embeddings "$TRAIN_EMBEDDINGS" \
  --oos-embeddings "$OOS_EMBEDDINGS" \
  --train-calendar "$TRAIN_CALENDAR" \
  --oos-calendar "$OOS_CALENDAR" \
  --train-universe "$TRAIN_UNIVERSE" \
  --oos-universe "$OOS_UNIVERSE" \
  --return-spec "$RETURN_SPEC" \
  --min-names-per-day "$MIN_NAMES_PER_DAY" \
  --out "$STRICT_DIR/preflight.json"

echo "==> 构建严格训练标签：$RETURN_SPEC"
"$PYTHON" -m quant_fm.downstream.build_panel_from_minio \
  --from-embeddings "$TRAIN_EMBEDDINGS" \
  --calendar-file "$TRAIN_CALENDAR" \
  --return-spec "$RETURN_SPEC" \
  --out "$TRAIN_PANEL"

echo "==> 构建同口径 2026 OOS panel：$RETURN_SPEC"
"$PYTHON" -m quant_fm.downstream.build_panel_from_minio \
  --from-embeddings "$OOS_EMBEDDINGS" \
  --calendar-file "$OOS_CALENDAR" \
  --return-spec "$RETURN_SPEC" \
  --out "$OOS_PANEL"

echo "==> 按时间留出 + purge + Multi-K LambdaNDCG 重新训练 Ranker"
delivery_args=(
  -m quant_fm.scripts.build_oos_delivery
  --train-emb-dir "$TRAIN_EMB_DIR"
  --train-panel "$TRAIN_PANEL"
  --train-universe "$TRAIN_UNIVERSE"
  --train-calendar "$TRAIN_CALENDAR"
  --test-emb "$OOS_EMBEDDINGS"
  --test-universe "$OOS_UNIVERSE"
  --pretrain-acceptance "$PRETRAIN_ACCEPTANCE"
  --out-dir "$DELIVERY_DIR"
  --epochs "$EPOCHS"
  --seed "$SEED"
  --device "$DEVICE"
  --min-names-per-day "$MIN_NAMES_PER_DAY"
  --ranker-val-days "$RANKER_VAL_DAYS"
  --ranker-purge-days "$RANKER_PURGE_DAYS"
  --ranker-patience "$RANKER_PATIENCE"
  --ndcg-ks "50,300,350"
  --ndcg-k-weights "0.20,0.60,0.20"
  --head-loss-weight "1.0"
  --global-ic-weight "0.30"
  --aux-huber-weight "0.05"
  --aux-huber-beta "0.5"
  --pair-samples-per-day "8192"
  --hard-pair-fraction "0.75"
  --min-label-rank-gap "0.02"
  --score-temperature "1.0"
)
if [[ -n "$FM_TRAINING_END_DATE" ]]; then
  delivery_args+=(--fm-training-end-date "$FM_TRAINING_END_DATE")
fi
"$PYTHON" "${delivery_args[@]}"

evaluation_args=(
  -m quant_fm.downstream.run_score_evaluation
  --scores "$DELIVERY_DIR/scores.parquet"
  --panel "$OOS_PANEL"
  --calendar "$OOS_CALENDAR"
  --out-dir "$EVALUATION_DIR"
)
if [[ -n "$MARKET_BENCHMARK_PANEL" ]]; then
  evaluation_args+=(--market-benchmark-panel "$MARKET_BENCHMARK_PANEL")
fi

echo "==> 在同一可交易 panel 上运行低换手主组合与参数网格"
"$PYTHON" "${evaluation_args[@]}"

echo "==> 完成：$EVALUATION_DIR/metrics.json"
