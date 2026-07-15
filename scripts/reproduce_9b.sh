#!/usr/bin/env bash
# Reproduce the 9B MATH-500 active-GRPO experiment exactly.
#
# This runs the focused variant set (Map + best LoRA r8-lr3e-5) with bank
# refresh and deterministic mode enabled. The reproducibility.json written to
# the output dir captures the exact environment (git commit, versions, args,
# env vars) needed to verify a reproduction.
#
# Usage:
#   ./scripts/reproduce_9b.sh
#   HF_TOKEN=hf_... ./scripts/reproduce_9b.sh
#   BANK_REFRESH_INTERVAL=50 TARGET_UPDATES=100 ./scripts/reproduce_9b.sh
#
# Requirements: a CUDA GPU with >=24GB memory (9B model bf16 + training buffers).
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

# ---- Configurable knobs (env overrides) ------------------------------------
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
OUT_DIR="${OUT_DIR:-results/9b-math500/reproduce}"
SEED="${SEED:-0}"
TARGET_UPDATES="${TARGET_UPDATES:-100}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-200}"
TIME_BUDGET_S="${TIME_BUDGET_S:-1200}"
BANK_REFRESH_INTERVAL="${BANK_REFRESH_INTERVAL:-50}"
CANDIDATE_N="${CANDIDATE_N:-100}"
PROBE_K="${PROBE_K:-8}"
K="${K:-8}"
MAX_NEW="${MAX_NEW:-64}"
MAX_NEW_EVAL="${MAX_NEW_EVAL:-128}"
EVAL_N="${EVAL_N:-200}"
TRAIN_BATCH="${TRAIN_BATCH:-1}"
MICRO_BATCH="${MICRO_BATCH:-1}"
BETA_KL="${BETA_KL:-0.05}"
LORA_R="${LORA_R:-8}"
LORA_LR="${LORA_LR:-3e-5}"

PROMPT_SUFFIX=(
  "Return only the final answer in \\boxed{...}. "
  "No explanation. No text after the box. /no_think"
)
CHAT_KWARGS='{"enable_thinking": false}'

echo "[reproduce] model=$MODEL"
echo "[reproduce] out_dir=$OUT_DIR"
echo "[reproduce] seed=$SEED target_updates=$TARGET_UPDATES bank_refresh_interval=$BANK_REFRESH_INTERVAL"

# ---- Memory / throughput env ----------------------------------------------
export PYTHONUNBUFFERED=1
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Install pinned deps ---------------------------------------------------
python3 -m pip install -q -r requirements.txt

# ---- Run the experiment ----------------------------------------------------
# Variant 1: Map-G2048 (builds the active bank + baseline eval).
python3 experiments/math500_active_grpo_9b.py \
  --model "$MODEL" \
  --dtype bf16 \
  --min-level 1 --max-level 3 \
  --candidate-n "$CANDIDATE_N" \
  --probe-k "$PROBE_K" \
  --K "$K" \
  --max-new "$MAX_NEW" \
  --max-new-eval "$MAX_NEW_EVAL" \
  --eval-n "$EVAL_N" \
  --eval-batch 1 \
  --target-updates "$TARGET_UPDATES" \
  --max-attempts "$MAX_ATTEMPTS" \
  --train-batch "$TRAIN_BATCH" \
  --micro-batch "$MICRO_BATCH" \
  --beta-kl "$BETA_KL" \
  --time-budget-s "$TIME_BUDGET_S" \
  --bank-refresh-interval "$BANK_REFRESH_INTERVAL" \
  --prompt-suffix "$PROMPT_SUFFIX" \
  --chat-template-kwargs "$CHAT_KWARGS" \
  --eval-after-train \
  --deterministic \
  --seed "$SEED" \
  --variants map \
  --out-dir "$OUT_DIR/map"

# Variant 2: LoRA r8-lr3e-5 (best LoRA). Reuses the Map bank, skips baseline eval.
python3 experiments/math500_active_grpo_9b.py \
  --model "$MODEL" \
  --dtype bf16 \
  --min-level 1 --max-level 3 \
  --candidate-n "$CANDIDATE_N" \
  --probe-k "$PROBE_K" \
  --K "$K" \
  --max-new "$MAX_NEW" \
  --max-new-eval "$MAX_NEW_EVAL" \
  --eval-n "$EVAL_N" \
  --eval-batch 1 \
  --target-updates "$TARGET_UPDATES" \
  --max-attempts "$MAX_ATTEMPTS" \
  --train-batch "$TRAIN_BATCH" \
  --micro-batch "$MICRO_BATCH" \
  --beta-kl "$BETA_KL" \
  --time-budget-s "$TIME_BUDGET_S" \
  --bank-refresh-interval "$BANK_REFRESH_INTERVAL" \
  --prompt-suffix "$PROMPT_SUFFIX" \
  --chat-template-kwargs "$CHAT_KWARGS" \
  --eval-after-train \
  --skip-baseline-eval \
  --active-bank-json "$OUT_DIR/map/active_bank.json" \
  --lora-r "$LORA_R" \
  --lora-lr "$LORA_LR" \
  --deterministic \
  --seed "$SEED" \
  --variants lora \
  --out-dir "$OUT_DIR/lora-r8-lr3e-5"

echo ""
echo "[reproduce] DONE. Artifacts in $OUT_DIR"
echo "[reproduce]   map/active_bank.json, map/reproducibility.json"
echo "[reproduce]   map/active_train_summary.json, lora-r8-lr3e-5/active_train_summary.json"
