#!/usr/bin/env bash
# Deploy the Qwen3.5-9B Map + LoRA lr/rank sweep via `colab run`.
#
# `colab run` rents a fresh GPU VM, runs the launcher script, and releases the
# VM when the script exits. This is more robust than `colab new` + `colab exec`
# (whose session can be reclaimed during the long model download).
#
#   ./scripts/deploy_qwen35_lora_sweep_colab.sh
#   LORA_VARIANTS=r8-lr1e-4,r8-lr3e-4 ./scripts/deploy_qwen35_lora_sweep_colab.sh
#   TIME_BUDGET_S=1800 TARGET_UPDATES=30 ./scripts/deploy_qwen35_lora_sweep_colab.sh
#
# The launcher clones the repo from GitHub (public), so no archive upload needed.
# Artifacts are base64-encoded to stdout; decode from the log after the run.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

GPU="${GPU:-G4}"
TIMEOUT="${TIMEOUT:-86400}"
TARGET_UPDATES="${TARGET_UPDATES:-50}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-200}"
TIME_BUDGET_S="${TIME_BUDGET_S:-1200}"
CANDIDATE_N="${CANDIDATE_N:-100}"
MAX_NEW="${MAX_NEW:-64}"
EVAL_N="${EVAL_N:-200}"
MICRO_BATCH="${MICRO_BATCH:-2}"
LORA_VARIANTS="${LORA_VARIANTS:-}"
LOG="/tmp/colab-qwen35-lora-sweep-$(date +%Y%m%d-%H%M).log"

echo "[deploy] launching via colab run (gpu=$GPU, timeout=${TIMEOUT}s)"
echo "[deploy] log: $LOG"

# Build script args (everything after the script path is forwarded as sys.argv).
SCRIPT_ARGS=(
  --stdout-artifact
  --candidate-n "$CANDIDATE_N"
  --max-new "$MAX_NEW"
  --eval-n "$EVAL_N"
  --micro-batch "$MICRO_BATCH"
  --target-updates "$TARGET_UPDATES"
  --max-attempts "$MAX_ATTEMPTS"
  --time-budget-s "$TIME_BUDGET_S"
)
if [[ -n "$LORA_VARIANTS" ]]; then
  SCRIPT_ARGS+=(--lora-variants "$LORA_VARIANTS")
fi

HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var first}" \
PYTHONUNBUFFERED=1 \
HF_XET_HIGH_PERFORMANCE=1 \
HF_HUB_DISABLE_PROGRESS_BARS=1 \
  colab run \
    --gpu "$GPU" \
    --timeout "$TIMEOUT" \
    scripts/run_qwen35_lora_sweep_colab.py \
    "${SCRIPT_ARGS[@]}" \
    2>&1 | tee "$LOG"

echo ""
echo "DONE. Log: $LOG"
echo "To decode artifacts from stdout base64:"
echo "  python scripts/decode_artifact.py '$LOG' results/9b-math500/artifact.tar.gz"
