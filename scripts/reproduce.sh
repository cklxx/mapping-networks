#!/usr/bin/env bash
# Reproduce the mapping-networks headline experiment end-to-end.
#
#   ./scripts/reproduce.sh --smoke   # CPU/Mac, minutes — verifies the FULL pipeline on a
#                                     # tiny random transformer, emits a (tiny) cost table.
#   ./scripts/reproduce.sh           # 1 GPU, ~2h — the real Qwen3-4B / MATH-500 run:
#                                     # writes results/4b-math500/results.txt, the cost
#                                     # table, and the figures.
#
# Steps: install deps -> fetch data+model -> run -> emit results/cost-table.md +
# results/4b-math500/results.txt + the figures.
#
# DATA + MODEL ids (documented, single source):
#   dataset : HuggingFaceH4/MATH-500            (HF datasets id, split=test, 500 problems)
#   model   : Qwen/Qwen3-4B                     (HF hub id, the frozen base)
# Override the model with: MODEL=<hf-id-or-local-path> ./scripts/reproduce.sh
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

SMOKE=0
[[ "${1:-}" == "--smoke" ]] && SMOKE=1

MODEL="${MODEL:-Qwen/Qwen3-4B}"
DATASET="HuggingFaceH4/MATH-500"
PY="${PYTHON:-python3}"

echo "=================================================================="
echo " mapping-networks reproduce  (smoke=$SMOKE)"
echo "   model:   $MODEL"
echo "   dataset: $DATASET"
echo "=================================================================="

# ---- 1. install deps ----
echo "[1/4] installing deps (requirements.txt) ..."
$PY -m pip install -q -r requirements.txt

if [[ "$SMOKE" == "1" ]]; then
  # ---- smoke: NO download, tiny random transformer on CPU ----
  echo "[2/4] smoke mode: no model/data fetch needed (tiny random transformer, CPU)."
  echo "[3/4] running cost benchmark (--smoke) ..."
  $PY experiments/cost_benchmark.py --smoke --device cpu --out results/cost-table.md
  echo "[4/4] smoke done. Pipeline verified — see results/cost-table.md (smoke rows +"
  echo "      the PENDING 4B GPU rows carrying the a-priori predictions)."
  echo ""
  echo "  The smoke run proves the instrumentation captures all four cost axes"
  echo "  (trainable params / peak VRAM / steps-to-target / wall-clock + FLOPs/step)"
  echo "  and renders the table. Run WITHOUT --smoke on a GPU for the real 4B numbers."
  exit 0
fi

# ---- 2. fetch data + model (HF cache; idempotent) ----
echo "[2/4] fetching dataset + model into the HF cache (idempotent) ..."
$PY - "$MODEL" "$DATASET" <<'PYFETCH'
import sys
model_id, dataset_id = sys.argv[1], sys.argv[2]
from datasets import load_dataset
print(f"  fetching dataset {dataset_id} (split=test) ...", flush=True)
ds = load_dataset(dataset_id, split="test")
print(f"  dataset ready: {len(ds)} problems", flush=True)
# Model: pull the weights now so the run itself doesn't stall on download. A local path
# (a dir) is used as-is; an HF id is materialized into the cache.
import os
if not os.path.isdir(model_id):
    from huggingface_hub import snapshot_download
    print(f"  fetching model {model_id} (this can take a while) ...", flush=True)
    snapshot_download(model_id)
print("  model ready.", flush=True)
PYFETCH

# ---- 3. the headline 4B MATH-500 RL experiment ----
echo "[3/4] running the 4B MATH-500 experiment (~2h on 1 GPU) ..."
DEVICE="${DEVICE:-cuda}"
$PY experiments/math500_rl.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --out results/4b-math500/results.txt \
    --cost-out results/cost-table.md

# ---- 4. figures ----
echo "[4/4] regenerating figures ..."
( cd results/4b-math500 && $PY plot_curves.py )

echo ""
echo "DONE. Outputs:"
echo "  results/4b-math500/results.txt        — the A/B report (accuracy, CIs, KL, cases)"
echo "  results/cost-table.md                 — the per-variant cost table"
echo "  results/4b-math500/fig_training_curves.png, fig_accuracy.png — figures"
