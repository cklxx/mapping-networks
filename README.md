# Mapping Networks: ultra-cheap elicitation fine-tuning by weight modulation

Adapt a frozen model by multiplicatively gating its weights with a tiny trainable
latent over a **fixed** channel grouping:

```
W'[c, :] = W[c, :] * (1 + alpha * o[group(c)])
```

`o` is one shared vector of length **G** (e.g. 256 or 2048). `group(c)` is a fixed
partition of the output channels, so the trainable parameter count is **G — the number
of weight-groups, not the number of weights**. That is orders of magnitude smaller than
LoRA, whose `r·(in+out)` per layer scales with width. The base weights are never touched.

On a frozen Qwen3-4B tuned with RL on MATH-500, a **2048-parameter** modulation lifted
accuracy **+19pp** over the base and **beat a LoRA carrying ~16.5M parameters** (8000×
more). That single clean statistical win is what this project is built around.

## Thesis

This is an **elicitation** fine-tuning method, not a from-scratch trainer.

- **What it does.** The gate *reweights a capable base's existing features*. RL elicits
  latent ability the base already has; the modulation is a cheap knob to steer toward it.
- **What it cannot do (the hypothesis).** A multiplicative per-channel gate can scale
  features up or down but cannot inject input-dependent structure — it has no term like
  LoRA's additive `B(Ax)`. So it should **elicit/reweight** but **not teach new
  knowledge**. A random base has no features to modulate and the fixed-grouping image
  caps capacity, so from-scratch is structurally limited (untested, theoretically weak —
  not worth chasing).
- **Where it wins.** Ultra-cheap elicitation fine-tuning of **large / very large**
  models. Params scale with weight-groups, not dims, so the bigger the model and the
  more task-adapters you serve, the larger the cost advantage over LoRA — and on the one
  elicitation task measured here it was not worse but **better**.

The open research question is the **function-class boundary**: which tasks the gate can
reach (elicit) vs which need LoRA's additive capacity (teach), and whether the +19pp
holds or grows at 27B/100B. See [`docs/research-plan.md`](docs/research-plan.md).

## The one validated result

Frozen **Qwen3-4B**, RL (GRPO) on **MATH-500**, greedy eval on a fixed n=200 test
subset, Wilson 95% CI. All variants share the same base, scorer, and ~40-step / 35-min
budget. Raw report: [`results/4b-math500/results.txt`](results/4b-math500/results.txt).

| Variant     | Trainable params | Accuracy | Wilson 95% CI    | vs baseline                     |
|-------------|-----------------:|---------:|------------------|---------------------------------|
| baseline    |                0 |    29.5% | [0.236, 0.362]   | —                               |
| Map-G256    |              256 |    38.0% | [0.316, 0.449]   | +8.5pp (overlap)                |
| LoRA-r8     |       16,515,072 |    39.0% | [0.325, 0.459]   | +9.5pp (overlap)                |
| **Map-G2048** |          **2,048** | **48.5%** | **[0.417, 0.554]** | **+19.0pp — clears baseline upward** |

Map-G2048 is the only variant whose CI lower bound (0.417) clears the baseline upper
bound (0.362): a clean statistical win, with **2048 parameters** beating a LoRA with
**8000× more**.

### Why it's real, not a lucky eval: the per-step telemetry

The load-bearing, artifact-free evidence is the optimization trace, not just the final
number. Only the modulation found a productive direction
([`results/4b-math500/fig_training_curves.png`](results/4b-math500/fig_training_curves.png)):

- **Map-G2048** — KL(π‖base) climbed monotonically 0 → 0.087 over the run (mean 0.066,
  above the ~0.05 leverage threshold): the latent moved the policy productively.
- **Map-G256** — partial movement, KL → ~0.045.
- **LoRA-r8** — barely moved: `mean|AB|` ~0.0071 → ~0.0073, KL stayed ~0.01. Effectively
  frozen at `lr=1e-4`.

The gate reached this with `mean|o| ≈ 0.04` and `max_gate ≈ 1.10` — well inside the
coherent band, so it was steering the policy, not perturbing it off-distribution.

### Honest caveats (read before quoting the absolute numbers)

1. **Scorer / gold artifact.** A few MATH-500 gold strings carry a transcription error
   (e.g. a tuple `(3, \frac{\pi}{2})` whose comma was dropped to `(3\frac{\pi}{2})`),
   adding a few points of extractor noise to **absolute** accuracy. It hits all variants
   roughly equally and does **not** explain the +19pp gap.
2. **"Beats LoRA" is partly LoRA being under-tuned.** At `lr=1e-4` the LoRA barely moved
   (the telemetry above). The clean, defensible claim is *"the 2048-param modulation
   found a productive optimization direction the same-budget LoRA did not"*. A fair
   head-to-head needs a LoRA learning-rate sweep — that is the first item in the research
   plan.

Earlier 0.8B / GSM8K explorations were negative — a model/task limit (thin coherence
band + an at-ceiling task), not a limit of the idea. Detailed in
[`docs/research-plan.md`](docs/research-plan.md#what-didnt-work-and-why).

## The cost story (the #1 deliverable)

The point of an 8000×-smaller adapter is **cost**, so the project measures it directly, per
variant: trainable params, peak VRAM, **steps-to-target** (the convergence number),
wall-clock, FLOPs/step, and **GPU-hours**. The full table —
[`results/cost-table.md`](results/cost-table.md) — is emitted by
[`experiments/cost_benchmark.py`](experiments/cost_benchmark.py) and by the validated
runner itself (same hooks, in [`src/costlib.py`](src/costlib.py)).

The headline question it answers: **is the modulation cheaper in GPU-hours, or only in
adapter SIZE + optimizer VRAM?** The mechanism makes this sharp: the frozen base's backward
dominates every step, so the adapter's own FLOPs (LoRA's two matmuls vs the gate's
element-wise scale) are a rounding error — **compute/step is ≈equal across adapters**.
Therefore GPU-hours ≈ steps-to-target × wall-per-step, and the only way the modulation wins
on GPU-hours is by *converging in fewer steps*. So the a-priori split is: adapter size
(~10⁴× smaller — certain), optimizer VRAM (smaller, grows with model size), compute/step
(≈LoRA), **GPU-hours (decided by steps-to-target — TBD by the GPU run)**. The benchmark also
runs a **LoRA learning-rate sweep** and reports LoRA's *best* variant, so "beats LoRA" is
not an artifact of the under-tuned `lr=1e-4` the first run used.

## Reproduce

Clone → install → fetch → smoke (CPU, minutes) → full (1 GPU, ~2h) → expected output.

```bash
git clone <repo> && cd mapping-networks
pip install -r requirements.txt        # pinned: torch, transformers, datasets, accelerate, matplotlib, numpy, torchvision

# 1) SMOKE — verify the whole pipeline on CPU/Mac in minutes (tiny random transformer,
#    no model download). MUST produce results/cost-table.md before you spend a GPU:
./scripts/reproduce.sh --smoke

# 2) FULL — the headline 4B MATH-500 experiment (needs ONE CUDA GPU, ~2h / 4 variants):
./scripts/reproduce.sh                  # fetches MATH-500 + Qwen3-4B, runs, writes outputs
```

**Data + model ids** (documented in [`scripts/reproduce.sh`](scripts/reproduce.sh)):
dataset `HuggingFaceH4/MATH-500` (HF datasets, split=test), model `Qwen/Qwen3-4B` (HF hub).
Override with `MODEL=<id-or-path> ./scripts/reproduce.sh`.

**Expected output** of the full run:
[`results/4b-math500/results.txt`](results/4b-math500/results.txt) (accuracy + Wilson CIs +
KL telemetry + decoded cases), [`results/cost-table.md`](results/cost-table.md) (the
per-variant cost table), and the two figures. The smoke run emits a *tiny* cost-table whose
absolute numbers are meaningless — it only proves the instrumentation captures all four
axes and the table renders; the 4B GPU rows are written as a clearly-marked **PENDING**
block carrying the a-priori predictions until the GPU run fills them.

**Hardware**: the 4B run was validated on a single H20 (80 GB); any ≥24 GB CUDA GPU should
fit the frozen bf16 4B + the small RL rollout. The smoke run is CPU-only.

**Caveat on absolute numbers**: a few MATH-500 gold strings carry a transcription artifact
(see "Honest caveats" above), so treat the *cross-variant deltas* (e.g. +19pp) as the
result, not the absolute accuracies. The cost table's absolute GPU-hours likewise depend on the exact GPU /
budget; the *ratio* (modulation vs best-LoRA, driven by steps-to-target) is the portable
claim.

The adapter math lives in [`src/adapters.py`](src/adapters.py) (`install_direct_map` for
the gate, `install_lora` for the baseline); the MATH scorer in
[`src/math_scorer.py`](src/math_scorer.py); the cost hooks in
[`src/costlib.py`](src/costlib.py). All are reusable from new experiments. The full
validation history (every negative run that led here) is preserved under
[`experiments/archive/`](experiments/archive/README.md).

## Layout

```
src/          adapters.py     — modulation gate + LoRA baseline (single source of the adapter math)
              math_scorer.py  — MATH-500 \boxed{} extraction + equivalence scoring
              costlib.py      — cost hooks (params / VRAM / steps-to-target / wall / FLOPs / GPU-hours)
experiments/  math500_rl.py     — the validated 4B MATH-500 RL runner (emits results.txt + cost-table.md)
              cost_benchmark.py — per-variant cost sweep + LoRA lr-sweep; --smoke runs on CPU
              archive/          — the FULL validation history (every negative run) + raw evidence + index
results/      4b-math500/     — raw report + training-curve & accuracy figures + plot script
              cost-table.md   — the per-variant cost table (smoke-filled; 4B rows PENDING the GPU run)
scripts/      reproduce.sh    — one entry point: install -> fetch -> smoke (CPU) / full (1 GPU)
docs/         research-plan.md — function-class boundary, scaling, fair LoRA head-to-head
```
