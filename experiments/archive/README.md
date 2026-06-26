# Archived validation history — the full evidence trail

Every exploration that led to the one validated result (frozen Qwen3-4B, MATH-500, a
2048-param modulation +19pp over baseline, beating a 16.5M-param LoRA — see
[`../../results/4b-math500/results.txt`](../../results/4b-math500/results.txt)). Most of
these runs are **negative**; they are preserved deliberately, because the README's "what
didn't work, and why" paragraph rests on them. A negative is a *case*, not a dead end —
each one ruled out a confound (wrong adapter design, wrong base size, wrong task) and
pointed at the next variable to isolate.

Light refactor only: hardcoded pod/absolute paths are now CLI flags (`--model-path`,
`--results-path`, `--device`), and where an archived script's adapter/scorer is
byte-identical to the productionized [`../../src/`](../../src/) version it now **imports**
from `src/` instead of carrying a duplicate. Run logic and every run-defining
hyperparameter are preserved. The `*_results.txt` / `*.log` files are the raw evidence,
copied verbatim.

## The arc (forward order: each row isolates ONE variable the previous row confounded)

| # | script | base / task | adapter | result | what it ruled out |
|---|--------|-------------|---------|--------|-------------------|
| 1 | [`phase1_fmnist.py`](phase1_fmnist.py) | from-scratch CNN / FashionMNIST | fixed random map of a latent `z` (Li-et-al-style) | **negative**: gap-suppression confirmed (every mapping gap < baseline; the 27M HyperNet control overfit worst) but test acc plateaued ~86% (d=4096), ~3pp **under** baseline 88.8% — never reached parity | from-scratch generation is capacity-limited with no features to modulate → this is an **elicitation** method, not a from-scratch trainer |
| 2 | [`phase2_gsm8k_mps.py`](phase2_gsm8k_mps.py) | Qwen3.5-0.8B / GSM8K (MPS) | **frozen random map** + tanh, α=0.1, 256-param `z` | **negative / inconclusive**: 0.417→0.350, within noise; underpowered (4-9 steps, n=60) | nothing yet — but the diagnostic showed the frozen-map+tanh+α=0.1 dampened the latent's KL leverage to ~0.002 |
| 3 | [`phase2_gsm8k_pod.py`](phase2_gsm8k_pod.py) | Qwen3.5-0.8B / GSM8K (H20) | same frozen random map (+ KL leash) | **negative**: Mapping 0.405 vs base 0.420 vs LoRA 0.440, **all CIs overlap** (n=200, properly powered) | confirms the *frozen-random-map adapter* is the problem (near-zero leverage), not power → redesign the adapter |
| 4 | [`phase2b.py`](phase2b.py) | Qwen3.5-0.8B / GSM8K (H20) | **DIRECT** gate `W'=W·(1+o[group(c)])`, α=1.0, o init 0, hard clamp | **negative**: now real leverage (KL 0.002→0.34/0.71) but acc fell **disjoint-below** baseline (0.42→0.245/0.235) | the gate *can* move the policy, but on this base/task it moves in a non-productive direction → is it over-driving, or no productive direction at all? |
| 5 | [`phase2b_coherent_probe.py`](phase2b_coherent_probe.py) | (re-uses #4's learned `o`) | scale sweep on the RL-learned direction | **negative, confound closed**: acc declines **monotonically** from identity outward, including fully-coherent low-`\|o\|` points | it is **not** an over-driven-end-state artifact — a genuine capacity/function-class limit *for 0.8B/GSM8K* |
| 6 | [`phase2_4b_gsm8k.py`](phase2_4b_gsm8k.py) | **Qwen3-4B** / GSM8K (H20) | DIRECT gate + LoRA-r8 (isolates base size) | **negative**: 4B is **at-ceiling** on GSM8K (~86.5% baseline) → zero RL advantage variance | GSM8K gives the 4B no headroom → the comparison is meaningless on GSM8K → pivot the **task** |
| 7 | [`phase2_4b_math.py`](phase2_4b_math.py) | Qwen3-4B / **MATH-500** (H20) | DIRECT gate (G∈{256,2048}) + LoRA-r8 | **POSITIVE — the validated result**: Map-G2048 **+19pp**, CI clears baseline upward, beats the 16.5M-param LoRA | both confounds removed (real headroom + wider coherent band) → the idea works on an elicit task |

The productionized, actively-maintained version of row 7 is
[`../math500_rl.py`](../math500_rl.py) (cleaner CLI, cost-instrumentation hooks). Row 7's
archived copy here is the exact provenance of the headline number.

## Why the negatives matter (the README's "what didn't work" rests on these)

- **Rows 2→3→4** trace the *adapter-design* dead end: a frozen random projection + tanh +
  small α has no leverage; the fix was the direct per-channel gate, not more compute.
- **Rows 4→5** are the **case-as-fact** discipline: a −18pp regression was not generalized
  into "the gate can't do math"; the coherent probe decoded *why* (monotone decline along
  the learned ray), proving it was a 0.8B/GSM8K function-class limit, not the idea's limit.
- **Rows 6→7** isolate base-size then task: only after both confounds (at-ceiling task,
  thin coherent band) were removed did the result flip to the clean +19pp win.

## Evidence files (raw, verbatim)

| file | the run it records |
|------|--------------------|
| [`phase1_results.txt`](phase1_results.txt) | row 1 — FashionMNIST compression/accuracy table |
| [`phase2_results.txt`](phase2_results.txt) | row 2 — MPS frozen-map null (n=60) |
| [`phase2_pod_results.txt`](phase2_pod_results.txt) | row 3 — H20 frozen-map null (n=200, Wilson CI) |
| [`phase2_run.log`](phase2_run.log) | row 2 — raw MPS training log |
| [`phase2b_results.txt`](phase2b_results.txt) | row 4 — direct gate, disjoint-below (n=200) |
| [`phase2b_coherent_probe.txt`](phase2b_coherent_probe.txt) | row 5 — monotone acc-vs-`o`-scale sweep |
| [`phase2_4b_math_results.txt`](phase2_4b_math_results.txt) | row 7 — **the validated +19pp** (same as `../../results/4b-math500/results.txt`) |

(No raw `*_results.txt` is checked in for row 6's GSM8K-at-ceiling 4B run; its negative
verdict is summarized in the table above and in [`../../docs/research-plan.md`](../../docs/research-plan.md).)

## Running an archived script

Each takes `--model-path` (local dir / HF id / snapshot glob), `--results-path`, and
`--device` (auto-detects cuda/mps/cpu). Example:

```bash
python experiments/archive/phase2_4b_math.py \
    --model-path Qwen/Qwen3-4B --results-path /tmp/phase2_4b_math.txt
python experiments/archive/phase1_fmnist.py --data-dir ~/data   # CPU/MPS, no flags needed
```

These reproduce historical negatives; the active experiment is
[`../math500_rl.py`](../math500_rl.py) (and its smoke-verifiable cost benchmark
[`../cost_benchmark.py`](../cost_benchmark.py)).
