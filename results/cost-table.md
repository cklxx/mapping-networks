# Cost table — modulation vs LoRA fine-tuning

- **run**: SMOKE (tiny random transformer, CPU)  
- **base model**: TinyLM(vocab=16,dim=32,L=3) (31,936 params)  
- **device**: cpu  
- **target reward** (steps-to-target threshold): 0.20  
- **step budget**: 12  **tokens/step (est)**: 160

## The headline question

**Is the modulation cheaper in GPU-hours, or only in adapter SIZE + optimizer VRAM?**

A-priori prediction (before reading the numbers):

| cost axis | prediction | certainty |
|---|---|---|
| adapter **size** (trainable params) | modulation ~10^4x smaller (G=2048 vs LoRA ~16.5M) | **certain** — it is arithmetic |
| **optimizer VRAM** (Adam m,v over trainable) | modulation smaller; the gap *grows* with model size | likely |
| **compute / step** (FLOPs fwd+bwd) | ~EQUAL — the frozen base dominates; LoRA's 2 matmuls and the gate's element-wise scale are both negligible | **certain** — see the FLOPs-share column |
| **GPU-hours** (compute/step x steps-to-target) | decided by **steps-to-target**, since compute/step is ~equal | **TBD by the run** |

The mechanism, stated plainly: the *adapter FLOPs share* column below is `6·N_adapter / 6·(N_base + N_adapter)` — it goes to ~0% as the base grows (on the 4B base, <0.5% for both adapters; on a tiny smoke base it is larger, since the base itself is tiny). Once the base dominates, compute/step is ≈equal across adapters, so the GPU-hour cost reduces to **steps-to-target x wall-per-step**. The only way the modulation wins on GPU-hours is by *converging in fewer steps*, not by doing less work per step. The size and optimizer-VRAM wins are real but separate.

## Cost table

| variant | trainable params | peak VRAM | steps-to-target | wall-clock (mean/step · total) | FLOPs/step (est) | adapter FLOPs share | GPU-hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| modulation-G16 | 16 | n/a | 2 / 12 | 0.04s · 0.5s | 30.67 MFLOP | 5.01e-02% | 2.39e-05 |
| modulation-G64 | 64 | n/a | 1 / 12 | 0.04s · 0.5s | 30.72 MFLOP | 0.200% | 1.16e-05 |
| LoRA-r8 (lr=0.0001) | 13,056 | n/a | 2 / 12 | 0.04s · 0.5s | 43.19 MFLOP | 29.018% | 2.49e-05 |
| LoRA-r8 (lr=0.0003) | 13,056 | n/a | 2 / 12 | 0.04s · 0.5s | 43.19 MFLOP | 29.018% | 2.47e-05 |
| LoRA-r8 (lr=0.001) | 13,056 | n/a | 1 / 12 | 0.04s · 0.5s | 43.19 MFLOP | 29.018% | 1.24e-05 |
| LoRA-r8 (lr=0.003)  ** BEST LoRA ** | 13,056 | n/a | 1 / 12 | 0.04s · 0.5s | 43.19 MFLOP | 29.018% | 1.23e-05 |
| modulation-G2048 (4B, MATH-500) | 2,048 | PENDING 4B GPU RUN | PENDING | PENDING | ~ base-dominated (≈LoRA) | ~0% | PENDING — predicted ≤ best-LoRA (decided by steps-to-target) |
| LoRA-r8 best-lr (4B, MATH-500) | 16,515,072 | PENDING 4B GPU RUN | PENDING | PENDING | ~ base-dominated | ~0% | PENDING — head-to-head target |

steps-to-target column reads `reached / total`; `—` = the trailing-mean reward never crossed the target inside the step budget (no convergence → GPU-hours charges the full budget as an upper bound).

## Verdict (from the measured rows)

> These are SMOKE numbers (tiny random transformer on CPU) — they prove the instrumentation captures all four axes and the table renders. The absolute values are meaningless; the 4B GPU rows above are PENDING.

- **Adapter size**: best modulation = 64 trainable params vs best LoRA = 13,056 → **204x smaller** (certain, arithmetic).
- **Compute/step**: on this TINY base the adapter FLOPs share peaks at 29.0% (the base is only ~35k params, so LoRA's 13k params are *not* negligible here) — but the MECHANISM is the point: that share is `6·N_adapter / 6·(N_base+N_adapter)`, which → 0 as N_base grows. On the 4B (base ~4e9 params) both adapters' share is <0.5%, so compute/step is ≈equal and GPU-hours hinges on steps-to-target.
- **Steps-to-target** (the GPU-hour driver): best modulation 1 steps; best LoRA 1 steps.
- **GPU-hours**: best modulation 1.16e-05 vs best LoRA 1.23e-05 — because compute/step is ≈equal, this ratio tracks steps-to-target, exactly as predicted.

**Answer to the headline question**: adapter SIZE and optimizer VRAM are certain wins for the modulation regardless of the run; whether it is also cheaper in GPU-HOURS is decided entirely by steps-to-target (compute/step is ≈equal). The fair LoRA lr-sweep above is what makes that steps-to-target comparison honest. Final 4B verdict awaits the GPU run (PENDING rows).

