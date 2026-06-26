# Research plan

One validated result anchors this project: on a frozen Qwen3-4B, a 2048-parameter
multiplicative weight gate lifted MATH-500 accuracy +19pp (CI clears baseline upward)
and beat a 16.5M-param LoRA at a matched RL budget. Everything below is designed to
either harden that result or find its edge — not to add scope for its own sake.

The three open questions, in priority order:

1. **Function-class boundary** — what can the gate reach, and what can it not?
2. **Fair head-to-head** — is "beats LoRA" real, or just an under-tuned LoRA?
3. **Scaling** — does the +19pp hold or grow at 27B / 100B?

---

## 1. The function-class boundary (the core question)

**Hypothesis.** A per-channel multiplicative gate `W'[c,:] = W[c,:]·(1 + α·o[group(c)])`
can only **rescale** a model's existing feature directions. It has no input-dependent
term — nothing like LoRA's additive `B(Ax)` that can write a *new* response as a function
of the input. Prediction:

- **ELICIT tasks** (the answer is a capability the base already has but under-expresses —
  e.g. "show the working", "use the latent math skill"): the gate **wins**, cheaply.
- **TEACH tasks** (the answer requires knowledge or a mapping the base does not contain —
  e.g. a private fact, a new label, a format the base never produces): the gate **loses**;
  LoRA's additive capacity is required.

MATH-500 on a 4B is an elicit task (the base solves 29.5% unaided — the skill is latent),
which is exactly why the gate worked. To draw the boundary we need a controlled pair on
the *same base* so the only variable is elicit-vs-teach.

**Probe design (2-3 tasks, same frozen base, same RL/SFT budget, same eval protocol):**

- **Probe A — ELICIT (predict: gate wins).** A reasoning task the base can do but
  under-expresses. Candidates: MATH-500 (already validated — keep as the positive
  anchor), or GSM8K *on a base with real headroom* (a 4B, not the at-ceiling 0.8B), or a
  "always show step-by-step then box the answer" format-elicitation task. Signal of a win:
  CI clears baseline upward at G≈2048, KL rises into the 0.05-0.10 band.
- **Probe B — TEACH new knowledge (predict: gate loses, LoRA wins).** A task whose
  answer is *not* recoverable by reweighting existing features:
  - a **synthetic key→value recall** task (memorize N random (key, value) pairs the base
    has never seen) — pure new knowledge, zero latent prior; or
  - a **new-symbol classification** task (map inputs to labels under a freshly-defined
    rule). Signal: LoRA learns it (accuracy climbs), the gate plateaus near chance even as
    KL rises — i.e. it *moves* the policy but in no productive direction (the exact
    signature already seen in the 0.8B/GSM8K probe below).
- **Probe C (optional) — boundary case.** A task that is *mostly* elicit with a small
  teach component (e.g. a domain QA where the base knows the domain but not the required
  output schema). Tells us whether the gate degrades gracefully or hits a hard wall.

**What a clean result looks like:** a 2×2 of {gate, LoRA} × {elicit, teach} where the gate
matches or beats LoRA on elicit at ~10⁴× fewer params and clearly trails it on teach. That
characterizes the boundary and turns "it worked once" into "here is the function class it
covers."

**Mechanism check (cheap, do alongside the probes):** for each probe, log the per-step
KL(π‖base) and `mean|o|`. The 0.8B/GSM8K probe already showed the diagnostic — the gate
can have real KL leverage (0.002 → 0.71) yet move along a *non-productive* ray. On a teach
task we expect the same: KL rises, accuracy does not. Decode actual generations per case;
do not trust the aggregate alone.

---

## 2. Fair LoRA head-to-head (de-confound "beats LoRA")

The validated run used `LoRA lr=1e-4`, and the telemetry shows the LoRA barely moved
(`mean|AB|` 0.0071→0.0073, KL ~0.01). So the head-to-head is currently confounded: we
showed the gate found a productive direction the same-budget LoRA did not, **not** that
the gate beats a well-tuned LoRA.

**Plan.** Sweep LoRA `lr ∈ {1e-4, 3e-4, 1e-3, 3e-3}` (and optionally rank `r ∈ {4, 8, 16}`)
on the identical MATH-500 / 4B setup, same step and wall budget, same scorer. Report the
**best** LoRA against Map-G2048. Decision:

- If best-LoRA still trails Map-G2048 → the "cheaper *and* better on this elicit task"
  claim survives, now properly.
- If best-LoRA matches/exceeds Map-G2048 → the claim narrows to **"matches LoRA at
  ~10⁴× fewer params on an elicit task"** — still the load-bearing cost story, just not a
  quality win. Either outcome is publishable; we just need to know which.

Also fix the scorer/gold artifact (de-noise the few mis-transcribed MATH-500 golds) so the
**absolute** numbers are clean — the deltas already survive it, but the absolutes should be
honest.

---

## 3. Scaling (does the advantage hold at 27B / 100B?)

The cost argument *grows* with model size: gate params depend on the group count G, not on
width, so a fixed G=2048 is an ever-shrinking fraction of the weights as the model grows,
while LoRA's `r·(in+out)` per layer grows with width. The thesis predicts the relative cost
advantage — and plausibly the elicitation headroom — increases with scale.

**Plan.** Re-run the validated MATH-500 / elicit setup at **27B** first (one step up, same
adapter, same G-sweep, same budget). Measure:

- does Map-G2048 still clear baseline upward, and by how much (≥ +19pp = holds/grows);
- the param ratio gate:LoRA at 27B (it should widen vs the 4B's 8000×);
- whether a larger base needs a different coherent band (`O_CLAMP`) — log `mean|o|` and
  `max_gate` and re-derive the band if the 4B's ±0.10 clamp over- or under-drives the 27B.

100B is the same experiment one rung higher; gate it on the 27B result clearing the bar
before spending the compute. Multi-seed (≥5) any sub-5pp comparison and report mean ± σ +
Wilson CI before claiming it — a single best-checkpoint pick is a positively-biased
estimator.

---

## What didn't work, and why

Carried forward as a one-paragraph record; the code is **not** migrated (it explored a
dead model/task combination, not a flaw in the idea).

- **From-scratch FashionMNIST (Phase 1).** A fixed-random mapping training only a latent
  `z` (Li et al. 2018 intrinsic-dimension style) suppressed overfitting vs the full net
  but **plateaued ~3pp under baseline accuracy** (86% at d=4096 vs 88.8%) and never
  reached parity. Lesson: with **no pretrained features to modulate**, the fixed-mapping
  image is capacity-limited — consistent with the thesis that this is an *elicitation*
  method, not a from-scratch trainer. Do not chase from-scratch.
- **0.8B / GSM8K (Phase 2 / 2b).** The direct gate had **real RL leverage** (KL rose
  0.002 → 0.34 at G=256, → 0.71 at G=2048) yet accuracy fell **disjoint-below** baseline
  (0.42 → 0.245 / 0.235). A coherence probe along the learned direction showed accuracy
  declining **monotonically** from identity outward at *every* scale, including
  fully-coherent low-`|o|` points — so it was not an over-driven end-state artifact. Two
  confounds made 0.8B/GSM8K the wrong testbed: (i) GSM8K is **at-ceiling** for the model
  (baseline 42%, little RL headroom — no elicitation to do), and (ii) the 0.8B's coherent
  band is very thin (`mean|o|` ≳ 0.05 already breaks reasoning), so the usable steering
  range is tiny. The 4B + MATH-500 pivot removed both confounds (real headroom, wider
  band) and the result flipped to the clean +19pp win. The negative was a **model/task
  limit, not the idea's limit** — but note the 0.8B/GSM8K signature (KL rises, accuracy
  doesn't) is exactly what a *teach* task should look like, which is why Probe B above
  reuses it deliberately as the predicted-loss arm.
