# 9B MATH-500 Experiment Plan

This is the execution plan for the 9B rerun. The previous long runs are treated as
diagnostic only because the training reward was zero for every step. A valid run must
first prove that the online RL loop has reward signal.

## Goal

Evaluate whether per-group multiplicative weight modulation can elicit MATH-500
reasoning from a frozen 9B model, compared with a properly tuned LoRA baseline, under
a reproducible Colab workflow.

Default model:

```text
01-ai/Yi-1.5-9B-Chat
```

Default accelerator:

```text
colab --gpu G4
```

The current Colab CLI maps `G4` to:

```text
NVIDIA RTX PRO 6000 Blackwell Server Edition, 94.97 GiB
```

## Colab Runtime Rules

The local `colab` CLI only supports these GPU accelerator names:

```text
T4, L4, G4, A100, H100
```

`Pro 6000` is not a valid CLI value. Request `G4` and verify the real device with:

```python
import torch
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1024**3)
```

Long-running commands must set an explicit timeout:

```bash
colab exec -s <session> -f <script.py> --timeout 86400
```

The timeout above is only the local Jupyter execution wait budget. It prevents the
CLI from giving up while the remote kernel is still healthy; it does not override
Colab's resource manager.

The CLI starts a keep-alive daemon for `colab new`; its source pings once per minute
and caps keep-alive at 24h. External Colab guidance also recommends checkpointing
long jobs and splitting them into resumable segments. The experiment must therefore
be chunked. Do not rely on one monolithic session.

Operational rules:

- No concurrent `colab exec` against a running training session.
- Use `status` sparingly; avoid probing the kernel while a long cell runs.
- Every chunk must finish in less than 24h.
- Every chunk must write local files under `results/9b-math500/`.
- Every chunk must emit small critical artifacts to stdout.
- Every chunk must produce `/content/mapping-networks-9b-artifacts.tar.gz`.
- Download artifacts immediately after the chunk finishes.
- Keep each chunk below 6h wall time by default and below 12h as a hard ceiling.
- Save curves/checkpoints every 5-10 steps during probes and every 10-20 steps during
  full runs.

## Fixed Data Split

Dataset:

```text
HuggingFaceH4/MATH-500, split=test
```

Evaluation:

```text
eval_items = first 200 records
```

Training:

```text
train_pool = remaining records after evaluation split, filtered to level 3-5
train_selection = stride over the filtered pool
```

Seed:

```text
torch.manual_seed(0)
```

## Scoring

Use the repository scorer:

```text
src/math_scorer.py
```

Primary reward is binary final-answer correctness after boxed-answer extraction and
normalization.

Additional metrics required before training:

```text
boxed extraction rate
completion token length
long-output rate
stop-token rate
reward hit rate
group reward variance rate
```

Important definitions:

```text
boxed_rate = fraction of completions containing a real \boxed{...} answer
extract_rate = fraction where the scorer can extract any final answer, including fallback
long_output_rate = fraction that reached max_new without an EOS/chat stop token
```

The runner passes EOS and chat stop ids to generation and trims padding after the first
stop token before scoring, length accounting, and KL accounting.

## Phase 1: Baseline Eval

Purpose: measure headroom and verify formatting.

Configuration:

```text
n_eval = 200
max_new_eval = 512
eval_batch = 4
do_sample = False
```

Required outputs:

```text
baseline.json
cases/baseline.json
target_modules.json
run_config.json
```

Go/no-go:

- Stop if baseline accuracy is greater than 85%.
- Stop and fix prompt/scorer if boxed extraction rate is low.
- Continue if there is headroom and output format is stable.

## Phase 2: Active Prompt Bank Probe

Purpose: build the exact prompt bank that online GRPO will train on. A global
reward probe is not enough. GRPO needs non-identical rewards inside the same prompt
completion group; otherwise the advantage is zero and the update has no learning
signal.

Candidate grid:

```text
candidate_n = 50 initially, then 100-200 if stable
candidate pool = held-out training records, level 3-5
candidate selection = stride over sorted pool
probe_K = 8 first, 16 if active bank is too small
max_new = 512
temperature = 0.8
top_p = 0.95
```

For each candidate prompt, record:

```text
dataset_idx
bank_id
level
gold
reward vector
num_correct
format vector
boxed rate
stop-token vector
lengths
predictions
first decoded samples
```

Partition prompts into three pools:

```text
active_prompt_bank: 1 <= num_correct <= K-1
hard_pool: num_correct == 0
easy_pool: num_correct == K
```

Only `active_prompt_bank` may be used by the first online training gate. Hard/easy
prompts can be revisited later with curriculum or exploration, but they must not be
allowed to dominate the initial GRPO steps.

Go/no-go:

```text
active_prompt_bank >= 20 prompts for target_updates=20
boxed_rate >= 90%
long_output_rate < 10%
active_prompt_bank is saved to active_bank.json
```

If this fails, do not train. Tune prompt, K, max_new, temperature/top_p, or candidate
selection first.

## Phase 3: Reward Function And Selective Rollouts

Training uses two reward views:

```text
correct_reward = 1.0 if final answer is correct else 0.0
format_reward = 1.0 if a real \boxed{...} answer appears else 0.0
shaped_reward = correct_reward + 0.1 * format_reward - 0.2 * overlong
```

The primary metric remains `correct_reward`; shaped reward is only a bootstrap signal
for optimization. Reports must show both curves separately and must never present
format reward as math accuracy.

Selective rollout rule:

```text
if std(shaped_reward_group) == 0:
    skip optimizer step
    log skipped_zero_variance
else:
    compute group-normalized advantage from shaped_reward_group
    update adapter
```

Required per-step logs:

```text
variant
attempt
update
bank_id
dataset_idx
gold
correct_rewards
shaped_rewards
format_rewards
overlong
lengths
preds
correct_var
shaped_var
KL
step time
tokens
```

## Phase 4: Adapter Sanity And 20-Update Training Gate

Run only after active prompt bank is built.

Configuration:

```text
active_prompt_bank = Phase 2 output
B = 1
K = 8 initially
target_updates = 20
max_attempts = 60
max_new = 512
print_every = every attempt
no final eval
no large final checkpoint
small stdout artifact required
```

Variants:

```text
Map-G2048
LoRA-r8 lr=1e-4
```

Adapter sanity checks before training:

```text
adapter init is exact identity
small perturbation changes logits
restore returns logits to base
target module count is nonzero
```

20-update gate:

```text
at least 20 optimizer updates completed
zero-variance skip rate <= 50%
at least 10/20 updates have correct_reward variance
at least one update has correct_reward > 0
KL trailing mean < 0.05 for this probe
output long_output_rate < 10%
```

If the gate fails, do not run full training. Diagnose prompt bank selection, reward
shaping, or generation settings first.

## Phase 5: Full Training

Run only after Phase 4 passes.

Variants:

```text
Map-G2048
Map-G256 optional
LoRA-r8 lr in {1e-4, 3e-4, 1e-3}
```

Default configuration:

```text
B = 1
K = best K from probe
max_new = best max_new from probe
max_steps = 150 first, extend to 350 only if learning
beta_kl = 0.05
o_clamp = 0.10
print_every = 5
save_every = 10
```

Stop rules:

- Hard stop if reward is all zero through step 30.
- Hard stop if KL is above 0.3 and reward does not improve.
- Hard stop if generation length explodes.
- Early stop if trailing reward does not improve for 50 steps.

## Phase 6: Evaluation

For every completed variant:

```text
n_eval = 200
max_new_eval = 512
do_sample = False
same prompt
same scorer
same eval subset
Wilson 95% CI
```

Report:

```text
accuracy
Wilson CI
delta vs baseline
CI overlap / clears baseline
reward curve
KL curve
steps to target reward
mean step time
GPU hours
peak VRAM
decoded cases
failure cases
```

## Valid Positive Result Criteria

A positive claim requires all of:

```text
reward curve is not all zero
variant CI lower bound clears baseline CI upper bound
or multi-seed delta is at least 5pp
KL is nonzero but not overdriven
decoded outputs are not long-output artifacts
```

If reward remains zero, the result is invalid as training evidence regardless of eval
accuracy noise.

## Artifact Contract

Every chunk writes:

```text
results/9b-math500/run_config.json
results/9b-math500/baseline.json
results/9b-math500/target_modules.json
results/9b-math500/progress.jsonl
results/9b-math500/curves/*.csv
results/9b-math500/curves/*.json
results/9b-math500/cases/*.json
results/9b-math500/variant_summaries/*.json
results/9b-math500/checkpoints/*_final.pt
results/9b-math500/map_params/*_o.json
results/9b-math500/results.json
results/9b-math500/REPORT.md
```

Map chunks also print periodic stdout recovery blobs:

```text
MAP_PARAM_<variant>-stepXXXX_BEGIN
...
MAP_PARAM_<variant>-stepXXXX_END
```

Final Map chunks print:

```text
MAP_PARAM_<variant>_BEGIN
...
MAP_PARAM_<variant>_END
```

Small artifacts may be printed as base64:

```text
ARTIFACT_TAR_GZ_B64_BEGIN
...
ARTIFACT_TAR_GZ_B64_END
```

Large LoRA artifacts must be downloaded immediately after completion.
