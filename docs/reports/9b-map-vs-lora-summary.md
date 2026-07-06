# 9B Map vs LoRA：模型参数、实验设计与优化方案

## 模型与硬件

| 项目 | 值 |
|---|---|
| Base model | `01-ai/Yi-1.5-9B-Chat` |
| Architecture | LLaMA-style causal LM |
| Base params | 8,829,407,232 |
| dtype | bfloat16 |
| layers | 48 |
| hidden_size | 4096 |
| intermediate_size | 11008 |
| attention heads | 32 |
| KV heads | 4 |
| vocab_size | 64000 |
| max_position_embeddings | 4096 |
| rope_theta | 5,000,000 |
| GPU | Colab G4 / RTX PRO 6000 Blackwell / 94.97GB |

## Adapter 定义

| 项目 | Map-G2048 | LoRA-r8 |
|---|---:|---:|
| Target modules | 336 linears: 48 layers x q/k/v/o/gate/up/down proj | same |
| Formula | `W'[c,:] = W[c,:] * (1 + o[group(c)])` | `W'x = Wx + scale * B(Ax)` |
| Trainable params | 2,048 | 27,230,208 |
| Checkpoint estimate | 4KB | 51.9MB |
| Param ratio | 1 / 13,296 of LoRA | 1x |

## Code Optimizations

1. Map forward uses activation-side scaling instead of materializing `W * gate[:, None]`:

```python
out = F.linear(x, W, None)
out = out * gate
if bias is not None:
    out = out + bias
```

2. GRPO logprob/KL is batched across surviving B*K completions:

```python
logps, kls = batched_logp_and_kl(...)
loss = (-(adv * logps) + beta_kl * kls).mean()
loss.backward()
```

3. Active bank and selective rollout:

- Probe candidate prompts first.
- Keep only prompts with `1 <= num_correct <= K-1`.
- Skip groups with zero shaped-reward variance.
- Reuse active bank with `--active-bank-json`.

## Experiment Design

| Stage | Config |
|---|---|
| Dataset | `HuggingFaceH4/MATH-500`, split=test |
| Eval split | first 200 records |
| Candidate pool | remaining records, level 3-5 |
| Active bank | candidate_n=50, probe_K=8, max_new=512, temp=0.8, top_p=0.95 |
| Selection reward | correct final answer after boxed extraction |
| Training reward | `correct + 0.1 * format - 0.2 * overlong` |
| Primary metric | correct reward and greedy eval accuracy |
| Training | target_updates=30, K=8, max_new=512, beta_kl=0.05 |
| Eval | eval_n=200, greedy, max_new_eval=512, Wilson CI |
| Attention | HF sdpa |

## Results

### Active Bank

| metric | value |
|---|---:|
| candidate_n | 50 |
| active_n | 29 |
| sample_correct_rate | 0.3175 |
| boxed_rate | 0.9075 |
| stopped_rate | 0.9400 |
| long_output_rate | 0.0600 |
| variance_prompt_rate | 0.5800 |

### Map vs LoRA

| metric | Baseline | Map-G2048 | LoRA-r8 | conclusion |
|---|---:|---:|---:|---|
| eval accuracy | 0.4050 | 0.4800 | 0.4200 | Map +6pp vs LoRA |
| correct / n | 81/200 | 96/200 | 84/200 | - |
| Wilson CI | [0.3394, 0.4742] | [0.4118, 0.5490] | [0.3537, 0.4893] | overlap |
| updates / skipped | - | 30 / 5 | 30 / 5 | same |
| train elapsed | - | 1358.7s | 1372.0s | Map slightly faster |
| mean step | - | 10.04s | 10.42s | Map 0.96x LoRA |
| tokens/s | - | 278.4 | 247.5 | Map 1.12x LoRA |
| trainable params | - | 2,048 | 27,230,208 | Map 13,296x fewer |
| checkpoint | - | 4KB | 51.9MB | Map 13,296x smaller |
| best train correct mean | - | 0.875 | 1.000 | LoRA peak higher |
| final train correct mean | - | 0.500 | 0.125 | Map final more stable |

## Why End-to-End Speed Is Not 13,296x

Map reduces trainable parameters, optimizer state, checkpoint size, and adapter state communication by 13,296x. It does not remove the shared 9B base model cost: rollout generation, policy forward/backward, KL reference forward, and eval generation. Therefore current HF eager end-to-end throughput gain is 1.12x, not tens of times.

## Extreme Performance Plan

| Layer | Plan | Expected Gain | Algorithm Change |
|---|---|---|---|
| Done | activation-side Map + B*K batched logprob/backward | removes obvious overhead | no |
| Minimal backward | `beta_kl=0` or intermittent KL | removes one 9B no-grad forward per update | mild constraint change |
| Rollout | vLLM/SGLang generation service | large rollout speedup | no |
| Batch | search `train_batch` vs `max_new`; 512 needs batch=1 on G4 | better GPU utilization | no |
| Kernel | fused output-scale or `torch.compile` | small Map-only gain | no |
| Schedule | fixed active bank + zero-variance skip | fewer wasted updates | no |

## Judgement

Current evidence supports: Map has clear engineering advantages and beats LoRA in one 30-update / eval200 run.

Current evidence does not prove: Map is statistically significantly better at final convergence. CI overlaps and only one seed was run.

## Next Training

1. Run at least 3 seeds.
2. Run update budgets 50 and 100.
3. Sweep Map: `G={2048,8192}`, `lr_o={0.003,0.005,0.01}`.
4. Sweep LoRA: `rank=8`, `lr={1e-4,3e-4}`.
5. Run fixed wall-clock comparison, e.g. 60min per variant.
6. If wall-clock is the target metric, integrate vLLM/SGLang rollout before hand-writing more Map kernels.

## Code

- `src/adapters.py`: Map/LoRA adapter; Map activation-side scaling.
- `experiments/math500_active_grpo_9b.py`: active bank, selective rollout, B*K batched logprob/backward, eval.
- `src/generation_utils.py`: stop token and completion trim.
- `docs/9b-experiment-plan.md`: full protocol.
