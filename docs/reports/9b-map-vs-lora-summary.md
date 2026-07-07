# 9B Map vs LoRA：实验设计、结果、性能对比

## 实验设计

| 项 | 配置 |
|---|---|
| Base model | `01-ai/Yi-1.5-9B-Chat`; 8,829,407,232 params; 48 layers; hidden 4096; FFN 11008; 32 heads; 4 KV heads; bf16 |
| 任务 | MATH-500；eval 使用前 200 条；训练候选使用剩余样本中 level 3-5 |
| Active bank | candidate_n=50；probe_K=8；max_new=512；active_n=29；boxed_rate=0.9075；variance_prompt_rate=0.58 |
| Reward | 训练用 shaped reward = correct + 0.1 * format - 0.2 * overlong；最终只看 correct/eval accuracy |
| 训练 | target_updates=30；K=8；max_new=512；beta_kl=0.05；zero-variance group skip |
| 评测 | eval_n=200；greedy；max_new_eval=512；Wilson CI |
| 硬件 | Colab G4 / RTX PRO 6000 / 94.97GB；HF sdpa |
| Map | G=2048；2,048 trainable params；shared output-channel multiplicative gate；target 336 linears |
| LoRA | rank=8；alpha=16；27,230,208 trainable params；target 同 336 linears |

## 实验结果

| 指标 | Baseline | Map-G2048 | LoRA-r8 |
|---|---:|---:|---:|
| Eval accuracy | 0.405 | 0.480 | 0.420 |
| Correct / n | 81 / 200 | 96 / 200 | 84 / 200 |
| Wilson CI | [0.339, 0.474] | [0.412, 0.549] | [0.354, 0.489] |
| Updates / skipped | - | 30 / 5 | 30 / 5 |
| Best train correct mean | - | 0.875 | 1.000 |
| Final train correct mean | - | 0.500 | 0.125 |

结论：单 seed 下 Map 比 LoRA 高 6pp，比 baseline 高 7.5pp；CI 重叠，不能声明统计显著。

## 性能对比

| 指标 | Map-G2048 | LoRA-r8 | 对比 |
|---|---:|---:|---:|
| Train elapsed | 1358.7s | 1372.0s | Map -1.0% |
| Mean step | 10.04s | 10.42s | Map -3.6% |
| Tokens/s | 278.4 | 247.5 | Map +12.5% |
| Trainable params | 2,048 | 27,230,208 | Map 少 13,296x |
| Checkpoint estimate | 4KB | 51.9MB | Map 小 13,296x |
| 完整实验 batch | train_batch=1 | train_batch=1 | max_new=512 下更大 batch OOM |

| Map 性能 sweep 配置 | Tokens/s | Mean step | Peak VRAM | 备注 |
|---|---:|---:|---:|---|
| max_new=512, train_batch=2, beta_kl=0.05 | 438.4 | 7.93s | 80.6GB | 短测最高吞吐；长跑 OOM 风险 |
| max_new=512, train_batch=1, beta_kl=0 | 379.3 | 5.92s | 46.5GB | 低显存稳定配置 |
| max_new=256, train_batch=2, beta_kl=0.05 | 437.5 | 7.90s | 80.1GB | 短输出高吞吐 |

最终推荐：稳定完整实验用 Map-G2048, train_batch=1, max_new=512, beta_kl=0.05；追求短跑吞吐可用 train_batch=2，但需接受 OOM 风险。

## Qwen3.5-9B 结果

| Prompt 配置 | active_n | boxed_rate | long_output_rate | 结论 |
|---|---:|---:|---:|---|
| enable_thinking=false + /no_think, max_new=256 | 2 / 30 | 0.079 | 0.921 | 未过 gate |
| enable_thinking=false + /no_think, max_new=512 | 8 / 30 | 0.392 | 0.613 | 未过 gate |
| enable_thinking=false, max_new=256 | 2 / 30 | 0.050 | 0.950 | 未过 gate |
| /no_think only, max_new=256 | 0 / 30 | 0.083 | 1.000 | 未过 gate |

Qwen3.5-9B 当前未进入 Map/LoRA 训练。原因：active bank、boxed_rate、long_output_rate 均不满足训练前置条件。

## Qwen3.5-9B Gate 结果

| Prompt 配置 | active_n | boxed_rate | long_output_rate | variance_prompt_rate | 结论 |
|---|---:|---:|---:|---:|---|
| answer_only + no_think, max_new=64 | 8 / 30 | 0.721 | 0.283 | 0.267 | 未过 gate |
| answer_only + no_think, max_new=128 | 8 / 30 | 0.746 | 0.258 | 0.267 | 未过 gate |
| answer_only + no_think, max_new=256 | 6 / 30 | 0.750 | 0.250 | 0.200 | 未过 gate |
| brief + no_think, max_new=64 | 0 / 30 | 0.008 | 0.992 | 0.000 | 未过 gate |
| brief + no_think, max_new=128 | 0 / 30 | 0.000 | 1.000 | 0.000 | 未过 gate |
| brief + no_think, max_new=256 | 3 / 30 | 0.146 | 0.854 | 0.100 | 未过 gate |

Qwen3.5-9B 未进入 Map/LoRA 训练。最佳配置仍未满足 active_n、boxed_rate、long_output_rate 三个 gate。

## Qwen3.5-9B Gate 结论

| 配置 | active_n | boxed_rate | long_output_rate | 结论 |
|---|---:|---:|---:|---|
| answer_only + no_think, max_new=64 | 7 / 30 | 0.721 | 0.283 | 未过 gate |
| answer_only + no_think, max_new=128 | 7 / 30 | 0.746 | 0.258 | 未过 gate |
| answer_only + no_think, max_new=256 | 7 / 30 | 0.750 | 0.250 | 未过 gate |
| XML answer tag, max_new=256 | 6 / 50 | 0.053 | 0.945 | 未过 gate |

已修复 scorer：支持 `<answer>...</answer>` 和 degree 归一化。修复后 Qwen3.5 仍未满足 active bank、boxed_rate、long_output_rate gate，因此没有进入 Map/LoRA 完整训练。

## Qwen3.5-9B 完整实验结果

| 指标 | Baseline | Map-G2048 | LoRA-r8 1e-4 | LoRA-r8 3e-4 |
|---|---:|---:|---:|---:|
| Eval accuracy | 0.355 | 0.395 | 0.305 | 0.150 |
| Correct / n | 71 / 200 | 79 / 200 | 61 / 200 | 30 / 200 |
| Wilson CI | [0.292, 0.423] | [0.330, 0.464] | [0.245, 0.372] | [0.107, 0.206] |
| Updates / skipped | - | 30 / 11 | 26 / 94 | 30 / 30 |
| Train elapsed | - | 84.2s | 111.3s | 89.4s |
| Mean step | - | 0.970s | 1.088s | 0.970s |
| Tokens/s | - | 950.0 | 839.5 | 847.9 |
| Peak VRAM | - | 73.2GB | 71.4GB | 71.4GB |
| Trainable params | - | 2,048 | 16,121,856 | 16,121,856 |

Qwen3.5-9B 使用 level1-3 active bank 后完成训练。Map 高于 baseline 4pp，高于最佳 LoRA 9pp；但仍是单 seed，CI 有重叠。

## Loop0 极致性能结果

| 配置 | tokens/s | step_s | peak VRAM | 结论 |
|---|---:|---:|---:|---|
| all layers, all targets, beta_kl=0 | 343.3 | 6.97s | 45.45GB | baseline local Map |
| all layers, o_down, beta_kl=0 | 468.5 | 4.25s | 35.37GB | 快 36.5% |
| all layers, down only, beta_kl=0.05 | 470.1 | 4.37s | 35.84GB | 最快且低显存 |
| last 8 layers, o_down, beta_kl=0.05 | 435.8 | 4.53s | 36.68GB | 局部层有效但不最优 |

Loop0 结论：局部 target 能把 Map 从 343 tok/s 提到 470 tok/s，显存从 45.5GB 降到 35.8GB，但未达到 3x。下一步进入 Loop1：SGLang/vLLM rollout 加速。

## Qwen3.5-9B 3-Seed 结果

| seed | Baseline | Map-G2048 | LoRA-r8 1e-4 | Map - LoRA |
|---|---:|---:|---:|---:|
| 0 | 0.355 | 0.385 | 0.265 | +0.120 |
| 1 | 0.355 | 0.375 | 0.350 | +0.025 |
| 2 | 0.355 | 0.390 | 0.365 | +0.025 |
| mean | 0.355 | 0.383 | 0.327 | +0.057 |
| stdev | 0.000 | 0.008 | 0.054 | - |

| 性能指标 | Map-G2048 | LoRA-r8 1e-4 |
|---|---:|---:|
| tokens/s mean | 916.9 | 869.2 |
| tokens/s stdev | 30.3 | 67.6 |
| peak VRAM mean | 77.1GB | 71.4GB |
| trainable params | 2,048 | 16,121,856 |

3-seed 结论：Map 平均 accuracy 0.383，高于 LoRA 0.327；Map 平均吞吐 916.9 tokens/s，高于 LoRA 869.2 tokens/s；Map 参数量少 7,872x。
