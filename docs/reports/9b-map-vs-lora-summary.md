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
