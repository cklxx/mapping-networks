# 9B Map vs LoRA：结论与训练配置

## 结论

| 维度 | Baseline | Map-G2048 | LoRA-r8 |
|---|---:|---:|---:|
| Eval accuracy | 40.5% | 48.0% | 42.0% |
| Correct / n | 81/200 | 96/200 | 84/200 |
| Wilson CI | [33.9, 47.4] | [41.2, 54.9] | [35.4, 48.9] |
| Train elapsed | - | 1358.7s | 1372.0s |
| Mean step | - | 10.04s | 10.42s |
| Tokens/s | - | 278.4 | 247.5 |
| Trainable params | - | 2,048 | 27,230,208 |
| Checkpoint | - | 4KB | 51.9MB |

单 seed / 30 update / eval200 下，Map 同时赢 eval、tokens/s、参数量和 checkpoint 大小。CI 仍重叠，不能说统计显著最终收敛优于 LoRA。

## 模型与 Adapter

| 项 | 值 |
|---|---|
| Base model | `01-ai/Yi-1.5-9B-Chat` |
| Base params | 8,829,407,232 |
| Layers / hidden / FFN | 48 / 4096 / 11008 |
| Heads / KV heads | 32 / 4 |
| dtype | bf16 |
| Target linears | 336 = 48 x q/k/v/o/gate/up/down |
| Map | shared 2048-group output-channel multiplicative gate |
| LoRA | rank=8, alpha=16, all target linears |

## 训练配置

| 项 | 值 |
|---|---|
| Dataset | MATH-500 test split |
| Eval | first 200 records, greedy, max_new_eval=512 |
| Candidate pool | remaining records, level 3-5 |
| Active bank | candidate_n=50, active_n=29, K=8, max_new=512 |
| Training | target_updates=30, K=8, max_new=512, beta_kl=0.05 |
| Reward | correct + 0.1 * format - 0.2 * overlong |
| Skip rule | skip zero-variance groups |
| GPU | Colab G4 / RTX PRO 6000 / 94.97GB |

## 代码方案

- `src/adapters.py`: Map forward 改为 activation-side scaling，不再 materialize `W * gate`。
- `experiments/math500_active_grpo_9b.py`: active bank、zero-variance skip、B*K batched logprob/backward。
- `experiments/perf_sweep_active_grpo_9b.py`: Map 性能 sweep，搜索 max_new / train_batch / beta_kl。

```python
# Map forward
out = F.linear(x, W, None)
out = out * gate
out = out + bias

# GRPO update
logps, kls = batched_logp_and_kl(...)
loss = (-(adv * logps) + beta_kl * kls).mean()
loss.backward()
```

## 下一步

1. 跑 3 seeds，确认 Map 48% vs LoRA 42% 是否稳定。
2. 跑 target_updates=50/100。
3. Map sweep: G=2048/8192, lr_o=0.003/0.005/0.01。
4. LoRA sweep: lr=1e-4/3e-4。
5. 若要 wall-clock 极致，接入 vLLM/SGLang rollout；继续手写 Map scale kernel 收益有限。

## 性能 Sweep

| 配置 | tokens/s | mean step | peak VRAM | 备注 |
|---|---:|---:|---:|---|
| max_new=512, train_batch=2, beta_kl=0.05 | 438.4 | 7.93s | 80.6GB | 最高吞吐，接近显存上限 |
| max_new=512, train_batch=1, beta_kl=0 | 379.3 | 5.92s | 46.5GB | 低显存稳定配置 |
| max_new=256, train_batch=2, beta_kl=0.05 | 437.5 | 7.90s | 80.1GB | 短输出下也接近最高吞吐 |
| max_new=512, train_batch=1, beta_kl=0.05 | 327.7 | 7.03s | 50.9GB | 带 KL 的稳健单 batch |

结论：如果追求最高吞吐，优先用 train_batch=2；如果追求稳定和可扩展，优先用 beta_kl=0、train_batch=1。真正的数量级加速需要把 rollout 切到 vLLM/SGLang。
