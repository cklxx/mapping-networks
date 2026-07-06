# 9B Map vs LoRA 实验结论

## 结论

在当前 9B / MATH-500 / active-bank RL 设置下，优化后的 Map-G2048 在 30 update、eval_n=200 的实验中优于 LoRA-r8：

| 维度 | Baseline | Map-G2048 | LoRA-r8 | 结论 |
|---|---:|---:|---:|---|
| Eval accuracy | 40.5% | 48.0% | 42.0% | Map 高 LoRA 6pp |
| Wilson CI | [33.9, 47.4] | [41.2, 54.9] | [35.4, 48.9] | CI 重叠，不能当统计终局 |
| Train updates / skip | - | 30 / 5 | 30 / 5 | 相同 |
| Train elapsed | - | 1358.7s | 1372.0s | 基本相同，Map 略快 |
| Mean step | - | 10.04s | 10.42s | Map 略快 |
| Tokens/s | - | 278.4 | 247.5 | Map +12% |
| Trainable params | - | 2,048 | 27,230,208 | Map 少 13,296x |
| Checkpoint size est. | - | 4KB | 51.9MB | Map 小 13,296x |

## 为什么之前性能差

原实现每次 forward 都 materialize `W * gate[:, None]`，会重扫 dense weight，内存带宽浪费很大。现在改成等价的 activation-side scaling：

```python
out = linear(x, W, None)
out = out * gate
out = out + bias
```

这个改动后，Map 从慢于 LoRA 变成略快于 LoRA。

## 为什么不会快几十倍

Map 参数量少 13,296x，但训练主要开销不是 adapter 参数，而是完整 9B 模型的：

- rollout generation
- policy logprob forward/backward
- KL reference forward
- eval generation

Map 和 LoRA 都要跑完整 9B base model，所以不会因为 adapter 参数少就快几十倍。Map 的主要工程优势是：参数、optimizer state、checkpoint 和额外 adapter 计算极小。

## 已做的代码优化

1. Map forward 改为 activation-side scaling，避免 materialize scaled weight。
2. GRPO logprob/KL 从逐 completion 计算改为 batched group 计算。
3. 支持多个 prompt group 的 B*K batch，但在 G4 + max_new=512 下，大 batch 会 OOM。
4. 支持 active bank 复用，避免重复 probe。
5. 支持 zero-variance group skip。
6. 支持固定时间和收敛阈值参数。

## 系统发现

- `train_batch=4` 的 Map 在 max_new=512 下 OOM。
- `train_batch=2` 的 LoRA 在 max_new=512 下 OOM。
- 完整 30-update / eval200 对比中，两者都使用 `train_batch=1`。
- G4 是当前最合适 GPU；A100 40GB、L4/T4 显存不足以稳妥跑该配置。

## Solid 判断

当前证据支持：Map 在这个设置下有明显工程优势，并且在一次完整 30-update/eval200 实验中效果也高于 LoRA。

当前证据不支持：Map 已经统计显著地最终收敛优于 LoRA。原因是 Wilson CI 仍重叠，且只有单 seed。

## 下一步

要把结论做扎实，需要：

1. 多 seed：至少 3 个 seed。
2. 更大 budget：target_updates 50/100。
3. Map sweep：G=2048/8192，lr_o=0.003/0.005/0.01。
4. LoRA sweep：lr=1e-4/3e-4，rank=8。
5. 固定 wall-clock 对比：例如每个 variant 60 分钟。

## 代码与产物

- 代码提交：`eb8e5ff` 之后已包含优化和 active-bank harness。
- 本地报告：`results/9b-math500/FULL_ACTIVE_COMPARE_REPORT.md`
- 本地 summary：`results/9b-math500/map30-summary.json`、`results/9b-math500/staged-lora-summary.json`
