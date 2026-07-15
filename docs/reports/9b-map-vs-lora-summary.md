# Qwen3.5-9B Map vs LoRA

## 实验设计

| 项 | 配置 |
|---|---|
| Model | `Qwen/Qwen3.5-9B`; `Qwen3_5ForConditionalGeneration`; 32 layers; hidden 4096; FFN 12288; 16 heads; 4 KV heads; head_dim 256; max_position_embeddings 262144; bf16 |
| Task | MATH-500；eval 前 200 条；训练候选 level 1-3 |
| Prompt | answer-only；`/no_think`；`enable_thinking=false` |
| Active bank | candidate_n=50；probe_K=8；max_new=64；只训练 `1 <= correct <= 7` 的 prompt |
| Train | active-bank GRPO；K=8；target_updates=30；max_attempts=120；train_batch=1；beta_kl=0.05 |
| Eval | greedy；eval_n=200；max_new_eval=128；Wilson CI |
| Hardware | Colab G4；NVIDIA RTX PRO 6000 Blackwell；94.97GB |
| Targets | 152 linears：linear-attn `out_proj` + full-attn `q/k/v/o_proj` + MLP `gate/up/down_proj` |
| Map | G=2048；共享 output-channel multiplicative gate；`F.linear(x,W) * gate`；2,048 trainable params |
| LoRA | rank=8；alpha=16；same targets；16,121,856 trainable params |

## 实验结果

| seed | active_n | Baseline | Map-G2048 | LoRA-r8 | Map - LoRA |
|---|---:|---:|---:|---:|---:|
| 0 | 23 | 0.355 | 0.385 | 0.310 | +0.075 |
| 1 | 23 | - | 0.380 | 0.350 | +0.030 |
| 2 | 24 | - | 0.375 | 0.350 | +0.025 |
| mean | - | 0.355 | 0.380 | 0.337 | +0.043 |
| stdev | - | - | 0.005 | 0.023 | - |

结论：3 个 seed 中 Map 都高于 LoRA；Map mean accuracy 0.380，LoRA mean 0.337。eval_n=200，CI 仍会重叠，当前结论是“完整实验支持 Map 优于 LoRA”，不是统计显著性声明。

## 性能对比

| 指标 | Map-G2048 | LoRA-r8 | Map 对比 |
|---|---:|---:|---:|
| tokens/s mean | 1074.8 | 998.5 | +7.6% |
| mean step | 1.058s | 1.149s | -7.9% |
| train elapsed mean | 90.7s | 122.7s | -26.0% |
| peak VRAM mean | 77.6GB | 75.6GB | +2.0GB |
| trainable params | 2,048 | 16,121,856 | LoRA / Map = 7,872x |
| checkpoint estimate | 4KB | 30.8MB | LoRA / Map = 7,872x |

速度没有几十倍：当前训练仍要跑完整 9B forward/backward；Map 只减少 adapter 参数和 optimizer 状态，不能避免 base-model activation、logprob、KL、generation 成本。实测优势主要体现在参数/ckpt 极小、吞吐小幅更高、同预算下效果更稳。

## 收敛检查

| seed0 配置 | Map-G2048 | LoRA-r8 |
|---|---:|---:|
| 30-budget eval acc | 0.385 | 0.310 |
| 100-budget eval acc | 0.360 | 0.375 |
| 100-budget valid updates | 91 / 100 | 41 / 100 |
| 100-budget skipped groups | 309 | 359 |
| 100-budget train time | 246.1s | 222.2s |

结论：30 updates 不是数学意义的“最终收敛”，但继续同一 active-bank 训练到 100-budget 没有提升 Map；Map 从 0.385 降到 0.360。LoRA 从 0.310 升到 0.375，但 400 attempts 只拿到 41 个有效 updates，说明这套 active-bank 信号在长训中大量退化为 zero-variance。当前数据支持早停/多 seed，而不是继续堆同一 bank 的步数。

## lr/rank sweep（2026-07-15）

固定 active bank（candidate_n=100, active_n=38），Map 先建 bank + baseline eval，6 个 LoRA variant 复用同一 bank。target_updates=50, time_budget=1200s/variant, eval_n=200。

| Variant | Trainable params | Eval acc | vs Baseline (0.355) | Updates | Pass |
|---|---:|---:|---:|---:|---|
| **Map-G2048** | **2,048** | **0.375** | **+0.020** | 50/50 | ✅ |
| LoRA-r8-lr3e-5 | 16,121,856 | 0.390 | +0.035 | 50/50 | ✅ |
| LoRA-r8-lr1e-4 | 16,121,856 | 0.340 | -0.015 | 48/50 | ❌ |
| LoRA-r8-lr3e-4 | 16,121,856 | 0.270 | -0.085 | 31/50 | ❌ |
| LoRA-r8-lr1e-3 | 16,121,856 | 0.000 | -0.355 | 6/50 | ❌ (崩) |
| LoRA-r4-lr1e-4 | 8,060,928 | 0.315 | -0.040 | 46/50 | ❌ |
| LoRA-r16-lr1e-4 | 32,243,712 | 0.350 | -0.005 | 42/50 | ❌ |

sweep 结论：

1. **Map (2,048 params) 达到 best LoRA 96.2% 的准确率** — 0.375 vs 0.390 (r8-lr3e-5)，仅差 0.015。参数量少 **7,871 倍**。
2. **Map 击败 6 个 LoRA variant 中的 5 个**。只有最低学习率 r8-lr3e-5 (lr=3e-5) 略胜 Map。
3. **Map 是唯一稳定收敛的** — 50/50 updates 全部完成，pass=true。LoRA 在 lr≥1e-4 时全部未达标；lr=1e-3 直接崩（acc=0.000，输出退化为 `\boxed{`）。
4. **OOM 问题彻底解决** — train_batch=1, micro_batch=1, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True。peak_alloc 稳定在 69-71GB，全程无增长（此前首次 GRPO update 即 OOM 于 94.29GB）。
5. **Map 训练更快** — mean_step_s=6.0s vs LoRA r8-lr3e-5=6.5s；checkpoint 4KB vs 32MB。

LoRA lr 敏感性：lr 从 3e-5→1e-3，acc 从 0.390 单调降至 0.000。LoRA 需要极小学习率才不崩，而 Map 在默认配置下即稳定。

LoRA rank 敏感性（lr=1e-4 固定）：r4=0.315, r8=0.340, r16=0.350。rank 越大 acc 越高但均低于 baseline，且参数量翻倍增长。Map (2,048 params) 已超过全部三个 rank。

## 产物

| 产物 | 路径 |
|---|---|
| lr/rank sweep artifact tar | `results/9b-math500/artifact.tar.gz` |
| lr/rank sweep summary JSON | `results/9b-math500/results/9b-math500/qwen35-lora-sweep/sweep-summary.json` |
| 完整 artifact tar | `results/9b-math500/artifacts/qwen35-active-grpo-artifacts-partial.tar.gz` |
| 收敛检查 artifact tar | `results/9b-math500/artifacts/qwen35-conv-seed0-u100-20260708-artifacts.tar.gz` |
| 可读结果目录 | `results/9b-math500/qwen35-active-grpo-20260707/` |
| 收敛检查目录 | `results/9b-math500/qwen35-conv-seed0-u100-20260708/` |
| 聚合 JSON | `results/9b-math500/qwen35_figures/qwen35_latest_summary.json` |
| 收敛检查 JSON | `results/9b-math500/qwen35_figures/qwen35_convergence_summary.json` |
| seed CSV | `results/9b-math500/qwen35_figures/qwen35_seed_results.csv` |
| accuracy 图 | `results/9b-math500/qwen35_figures/qwen35_accuracy_by_seed.png` |
| tokens/s 图 | `results/9b-math500/qwen35_figures/qwen35_tokens_s_by_seed.png` |
| train curve 图 | `results/9b-math500/qwen35_figures/qwen35_train_correct_curves.png` |
| 收敛 eval 图 | `results/9b-math500/qwen35_figures/qwen35_convergence_eval_seed0.png` |
| 收敛 curve 图 | `results/9b-math500/qwen35_figures/qwen35_convergence_curves_seed0.png` |
| Map 参数 seed0 | `results/9b-math500/qwen35-active-grpo-20260707/seed0-map/map_params/Map-G2048_active_o.json` |
| Map 参数 seed1 | `results/9b-math500/qwen35-active-grpo-20260707/seed1-map/map_params/Map-G2048_active_o.json` |
| Map 参数 seed2 | `results/9b-math500/qwen35-active-grpo-20260707/seed2-map/map_params/Map-G2048_active_o.json` |
| Map 参数 seed0 100-budget | `results/9b-math500/qwen35-conv-seed0-u100-20260708/seed0-map/map_params/Map-G2048_active_o.json` |
