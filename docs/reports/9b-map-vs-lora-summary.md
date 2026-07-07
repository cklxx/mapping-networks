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

## 产物

| 产物 | 路径 |
|---|---|
| 完整 artifact tar | `results/9b-math500/artifacts/qwen35-active-grpo-artifacts-partial.tar.gz` |
| 可读结果目录 | `results/9b-math500/qwen35-active-grpo-20260707/` |
| 聚合 JSON | `results/9b-math500/qwen35_figures/qwen35_latest_summary.json` |
| seed CSV | `results/9b-math500/qwen35_figures/qwen35_seed_results.csv` |
| accuracy 图 | `results/9b-math500/qwen35_figures/qwen35_accuracy_by_seed.png` |
| tokens/s 图 | `results/9b-math500/qwen35_figures/qwen35_tokens_s_by_seed.png` |
| train curve 图 | `results/9b-math500/qwen35_figures/qwen35_train_correct_curves.png` |
| Map 参数 seed0 | `results/9b-math500/qwen35-active-grpo-20260707/seed0-map/map_params/Map-G2048_active_o.json` |
| Map 参数 seed1 | `results/9b-math500/qwen35-active-grpo-20260707/seed1-map/map_params/Map-G2048_active_o.json` |
| Map 参数 seed2 | `results/9b-math500/qwen35-active-grpo-20260707/seed2-map/map_params/Map-G2048_active_o.json` |
