# 9B Map vs LoRA：性能结果

## 结果

| 对比项 | 结果 |
|---|---|
| Map forward 优化收益 | 147.4 -> 251.2 tokens/s，+70.4%；17.17s -> 8.50s/step，-50.5% |
| Map 最快短测 | 438.4 tokens/s；7.93s/step；peak VRAM 80.6GB |
| Map 稳定低显存配置 | 379.3 tokens/s；5.92s/step；peak VRAM 46.5GB |
| 完整实验设置 | target_updates=30，eval_n=200，max_new=512，beta_kl=0.05 |
| 完整实验 accuracy | Baseline 40.5%；Map 48.0%；LoRA 42.0% |
| 完整实验吞吐 | Map 278.4 tokens/s；LoRA 247.5 tokens/s；Map +12.5% |
| 完整实验 step time | Map 10.04s；LoRA 10.42s；Map -3.6% |
| 完整实验训练耗时 | Map 1358.7s；LoRA 1372.0s；Map -1.0% |
| 参数量 | Map 2,048；LoRA 27,230,208；Map 少 13,296x |
| Checkpoint | Map 4KB；LoRA 51.9MB；Map 小 13,296x |
| 显存约束 | max_new=512 下，Map train_batch=4 OOM；LoRA train_batch=2 OOM；完整实验均用 train_batch=1 |

## 结论

- Map 在完整实验中效果高于 LoRA：48.0% vs 42.0%。
- Map 在完整实验中吞吐高于 LoRA：+12.5%。
- Map 的参数量和 checkpoint 优势是 13,296x。
- Map 最快短测可到 438.4 tokens/s，但显存接近上限，不作为完整训练默认配置。
- 当前结果是单 seed，CI 重叠，不能声明统计显著最终收敛优于 LoRA。

## 最终推荐

| 目标 | 推荐配置 |
|---|---|
| 稳定完整实验 | Map-G2048，train_batch=1，max_new=512，beta_kl=0.05 |
| 最高吞吐短跑 | Map-G2048，train_batch=2，max_new=512，beta_kl=0.05 |
| 低显存稳定 | Map-G2048，train_batch=1，max_new=512，beta_kl=0 |
| 下一步确认 | 3 seeds；target_updates=50/100；eval_n=200 |
