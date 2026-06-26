"""MATH-500 RL: per-group weight modulation vs LoRA on a FROZEN Qwen3-4B.

This is the experiment that produced the one validated result (see results/4b-math500/).
It RL-tunes a frozen base with GRPO and compares, at matched ~40-step / 35-min budgets:
  - Map-G{256,2048}: the multiplicative modulation gate (G trainable params), and
  - LoRA-r8        : the additive low-rank baseline (~16.5M trainable params).

Why MATH-500 and not GSM8K: the 4B sits well below ceiling on MATH-500 (~30% baseline),
so RL has real headroom and the small-modulation-vs-LoRA comparison is meaningful. GSM8K
was at-ceiling for both the 0.8B and 4B and gave RL no advantage variance (see
docs/research-plan.md "What didn't work").

Adapter math + the MATH scorer live in src/ (single source of truth). This file owns only
the RL loop, the eval, and the report.

Usage:
    python experiments/math500_rl.py --model Qwen/Qwen3-4B --out results/4b-math500/results.txt
    python experiments/math500_rl.py --baseline-only      # just measure the base (headroom gate)
"""
import argparse
import json
import math
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.adapters import (  # noqa: E402
    ALPHA_MOD,
    base_forward,
    get_parent,
    install_direct_map,
    install_lora,
    num_layers_of,
    restore,
    target_modules,
)
from src.math_scorer import extract_answer, gold_answer, reward_of  # noqa: E402
from src import costlib  # noqa: E402 — same cost hooks as experiments/cost_benchmark.py

torch.manual_seed(0)

# steps-to-target threshold: the first step whose 3-step trailing-mean reward reaches this.
COST_TARGET_REWARD = 0.20

# ---- RL / eval config (the validated 4B-MATH settings) ----
G_SWEEP = [256, 2048]      # 256 = high-leverage low budget; 2048 = ResNet50 §5.4 size
LR_O = 0.01                # 4B-MATH has signal; between 0.8B-hot 0.02 and 4B-GSM8K-frozen 0.002
O_CLAMP = 0.10             # HARD CLAMP |o| <= 0.10 after each step (4B coherent band ~0.10-0.15)
LORA_R = 8
LR_LORA = 1e-4             # legacy single-lr default (kept for reference); the run sweeps below
# FAIR head-to-head: sweep the LoRA learning rate and report LoRA's BEST variant, so
# "modulation beats LoRA" is not an artifact of an under-tuned lr=1e-4 (docs/research-plan.md §2).
LORA_LR_SWEEP = [1e-4, 3e-4, 1e-3, 3e-3]
B, K = 6, 4                # GRPO: questions/step, completions/question
MAX_NEW, MAX_NEW_EVAL = 512, 1024   # train rollout short (reward variance); eval long (no truncation)
EVAL_BATCH = 16
N_EVAL = 200               # fixed MATH-500 test subset, greedy
TIME_BUDGET_S, MAX_STEPS = 35 * 60, 40
BETA_KL = 0.05             # KL leash, looser than GSM8K's 0.1 to let MATH signal move
N_CASES = 3

SYS = (
    "Solve the math problem. Reason briefly, then put the final answer in \\boxed{}. "
    "Do not write anything after the boxed answer."
)


def build_prompt(tok, q):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def comp_logp_and_kl(model, names, prompt_ids, comp_ids, dev):
    """ONE policy forward + ONE base forward over prompt+completion -> (sum_logp, mean_KL)."""
    ids = torch.cat([prompt_ids, comp_ids], 0)[None].to(dev)
    logits = model(ids).logits[0, :-1].float()  # policy, predict token t+1
    logp = torch.log_softmax(logits, -1)
    tgt = ids[0, 1:]
    n_prompt = prompt_ids.numel()
    comp_mask = torch.zeros_like(tgt, dtype=torch.bool)
    comp_mask[n_prompt - 1:] = True  # completion-token positions only
    tok_lp = logp.gather(1, tgt[:, None]).squeeze(1)
    sum_lp = tok_lp[comp_mask].sum()
    with torch.no_grad(), base_forward(model, names):
        base_logits = model(ids).logits[0, :-1].float()
        base_logp = torch.log_softmax(base_logits, -1)
    p = logp.exp()
    kl_per_pos = (p * (logp - base_logp)).sum(-1)  # KL(policy||base) per position
    kl = kl_per_pos[comp_mask].mean()
    return sum_lp, kl


@torch.no_grad()
def evaluate(model, tok, items, dev, label="", collect_cases=0):
    """Batched greedy accuracy on a fixed test subset (left-pad)."""
    model.eval()
    correct = 0
    t0 = time.time()
    cases = []
    prev_side = tok.padding_side
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    done = 0
    for b0 in range(0, len(items), EVAL_BATCH):
        batch = items[b0:b0 + EVAL_BATCH]
        prompts = [build_prompt(tok, q) for q, _ in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
        out = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW_EVAL, pad_token_id=pad_id)
        gen = out[:, enc.input_ids.shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for (q, gold), text in zip(batch, texts):
            pred = extract_answer(text)
            correct += int(pred == gold and bool(gold))
            if len(cases) < collect_cases:
                cases.append((q, text, pred, gold))
        done += len(batch)
        print(f"  [eval {label}] {done}/{len(items)}  acc_sofar={correct/max(1,done):.3f}  "
              f"{(time.time()-t0)/max(1,done):.1f}s/q", flush=True)
    tok.padding_side = prev_side
    return correct, len(items), cases


def train_grpo(model, tok, names, train_items, trainable, lr, dev, label, telem_fn=None, clamp_o=None):
    """GRPO + KL leash + optional hard clamp. Returns (reward_curve, kl_curve, telem, timer,
    tokens_per_step). The timer + token count feed the cost hooks (src/costlib.py) so this
    validated runner emits the SAME per-variant cost row as experiments/cost_benchmark.py."""
    model.train()
    opt = torch.optim.Adam(trainable, lr=lr)
    t0 = time.time()
    curve, kl_curve, telem = [], [], []
    timer = costlib.StepTimer().start()
    tok_acc = []  # total prompt+completion tokens the step's forwards saw (for FLOPs/step est)
    for step in range(MAX_STEPS):
        if time.time() - t0 > TIME_BUDGET_S:
            print(f"[{label}] time budget hit at step {step}", flush=True)
            break
        batch = [train_items[(step * B + i) % len(train_items)] for i in range(B)]
        step_r, nz_groups, kl_acc, did_backward = [], 0, [], False
        step_tokens = 0
        opt.zero_grad()
        for q, gold in batch:
            prompt = build_prompt(tok, q)
            pids = tok(prompt, return_tensors="pt").input_ids[0].to(dev)
            with torch.no_grad():
                gen = model.generate(pids[None], do_sample=True, temperature=0.8, top_p=0.95,
                                     num_return_sequences=K, max_new_tokens=MAX_NEW,
                                     pad_token_id=tok.eos_token_id)
            comps = [gen[k, pids.numel():] for k in range(K)]
            texts = [tok.decode(c, skip_special_tokens=True) for c in comps]
            rs = torch.tensor([reward_of(t, gold) for t in texts], dtype=torch.float32)
            step_r.append(rs.mean().item())
            if rs.std() > 1e-6:
                nz_groups += 1
            adv = (rs - rs.mean()) / (rs.std() + 1e-4)
            for k in range(K):
                step_tokens += pids.numel() + comps[k].numel()
                lp, kl = comp_logp_and_kl(model, names, pids, comps[k], dev)
                kl_acc.append(kl.item())
                pg = -adv[k].to(dev).detach() * lp if adv[k].abs() >= 1e-6 else 0.0 * lp
                comp_loss = (pg + BETA_KL * kl) / (B * K)
                comp_loss.backward()  # free this completion's graph now
                did_backward = True
        if did_backward:
            opt.step()
            if clamp_o is not None:
                with torch.no_grad():
                    trainable[0].clamp_(-clamp_o, clamp_o)  # project o into the coherent band
        mr = sum(step_r) / len(step_r)
        mkl = sum(kl_acc) / len(kl_acc) if kl_acc else 0.0
        curve.append(mr)
        kl_curve.append(mkl)
        tok_acc.append(step_tokens)
        timer.tick()
        tline = telem_fn(trainable) if telem_fn else ""
        telem.append(tline)
        print(f"[{label}] step {step:3d}  mean_reward={mr:.3f}  signal_groups={nz_groups}/{B}  "
              f"mean_kl={mkl:.4f}  {tline}  step_s={timer.per_step[-1]:.1f}  "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
    tokens_per_step = int(sum(tok_acc) / len(tok_acc)) if tok_acc else 0
    return curve, kl_curve, telem, timer, tokens_per_step


def o_telem(trainable):
    o = trainable[0].detach()
    return f"mean|o|={o.abs().mean().item():.4f} max_gate={(1.0 + ALPHA_MOD * o).abs().max().item():.3f}"


def lora_telem(trainable):
    mag = torch.stack([p.detach().abs().mean() for p in trainable]).mean().item()
    return f"mean|AB|={mag:.4f}"


def fmt_cases(cases):
    out = []
    for j, (q, text, pred, gold) in enumerate(cases):
        out.append(f"  --- case {j+1} ---")
        out.append(f"  PROBLEM: {q.strip()[:400]}")
        out.append(f"  MODEL: {text.strip()[:1100]}")
        out.append(f"  extracted_boxed={pred!r}  gold={gold!r}  "
                   f"{'CORRECT' if (pred == gold and gold) else 'WRONG'}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B",
                    help="HF model id or local path of the frozen base")
    ap.add_argument("--out", default="results/4b-math500/results.txt",
                    help="where to write the report")
    ap.add_argument("--cost-out", default="results/cost-table.md",
                    help="where to write the per-variant cost table (markdown)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--baseline-only", action="store_true",
                    help="measure only the base accuracy (headroom gate) and stop")
    args = ap.parse_args()

    dev = args.device
    dt = torch.bfloat16
    print(f"device={dev}  dtype={dt}  model={args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt, trust_remote_code=True).to(dev)
    model.requires_grad_(False)
    nlayers = num_layers_of(model)
    base_params = sum(p.numel() for p in model.parameters())  # for the FLOPs/step cost est
    print(f"num_hidden_layers (from config) = {nlayers}  base_params={base_params:,}", flush=True)

    print(f"config: B={B} K={K} MAX_STEPS={MAX_STEPS} time_box={TIME_BUDGET_S}s MAX_NEW={MAX_NEW} "
          f"N_EVAL={N_EVAL} BETA_KL={BETA_KL} LR_O={LR_O} O_CLAMP={O_CLAMP} LORA_R={LORA_R} "
          f"ALPHA_MOD={ALPHA_MOD} G_SWEEP={G_SWEEP}  baseline_only={args.baseline_only}", flush=True)

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    all_items = [(r["problem"], gold_answer(r)) for r in ds]
    eval_items = all_items[:N_EVAL]
    # Curriculum fix: the raw index-200:500 tail is harder than eval (56% vs 47% Level-4/5),
    # so GRPO drew all-wrong batches -> zero advantage -> o frozen at identity. Sorting the
    # TRAIN tail easy->hard gives some solvable problems per batch -> non-zero advantage. Eval
    # is untouched, so the A/B verdict stays unbiased.
    train_records = [r for r in ds][N_EVAL:] if len(ds) > N_EVAL else [r for r in ds]
    train_records.sort(key=lambda r: int(r.get("level") or 5))
    train_pool = [(r["problem"], gold_answer(r)) for r in train_records]
    train_items = [train_pool[i % len(train_pool)] for i in range(B * MAX_STEPS + 64)]
    print(f"train pool: {len(train_pool)} problems, sorted easy->hard (curriculum)", flush=True)

    names = target_modules(model)
    print(f"target projection linears: {len(names)}  (all {nlayers} layers)", flush=True)

    # ---- baseline (no adapter) ----
    k_base, n_base, cases_base = evaluate(model, tok, eval_items, dev, label="base", collect_cases=N_CASES)
    acc_base = k_base / n_base
    ci_base = wilson_ci(k_base, n_base)
    print(f"[baseline] MATH-500 acc = {acc_base:.4f} ({k_base}/{n_base})  "
          f"CI [{ci_base[0]:.3f},{ci_base[1]:.3f}]", flush=True)

    # Headroom gate: if at ceiling, MATH-500 gave no RL headroom -> stop.
    if acc_base >= 0.85:
        print(f"\nSTOP: baseline acc={acc_base:.4f} >= 0.85 (at ceiling). Need a harder task.\n", flush=True)
        return
    print(f"[gate] baseline {acc_base:.4f} < 0.85 -> headroom confirmed.", flush=True)
    if args.baseline_only:
        print("[baseline-only] stopping after baseline.", flush=True)
        return

    results = {}
    cost_records = []  # per-variant cost rows (same hooks as experiments/cost_benchmark.py)

    # ---- Map-RL sweep ----
    for G in G_SWEEP:
        key = f"Map-G{G}"
        print(f"\n========== DIRECT MAP sweep G={G} ==========", flush=True)
        o_orig = [getattr(*get_parent(model, n)) for n in names]
        costlib.reset_peak_vram(dev)
        params, total_out = install_direct_map(model, names, G)
        n_par = sum(p.numel() for p in params)
        print(f"[{key}] trainable params = {n_par}  (TOTAL_OUT={total_out})", flush=True)
        curve, kl_curve, _, timer, tps = train_grpo(model, tok, names, train_items, params, LR_O, dev,
                                                     key, telem_fn=o_telem, clamp_o=O_CLAMP)
        cost_records.append(costlib.cost_record(
            key, params, base_params, curve, timer, dev, tps, COST_TARGET_REWARD))
        final_mean_abs_o = params[0].detach().abs().mean().item()
        final_max_gate = (1.0 + ALPHA_MOD * params[0].detach()).abs().max().item()
        k_g, n_g, cases_g = evaluate(model, tok, eval_items, dev, label=key, collect_cases=N_CASES)
        restore(model, names, o_orig)
        acc_g = k_g / n_g
        ci_g = wilson_ci(k_g, n_g)
        final_kl = sum(kl_curve[-5:]) / max(1, len(kl_curve[-5:])) if kl_curve else 0.0
        print(f"[{key}] MATH-500 acc = {acc_g:.4f} ({k_g}/{n_g})  CI [{ci_g[0]:.3f},{ci_g[1]:.3f}]  "
              f"final_mean_KL(last5)={final_kl:.4f}", flush=True)
        results[key] = dict(kind="map", n_par=n_par, curve=curve, kl_curve=kl_curve, k=k_g, n=n_g,
                            acc=acc_g, ci=ci_g, final_kl=final_kl, cases=cases_g,
                            mean_abs_o=final_mean_abs_o, max_gate=final_max_gate)

    # ---- LoRA-RL comparator — FAIR lr-sweep, report BEST (docs/research-plan.md §2) ----
    lora_variants = []  # (key, lr) in run order, for the report
    for lr_lora in LORA_LR_SWEEP:
        key = f"LoRA-r{LORA_R}-lr{lr_lora:g}"
        print(f"\n========== LoRA-RL r={LORA_R} lr={lr_lora:g} ==========", flush=True)
        o_orig = [getattr(*get_parent(model, n)) for n in names]
        costlib.reset_peak_vram(dev)
        params = install_lora(model, names, LORA_R)
        n_par = sum(p.numel() for p in params)
        print(f"[{key}] trainable params = {n_par}", flush=True)
        curve, kl_curve, _, timer, tps = train_grpo(model, tok, names, train_items, params, lr_lora, dev,
                                                    key, telem_fn=lora_telem)
        cost_records.append(costlib.cost_record(
            key, params, base_params, curve, timer, dev, tps, COST_TARGET_REWARD))
        k_l, n_l, cases_l = evaluate(model, tok, eval_items, dev, label=key, collect_cases=N_CASES)
        restore(model, names, o_orig)
        acc_l = k_l / n_l
        ci_l = wilson_ci(k_l, n_l)
        final_kl = sum(kl_curve[-5:]) / max(1, len(kl_curve[-5:])) if kl_curve else 0.0
        print(f"[{key}] MATH-500 acc = {acc_l:.4f} ({k_l}/{n_l})  "
              f"CI [{ci_l[0]:.3f},{ci_l[1]:.3f}]  final_mean_KL(last5)={final_kl:.4f}", flush=True)
        results[key] = dict(kind="lora", lr=lr_lora, n_par=n_par, curve=curve, kl_curve=kl_curve,
                            k=k_l, n=n_l, acc=acc_l, ci=ci_l, final_kl=final_kl, cases=cases_l)
        lora_variants.append(key)

    # pick BEST LoRA: highest accuracy, tie-break fewest steps-to-target then final KL.
    def lora_score(key):
        r = results[key]
        s2t = costlib.steps_to_target(r["curve"], COST_TARGET_REWARD)
        return (-r["acc"], s2t if s2t is not None else 10**9, -r["final_kl"])
    best_lora_key = min(lora_variants, key=lora_score)
    results[best_lora_key]["is_best_lora"] = True
    print(f"\n[best-LoRA] -> {best_lora_key} (acc={results[best_lora_key]['acc']:.4f})", flush=True)

    # ---- report ----
    def overlap(a, b):
        return a[0] <= b[1] and b[0] <= a[1]

    order = [f"Map-G{G}" for G in G_SWEEP] + lora_variants
    lines = ["=" * 78,
             "MATH-500 — per-group modulation vs LoRA on a frozen Qwen3-4B (RL)",
             "=" * 78,
             f"model: {args.model}",
             f"num_hidden_layers (config) = {nlayers}   target *_proj linears = {len(names)}",
             f"config: B={B} K={K} steps<={MAX_STEPS} time_box={TIME_BUDGET_S//60}min/variant "
             f"MAX_NEW={MAX_NEW} N_EVAL={N_EVAL} BETA_KL={BETA_KL} LR_o={LR_O} O_CLAMP={O_CLAMP} "
             f"LoRA_lr_sweep={LORA_LR_SWEEP} best_LoRA={best_lora_key} "
             f"ALPHA_MOD={ALPHA_MOD} dtype={dt}",
             "Map adapter: W'[c,:] = W[c,:] * (1 + alpha*o[group(c)]), o init ZEROS (identity), "
             f"HARD CLAMP |o|<={O_CLAMP}, params = G",
             f"LoRA adapter: W' = W + (alpha/r) B A, r={LORA_R}, B init zeros (identity)",
             "scorer: \\boxed{} math-equivalence (src/math_scorer.py)",
             "",
             f"baseline (no adapter): {acc_base:.4f}  ({k_base}/{n_base})   "
             f"CI [{ci_base[0]:.3f}, {ci_base[1]:.3f}]",
             "",
             f"MATH-500 greedy acc (n={N_EVAL}), Wilson 95% CI:"]
    for key in order:
        r = results[key]
        mark = "  ** BEST LoRA **" if r.get("is_best_lora") else ""
        lines.append(f"  {key:<18s} params={r['n_par']:<8d}: acc={r['acc']:.4f}  ({r['k']}/{r['n']})   "
                     f"CI [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]   final_mean_KL(last5)={r['final_kl']:.4f}{mark}")
    lines.append("")
    for key in order:
        r = results[key]
        cleared_up = r['ci'][0] > ci_base[1]
        ob = overlap(r['ci'], ci_base)
        verdict = ("CLEARS baseline CI UPWARD" if cleared_up else
                   "OVERLAP (within noise)" if ob else "DISJOINT-BELOW (regressed)")
        lines.append(f"  {key} vs baseline: {r['acc']-acc_base:+.4f}  -> {verdict}")
    lines.append("")
    lines.append("KL / coherence-band (Map variants — latent has leverage iff mean KL > ~0.05):")
    for key in order:
        r = results[key]
        if r['kind'] != 'map':
            lines.append(f"  {key}: final_mean_KL={r['final_kl']:.4f}  (LoRA, no o-band)")
            continue
        lev = ("RISES (leverage confirmed)" if r['final_kl'] > 0.05
               else "STILL ~0 (no leverage)" if r['final_kl'] < 0.01 else "partial")
        coh = ("IN-BAND (<=clamp)" if r['mean_abs_o'] <= O_CLAMP + 1e-3 else "OUT-OF-BAND")
        lines.append(f"  {key}: final_mean_KL={r['final_kl']:.4f} -> {lev}   "
                     f"final_mean|o|={r['mean_abs_o']:.4f} max_gate={r['max_gate']:.3f} -> {coh}")
    lines.append("")
    for key in order:
        r = results[key]
        lines.append(f"mean_reward curve {key} ({len(r['curve'])} steps): " +
                     " ".join(f"{x:.2f}" for x in r['curve']))
        lines.append(f"mean_KL     curve {key} ({len(r['kl_curve'])} steps): " +
                     " ".join(f"{x:.3f}" for x in r['kl_curve']))
    lines.append("")
    lines.append("DECODED CASES (baseline):")
    lines.append(fmt_cases(cases_base))
    for key in order:
        lines.append("")
        lines.append(f"DECODED CASES ({key}):")
        lines.append(fmt_cases(results[key]['cases']))
    lines.append("=" * 78)

    report = "\n".join(lines)
    print("\n" + report, flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(f"\nwrote {args.out}", flush=True)

    # ---- cost table (same hooks as experiments/cost_benchmark.py) ----
    # Mark the chosen best-LoRA row in the cost table for the fair head-to-head.
    for rec in cost_records:
        if rec["variant"] == best_lora_key:
            rec["variant"] = rec["variant"] + "  ** BEST LoRA **"
    meta = dict(label=f"4B MATH-500 ({args.model})", model=args.model,
                base_params=base_params, device=dev, target_reward=COST_TARGET_REWARD,
                max_steps=MAX_STEPS,
                tokens_per_step=int(cost_records[0]["base_flops_step"] / (6.0 * base_params))
                if base_params else 0)
    cost_table = costlib.render_cost_table(cost_records, meta)
    os.makedirs(os.path.dirname(args.cost_out) or ".", exist_ok=True)
    with open(args.cost_out, "w") as f:
        f.write(cost_table + "\n")
    print(f"wrote {args.cost_out}", flush=True)

    # ---- structured JSON (everything needed for charts + the README/cost-table edits) ----
    payload = dict(
        model=args.model, device=str(dev), dtype=str(dt), num_hidden_layers=nlayers,
        target_proj_linears=len(names), base_params=base_params,
        config=dict(B=B, K=K, max_steps=MAX_STEPS, time_budget_s=TIME_BUDGET_S,
                    max_new=MAX_NEW, n_eval=N_EVAL, beta_kl=BETA_KL, lr_o=LR_O,
                    o_clamp=O_CLAMP, lora_r=LORA_R, alpha_mod=ALPHA_MOD,
                    g_sweep=G_SWEEP, lora_lr_sweep=LORA_LR_SWEEP,
                    cost_target_reward=COST_TARGET_REWARD),
        baseline=dict(acc=acc_base, k=k_base, n=n_base, ci=list(ci_base),
                      cases=[dict(problem=q, model=t, pred=p, gold=g) for q, t, p, g in cases_base]),
        best_lora_key=best_lora_key,
        order=order,
        variants={},
        cost_records=cost_records,
    )
    for key in order:
        r = results[key]
        s2t = costlib.steps_to_target(r["curve"], COST_TARGET_REWARD)
        payload["variants"][key] = dict(
            kind=r["kind"], lr=r.get("lr"), n_par=r["n_par"], acc=r["acc"],
            k=r["k"], n=r["n"], ci=list(r["ci"]), final_kl=r["final_kl"],
            steps_to_target=s2t, is_best_lora=bool(r.get("is_best_lora")),
            mean_abs_o=r.get("mean_abs_o"), max_gate=r.get("max_gate"),
            reward_curve=r["curve"], kl_curve=r["kl_curve"],
            cases=[dict(problem=q, model=t, pred=p, gold=g) for q, t, p, g in r["cases"]],
        )
    json_out = os.path.join(os.path.dirname(args.out) or ".", "results.json")
    with open(json_out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {json_out}", flush=True)


if __name__ == "__main__":
    main()
