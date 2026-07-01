"""L4-optimized 8B experiment: B=1 K=3, shorter rollouts, gradient checkpointing."""
import argparse, json, math, os, sys, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/content")
from src.adapters import (
    ALPHA_MOD, base_forward, get_parent, install_direct_map,
    install_lora, num_layers_of, restore, target_modules,
)
from src.math_scorer import extract_answer, gold_answer, reward_of
from src import costlib

torch.manual_seed(0)
COST_TARGET_REWARD = 0.20
G_SWEEP = [256, 2048]
LR_O = 0.005
O_CLAMP = 0.10
LORA_R = 8
LORA_LR_SWEEP = [1e-4, 1e-3]
B, K = 1, 3                      # tiny batch for L4 22GB
MAX_NEW, MAX_NEW_EVAL = 256, 512 # shorter rollouts
EVAL_BATCH = 4
N_EVAL = 200
TIME_BUDGET_S, MAX_STEPS = 6 * 3600, 350
BETA_KL = 0.05
N_CASES = 3

SYS = "Solve the math problem. Reason briefly, then put the final answer in \\boxed{}. Do not write anything after the boxed answer."

def build_prompt(tok, q):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)

def comp_logp_and_kl(model, names, prompt_ids, comp_ids, dev):
    """Base forward FIRST (no grad), then policy forward. This avoids holding both graphs."""
    ids = torch.cat([prompt_ids, comp_ids], 0)[None].to(dev)
    # Base forward first (no grad, lighter)
    with torch.no_grad(), base_forward(model, names):
        base_logits = model(ids).logits[0, :-1].float()
        base_logp = torch.log_softmax(base_logits, -1)
    
    # Policy forward (with grad)
    logits = model(ids).logits[0, :-1].float()
    logp = torch.log_softmax(logits, -1)
    
    tgt = ids[0, 1:]
    n_prompt = prompt_ids.numel()
    comp_mask = torch.zeros_like(tgt, dtype=torch.bool)
    comp_mask[n_prompt - 1:] = True
    tok_lp = logp.gather(1, tgt[:, None]).squeeze(1)
    sum_lp = tok_lp[comp_mask].sum()
    
    p = logp.exp()
    kl_per_pos = (p * (logp - base_logp)).sum(-1)
    kl = kl_per_pos[comp_mask].mean()
    return sum_lp, kl

@torch.no_grad()
def evaluate(model, tok, items, dev, label="", collect_cases=0):
    model.eval()
    correct, cases = 0, []
    t0 = time.time()
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
        print(f"  [eval {label}] {done}/{len(items)}  acc_sofar={correct/max(1,done):.3f}", flush=True)
    tok.padding_side = prev_side
    return correct, len(items), cases

def train_grpo(model, tok, names, train_items, trainable, lr, dev, label, telem_fn=None, clamp_o=None):
    model.train()
    opt = torch.optim.Adam(trainable, lr=lr)
    t0 = time.time()
    curve, kl_curve = [], []
    timer = costlib.StepTimer().start()
    tok_acc = []
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
            if rs.std() > 1e-6: nz_groups += 1
            adv = (rs - rs.mean()) / (rs.std() + 1e-4)
            for k in range(K):
                step_tokens += pids.numel() + comps[k].numel()
                lp, kl = comp_logp_and_kl(model, names, pids, comps[k], dev)
                kl_acc.append(kl.item())
                pg = -adv[k].to(dev).detach() * lp if adv[k].abs() >= 1e-6 else 0.0 * lp
                (pg + BETA_KL * kl).backward()
                did_backward = True
        if did_backward:
            opt.step()
            if clamp_o is not None:
                with torch.no_grad(): trainable[0].clamp_(-clamp_o, clamp_o)
        mr = sum(step_r) / len(step_r)
        mkl = sum(kl_acc) / len(kl_acc) if kl_acc else 0.0
        curve.append(mr); kl_curve.append(mkl)
        tok_acc.append(step_tokens)
        timer.tick()
        tline = telem_fn(trainable) if telem_fn else ""
        if step % 20 == 0 or step < 5:
            print(f"[{label}] step {step:3d}  reward={mr:.3f}  nz={nz_groups}/{B}  KL={mkl:.4f}  "
                  f"{tline}  step_s={timer.per_step[-1]:.1f}  elapsed={time.time()-t0:.0f}s", flush=True)
    tokens_per_step = int(sum(tok_acc) / len(tok_acc)) if tok_acc else 0
    return curve, kl_curve, timer, tokens_per_step

def o_telem(trainable):
    o = trainable[0].detach()
    return f"mean|o|={o.abs().mean().item():.4f} max_gate={(1.0+ALPHA_MOD*o).abs().max().item():.3f}"

def lora_telem(trainable):
    mag = torch.stack([p.detach().abs().mean() for p in trainable]).mean().item()
    return f"mean|AB|={mag:.4f}"

def fmt_cases(cases):
    out = []
    for j, (q, text, pred, gold) in enumerate(cases):
        out.append(f"  --- case {j+1} ---")
        out.append(f"  PROBLEM: {q.strip()[:400]}")
        out.append(f"  MODEL: {text.strip()[:1100]}")
        out.append(f"  pred={pred!r} gold={gold!r} {'OK' if (pred==gold and gold) else 'WRONG'}")
    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default="results/8b-math500/results.txt")
    ap.add_argument("--cost-out", default="results/cost-table-8b.md")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device
    dt = torch.bfloat16
    print(f"device={dev} dtype={dt} model={args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt, trust_remote_code=True).to(dev)
    model.requires_grad_(False)
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    nlayers = num_layers_of(model)
    base_params = sum(p.numel() for p in model.parameters())
    print(f"num_hidden_layers={nlayers} base_params={base_params:,} grad_ckpt=True", flush=True)
    print(f"config: B={B} K={K} MAX_STEPS={MAX_STEPS} MAX_NEW={MAX_NEW} MAX_NEW_EVAL={MAX_NEW_EVAL} "
          f"LR_O={LR_O} O_CLAMP={O_CLAMP} LORA_R={LORA_R}", flush=True)

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    all_items = [(r["problem"], gold_answer(r)) for r in ds]
    eval_items = all_items[:N_EVAL]
    train_records = [r for r in ds][N_EVAL:]
    train_records.sort(key=lambda r: int(r.get("level") or 5))
    train_pool = [(r["problem"], gold_answer(r)) for r in train_records]
    train_items = [train_pool[i % len(train_pool)] for i in range(B * MAX_STEPS + 64)]
    print(f"train pool: {len(train_pool)} problems", flush=True)

    names = target_modules(model)
    print(f"target *_proj linears: {len(names)}", flush=True)

    k_base, n_base, cases_base = evaluate(model, tok, eval_items, dev, label="base", collect_cases=N_CASES)
    acc_base = k_base / n_base
    ci_base = wilson_ci(k_base, n_base)
    print(f"[baseline] MATH-500 acc = {acc_base:.4f} ({k_base}/{n_base}) CI [{ci_base[0]:.3f},{ci_base[1]:.3f}]", flush=True)
    if acc_base >= 0.85: return

    results = {}
    cost_records = []

    for G in G_SWEEP:
        key = f"Map-G{G}"
        print(f"\n{'='*50}\nDIRECT MAP G={G}\n{'='*50}", flush=True)
        o_orig = [getattr(*get_parent(model, n)) for n in names]
        costlib.reset_peak_vram(dev)
        params, total_out = install_direct_map(model, names, G)
        n_par = sum(p.numel() for p in params)
        print(f"[{key}] trainable params = {n_par}", flush=True)
        curve, kl_curve, timer, tps = train_grpo(model, tok, names, train_items, params, LR_O, dev, key, o_telem, O_CLAMP)
        cost_records.append(costlib.cost_record(key, params, base_params, curve, timer, dev, tps, COST_TARGET_REWARD))
        final_mao = params[0].detach().abs().mean().item()
        final_mg = (1.0 + ALPHA_MOD * params[0].detach()).abs().max().item()
        k_g, n_g, cases_g = evaluate(model, tok, eval_items, dev, label=key, collect_cases=N_CASES)
        restore(model, names, o_orig)
        acc_g = k_g / n_g
        ci_g = wilson_ci(k_g, n_g)
        fkl = sum(kl_curve[-5:]) / max(1, len(kl_curve[-5:])) if kl_curve else 0.0
        print(f"[{key}] acc={acc_g:.4f} ({k_g}/{n_g}) CI [{ci_g[0]:.3f},{ci_g[1]:.3f}] KL={fkl:.4f}", flush=True)
        results[key] = dict(kind="map", n_par=n_par, curve=curve, kl_curve=kl_curve, k=k_g, n=n_g,
                            acc=acc_g, ci=ci_g, final_kl=fkl, cases=cases_g,
                            mean_abs_o=final_mao, max_gate=final_mg)

    lora_variants = []
    for lr_lora in LORA_LR_SWEEP:
        key = f"LoRA-r{LORA_R}-lr{lr_lora:g}"
        print(f"\n{'='*50}\nLoRA r={LORA_R} lr={lr_lora:g}\n{'='*50}", flush=True)
        o_orig = [getattr(*get_parent(model, n)) for n in names]
        costlib.reset_peak_vram(dev)
        params = install_lora(model, names, LORA_R)
        n_par = sum(p.numel() for p in params)
        print(f"[{key}] trainable params = {n_par}", flush=True)
        curve, kl_curve, timer, tps = train_grpo(model, tok, names, train_items, params, lr_lora, dev, key, lora_telem)
        cost_records.append(costlib.cost_record(key, params, base_params, curve, timer, dev, tps, COST_TARGET_REWARD))
        k_l, n_l, cases_l = evaluate(model, tok, eval_items, dev, label=key, collect_cases=N_CASES)
        restore(model, names, o_orig)
        acc_l = k_l / n_l
        ci_l = wilson_ci(k_l, n_l)
        fkl = sum(kl_curve[-5:]) / max(1, len(kl_curve[-5:])) if kl_curve else 0.0
        print(f"[{key}] acc={acc_l:.4f} ({k_l}/{n_l}) CI [{ci_l[0]:.3f},{ci_l[1]:.3f}] KL={fkl:.4f}", flush=True)
        results[key] = dict(kind="lora", lr=lr_lora, n_par=n_par, curve=curve, kl_curve=kl_curve,
                            k=k_l, n=n_l, acc=acc_l, ci=ci_l, final_kl=fkl, cases=cases_l)
        lora_variants.append(key)

    def lora_score(k):
        r = results[k]
        s2t = costlib.steps_to_target(r["curve"], COST_TARGET_REWARD)
        return (-r["acc"], s2t if s2t else 10**9, -r["final_kl"])
    best_lora_key = min(lora_variants, key=lora_score)
    results[best_lora_key]["is_best_lora"] = True
    print(f"\n[best-LoRA] -> {best_lora_key} (acc={results[best_lora_key]['acc']:.4f})", flush=True)

    def overlap(a, b): return a[0] <= b[1] and b[0] <= a[1]
    order = [f"Map-G{G}" for G in G_SWEEP] + lora_variants
    lines = ["=" * 78,
             "MATH-500 — modulation vs LoRA on frozen Qwen3-8B (L4, 350 steps RL)",
             "=" * 78,
             f"model: {args.model}",
             f"num_hidden_layers={nlayers} target linears={len(names)}",
             f"config: B={B} K={K} MAX_STEPS={MAX_STEPS} MAX_NEW={MAX_NEW} MAX_NEW_EVAL={MAX_NEW_EVAL} "
             f"LR_o={LR_O} O_CLAMP={O_CLAMP} LORA_R={LORA_R} best_LoRA={best_lora_key}",
             f"baseline: {acc_base:.4f} ({k_base}/{n_base}) CI [{ci_base[0]:.3f}, {ci_base[1]:.3f}]",
             "",
             f"MATH-500 greedy acc (n={N_EVAL}), Wilson 95% CI:"]
    for key in order:
        r = results[key]
        mark = "  ** BEST LoRA **" if r.get("is_best_lora") else ""
        lines.append(f"  {key:<22s} params={r['n_par']:<10d}: acc={r['acc']:.4f} ({r['k']}/{r['n']}) "
                     f"CI [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] KL={r['final_kl']:.4f}{mark}")
    lines.append("")
    for key in order:
        r = results[key]
        cl = r['ci'][0] > ci_base[1]
        ob = overlap(r['ci'], ci_base)
        v = ("CLEARS baseline" if cl else "OVERLAP" if ob else "BELOW")
        lines.append(f"  {key} vs baseline: {r['acc']-acc_base:+.4f} -> {v}")
    lines.append("")
    for key in order:
        r = results[key]
        if r['kind'] == 'map':
            lev = "RISES" if r['final_kl'] > 0.05 else ("partial" if r['final_kl'] > 0.01 else "~0")
            coh = "IN-BAND" if r['mean_abs_o'] <= O_CLAMP + 1e-3 else "OUT-OF-BAND"
            lines.append(f"  {key}: KL={r['final_kl']:.4f} -> {lev} mean|o|={r['mean_abs_o']:.4f} max_gate={r['max_gate']:.3f} -> {coh}")
        else:
            lines.append(f"  {key}: KL={r['final_kl']:.4f} (LoRA)")
    lines.append("")
    for key in order:
        r = results[key]
        lines.append(f"mean_reward {key} ({len(r['curve'])} steps): " + " ".join(f"{x:.2f}" for x in r['curve']))
        lines.append(f"mean_KL     {key} ({len(r['kl_curve'])} steps): " + " ".join(f"{x:.3f}" for x in r['kl_curve']))
    lines.append("\nDECODED CASES (baseline):\n" + fmt_cases(cases_base))
    for key in order:
        lines.append(f"\nDECODED CASES ({key}):\n" + fmt_cases(results[key]['cases']))

    report = "\n".join(lines)
    print("\n" + report, flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f: f.write(report + "\n")
    print(f"wrote {args.out}", flush=True)

    for rec in cost_records:
        if rec["variant"] == best_lora_key:
            rec["variant"] = rec["variant"] + "  ** BEST LoRA **"
    tps_val = int(cost_records[0]["base_flops_step"] / (6.0 * base_params)) if base_params and cost_records else 0
    meta = dict(label=f"8B MATH-500 on L4", model=args.model, base_params=base_params, device=dev,
                target_reward=COST_TARGET_REWARD, max_steps=MAX_STEPS, tokens_per_step=tps_val)
    cost_table = costlib.render_cost_table(cost_records, meta)
    os.makedirs(os.path.dirname(args.cost_out) or ".", exist_ok=True)
    with open(args.cost_out, "w") as f: f.write(cost_table + "\n")
    print(f"wrote {args.cost_out}", flush=True)

    payload = dict(
        model=args.model, device=str(dev), dtype=str(dt), num_hidden_layers=nlayers,
        target_proj_linears=len(names), base_params=base_params,
        config=dict(B=B, K=K, max_steps=MAX_STEPS, time_budget_s=TIME_BUDGET_S,
                    max_new=MAX_NEW, n_eval=N_EVAL, beta_kl=BETA_KL, lr_o=LR_O,
                    o_clamp=O_CLAMP, lora_r=LORA_R, g_sweep=G_SWEEP, lora_lr_sweep=LORA_LR_SWEEP),
        baseline=dict(acc=acc_base, k=k_base, n=n_base, ci=list(ci_base),
                      cases=[dict(problem=q, model=t, pred=p, gold=g) for q, t, p, g in cases_base]),
        best_lora_key=best_lora_key, order=order, cost_records=cost_records, variants={},
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
    with open(json_out, "w") as f: json.dump(payload, f, indent=2)
    print(f"wrote {json_out}", flush=True)

if __name__ == "__main__":
    main()
