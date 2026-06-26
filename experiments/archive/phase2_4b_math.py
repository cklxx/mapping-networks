"""[ARCHIVED — the run that produced THE validated result] Phase 2 (4B / MATH-500):
DIRECT §5.4 modulation vs LoRA on MATH-500 RL — the GSM8K->MATH pivot.

This is the historical script behind results/4b-math500/results.txt and the README's one
validated result: on a frozen Qwen3-4B, a 2048-param modulation lifted MATH-500 accuracy
+19pp (CI clears baseline upward) and beat a 16.5M-param LoRA at a matched ~40-step/35-min
budget. The productionized, actively-maintained version is ../math500_rl.py (cleaner CLI,
cost hooks). This copy is preserved as the exact provenance of the headline number.

Why MATH-500 not GSM8K: GSM8K was at-ceiling for the 4B (phase2_4b_gsm8k.py, baseline
~86.5%, zero RL advantage variance). MATH-500 sits the 4B well below ceiling (~30%
baseline) -> real RL headroom -> the small-modulation-vs-LoRA comparison becomes meaningful.

TUNED RL: 4B coherent band ~0.10-0.15, HARD CLAMP |o|<=0.10, lr=0.01, KL leash beta=0.05.
TRAIN rollout MAX_NEW=512 (reward variance); EVAL 1024 (no truncation). Curriculum fix:
sort the train tail easy->hard so GRPO sees solvable problems -> non-zero advantage.

REFACTOR (vs the original repro script): the MATH-500 scorer is IMPORTED from
src/math_scorer.py and the DirectMap gate / LoRA / plumbing from src/adapters.py (single
source — the original inlined verbatim copies of both). Hardcoded pod paths are
--model-path / --results-path flags. Hyperparameters preserved byte-for-byte.
"""
import argparse
import glob
import math
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.adapters import (  # noqa: E402 — single source of the adapter math
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

torch.manual_seed(0)

G_SWEEP = [256, 2048]
LR_O = 0.01          # 4B-MATH has signal; between 0.8B-hot 0.02 and 4B-GSM8K-frozen 0.002
O_CLAMP = 0.10       # HARD CLAMP |o| <= 0.10 (4B coherent band ~0.10-0.15)
LORA_R = 8
LR_LORA = 1e-4       # NOTE: under-tuned — the fair head-to-head needs a lr sweep (cost_benchmark.py)
B, K = 6, 4
MAX_NEW, MAX_NEW_EVAL = 512, 1024
EVAL_BATCH = 16
N_EVAL = 200
TIME_BUDGET_S, MAX_STEPS = 35 * 60, 40
BETA_KL = 0.05
N_CASES = 3

SYS = ("Solve the math problem. Reason briefly, then put the final answer in \\boxed{}. "
       "Do not write anything after the boxed answer.")


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
    ids = torch.cat([prompt_ids, comp_ids], 0)[None].to(dev)
    logits = model(ids).logits[0, :-1].float()
    logp = torch.log_softmax(logits, -1)
    tgt = ids[0, 1:]
    comp_mask = torch.zeros_like(tgt, dtype=torch.bool)
    comp_mask[prompt_ids.numel() - 1:] = True
    tok_lp = logp.gather(1, tgt[:, None]).squeeze(1)
    sum_lp = tok_lp[comp_mask].sum()
    with torch.no_grad(), base_forward(model, names):
        base_logits = model(ids).logits[0, :-1].float()
        base_logp = torch.log_softmax(base_logits, -1)
    p = logp.exp()
    kl_per_pos = (p * (logp - base_logp)).sum(-1)
    return sum_lp, kl_per_pos[comp_mask].mean()


@torch.no_grad()
def evaluate(model, tok, items, dev, label="", collect_cases=0):
    """BATCHED greedy on a fixed test subset (left-pad)."""
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
    model.train()
    opt = torch.optim.Adam(trainable, lr=lr)
    t0 = time.time()
    curve, kl_curve = [], []
    for step in range(MAX_STEPS):
        if time.time() - t0 > TIME_BUDGET_S:
            print(f"[{label}] time budget hit at step {step}", flush=True)
            break
        batch = [train_items[(step * B + i) % len(train_items)] for i in range(B)]
        step_r, nz_groups, kl_acc, did = [], 0, [], False
        opt.zero_grad()
        for q, gold in batch:
            pids = tok(build_prompt(tok, q), return_tensors="pt").input_ids[0].to(dev)
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
                lp, kl = comp_logp_and_kl(model, names, pids, comps[k], dev)
                kl_acc.append(kl.item())
                pg = -adv[k].to(dev).detach() * lp if adv[k].abs() >= 1e-6 else 0.0 * lp
                ((pg + BETA_KL * kl) / (B * K)).backward()
                did = True
        if did:
            opt.step()
            if clamp_o is not None:
                with torch.no_grad():
                    trainable[0].clamp_(-clamp_o, clamp_o)
        mr = sum(step_r) / len(step_r)
        mkl = sum(kl_acc) / len(kl_acc) if kl_acc else 0.0
        curve.append(mr)
        kl_curve.append(mkl)
        tline = telem_fn(trainable) if telem_fn else ""
        print(f"[{label}] step {step:3d}  mean_reward={mr:.3f}  signal_groups={nz_groups}/{B}  "
              f"mean_kl={mkl:.4f}  {tline}  elapsed={time.time()-t0:.0f}s", flush=True)
    return curve, kl_curve


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


def resolve_model(model_path):
    if os.path.isdir(model_path):
        return model_path
    hits = glob.glob(model_path)
    return hits[0] if hits else model_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B",
                    help="local dir / HF id / snapshot glob (was a hardcoded pod path)")
    ap.add_argument("--results-path", default="phase2_4b_math_results.txt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--baseline-only", action="store_true")
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    snap = resolve_model(args.model_path)
    print(f"device={dev}  dtype={dt}  model={snap}", flush=True)
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=dt, trust_remote_code=True).to(dev)
    model.requires_grad_(False)
    nlayers = num_layers_of(model)
    print(f"num_hidden_layers (from config) = {nlayers}", flush=True)

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    all_items = [(r["problem"], gold_answer(r)) for r in ds]
    eval_items = all_items[:N_EVAL]
    # Curriculum fix: sort the TRAIN tail easy->hard so GRPO sees solvable problems per batch
    # -> non-zero advantage. Eval is untouched, so the A/B verdict stays unbiased.
    train_records = [r for r in ds][N_EVAL:] if len(ds) > N_EVAL else [r for r in ds]
    train_records.sort(key=lambda r: int(r.get("level") or 5))
    train_pool = [(r["problem"], gold_answer(r)) for r in train_records]
    train_items = [train_pool[i % len(train_pool)] for i in range(B * MAX_STEPS + 64)]
    print(f"train pool: {len(train_pool)} problems, sorted easy->hard (curriculum)", flush=True)

    names = target_modules(model)
    print(f"target projection linears: {len(names)}  (all {nlayers} layers)", flush=True)

    k_base, n_base, cases_base = evaluate(model, tok, eval_items, dev, label="base", collect_cases=N_CASES)
    acc_base = k_base / n_base
    ci_base = wilson_ci(k_base, n_base)
    print(f"[baseline] MATH-500 acc = {acc_base:.4f} ({k_base}/{n_base})  "
          f"CI [{ci_base[0]:.3f},{ci_base[1]:.3f}]", flush=True)

    if acc_base >= 0.85:
        print(f"\nSTOP: baseline acc={acc_base:.4f} >= 0.85 (still at ceiling).\n", flush=True)
        return
    print(f"[gate] baseline {acc_base:.4f} < 0.85 -> headroom confirmed.", flush=True)
    if args.baseline_only:
        print("[baseline-only] stopping after baseline.", flush=True)
        return

    results = {}
    for G in G_SWEEP:
        key = f"Map-G{G}"
        print(f"\n========== DIRECT MAP sweep G={G} ==========", flush=True)
        o_orig = [getattr(*get_parent(model, n)) for n in names]
        params, total_out = install_direct_map(model, names, G)
        n_par = sum(p.numel() for p in params)
        print(f"[{key}] trainable params = {n_par}  (TOTAL_OUT={total_out})", flush=True)
        curve, kl_curve = train_grpo(model, tok, names, train_items, params, LR_O, dev, key,
                                     telem_fn=o_telem, clamp_o=O_CLAMP)
        final_mean_abs_o = params[0].detach().abs().mean().item()
        final_max_gate = (1.0 + ALPHA_MOD * params[0].detach()).abs().max().item()
        k_g, n_g, cases_g = evaluate(model, tok, eval_items, dev, label=key, collect_cases=N_CASES)
        restore(model, names, o_orig)
        acc_g = k_g / n_g
        ci_g = wilson_ci(k_g, n_g)
        final_kl = sum(kl_curve[-5:]) / max(1, len(kl_curve[-5:])) if kl_curve else 0.0
        print(f"[{key}] MATH-500 acc = {acc_g:.4f} ({k_g}/{n_g})  CI [{ci_g[0]:.3f},{ci_g[1]:.3f}]  "
              f"final_mean_KL(last5)={final_kl:.4f}", flush=True)
        results[key] = dict(kind="map", n_par=n_par, k=k_g, n=n_g, acc=acc_g, ci=ci_g, final_kl=final_kl,
                            cases=cases_g, mean_abs_o=final_mean_abs_o, max_gate=final_max_gate)

    print(f"\n========== LoRA-RL r={LORA_R} ==========", flush=True)
    o_orig = [getattr(*get_parent(model, n)) for n in names]
    params = install_lora(model, names, LORA_R)
    n_par = sum(p.numel() for p in params)
    print(f"[LoRA-r{LORA_R}] trainable params = {n_par}", flush=True)
    curve, kl_curve = train_grpo(model, tok, names, train_items, params, LR_LORA, dev,
                                 f"LoRA-r{LORA_R}", telem_fn=lora_telem)
    k_l, n_l, cases_l = evaluate(model, tok, eval_items, dev, label=f"LoRA-r{LORA_R}", collect_cases=N_CASES)
    restore(model, names, o_orig)
    acc_l = k_l / n_l
    ci_l = wilson_ci(k_l, n_l)
    final_kl = sum(kl_curve[-5:]) / max(1, len(kl_curve[-5:])) if kl_curve else 0.0
    print(f"[LoRA-r{LORA_R}] MATH-500 acc = {acc_l:.4f} ({k_l}/{n_l})  CI [{ci_l[0]:.3f},{ci_l[1]:.3f}]  "
          f"final_mean_KL(last5)={final_kl:.4f}", flush=True)
    results[f"LoRA-r{LORA_R}"] = dict(kind="lora", n_par=n_par, k=k_l, n=n_l, acc=acc_l, ci=ci_l,
                                      final_kl=final_kl, cases=cases_l)

    def overlap(a, b):
        return a[0] <= b[1] and b[0] <= a[1]
    order = [f"Map-G{G}" for G in G_SWEEP] + [f"LoRA-r{LORA_R}"]
    lines = ["=" * 78,
             "PHASE 2 (4B) MATH-500 — DIRECT §5.4 modulation vs LoRA on MATH-500 RL (GSM8K pivot)",
             "=" * 78,
             f"base: {snap}   num_hidden_layers={nlayers}   target *_proj linears={len(names)}",
             f"config: B={B} K={K} steps<={MAX_STEPS} time_box={TIME_BUDGET_S//60}min/variant MAX_NEW={MAX_NEW} "
             f"N_EVAL={N_EVAL} BETA_KL={BETA_KL} LR_o={LR_O} O_CLAMP={O_CLAMP} LR_lora={LR_LORA} "
             f"ALPHA_MOD={ALPHA_MOD} dtype={dt}",
             "Map adapter: W'[c,:]=W[c,:]*(1+alpha*o[group(c)]), o init ZEROS, HARD CLAMP, params=G (src/adapters.py)",
             "scorer: \\boxed{} math-equivalence (src/math_scorer.py)",
             "",
             f"baseline: {acc_base:.4f}  ({k_base}/{n_base})   CI [{ci_base[0]:.3f}, {ci_base[1]:.3f}]",
             "",
             f"MATH-500 greedy acc (n={N_EVAL}), Wilson 95% CI:"]
    for key in order:
        r = results[key]
        lines.append(f"  {key:<10s} params={r['n_par']:<8d}: acc={r['acc']:.4f}  ({r['k']}/{r['n']})   "
                     f"CI [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]   final_mean_KL(last5)={r['final_kl']:.4f}")
    lines.append("")
    for key in order:
        r = results[key]
        cleared_up = r['ci'][0] > ci_base[1]
        ob = overlap(r['ci'], ci_base)
        verdict = ("CLEARS baseline CI UPWARD" if cleared_up else
                   "OVERLAP (within noise)" if ob else "DISJOINT-BELOW (regressed)")
        lines.append(f"  {key} vs baseline: {r['acc']-acc_base:+.4f}  -> {verdict}")
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
    with open(args.results_path, "w") as f:
        f.write(report + "\n")
    print(f"\nwrote {args.results_path}", flush=True)


if __name__ == "__main__":
    main()
