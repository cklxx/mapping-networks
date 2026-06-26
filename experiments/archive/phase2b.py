"""[ARCHIVED — negative result] Phase 2b (0.8B / GSM8K): DIRECT high-leverage §5.4
modulation vs baseline on GSM8K RL.

RESULT (negative, see phase2b_results.txt + phase2b_coherent_probe.txt): the DIRECT gate
W'[c,:]=W[c,:]*(1+o[group(c)]) now had REAL RL leverage (KL rose 0.002 -> 0.34 at G=256,
0.71 at G=2048) yet accuracy fell DISJOINT-BELOW baseline (0.42 -> 0.245 / 0.235). The
coherent probe (phase2b_coherent_probe.py) then showed accuracy declining MONOTONICALLY
from identity outward at every scale, including fully-coherent low-|o| points — so it was
NOT an over-driven end-state artifact. Two confounds made 0.8B/GSM8K the wrong testbed:
(i) GSM8K is at-ceiling for the 0.8B (~42% baseline, little headroom), and (ii) the 0.8B's
coherent band is very thin. The 4B + MATH-500 pivot removed both and flipped to the clean
+19pp win (../math500_rl.py). Preserved as the evidence behind the README's "what didn't
work" paragraph; the negative was a model/task limit, not the idea's limit.

This re-tested a DIRECT, high-leverage modulation (no frozen random map, no tanh, alpha=1.0,
o init ZEROS = identity) after phase2.py's frozen-random-map variant gave a null leverage
result — do not kill the hypothesis on a weak adapter. A HARD CLAMP projects o into
[-O_CLAMP, O_CLAMP] after each step (coherent by construction, collapse impossible).

REFACTOR (vs the original repro script): the DirectMap gate, install/restore plumbing,
base_forward KL context, and LoRA are now IMPORTED from src/adapters.py (single source of
the adapter math — byte-identical to the original). Hardcoded pod paths are --model-path /
--results-path flags. The GSM8K scorer (#### regex) stays local (it is the 0.8B-specific
brevity-prompt extractor, not the MATH scorer in src/). Every run-defining hyperparameter
is PRESERVED.
"""
import argparse
import glob
import math
import os
import re
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
    restore,
    target_modules,
)

torch.manual_seed(0)

G_SWEEP = [256, 2048]
O_CLAMP = 0.10        # per-element |o| ceiling -> gate in [0.90, 1.10], measured-coherent
LR_O = 0.02          # healthy: reaches the clamp band in ~5-10 steps; clamp prevents runaway
B, K = 8, 4
MAX_NEW, MAX_NEW_EVAL = 256, 256
N_EVAL = 200
TIME_BUDGET_S, MAX_STEPS = 35 * 60, 50
BETA_KL = 0.02       # LIGHT leash: the clamp (not the leash) bounds the band
N_CASES = 3

SYS = ("Solve the math problem. Think briefly in at most 3 short steps, then output the final "
       "answer as a single line: #### <number>. Keep it under 120 words. Do NOT use markdown headings.")
# hardened '####' regex: require end/newline/non-'.<letter>' after to dodge '#### 4. Conclusion'
ANS_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)(?!\.\s*[A-Za-z])")


def build_prompt(tok, q):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def gold_of(answer):
    return answer.split("####")[-1].strip().replace(",", "")


def pred_of(text):
    ms = ANS_RE.findall(text)
    return ms[-1].replace(",", "") if ms else None


def reward_of(text, gold):
    return 1.0 if pred_of(text) == gold else 0.0


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
    model.eval()
    correct = 0
    t0 = time.time()
    cases = []
    for i, (q, gold) in enumerate(items):
        ids = tok(build_prompt(tok, q), return_tensors="pt").input_ids.to(dev)
        out = model.generate(ids, do_sample=False, max_new_tokens=MAX_NEW_EVAL,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        pred = pred_of(text)
        correct += int(pred == gold)
        if len(cases) < collect_cases:
            cases.append((q, text, pred, gold))
        if (i + 1) % 25 == 0:
            print(f"  [eval {label}] {i+1}/{len(items)}  acc_sofar={correct/(i+1):.3f}  "
                  f"{(time.time()-t0)/(i+1):.1f}s/q", flush=True)
    return correct, len(items), cases


def train_grpo(model, tok, names, train_items, trainable, lr, dev, label):
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
            with torch.no_grad():
                trainable[0].clamp_(-O_CLAMP, O_CLAMP)  # project o into the coherent band
        mr = sum(step_r) / len(step_r)
        mkl = sum(kl_acc) / len(kl_acc) if kl_acc else 0.0
        curve.append(mr)
        kl_curve.append(mkl)
        o = trainable[0].detach()
        print(f"[{label}] step {step:3d}  mean_reward={mr:.3f}  signal_groups={nz_groups}/{B}  "
              f"mean_kl={mkl:.4f}  mean|o|={o.abs().mean().item():.4f}  "
              f"max_gate={(1.0 + ALPHA_MOD * o).abs().max().item():.3f}  "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
    return curve, kl_curve


def fmt_cases(cases):
    out = []
    for j, (q, text, pred, gold) in enumerate(cases):
        out.append(f"  --- case {j+1} ---")
        out.append(f"  Q: {q.strip()[:400]}")
        out.append(f"  MODEL: {text.strip()[:900]}")
        out.append(f"  extracted={pred!r}  gold={gold!r}  {'CORRECT' if pred==gold else 'WRONG'}")
    return "\n".join(out)


def resolve_model(model_path):
    """A local path, or a glob (the original used a pod HF-cache snapshot glob)."""
    if os.path.isdir(model_path):
        return model_path
    hits = glob.glob(model_path)
    if hits:
        return hits[0]
    return model_path  # let HF resolve it as a hub id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3.5-0.8B",
                    help="local dir / HF id / snapshot glob (was a hardcoded pod path)")
    ap.add_argument("--results-path", default="phase2b_results.txt",
                    help="where to write the report (was a hardcoded /host path)")
    ap.add_argument("--device", default=None, help="cuda / cpu / mps (default: auto)")
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    snap = resolve_model(args.model_path)
    print(f"device={dev}  dtype={dt}  model={snap}", flush=True)
    print(f"config: B={B} K={K} MAX_STEPS={MAX_STEPS} time_box={TIME_BUDGET_S}s "
          f"MAX_NEW={MAX_NEW} N_EVAL={N_EVAL} BETA_KL={BETA_KL} LR_O={LR_O} ALPHA_MOD={ALPHA_MOD} "
          f"O_CLAMP={O_CLAMP} G_SWEEP={G_SWEEP}", flush=True)
    tok = AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=dt, trust_remote_code=True).to(dev)
    model.requires_grad_(False)

    ds = load_dataset("openai/gsm8k", "main")
    train_items = [(r["question"], gold_of(r["answer"])) for r in ds["train"].select(range(B * MAX_STEPS + 64))]
    eval_items = [(r["question"], gold_of(r["answer"])) for r in ds["test"].select(range(N_EVAL))]

    names = target_modules(model)
    print(f"target projection linears: {len(names)}", flush=True)

    k_base, n_base, cases_base = evaluate(model, tok, eval_items, dev, label="base", collect_cases=N_CASES)
    acc_base = k_base / n_base
    ci_base = wilson_ci(k_base, n_base)
    print(f"[baseline] GSM8K acc = {acc_base:.4f} ({k_base}/{n_base})  "
          f"CI [{ci_base[0]:.3f},{ci_base[1]:.3f}]", flush=True)

    results = {}
    for G in G_SWEEP:
        print(f"\n========== DIRECT MAP sweep G={G} ==========", flush=True)
        o_orig = [getattr(*get_parent(model, n)) for n in names]
        params, total_out = install_direct_map(model, names, G)
        n_par = sum(p.numel() for p in params)
        print(f"[G={G}] trainable params = {n_par}  (TOTAL_OUT={total_out})", flush=True)
        curve, kl_curve = train_grpo(model, tok, names, train_items, params, LR_O, dev, f"Map-G{G}")
        final_mean_abs_o = params[0].detach().abs().mean().item()
        final_max_gate = (1.0 + ALPHA_MOD * params[0].detach()).abs().max().item()
        k_g, n_g, cases_g = evaluate(model, tok, eval_items, dev, label=f"Map-G{G}", collect_cases=N_CASES)
        restore(model, names, o_orig)
        acc_g = k_g / n_g
        ci_g = wilson_ci(k_g, n_g)
        final_kl = sum(kl_curve[-5:]) / max(1, len(kl_curve[-5:])) if kl_curve else 0.0
        print(f"[Map-G{G}] GSM8K acc = {acc_g:.4f} ({k_g}/{n_g})  CI [{ci_g[0]:.3f},{ci_g[1]:.3f}]  "
              f"final_mean_kl(last5)={final_kl:.4f}", flush=True)
        results[G] = dict(n_par=n_par, curve=curve, kl_curve=kl_curve, k=k_g, n=n_g, acc=acc_g,
                          ci=ci_g, final_kl=final_kl, cases=cases_g,
                          mean_abs_o=final_mean_abs_o, max_gate=final_max_gate)

    def overlap(a, b):
        return a[0] <= b[1] and b[0] <= a[1]
    lines = ["=" * 78, "PHASE 2b (0.8B/GSM8K) — DIRECT high-leverage §5.4 modulation on GSM8K RL", "=" * 78,
             f"config: B={B} K={K} steps<={MAX_STEPS} time_box={TIME_BUDGET_S//60}min/G MAX_NEW={MAX_NEW} "
             f"N_EVAL={N_EVAL} BETA_KL={BETA_KL} LR_o={LR_O} ALPHA_MOD={ALPHA_MOD} O_CLAMP={O_CLAMP} dtype={dt}",
             "adapter: W'[c,:]=W[c,:]*(1+alpha*o[group(c)]), o init ZEROS (identity), params=G (from src/adapters.py)",
             "",
             f"baseline: {acc_base:.4f}  ({k_base}/{n_base})   CI [{ci_base[0]:.3f}, {ci_base[1]:.3f}]",
             "",
             f"GSM8K greedy acc (n={N_EVAL}), Wilson 95% CI:"]
    for G in G_SWEEP:
        r = results[G]
        lines.append(f"  G={G:<5d} params={r['n_par']:<6d}: acc={r['acc']:.4f}  ({r['k']}/{r['n']})   "
                     f"CI [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]   final_mean_KL(last5)={r['final_kl']:.4f}")
    lines.append("")
    for G in G_SWEEP:
        r = results[G]
        ob = overlap(r['ci'], ci_base)
        lines.append(f"  G={G} vs baseline: {r['acc']-acc_base:+.4f}  -> CIs "
                     f"{'OVERLAP (within noise)' if ob else 'DISJOINT (clears noise)'}")
    lines.append("")
    lines.append("DECODED CASES (baseline):")
    lines.append(fmt_cases(cases_base))
    for G in G_SWEEP:
        lines.append("")
        lines.append(f"DECODED CASES (Map-G{G}):")
        lines.append(fmt_cases(results[G]['cases']))
    lines.append("=" * 78)

    report = "\n".join(lines)
    print("\n" + report, flush=True)
    with open(args.results_path, "w") as f:
        f.write(report + "\n")
    print(f"\nwrote {args.results_path}", flush=True)


if __name__ == "__main__":
    main()
