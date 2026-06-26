"""[ARCHIVED — negative result, isolates the base-size variable] Phase 2 (4B / GSM8K):
isolate base model size 0.8B -> 4B, DIRECT §5.4 modulation vs LoRA on GSM8K RL.

The ONLY intended variable vs phase2b.py is the base model: Qwen3-4B instead of
Qwen3.5-0.8B. Everything else (DIRECT trainable o, gate=1+alpha*o, init zeros, GRPO B=8/K=4,
brevity prompt, #### regex, n=200 Wilson-CI eval) is preserved.

RESULT (negative for GSM8K — leads to the MATH-500 pivot): the 4B is AT-CEILING on GSM8K
(baseline ~86.5%), so RL had ZERO advantage variance (signal_groups 0-3/8) and the
small-modulation-vs-LoRA comparison was meaningless. This is exactly what motivated the
pivot to MATH-500 (phase2_4b_math.py -> the validated +19pp win in ../math500_rl.py): a
task where the 4B sits well below ceiling, giving RL real headroom. Also includes a cheap
--probe coherence mode: set a uniform-direction o at magnitudes {0.05,0.10,0.20,0.40} and
report at what mean|o| the 4B stays coherent (the 4B band is ~0.10-0.15, wider than the
0.8B's ~0.05).

REFACTOR (vs the original repro script): DirectMap gate / LoRA / install plumbing / KL
context are IMPORTED from src/adapters.py. Hardcoded pod paths are --model-path /
--results-path flags. GSM8K scorer stays local. Hyperparameters preserved.
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
    num_layers_of,
    restore,
    target_modules,
)

torch.manual_seed(0)

G_SWEEP = [256, 2048]
LR_O = 0.002         # tamed (0.8B: 0.05 -> collapse)
LORA_R = 8
LR_LORA = 1e-4
B, K = 8, 4
MAX_NEW, MAX_NEW_EVAL = 512, 512   # 4B uses <think>; 256 truncates the answer on harder Qs
N_EVAL = 200
TIME_BUDGET_S, MAX_STEPS = 35 * 60, 40
BETA_KL = 0.1        # FIRM leash: hold KL in the coherent band (~0.05-0.15)
N_CASES = 3

SYS = ("Solve the math problem. Think briefly in at most 3 short steps, then output the final "
       "answer as a single line: #### <number>. Keep it under 120 words. Do NOT use markdown headings.")
ANS_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)(?!\.\s*[A-Za-z])")
# 4B-format-robust fallback: Qwen3-4B reasons correctly but ignores '####' (writes $18,
# 'X = 18', a trailing number). Extract AFTER </think>, preferring an explicit marker.
ANS_MARK_RE = re.compile(r"(?:####|answer\s*[:=]?|\bis\b|=|\$)\s*\**\s*(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
NUM_RE = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)")


def build_prompt(tok, q):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def gold_of(answer):
    return answer.split("####")[-1].strip().replace(",", "")


def pred_of(text):
    ms = ANS_RE.findall(text)
    if ms:
        return ms[-1].replace(",", "")
    span = text.split("</think>")[-1] if "</think>" in text else text
    mk = ANS_MARK_RE.findall(span)
    if mk:
        return mk[-1].replace(",", "")
    nm = NUM_RE.findall(span)
    return nm[-1].replace(",", "") if nm else None


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


def train_grpo(model, tok, names, train_items, trainable, lr, dev, label, telem_fn=None):
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
        out.append(f"  Q: {q.strip()[:400]}")
        out.append(f"  MODEL: {text.strip()[:900]}")
        out.append(f"  extracted={pred!r}  gold={gold!r}  {'CORRECT' if pred==gold else 'WRONG'}")
    return "\n".join(out)


def resolve_model(model_path):
    if os.path.isdir(model_path):
        return model_path
    hits = glob.glob(model_path)
    return hits[0] if hits else model_path


def run_probe(model, tok, dev):
    ds = load_dataset("openai/gsm8k", "main")
    items = [(r["question"], gold_of(r["answer"])) for r in ds["test"].select(range(3))]
    names = target_modules(model)
    print(f"[probe] target *_proj linears: {len(names)}  num_layers={num_layers_of(model)}", flush=True)
    torch.manual_seed(1)
    params, total_out = install_direct_map(model, names, 256)
    o = params[0]
    base_dir = torch.randn(256, device=dev, dtype=o.dtype)
    base_dir = base_dir / base_dir.abs().mean()       # mean|dir|=1 so s == mean|o|
    print(f"[probe] TOTAL_OUT={total_out}", flush=True)
    for s in [0.05, 0.10, 0.20, 0.40]:
        with torch.no_grad():
            o.zero_()
            o.add_(base_dir * s)
        n_ok = 0
        samples = []
        for q, gold in items:
            ids = tok(build_prompt(tok, q), return_tensors="pt").input_ids.to(dev)
            out = model.generate(ids, do_sample=False, max_new_tokens=120, pad_token_id=tok.eos_token_id)
            txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            ok = pred_of(txt) == gold
            n_ok += ok
            samples.append(("OK " if ok else "BAD") + " " + repr(txt[:110].replace("\n", " ")))
        max_gate = (1 + ALPHA_MOD * o).abs().max().item()
        print("s=%.2f  mean|o|=%.3f  max_gate=%.2f  correct=%d/3" %
              (s, o.abs().mean().item(), max_gate, n_ok), flush=True)
        for line in samples:
            print("    " + line, flush=True)
    print("PROBE_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B",
                    help="local dir / HF id / snapshot glob (was a hardcoded pod path)")
    ap.add_argument("--results-path", default="phase2_4b_gsm8k_results.txt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--probe", action="store_true", help="cheap coherence probe, then stop")
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

    if args.probe:
        run_probe(model, tok, dev)
        return

    ds = load_dataset("openai/gsm8k", "main")
    train_items = [(r["question"], gold_of(r["answer"])) for r in ds["train"].select(range(B * MAX_STEPS + 64))]
    eval_items = [(r["question"], gold_of(r["answer"])) for r in ds["test"].select(range(N_EVAL))]

    names = target_modules(model)
    print(f"target projection linears: {len(names)}  (all {nlayers} layers)", flush=True)

    k_base, n_base, cases_base = evaluate(model, tok, eval_items, dev, label="base", collect_cases=N_CASES)
    acc_base = k_base / n_base
    ci_base = wilson_ci(k_base, n_base)
    print(f"[baseline] GSM8K acc = {acc_base:.4f} ({k_base}/{n_base})  "
          f"CI [{ci_base[0]:.3f},{ci_base[1]:.3f}]", flush=True)

    results = {}
    for G in G_SWEEP:
        key = f"Map-G{G}"
        print(f"\n========== DIRECT MAP sweep G={G} ==========", flush=True)
        o_orig = [getattr(*get_parent(model, n)) for n in names]
        params, total_out = install_direct_map(model, names, G)
        n_par = sum(p.numel() for p in params)
        print(f"[{key}] trainable params = {n_par}  (TOTAL_OUT={total_out})", flush=True)
        curve, kl_curve = train_grpo(model, tok, names, train_items, params, LR_O, dev, key, telem_fn=o_telem)
        k_g, n_g, cases_g = evaluate(model, tok, eval_items, dev, label=key, collect_cases=N_CASES)
        restore(model, names, o_orig)
        acc_g = k_g / n_g
        ci_g = wilson_ci(k_g, n_g)
        final_kl = sum(kl_curve[-5:]) / max(1, len(kl_curve[-5:])) if kl_curve else 0.0
        print(f"[{key}] GSM8K acc = {acc_g:.4f} ({k_g}/{n_g})  CI [{ci_g[0]:.3f},{ci_g[1]:.3f}]  "
              f"final_mean_kl(last5)={final_kl:.4f}", flush=True)
        results[key] = dict(kind="map", n_par=n_par, k=k_g, n=n_g, acc=acc_g, ci=ci_g,
                            final_kl=final_kl, cases=cases_g)

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
    print(f"[LoRA-r{LORA_R}] GSM8K acc = {acc_l:.4f} ({k_l}/{n_l})  CI [{ci_l[0]:.3f},{ci_l[1]:.3f}]  "
          f"final_mean_kl(last5)={final_kl:.4f}", flush=True)
    results[f"LoRA-r{LORA_R}"] = dict(kind="lora", n_par=n_par, k=k_l, n=n_l, acc=acc_l, ci=ci_l,
                                      final_kl=final_kl, cases=cases_l)

    def overlap(a, b):
        return a[0] <= b[1] and b[0] <= a[1]
    order = [f"Map-G{G}" for G in G_SWEEP] + [f"LoRA-r{LORA_R}"]
    lines = ["=" * 78,
             "PHASE 2 (4B/GSM8K) — ISOLATE base size 0.8B->4B: DIRECT §5.4 modulation vs LoRA",
             "=" * 78,
             f"base: {snap}   num_hidden_layers={nlayers}   target *_proj linears={len(names)}",
             f"config: B={B} K={K} steps<={MAX_STEPS} time_box={TIME_BUDGET_S//60}min/variant MAX_NEW={MAX_NEW} "
             f"N_EVAL={N_EVAL} BETA_KL={BETA_KL} LR_o={LR_O} LR_lora={LR_LORA} ALPHA_MOD={ALPHA_MOD} dtype={dt}",
             "adapters from src/adapters.py: DirectMap gate (params=G) + LoRA (r=8)",
             "",
             f"baseline: {acc_base:.4f}  ({k_base}/{n_base})   CI [{ci_base[0]:.3f}, {ci_base[1]:.3f}]",
             "",
             f"GSM8K greedy acc (n={N_EVAL}), Wilson 95% CI:"]
    for key in order:
        r = results[key]
        lines.append(f"  {key:<10s} params={r['n_par']:<8d}: acc={r['acc']:.4f}  ({r['k']}/{r['n']})   "
                     f"CI [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]   final_mean_KL(last5)={r['final_kl']:.4f}")
    lines.append("")
    for key in order:
        r = results[key]
        ob = overlap(r['ci'], ci_base)
        lines.append(f"  {key} vs baseline: {r['acc']-acc_base:+.4f}  -> CIs "
                     f"{'OVERLAP (within noise)' if ob else 'DISJOINT (clears noise)'}")
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
