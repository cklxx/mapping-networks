"""[ARCHIVED — negative result, the ABANDONED frozen-random-map adapter design] Phase 2
(POD / H20 CUDA): §5.4 fine-tuning modulation of Mapping Networks vs LoRA on GSM8K RL —
properly-powered re-run.

RESULT: the properly-powered H20 re-run (B=8/K=4, ~24 steps, n=200 Wilson CI, + a KL leash)
confirmed the MPS null — frozen-random-map Mapping-RL 0.405 vs baseline 0.420 vs LoRA 0.440,
ALL CIs OVERLAP (within noise): the frozen-random-map adapter still had near-zero leverage.
This is the ABANDONED adapter design — a frozen random projection map + tanh squash — which
predates src/adapters.py and gave a NULL leverage result. It again pointed to the DIRECT
high-leverage redesign in phase2b.py (and the src/ DirectMap that followed). Because the whole
point of preserving this file is that this design is distinct from the src/ DirectMap, it KEEPS
its own bespoke MapLinear/LoRALinear and does NOT import the adapter from src/. The ONLY refactor
vs the original repro script: hardcoded pod/absolute paths -> --model-path / --results-path
flags; device/dtype auto-detect -> --device flag (bfloat16 on cuda, else float32). Every
hyperparameter, adapter definition, and comment is preserved. See phase2_pod_results.txt.

Properly-powered re-run of phase2_gsm8k_rl.py on an H20 GPU. The MPS run was underpowered
(4-9 GRPO steps, n=60 eval, all diffs within noise). This scales to B=16/K=8, ~200 steps
(30-min wall box per adapter), n=200 greedy eval with Wilson 95% CI.

ADDED vs the MPS script: a KL leash on the GRPO loss (loss += beta*KL(policy||frozen-base) on
completion tokens, beta=0.04) computed from one extra frozen-base forward, to stop the Mapping
latent's verbosity runaway the MPS run hit.

PRESERVED from the MPS script (do not touch): the brevity-forcing SYS prompt (the 0.8B is too
verbose and never reaches '####' otherwise), the hardened '####' ANS_RE (dodges markdown-heading
collision), the 114 *_proj target set, LATENT=256 / ALPHA_MOD=0.1 modulation W'=W*(1+0.1*tanh(s)),
LoRA r=8/alpha=16, and the adapter-changes-logits self-checks.

Base = Qwen3.5-0.8B (HYBRID Qwen3_5GatedDeltaNet: 18 linear-attn + 6 full-attn layers), ALL
base weights frozen. Two adapters compute W' per-forward from the frozen W:
  LoRA : W' = W + (alpha/r)*B@A         (trains A,B per target; ~thousands/target)
  Map  : W' = W * (1 + alpha_mod*tanh(s)); s = slice of (FROZEN [TOTAL_OUT,256] @ z)
         trains z ONLY (256 params, shared across ALL targets via one frozen mapping)
"""
import argparse, glob, os, re, time, math, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEV = None  # resolved in main() after parsing args
DT = None   # resolved in main() after parsing args
SNAP = None  # resolved in main() after parsing args
RESULTS_PATH = None  # resolved in main() after parsing args
torch.manual_seed(0)

R, ALPHA = 8, 16                         # LoRA rank/alpha
LATENT, ALPHA_MOD = 256, 0.1            # Mapping latent dim / modulation strength
# Measured on H20 GPU5: B=8/K=4 = 32 completions/step ~= 73s/step (gen + 2 fwds each, peak ~10GB).
# The 30-min box -> ~24 steps/adapter (vs the MPS run's 4-9) — meaningfully powered. B=16/K=8 was
# ~5min/step -> only ~5 steps, still underpowered, so K reduced to 4 (still gives group-reward
# variance for the GRPO advantage: measured signal_groups 4-5/8). Both adapters share this config.
B, K = 8, 4                             # GRPO: questions/step, completions/question
MAX_NEW, MAX_NEW_EVAL = 256, 256        # brief's 256; brevity prompt -> early EOS, so cheap
N_EVAL = 200                            # fixed GSM8K test subset, greedy
TIME_BUDGET_S, MAX_STEPS = 30 * 60, 120  # matched 30-min wall box / 120-step cap per adapter
LAST_N_LAYERS = 24                       # all 24 decoder layers
BETA_KL = 0.04                           # KL leash strength: loss += BETA_KL * KL(policy||base)
N_CASES = 3                              # decoded GSM8K cases to dump per variant

# Brevity-forcing prompt (PRESERVED): Qwen3.5-0.8B is verbose and never reaches '####' without it.
SYS = ("Solve the math problem. Think briefly in at most 3 short steps, then output the final "
       "answer as a single line: #### <number>. Keep it under 120 words. Do NOT use markdown headings.")
# hardened '####' regex (PRESERVED): require end/newline/non-'.<letter>' after to dodge '#### 4. Conclusion'
ANS_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)(?!\.\s*[A-Za-z])")

# ---------- adapters: wrap a frozen Linear, compute W' on the fly ----------
class LoRALinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base                                  # frozen
        out_f, in_f = base.weight.shape
        self.A = nn.Parameter(torch.randn(R, in_f, device=DEV, dtype=DT) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, R, device=DEV, dtype=DT))
    def forward(self, x):
        A, B = self.A.to(x.dtype), self.B.to(x.dtype)
        return self.base(x) + (ALPHA / R) * (x @ A.t() @ B.t())

class MapLinear(nn.Module):
    """W' = W * (1 + alpha_mod*tanh(s)); s = this target's slice of mapping(z). Shares z+map."""
    def __init__(self, base, scale_fn, off, out_f):
        super().__init__()
        self.base, self.scale_fn, self.off, self.out_f = base, scale_fn, off, out_f
    def forward(self, x):
        s = self.scale_fn()[self.off:self.off + self.out_f].to(x.dtype)
        gate = 1.0 + ALPHA_MOD * torch.tanh(s)
        return torch.nn.functional.linear(x, self.base.weight.to(x.dtype) * gate[:, None],
                                          None if self.base.bias is None else self.base.bias.to(x.dtype))

def target_modules(model):
    out = []
    for name, mod in model.named_modules():
        m = re.search(r"\.layers\.(\d+)\.", name)
        if not isinstance(mod, nn.Linear) or not m:
            continue
        if int(m.group(1)) < 24 - LAST_N_LAYERS:
            continue
        if name.endswith("_proj"):
            out.append(name)
    return out

def get_parent(model, dotted):
    parent = model
    *path, leaf = dotted.split(".")
    for p in path:
        parent = getattr(parent, p)
    return parent, leaf

def install_lora(model, names):
    adapters = []
    for n in names:
        parent, leaf = get_parent(model, n)
        w = LoRALinear(getattr(parent, leaf)).to(DEV)
        setattr(parent, leaf, w)
        adapters += [w.A, w.B]
    return adapters

def install_map(model, names):
    outs = [getattr(*get_parent(model, n)).weight.shape[0] for n in names]
    total_out = sum(outs)
    z = nn.Parameter(torch.randn(LATENT, device=DEV, dtype=DT) * 0.1)
    W_map = (torch.randn(total_out, LATENT, device=DEV, dtype=DT) * (1.0 / LATENT) ** 0.5)  # frozen
    scale_fn = lambda: W_map @ z
    off = 0
    for n, o in zip(names, outs):
        parent, leaf = get_parent(model, n)
        setattr(parent, leaf, MapLinear(getattr(parent, leaf), scale_fn, off, o).to(DEV))
        off += o
    return [z], total_out

def restore(model, names, originals):
    for n, orig in zip(names, originals):
        parent, leaf = get_parent(model, n)
        setattr(parent, leaf, orig)

# Context manager: temporarily swap adapters OUT so a forward runs on the frozen base (for KL ref).
class base_forward:
    def __init__(self, model, names):
        self.model, self.names = model, names
    def __enter__(self):
        self.cur = [getattr(*get_parent(self.model, n)) for n in self.names]
        self.orig = [m.base for m in self.cur]            # each adapter wraps .base = frozen Linear
        restore(self.model, self.names, self.orig)
        return self
    def __exit__(self, *a):
        restore(self.model, self.names, self.cur)

# ---------- data / prompting ----------
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

# ---------- logprobs + KL: ONE policy forward + ONE base forward over prompt+completion ----------
def comp_logp_and_kl(model, tok, names, prompt_ids, comp_ids):
    """Returns (sum policy logprob over completion tokens, mean per-token KL(policy||base) over completion).
    KL is full-distribution: sum_v p_pol(v)*(logp_pol(v)-logp_base(v)), averaged over completion tokens."""
    ids = torch.cat([prompt_ids, comp_ids], 0)[None].to(DEV)
    logits = model(ids).logits[0, :-1].float()             # policy, predict token t+1
    logp = torch.log_softmax(logits, -1)
    tgt = ids[0, 1:]
    n_prompt = prompt_ids.numel()
    comp_mask = torch.zeros_like(tgt, dtype=torch.bool)
    comp_mask[n_prompt - 1:] = True                        # completion-token positions only
    tok_lp = logp.gather(1, tgt[:, None]).squeeze(1)
    sum_lp = tok_lp[comp_mask].sum()
    # frozen-base forward on the same ids (one extra forward, no grad through base)
    with torch.no_grad(), base_forward(model, names):
        base_logits = model(ids).logits[0, :-1].float()
        base_logp = torch.log_softmax(base_logits, -1)
    p = logp.exp()
    kl_per_pos = (p * (logp - base_logp)).sum(-1)          # KL(policy||base) per position
    kl = kl_per_pos[comp_mask].mean()
    return sum_lp, kl

# ---------- eval: greedy on a fixed test subset ----------
@torch.no_grad()
def evaluate(model, tok, items, label="", collect_cases=0):
    model.eval(); correct = 0; t0 = time.time(); cases = []
    for i, (q, gold) in enumerate(items):
        ids = tok(build_prompt(tok, q), return_tensors="pt").input_ids.to(DEV)
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

# ---------- GRPO training loop (+ KL leash) ----------
def train_grpo(model, tok, names, train_items, trainable, lr, label):
    model.train()
    opt = torch.optim.Adam(trainable, lr=lr)
    t0 = time.time(); curve = []
    for step in range(MAX_STEPS):
        if time.time() - t0 > TIME_BUDGET_S:
            print(f"[{label}] time budget hit at step {step}", flush=True); break
        batch = [train_items[(step * B + i) % len(train_items)] for i in range(B)]
        step_r, nz_groups, kl_acc, did_backward = [], 0, [], False
        opt.zero_grad()
        # Accumulate gradients per-completion (backward() immediately, free each graph) so live
        # autograd memory is capped at one completion — 64 retained graphs over 151k-vocab fp32
        # logits OOMs an H20 otherwise.
        for q, gold in batch:
            prompt = build_prompt(tok, q)
            pids = tok(prompt, return_tensors="pt").input_ids[0].to(DEV)
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
                # KL term applies to EVERY completion (leash), policy-gradient term only when adv!=0
                lp, kl = comp_logp_and_kl(model, tok, names, pids, comps[k])
                kl_acc.append(kl.item())
                pg = -adv[k].to(DEV).detach() * lp if adv[k].abs() >= 1e-6 else 0.0 * lp
                comp_loss = (pg + BETA_KL * kl) / (B * K)
                comp_loss.backward()                              # free this completion's graph now
                did_backward = True
        if did_backward:
            opt.step()
        mr = sum(step_r) / len(step_r)
        mkl = sum(kl_acc) / len(kl_acc) if kl_acc else 0.0
        curve.append(mr)
        print(f"[{label}] step {step:3d}  mean_reward={mr:.3f}  signal_groups={nz_groups}/{B}  "
              f"mean_kl={mkl:.4f}  elapsed={time.time()-t0:.0f}s", flush=True)
    return curve

# ---------- self-check: adapter actually changes logits ----------
def assert_changes_logits(model, tok, names, install, probe_ids):
    with torch.no_grad():
        base = model(probe_ids).logits.clone()
    originals = [getattr(*get_parent(model, n)) for n in names]
    trainable = install(model, names)
    if isinstance(trainable, tuple):
        trainable = trainable[0]
    with torch.no_grad():
        for p in trainable:
            p.add_(0.05)
        adapted = model(probe_ids).logits
    ok = (adapted != base).any().item()
    restore(model, names, originals)
    assert ok, "adapter did not change logits!"

def fmt_cases(cases):
    out = []
    for j, (q, text, pred, gold) in enumerate(cases):
        out.append(f"  --- case {j+1} ---")
        out.append(f"  Q: {q.strip()[:400]}")
        out.append(f"  MODEL: {text.strip()[:900]}")
        out.append(f"  extracted={pred!r}  gold={gold!r}  {'CORRECT' if pred==gold else 'WRONG'}")
    return "\n".join(out)

def resolve_model_path(path):
    """if a dir, use it; elif a glob hits, use hits[0]; else pass through as an HF id."""
    if os.path.isdir(path):
        return path
    hits = glob.glob(path)
    if hits:
        return hits[0]
    return path

def main():
    global DEV, DT, SNAP, RESULTS_PATH
    p = argparse.ArgumentParser(description="Phase 2 (H20) frozen-random-map §5.4 modulation vs LoRA on GSM8K RL.")
    p.add_argument("--model-path", default="Qwen/Qwen3.5-0.8B",
                   help="dir / glob / HF id for the base model snapshot")
    p.add_argument("--device", default=None,
                   help="torch device (default: auto — mps if available, else cuda, else cpu)")
    p.add_argument("--results-path", default="phase2_pod_results.txt",
                   help="where to write the results report")
    args = p.parse_args()

    if args.device is not None:
        DEV = args.device
    elif torch.backends.mps.is_available():
        DEV = "mps"
    elif torch.cuda.is_available():
        DEV = "cuda"
    else:
        DEV = "cpu"
    DT = torch.bfloat16 if DEV == "cuda" else torch.float32
    SNAP = resolve_model_path(args.model_path)
    RESULTS_PATH = args.results_path

    print(f"device={DEV}  dtype={DT}  snapshot={SNAP.split('/')[-2][:12] if SNAP.endswith('/') else os.path.basename(SNAP.rstrip('/'))[:12]}", flush=True)
    print(f"config: B={B} K={K} MAX_STEPS={MAX_STEPS} time_box={TIME_BUDGET_S}s "
          f"MAX_NEW={MAX_NEW} N_EVAL={N_EVAL} BETA_KL={BETA_KL}", flush=True)
    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=DT, trust_remote_code=True).to(DEV)
    model.requires_grad_(False)

    ds = load_dataset("openai/gsm8k", "main")
    train_items = [(r["question"], gold_of(r["answer"])) for r in ds["train"].select(range(B * MAX_STEPS + 64))]
    eval_items = [(r["question"], gold_of(r["answer"])) for r in ds["test"].select(range(N_EVAL))]

    names = target_modules(model)
    probe = tok(build_prompt(tok, train_items[0][0]), return_tensors="pt").input_ids.to(DEV)
    print(f"target projection linears: {len(names)}  (last {LAST_N_LAYERS} layers)", flush=True)

    o = [getattr(*get_parent(model, n)) for n in names]
    lora_params = install_lora(model, names)
    n_lora = sum(p.numel() for p in lora_params)
    restore(model, names, o)
    o = [getattr(*get_parent(model, n)) for n in names]
    map_params, total_out = install_map(model, names)
    n_map = sum(p.numel() for p in map_params)
    restore(model, names, o)
    print(f"trainable params  LoRA={n_lora}  Mapping={n_map}  (TOTAL_OUT={total_out})", flush=True)

    assert_changes_logits(model, tok, names, install_lora, probe)
    assert_changes_logits(model, tok, names, install_map, probe)
    print("self-check OK: both adapters change logits", flush=True)

    # (a) baseline, no adapter
    k_base, n_base, cases_base = evaluate(model, tok, eval_items, label="base", collect_cases=N_CASES)
    acc_base = k_base / n_base
    print(f"[baseline] GSM8K acc = {acc_base:.4f} ({k_base}/{n_base})", flush=True)

    # (b) LoRA RL
    o = [getattr(*get_parent(model, n)) for n in names]
    lora_params = install_lora(model, names)
    curve_lora = train_grpo(model, tok, names, train_items, lora_params, lr=1e-4, label="LoRA")
    k_lora, n_lora_e, cases_lora = evaluate(model, tok, eval_items, label="LoRA", collect_cases=N_CASES)
    restore(model, names, o)
    acc_lora = k_lora / n_lora_e
    print(f"[LoRA-RL] GSM8K acc = {acc_lora:.4f} ({k_lora}/{n_lora_e})", flush=True)

    # (c) Mapping-modulation RL
    o = [getattr(*get_parent(model, n)) for n in names]
    map_params, _ = install_map(model, names)
    curve_map = train_grpo(model, tok, names, train_items, map_params, lr=2e-3, label="Map")
    k_map, n_map_e, cases_map = evaluate(model, tok, eval_items, label="Map", collect_cases=N_CASES)
    restore(model, names, o)
    acc_map = k_map / n_map_e
    print(f"[Mapping-RL] GSM8K acc = {acc_map:.4f} ({k_map}/{n_map_e})", flush=True)

    ci_base, ci_lora, ci_map = wilson_ci(k_base, n_base), wilson_ci(k_lora, n_lora_e), wilson_ci(k_map, n_map_e)
    delta = acc_map - acc_base
    overlap_bl = ci_map[0] <= ci_base[1] and ci_base[0] <= ci_map[1]   # CIs overlap -> within noise
    overlap_ml = ci_map[0] <= ci_lora[1] and ci_lora[0] <= ci_map[1]

    lines = []
    lines.append("=" * 78)
    lines.append("PHASE 2 (H20 CUDA) — Mapping-modulation vs LoRA on GSM8K RL")
    lines.append("=" * 78)
    lines.append(f"config: B={B} K={K} steps<= {MAX_STEPS} time_box={TIME_BUDGET_S//60}min/adapter "
                 f"MAX_NEW={MAX_NEW} N_EVAL={N_EVAL} BETA_KL={BETA_KL} dtype={DT}")
    lines.append(f"trainable params:  LoRA={n_lora}   Mapping={n_map}   ({n_lora/n_map:.0f}x more for LoRA)")
    lines.append("")
    lines.append(f"GSM8K greedy acc (n={N_EVAL}), Wilson 95% CI:")
    lines.append(f"  baseline    : {acc_base:.4f}  ({k_base}/{n_base})   CI [{ci_base[0]:.3f}, {ci_base[1]:.3f}]")
    lines.append(f"  LoRA-RL     : {acc_lora:.4f}  ({k_lora}/{n_lora_e})   CI [{ci_lora[0]:.3f}, {ci_lora[1]:.3f}]")
    lines.append(f"  Mapping-RL  : {acc_map:.4f}  ({k_map}/{n_map_e})   CI [{ci_map[0]:.3f}, {ci_map[1]:.3f}]")
    lines.append("")
    lines.append(f"Mapping vs baseline: {delta:+.4f}  -> CIs {'OVERLAP (within noise)' if overlap_bl else 'DISJOINT (clears noise)'}")
    lines.append(f"Mapping vs LoRA    : {acc_map-acc_lora:+.4f}  -> CIs {'OVERLAP (within noise)' if overlap_ml else 'DISJOINT (clears noise)'}")
    lines.append("")
    lines.append(f"mean_reward curve LoRA ({len(curve_lora)} steps): " +
                 " ".join(f"{x:.2f}" for x in curve_lora))
    lines.append(f"mean_reward curve Map  ({len(curve_map)} steps): " +
                 " ".join(f"{x:.2f}" for x in curve_map))
    lines.append("")
    lines.append("DECODED CASES (baseline):"); lines.append(fmt_cases(cases_base))
    lines.append(""); lines.append("DECODED CASES (LoRA-RL):"); lines.append(fmt_cases(cases_lora))
    lines.append(""); lines.append("DECODED CASES (Mapping-RL):"); lines.append(fmt_cases(cases_map))
    lines.append("=" * 78)

    report = "\n".join(lines)
    print("\n" + report, flush=True)
    with open(RESULTS_PATH, "w") as f:
        f.write(report + "\n")
    print(f"\nwrote {RESULTS_PATH}", flush=True)

if __name__ == "__main__":
    main()
