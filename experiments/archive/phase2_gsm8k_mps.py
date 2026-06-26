"""[ARCHIVED — negative result, the ABANDONED frozen-random-map adapter design] Phase 2:
§5.4 fine-tuning modulation of Mapping Networks (arXiv:2602.19134) on GSM8K math RL — MPS,
underpowered first attempt.

RESULT: on Qwen3.5-0.8B / GSM8K (MPS, underpowered — only 4-9 GRPO steps, n=60 eval), the
frozen-random-map §5.4 modulation W'=W*(1+0.1*tanh(FROZEN_random_map@z)) with a 256-param
latent z gave NO improvement (0.417 -> 0.350) and all diffs were within noise; the diagnostic
showed the frozen random map + alpha=0.1 + tanh dampened the latent's leverage to near-zero
(KL ~0.002). This is the ABANDONED adapter design — a frozen random projection map + tanh
squash — which predates src/adapters.py and gave a NULL leverage result. It motivated the
DIRECT high-leverage redesign in phase2b.py (and the src/ DirectMap that followed). Because the
whole point of preserving this file is that this design is distinct from the src/ DirectMap, it
KEEPS its own bespoke MapLinear/LoRALinear and does NOT import the adapter from src/. The ONLY
refactor vs the original repro script: hardcoded pod/absolute paths -> --model-path /
--results-path flags; device/dtype auto-detect -> --device flag. Every hyperparameter, adapter
definition, and comment is preserved. See phase2_results.txt.

Question: can an ultra-small trainable latent z (256 params) through a FIXED random
mapping that MODULATES the frozen base weights drive GSM8K RL fine-tuning as well as
LoRA (which trains thousands of params per target)? Phase 1 showed the gap-suppression
comes from parameter compression; here we test the §5.4 *modulation* variant under GRPO.

Base = Qwen3.5-0.8B, ALL base weights frozen. Two adapters compute W' per-forward from
the frozen W (no second persistent weight set):
  LoRA  : W' = W + (alpha/r)·B@A         (trains A,B per target; ~thousands/target)
  Map   : W' = W * (1 + alpha_mod·tanh(s));  s = slice of (FROZEN [TOTAL_OUT,256] @ z)
          trains z ONLY (256 params, shared across ALL targets via one frozen mapping)

DEVIATION FROM BRIEF (documented): Qwen3.5-0.8B is a HYBRID linear/full-attention model
(Qwen3_5GatedDeltaNet on 18 layers + Qwen3_5Attention on 6 layers), so q/k/v/o_proj exist
on only 6 layers. We honor the brief's intent ("adapt every decoder layer's projections,
base frozen") by targeting every decoder-layer Linear whose name ends in '_proj' = 114
modules: 6 full-attn layers' q/k/v/o_proj (24), all 24 mlp gate/up/down_proj (72), the 18
DeltaNet layers' out_proj (18). (The DeltaNet in_proj_{qkv,z,b,a} end in _qkv/_z/_b/_a, not
_proj, so they're excluded — the adapted set still spans attention-out + all qkvo + all MLP.)
TOTAL_OUT = sum of out_features over the 114 targets = 251904.

ponytail: minimal GRPO, task reward only, no KL/PPO-clip — group-normalized REINFORCE.
"""
import argparse, glob, os, re, time, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

DEV = None  # resolved in main() after parsing args
DT = None   # resolved in main() after parsing args
SNAP = None  # resolved in main() after parsing args
RESULTS_PATH = None  # resolved in main() after parsing args
torch.manual_seed(0)

R, ALPHA = 8, 16                         # LoRA rank/alpha
LATENT, ALPHA_MOD = 256, 0.1            # Mapping latent dim / modulation strength
# REDUCED from brief (MPS is slow: ~13s for K=4 batched 160-tok sampling, ~17s/256-tok greedy).
# Brief allowed reducing max_new_tokens / test subset. B=6/K=4 -> ~110s/step; 25min ~= 13 steps.
B, K = 6, 4                              # GRPO: questions/step, completions/question
# With the brevity prompt the model finishes in ~80 tok; 160 cap is ample and ~2x faster.
MAX_NEW, MAX_NEW_EVAL = 160, 160        # reduced from 256 (brevity prompt -> short outputs)
N_EVAL = 60                              # reduced from 150 (60 x ~8s x 3 evals ~= 24 min)
TIME_BUDGET_S, MAX_STEPS = 25 * 60, 40
LAST_N_LAYERS = 24                       # reduce (e.g. 12) if MPS OOMs

# Brevity-forcing prompt: Qwen3.5-0.8B is verbose and runs out of tokens before finishing
# the calc (baseline 0/12 at 300 tok). Forcing <=3 short steps -> 4/12 correct, all finish
# in ~80 tok, with reward variance -> GRPO actually has a learning signal. (Harness fix, not bias.)
SYS = ("Solve the math problem. Think briefly in at most 3 short steps, then output the final "
       "answer as a single line: #### <number>. Keep it under 120 words. Do NOT use markdown headings.")
# take #### <number>, but the number must NOT be a markdown-heading ordinal like "#### 4. Conclusion"
# (require end-of-string/newline/non-'.<letter>' after) to dodge the model's heading collision.
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
        A, B = self.A.to(x.dtype), self.B.to(x.dtype)     # match activation dtype (model runs bf16 internally)
        return self.base(x) + (ALPHA / R) * (x @ A.t() @ B.t())

class MapLinear(nn.Module):
    """W' = W * (1 + alpha_mod·tanh(s)); s = this target's slice of mapping(z). Shares z+map."""
    def __init__(self, base, scale_fn, off, out_f):
        super().__init__()
        self.base, self.scale_fn, self.off, self.out_f = base, scale_fn, off, out_f
    def forward(self, x):
        s = self.scale_fn()[self.off:self.off + self.out_f].to(x.dtype)   # R^{out}, match activation dtype
        gate = 1.0 + ALPHA_MOD * torch.tanh(s)
        return torch.nn.functional.linear(x, self.base.weight.to(x.dtype) * gate[:, None],
                                          None if self.base.bias is None else self.base.bias.to(x.dtype))

def target_modules(model):
    """All decoder-layer projection Linears in the last LAST_N_LAYERS layers."""
    out = []
    for name, mod in model.named_modules():
        m = re.search(r"\.layers\.(\d+)\.", name)
        if not isinstance(mod, nn.Linear) or not m:
            continue
        if int(m.group(1)) < 24 - LAST_N_LAYERS:
            continue
        if name.endswith("_proj"):                         # *_proj covers all target projections
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
    return adapters                                        # trainable params

def install_map(model, names):
    outs = [getattr(*get_parent(model, n)).weight.shape[0] for n in names]
    total_out = sum(outs)
    z = nn.Parameter(torch.randn(LATENT, device=DEV, dtype=DT) * 0.1)
    W_map = (torch.randn(total_out, LATENT, device=DEV, dtype=DT) * (1.0 / LATENT) ** 0.5)  # frozen
    scale_fn = lambda: W_map @ z                           # cached per-forward by autograd graph reuse
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

# ---------- logprobs: one teacher-forced forward over prompt+completion, mask prompt ----------
def completion_logprob(model, tok, prompt_ids, comp_ids):
    ids = torch.cat([prompt_ids, comp_ids], 0)[None].to(DEV)
    logits = model(ids).logits[0, :-1]                     # predict token t+1
    logp = torch.log_softmax(logits, -1)
    tgt = ids[0, 1:]
    tok_lp = logp.gather(1, tgt[:, None]).squeeze(1)
    comp_mask = torch.zeros_like(tgt, dtype=torch.bool)
    comp_mask[prompt_ids.numel() - 1:] = True              # only completion tokens contribute
    return tok_lp[comp_mask].sum()

# ---------- eval: greedy on a fixed test subset ----------
@torch.no_grad()
def evaluate(model, tok, items, label=""):
    model.eval(); correct = 0; t0 = time.time()
    for i, (q, gold) in enumerate(items):
        ids = tok(build_prompt(tok, q), return_tensors="pt").input_ids.to(DEV)
        out = model.generate(ids, do_sample=False, max_new_tokens=MAX_NEW_EVAL,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        correct += int(pred_of(text) == gold)
        if (i + 1) % 10 == 0:
            print(f"  [eval {label}] {i+1}/{len(items)}  acc_sofar={correct/(i+1):.3f}  "
                  f"{(time.time()-t0)/(i+1):.1f}s/q", flush=True)
    return correct / len(items)

# ---------- GRPO training loop ----------
def train_grpo(model, tok, train_items, trainable, lr, label):
    model.train()
    opt = torch.optim.Adam(trainable, lr=lr)
    t0 = time.time()
    for step in range(MAX_STEPS):
        if time.time() - t0 > TIME_BUDGET_S:
            print(f"[{label}] time budget hit at step {step}"); break
        batch = [train_items[(step * B + i) % len(train_items)] for i in range(B)]
        step_loss, step_r, nz_groups = [], [], 0   # nz_groups: groups with reward variance (learning signal)
        for q, gold in batch:
            prompt = build_prompt(tok, q)
            pids = tok(prompt, return_tensors="pt").input_ids[0].to(DEV)
            with torch.no_grad():
                gen = model.generate(pids[None], do_sample=True, temperature=0.8, top_p=0.95,
                                     num_return_sequences=K, max_new_tokens=MAX_NEW,
                                     pad_token_id=tok.eos_token_id)
            comps = [gen[k, pids.numel():] for k in range(K)]
            texts = [tok.decode(c, skip_special_tokens=True) for c in comps]
            rs = torch.tensor([reward_of(t, gold) for t in texts], dtype=DT)
            step_r.append(rs.mean().item())
            if rs.std() > 1e-6:
                nz_groups += 1                                            # this group has learnable signal
            adv = (rs - rs.mean()) / (rs.std() + 1e-4)                     # group-normalized
            for k in range(K):
                if adv[k].abs() < 1e-6:
                    continue                                              # zero advantage -> skip
                lp = completion_logprob(model, tok, pids, comps[k])
                step_loss.append(-adv[k].to(DEV).detach() * lp)
        if step_loss:
            loss = torch.stack(step_loss).sum() / (B * K)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"[{label}] step {step:2d}  mean_reward={sum(step_r)/len(step_r):.3f}  "
              f"signal_groups={nz_groups}/{B}  elapsed={time.time()-t0:.0f}s", flush=True)
    return model

# ---------- self-check: adapter actually changes logits ----------
def assert_changes_logits(model, tok, names, install, probe_ids):
    # LoRA inits B=0 (W'=W at init by design), so perturb the trainable params first:
    # this proves the adapter is wired into the logits and is gradient-reachable.
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
    assert (adapted != base).any(), "adapter did not change logits!"
    restore(model, names, originals)

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
    p = argparse.ArgumentParser(description="Phase 2 frozen-random-map §5.4 modulation vs LoRA on GSM8K RL (MPS).")
    p.add_argument("--model-path", default="Qwen/Qwen3.5-0.8B",
                   help="dir / glob / HF id for the base model snapshot")
    p.add_argument("--device", default=None,
                   help="torch device (default: auto — mps if available, else cuda, else cpu)")
    p.add_argument("--results-path", default="phase2_results.txt",
                   help="where to write the results summary")
    args = p.parse_args()

    if args.device is not None:
        DEV = args.device
    elif torch.backends.mps.is_available():
        DEV = "mps"
    elif torch.cuda.is_available():
        DEV = "cuda"
    else:
        DEV = "cpu"
    DT = torch.float32                       # MPS-safe
    SNAP = resolve_model_path(args.model_path)
    RESULTS_PATH = args.results_path

    print(f"device={DEV}  dtype={DT}  snapshot={SNAP.split('/')[-2][:12] if SNAP.endswith('/') else os.path.basename(SNAP.rstrip('/'))[:12]}")
    tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=DT, trust_remote_code=True).to(DEV)
    model.requires_grad_(False)                             # FREEZE base

    ds = load_dataset("openai/gsm8k", "main")
    train_items = [(r["question"], gold_of(r["answer"])) for r in ds["train"].select(range(B * MAX_STEPS + 64))]
    eval_items = [(r["question"], gold_of(r["answer"])) for r in ds["test"].select(range(N_EVAL))]

    names = target_modules(model)
    probe = tok(build_prompt(tok, train_items[0][0]), return_tensors="pt").input_ids.to(DEV)
    print(f"target projection linears: {len(names)}  (last {LAST_N_LAYERS} layers)")

    # param counts (build temporarily, count, restore)
    o = [getattr(*get_parent(model, n)) for n in names]
    lora_params = install_lora(model, names)
    n_lora = sum(p.numel() for p in lora_params)
    restore(model, names, o)
    o = [getattr(*get_parent(model, n)) for n in names]
    map_params, total_out = install_map(model, names)
    n_map = sum(p.numel() for p in map_params)
    restore(model, names, o)
    print(f"trainable params  LoRA={n_lora}  Mapping={n_map}  (TOTAL_OUT={total_out})")

    # self-checks: both adapters change the logits
    assert_changes_logits(model, tok, names, install_lora, probe)
    assert_changes_logits(model, tok, names, install_map, probe)
    print("self-check OK: both adapters change logits")

    # (a) baseline, no adapter
    acc_base = evaluate(model, tok, eval_items, label="base")
    print(f"[baseline] GSM8K acc = {acc_base:.4f}", flush=True)

    # (b) LoRA RL
    o = [getattr(*get_parent(model, n)) for n in names]
    lora_params = install_lora(model, names)
    train_grpo(model, tok, train_items, lora_params, lr=1e-4, label="LoRA")
    acc_lora = evaluate(model, tok, eval_items, label="LoRA")
    restore(model, names, o)
    print(f"[LoRA-RL] GSM8K acc = {acc_lora:.4f}", flush=True)

    # (c) Mapping-modulation RL
    o = [getattr(*get_parent(model, n)) for n in names]
    map_params, _ = install_map(model, names)
    train_grpo(model, tok, train_items, map_params, lr=2e-3, label="Map")
    acc_map = evaluate(model, tok, eval_items, label="Map")
    restore(model, names, o)
    print(f"[Mapping-RL] GSM8K acc = {acc_map:.4f}", flush=True)

    delta = acc_map - acc_base
    verdict = (f"256-param latent {'IMPROVED' if delta > 0 else 'did NOT improve'} GSM8K "
               f"({acc_base:.3f}->{acc_map:.3f}, {delta:+.3f}); LoRA={acc_lora:.3f} with "
               f"{n_lora} params ({n_lora/n_map:.0f}x more params than Mapping's {n_map}).")
    lines = [
        f"trainable params:  LoRA={n_lora}   Mapping={n_map}",
        f"GSM8K acc (n={N_EVAL}):  baseline={acc_base:.4f}   LoRA-RL={acc_lora:.4f}   Mapping-RL={acc_map:.4f}",
        f"verdict: {verdict}",
    ]
    print("\n" + "\n".join(lines))
    with open(RESULTS_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
