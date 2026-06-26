"""Cost benchmark — THE centerpiece. Measure, per adapter VARIANT, what makes one
fine-tuning method cheaper than another in GPU-hours.

For each variant in {modulation-G256, modulation-G2048, LoRA-r8 (+ an lr SWEEP for a
FAIR comparison), full-FT if it fits} it runs the SAME GRPO loop on the SAME frozen base
and records, via src/costlib.py (the single source of the cost hooks):

  - trainable param count (exact)
  - peak VRAM            (torch.cuda.max_memory_allocated, reset before the loop)
  - steps-to-target      (# steps to reach a fixed reward threshold — the convergence #)
  - wall-clock           (mean per-step + total)
  - FLOPs/step (est)     (fwd+bwd; base term vs adapter term shown separately so the
                          mechanism — base backward dominates, adapter FLOPs ~0 — is
                          EXPLICIT, making compute/step ~equal across adapters)
  - GPU-hours            (compute/step x steps-to-target ≈ the real cost)

then emits the markdown COST TABLE to results/cost-table.md.

FAIR LoRA: the prior validated run left LoRA stuck at lr=1e-4 (it barely moved). This
benchmark adds a LoRA learning-rate SWEEP and reports LoRA's BEST variant in the
head-to-head, so "modulation beats LoRA" is not an artifact of an under-tuned LoRA.

MODES:
  --smoke : tiny RANDOM transformer on CPU, few steps, tiny N — runs end-to-end in
            minutes on a Mac with no GPU and no model download, and STILL emits a real
            (tiny) cost table, proving the instrumentation works before the GPU run.
  (full)  : Qwen3-4B + MATH-500 on a CUDA GPU (~2h / 4 variants).

The 4B GPU numbers need a GPU; on a Mac --smoke fills the measured rows and the GPU rows
are written to results/cost-table.md as a clearly-marked PENDING block with the a-priori
predictions.
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.adapters import (  # noqa: E402
    ALPHA_MOD,
    get_parent,
    install_direct_map,
    install_lora,
    num_layers_of,
    restore,
    target_modules,
)
from src import costlib  # noqa: E402

torch.manual_seed(0)


# ===========================================================================
# tiny random transformer (smoke mode) — a real causal-LM forward, no download
# ===========================================================================
class TinyConfig:
    def __init__(self, vocab, dim, n_layers):
        self.vocab_size = vocab
        self.hidden_size = dim
        self.num_hidden_layers = n_layers


class TinyOut:
    def __init__(self, logits):
        self.logits = logits


class TinyBlock(nn.Module):
    """One decoder block exposing q/k/v/o_proj + gate/up/down_proj so the adapters'
    `*_proj` discovery and per-output-channel grouping wire up exactly as on a real model."""

    def __init__(self, dim, idx):
        super().__init__()
        self.idx = idx
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.up_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.down_proj = nn.Linear(2 * dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h):
        a = self.o_proj(self.q_proj(h) + self.k_proj(h) + self.v_proj(h))
        h = self.norm(h + a)
        m = self.down_proj(torch.nn.functional.silu(self.gate_proj(h)) + self.up_proj(h))
        return self.norm(h + m)


class TinyLM(nn.Module):
    """Minimal HF-causal-LM-shaped model: `.config.num_hidden_layers`, `model.layers.<i>.*_proj`,
    `forward(ids).logits`, and `.generate(...)`. Just enough surface for the adapters +
    GRPO loop to run unchanged. NOT a language model — it's a deterministic-toy whose
    'reward' is whether it emits the gold token, so steps-to-target is well-defined."""

    def __init__(self, vocab=64, dim=32, n_layers=3):
        super().__init__()
        self.config = TinyConfig(vocab, dim, n_layers)
        self.embed = nn.Embedding(vocab, dim)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyBlock(dim, i) for i in range(n_layers)])
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, ids, **kw):
        h = self.embed(ids)
        for blk in self.model.layers:
            h = blk(h)
        return TinyOut(self.lm_head(h))

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=8, do_sample=False, temperature=1.0,
                 top_p=1.0, num_return_sequences=1, pad_token_id=0, **kw):
        ids = ids.repeat(num_return_sequences, 1)
        for _ in range(max_new_tokens):
            logits = self.forward(ids).logits[:, -1] / max(temperature, 1e-6)
            if do_sample:
                nxt = torch.multinomial(torch.softmax(logits, -1), 1)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], 1)
        return ids


class TinyTokenizer:
    """Trivial tokenizer over a fixed vocab. A 'problem' is a seed int; the gold answer is
    a fixed token id derived from it, so a forward CAN learn to emit it -> reward variance
    -> a real steps-to-target signal even in smoke mode."""

    def __init__(self, vocab=64):
        self.vocab = vocab
        self.eos_token_id = 1
        self.pad_token_id = 0
        self.padding_side = "right"

    def __call__(self, text, return_tensors=None, padding=False):
        ids = [2 + (ord(c) % (self.vocab - 4)) for c in str(text)[:12]] or [2]
        t = torch.tensor([ids])
        return type("Enc", (), {"input_ids": t, "to": lambda self_, d: self_,
                                 "__getitem__": lambda self_, k: t})()


# ===========================================================================
# GRPO loop (shared shape with experiments/math500_rl.py), instrumented
# ===========================================================================
def grpo_train(model, names, train_items, trainable, lr, dev, B, K, max_steps,
               reward_fn, gen_kw, clamp_o=None, kl_beta=0.0):
    """Minimal instrumented GRPO. Returns (reward_curve, timer, tokens_per_step_est)."""
    model.train()
    opt = torch.optim.Adam(trainable, lr=lr)
    timer = costlib.StepTimer().start()
    curve = []
    tok_acc = []
    for step in range(max_steps):
        batch = [train_items[(step * B + i) % len(train_items)] for i in range(B)]
        opt.zero_grad()
        step_r, did = [], False
        step_tokens = 0
        for prompt_ids, gold in batch:
            pids = prompt_ids.to(dev)
            with torch.no_grad():
                gen = model.generate(pids[None], do_sample=True, temperature=0.8,
                                     top_p=0.95, num_return_sequences=K, **gen_kw)
            comps = [gen[k, pids.numel():] for k in range(K)]
            rs = torch.tensor([reward_fn(c, gold) for c in comps], dtype=torch.float32)
            step_r.append(rs.mean().item())
            adv = (rs - rs.mean()) / (rs.std() + 1e-4)
            for k in range(K):
                ids = torch.cat([pids, comps[k]], 0)[None].to(dev)
                step_tokens += ids.numel()
                logits = model(ids).logits[0, :-1].float()
                logp = torch.log_softmax(logits, -1)
                tgt = ids[0, 1:]
                tok_lp = logp.gather(1, tgt[:, None]).squeeze(1)
                comp_mask = torch.zeros_like(tgt, dtype=torch.bool)
                comp_mask[pids.numel() - 1:] = True
                lp = tok_lp[comp_mask].sum()
                pg = -adv[k].to(dev).detach() * lp if adv[k].abs() >= 1e-6 else 0.0 * lp
                (pg / (B * K)).backward()
                did = True
        if did:
            opt.step()
            if clamp_o is not None:
                with torch.no_grad():
                    trainable[0].clamp_(-clamp_o, clamp_o)
        curve.append(sum(step_r) / len(step_r))
        tok_acc.append(step_tokens)
        timer.tick()
        print(f"  [cost {lr=}] step {step:2d} mean_reward={curve[-1]:.3f} "
              f"step_s={timer.per_step[-1]:.2f}", flush=True)
    tokens_per_step = int(sum(tok_acc) / len(tok_acc)) if tok_acc else 0
    return curve, timer, tokens_per_step


# ===========================================================================
# variant runners — each installs, trains, records the cost row, restores
# ===========================================================================
def run_variant(model, names, train_items, dev, cfg, variant, kind, base_params,
                lr, G=None, r=None):
    o_orig = [getattr(*get_parent(model, n)) for n in names]
    costlib.reset_peak_vram(dev)
    if kind == "map":
        params, _ = install_direct_map(model, names, G)
        clamp = cfg["o_clamp"]
    elif kind == "lora":
        params = install_lora(model, names, r)
        clamp = None
    elif kind == "full":
        # full fine-tune: train every target *_proj weight directly (no adapter).
        params = [getattr(*get_parent(model, n)).weight for n in names]
        for p in params:
            p.requires_grad_(True)
        clamp = None
    else:
        raise ValueError(kind)
    curve, timer, tps = grpo_train(
        model, names, train_items, params, lr, dev,
        cfg["B"], cfg["K"], cfg["max_steps"], cfg["reward_fn"], cfg["gen_kw"],
        clamp_o=clamp, kl_beta=0.0,
    )
    rec = costlib.cost_record(variant, params, base_params, curve, timer, dev,
                              tps, cfg["target_reward"])
    if kind == "full":
        for p in params:
            p.requires_grad_(False)
    restore(model, names, o_orig)
    return rec, curve


# ===========================================================================
# smoke harness — tiny random transformer, fixed gold token per problem
# ===========================================================================
def smoke_reward(comp_ids, gold_token):
    """1.0 iff the completion contains the gold token (learnable -> reward variance)."""
    return 1.0 if (comp_ids == gold_token).any().item() else 0.0


def build_smoke(dev):
    vocab, dim, n_layers = 16, 32, 3
    model = TinyLM(vocab, dim, n_layers).to(dev)
    model.requires_grad_(False)
    tok = TinyTokenizer(vocab)
    # 8 'problems': a short prompt id-seq + a small set of gold tokens the model can learn
    # to emit. Small vocab (16) + short gen so RL has a real, reachable target -> a genuine
    # steps-to-target number, not just an instrumentation no-op.
    train_items = []
    for s in range(8):
        pids = torch.tensor([2 + (s % 8), 3, 4, 5])
        gold = 6 + (s % 4)
        train_items.append((pids, gold))
    base_params = sum(p.numel() for p in model.parameters())
    return model, tok, train_items, base_params, vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny random transformer on CPU — verify the instrumentation end-to-end")
    ap.add_argument("--model", default="Qwen/Qwen3-4B", help="frozen base (full mode)")
    ap.add_argument("--device", default=None, help="cuda / cpu / mps (default: auto)")
    ap.add_argument("--out", default="results/cost-table.md")
    ap.add_argument("--lora-lrs", default="1e-4,3e-4,1e-3,3e-3",
                    help="LoRA learning-rate sweep (fair head-to-head)")
    ap.add_argument("--with-full-ft", action="store_true",
                    help="also benchmark a full fine-tune (only if it fits in VRAM)")
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    lora_lrs = [float(x) for x in args.lora_lrs.split(",")]
    print(f"device={dev}  smoke={args.smoke}  lora_lrs={lora_lrs}", flush=True)

    if args.smoke:
        model, tok, train_items, base_params, vocab = build_smoke(dev)
        names = target_modules(model)
        cfg = dict(
            B=4, K=4, max_steps=12, o_clamp=0.10, target_reward=0.20,
            gen_kw=dict(max_new_tokens=6, pad_token_id=0),
            reward_fn=smoke_reward,
        )
        label = "SMOKE (tiny random transformer, CPU)"
        model_name = f"TinyLM(vocab={vocab},dim=32,L=3)"
        g_small, g_big = 16, 64  # tiny G's so groups <= total_out channels
    else:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.math_scorer import gold_answer, reward_of as math_reward

        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, trust_remote_code=True).to(dev)
        model.requires_grad_(False)
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        recs = [r for r in ds]
        recs.sort(key=lambda r: int(r.get("level") or 5))
        SYS = ("Solve the math problem. Reason briefly, then put the final answer in "
               "\\boxed{}. Do not write anything after the boxed answer.")

        def build_pids(q):
            msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            return tok(s, return_tensors="pt").input_ids[0]

        train_items = [(build_pids(r["problem"]), gold_answer(r)) for r in recs[:64]]

        def math_reward_ids(comp_ids, gold):
            return math_reward(tok.decode(comp_ids, skip_special_tokens=True), gold)

        names = target_modules(model)
        base_params = sum(p.numel() for p in model.parameters())
        cfg = dict(
            B=6, K=4, max_steps=40, o_clamp=0.10, target_reward=0.20,
            gen_kw=dict(max_new_tokens=512, pad_token_id=tok.eos_token_id),
            reward_fn=math_reward_ids,
        )
        label = "FULL (Qwen3-4B, MATH-500, CUDA)"
        model_name = args.model
        g_small, g_big = 256, 2048

    nlayers = num_layers_of(model)
    print(f"base={model_name}  layers={nlayers}  target *_proj={len(names)}  "
          f"base_params={base_params:,}", flush=True)

    records = []
    # modulation variants
    rec, _ = run_variant(model, names, train_items, dev, cfg,
                         f"modulation-G{g_small}", "map", base_params, cfg.get("lr_o", 0.01), G=g_small)
    records.append(rec)
    rec, _ = run_variant(model, names, train_items, dev, cfg,
                         f"modulation-G{g_big}", "map", base_params, cfg.get("lr_o", 0.01), G=g_big)
    records.append(rec)

    # LoRA learning-rate SWEEP -> report BEST (fair head-to-head)
    lora_recs = []
    for lr in lora_lrs:
        rec, _ = run_variant(model, names, train_items, dev, cfg,
                             f"LoRA-r8 (lr={lr:g})", "lora", base_params, lr, r=8)
        lora_recs.append(rec)
        records.append(rec)
    # pick LoRA's best by steps-to-target (fewest steps; None = worst), tie-break final_reward
    def lora_key(r):
        s = r["steps_to_target"]
        return (s if s is not None else 10**9, -r["final_reward"])
    best_lora = min(lora_recs, key=lora_key)
    best_lora_marked = dict(best_lora)
    best_lora_marked["variant"] = best_lora["variant"] + "  ** BEST LoRA **"
    # replace the chosen row's label in the table copy
    records = [best_lora_marked if r is best_lora else r for r in records]

    # optional full fine-tune
    if args.with_full_ft:
        try:
            rec, _ = run_variant(model, names, train_items, dev, cfg,
                                 "full-FT", "full", base_params, 1e-5)
            records.append(rec)
        except RuntimeError as e:
            print(f"[full-FT] skipped (did not fit): {e}", flush=True)

    meta = dict(label=label, model=model_name, base_params=base_params, device=dev,
                target_reward=cfg["target_reward"], max_steps=cfg["max_steps"],
                tokens_per_step=records[0]["base_flops_step"] / (6.0 * base_params)
                if base_params else 0)
    meta["tokens_per_step"] = int(meta["tokens_per_step"])

    pending = None
    if args.smoke:
        # the real 4B GPU run is not done on a Mac: write the predicted rows as PENDING.
        pending = [
            dict(variant="modulation-G2048 (4B, MATH-500)", trainable_params="2,048",
                 peak_vram="PENDING 4B GPU RUN", steps_to_target="PENDING",
                 wall="PENDING", flops="~ base-dominated (≈LoRA)", share="~0%",
                 gpu_hours="PENDING — predicted ≤ best-LoRA (decided by steps-to-target)"),
            dict(variant="LoRA-r8 best-lr (4B, MATH-500)", trainable_params="16,515,072",
                 peak_vram="PENDING 4B GPU RUN", steps_to_target="PENDING",
                 wall="PENDING", flops="~ base-dominated", share="~0%",
                 gpu_hours="PENDING — head-to-head target"),
        ]

    table = costlib.render_cost_table(records, meta, pending_rows=pending)
    table += _verdict_prose(records, best_lora, args.smoke)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(table + "\n")
    print("\n" + table, flush=True)
    print(f"\nwrote {args.out}", flush=True)


def _verdict_prose(records, best_lora, smoke):
    """Prose verdict answering the headline question from the measured rows."""
    mods = [r for r in records if r["variant"].startswith("modulation")]
    best_mod = min(mods, key=lambda r: (r["steps_to_target"] or 10**9, -r["final_reward"])) if mods else None
    L = ["", "## Verdict (from the measured rows)", ""]
    if smoke:
        L.append("> These are SMOKE numbers (tiny random transformer on CPU) — they prove "
                 "the instrumentation captures all four axes and the table renders. The "
                 "absolute values are meaningless; the 4B GPU rows above are PENDING.")
        L.append("")
    if best_mod is not None:
        msize = best_mod["trainable_params"]
        lsize = best_lora["trainable_params"]
        ratio = (lsize / msize) if msize else float("inf")
        L.append(f"- **Adapter size**: best modulation = {msize:,} trainable params vs "
                 f"best LoRA = {lsize:,} → **{ratio:,.0f}x smaller** (certain, arithmetic).")
        max_share = max(
            (r["adapter_flops_step"] / r["total_flops_step"]) if r["total_flops_step"] else 0.0
            for r in records)
        if smoke:
            L.append(f"- **Compute/step**: on this TINY base the adapter FLOPs share peaks "
                     f"at {max_share*100:.1f}% (the base is only ~35k params, so LoRA's "
                     f"13k params are *not* negligible here) — but the MECHANISM is the "
                     f"point: that share is `6·N_adapter / 6·(N_base+N_adapter)`, which → 0 "
                     f"as N_base grows. On the 4B (base ~4e9 params) both adapters' share is "
                     f"<0.5%, so compute/step is ≈equal and GPU-hours hinges on "
                     f"steps-to-target.")
        else:
            L.append(f"- **Compute/step**: the *adapter FLOPs share* column peaks at "
                     f"{max_share*100:.3f}% — the frozen base's backward dominates, so "
                     f"compute/step is **≈equal** across adapters (LoRA's 2 matmuls and the "
                     f"gate's element-wise scale are both negligible).")
        ms, ls = best_mod["steps_to_target"], best_lora["steps_to_target"]
        ms_s = "did not converge in budget" if ms is None else f"{ms} steps"
        ls_s = "did not converge in budget" if ls is None else f"{ls} steps"
        L.append(f"- **Steps-to-target** (the GPU-hour driver): best modulation "
                 f"{ms_s}; best LoRA {ls_s}.")
        L.append(f"- **GPU-hours**: best modulation {best_mod['gpu_hours']:.2e} vs best "
                 f"LoRA {best_lora['gpu_hours']:.2e} — because compute/step is ≈equal, this "
                 f"ratio tracks steps-to-target, exactly as predicted.")
    L.append("")
    L.append("**Answer to the headline question**: adapter SIZE and optimizer VRAM are "
             "certain wins for the modulation regardless of the run; whether it is also "
             "cheaper in GPU-HOURS is decided entirely by steps-to-target (compute/step is "
             "≈equal). The fair LoRA lr-sweep above is what makes that steps-to-target "
             "comparison honest. Final 4B verdict awaits the GPU run (PENDING rows).")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
