"""MATH-500 RL for a 9B Llama-style model on Colab.

This runner is the low-memory version of the 4B MATH-500 experiment. It keeps the
same comparison surface:
  - Map-G{256,2048}: one shared multiplicative per-group gate, and
  - LoRA-r8: additive low-rank adapters with a small learning-rate sweep.

Defaults target a public, non-gated 9B model whose modules match src.adapters'
`model.layers.*_proj` discovery:

    python experiments/math500_rl_9b.py --model 01-ai/Yi-1.5-9B-Chat

It writes a text report, a structured results.json for figures, and a cost table.
"""

import argparse
import base64
import csv
import gc
import json
import math
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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
from src.generation_utils import generation_kwargs, stop_token_ids, trim_completion  # noqa: E402
from src.math_scorer import extract_answer, gold_answer, reward_of  # noqa: E402
from src import costlib  # noqa: E402


torch.manual_seed(0)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

COST_TARGET_REWARD = 0.20
SYS = (
    "Solve the math problem. Reason briefly. The final line must contain only "
    "\\boxed{...} with the final answer inside the braces. Stop immediately after "
    "the closing brace."
)


def parse_ints(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_floats(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def safe_key(key):
    return key.replace("/", "_").replace(" ", "_").replace("*", "").replace("=", "-")


def ensure_dirs(root):
    for name in ("checkpoints", "curves", "cases", "variant_summaries", "map_params"):
        os.makedirs(os.path.join(root, name), exist_ok=True)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def write_curve_files(root, key, reward_curve, kl_curve, step_times, token_counts):
    skey = safe_key(key)
    rows = []
    for i, r in enumerate(reward_curve):
        rows.append({
            "step": i,
            "reward": r,
            "kl": kl_curve[i] if i < len(kl_curve) else None,
            "step_s": step_times[i] if i < len(step_times) else None,
            "tokens": token_counts[i] if i < len(token_counts) else None,
        })
    write_json(os.path.join(root, "curves", f"{skey}.json"), rows)
    csv_path = os.path.join(root, "curves", f"{skey}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "reward", "kl", "step_s", "tokens"])
        w.writeheader()
        w.writerows(rows)


def adapter_state_dict(kind, trainable):
    if kind == "map":
        return {"o": trainable[0].detach().float().cpu()}
    return {f"param_{i}": p.detach().cpu() for i, p in enumerate(trainable)}


def save_adapter_checkpoint(root, key, kind, trainable, meta, step=None, final=False):
    skey = safe_key(key)
    payload = dict(meta)
    payload.update({
        "variant": key,
        "kind": kind,
        "step": step,
        "final": final,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "state": adapter_state_dict(kind, trainable),
    })
    latest = os.path.join(root, "checkpoints", f"{skey}_latest.pt")
    torch.save(payload, latest)
    if final:
        torch.save(payload, os.path.join(root, "checkpoints", f"{skey}_final.pt"))
    elif kind == "map" and step is not None:
        # Map checkpoints are tiny; keep a step trace so the learned trajectory is reproducible.
        torch.save(payload, os.path.join(root, "checkpoints", f"{skey}_step{step:04d}.pt"))


def print_recovery_blob(tag, payload):
    raw = json.dumps(payload, separators=(",", ":")).encode()
    b64 = base64.b64encode(raw).decode()
    print(f"{tag}_BEGIN", flush=True)
    for i in range(0, len(b64), 4096):
        print(b64[i:i + 4096], flush=True)
    print(f"{tag}_END", flush=True)


def emit_map_recovery(root, key, trainable, meta, reward_curve, kl_curve):
    payload = dict(meta)
    payload.update({
        "variant": key,
        "kind": "map",
        "o": [float(x) for x in trainable[0].detach().float().cpu().tolist()],
        "reward_curve": [float(x) for x in reward_curve],
        "kl_curve": [float(x) for x in kl_curve],
    })
    write_json(os.path.join(root, "map_params", f"{safe_key(key)}_o.json"), payload)
    print_recovery_blob(f"MAP_PARAM_{safe_key(key)}", payload)


def save_cases(root, key, cases):
    write_json(
        os.path.join(root, "cases", f"{safe_key(key)}.json"),
        [dict(problem=q, model=t, pred=p, gold=g) for q, t, p, g in cases],
    )


def build_prompt(tok, q):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"System: {SYS}\n\nUser: {q}\n\nAssistant:"


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def comp_logp_and_kl(model, names, prompt_ids, comp_ids, dev):
    """Base forward first, then policy forward, keeping KL tensors to completion tokens."""
    ids = torch.cat([prompt_ids, comp_ids], 0)[None].to(dev)
    tgt = ids[0, 1:]
    n_prompt = prompt_ids.numel()
    comp_start = max(0, n_prompt - 1)
    tgt_comp = tgt[comp_start:]

    with torch.no_grad(), base_forward(model, names):
        base_logits = model(ids).logits[0, comp_start:-1].float()
        base_logp = torch.log_softmax(base_logits, -1)

    logits = model(ids).logits[0, comp_start:-1].float()
    logp = torch.log_softmax(logits, -1)
    tok_lp = logp.gather(1, tgt_comp[:, None]).squeeze(1)
    sum_lp = tok_lp.sum()

    p = logp.exp()
    kl = (p * (logp - base_logp)).sum(-1).mean()
    return sum_lp, kl


@torch.no_grad()
def evaluate(model, tok, items, dev, label, eval_batch, max_new_eval, collect_cases, gen_extra, stop_ids):
    model.eval()
    correct, cases = 0, []
    t0 = time.time()
    prev_side = tok.padding_side
    tok.padding_side = "left"
    done = 0
    for b0 in range(0, len(items), eval_batch):
        batch = items[b0:b0 + eval_batch]
        prompts = [build_prompt(tok, q) for q, _ in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
        out = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_eval,
            **gen_extra,
        )
        gen = out[:, enc.input_ids.shape[1]:]
        texts = [
            tok.decode(trim_completion(row, stop_ids)[0], skip_special_tokens=True)
            for row in gen
        ]
        for (q, gold), text in zip(batch, texts):
            pred = extract_answer(text)
            correct += int(pred == gold and bool(gold))
            if len(cases) < collect_cases:
                cases.append((q, text, pred, gold))
        done += len(batch)
        print(
            f"  [eval {label}] {done}/{len(items)}  "
            f"acc_sofar={correct/max(1, done):.3f}  "
            f"{(time.time()-t0)/max(1, done):.1f}s/q",
            flush=True,
        )
    tok.padding_side = prev_side
    return correct, len(items), cases


def train_grpo(model, tok, names, train_items, trainable, cfg, dev, label, telem_fn=None, clamp_o=None):
    model.train()
    opt = torch.optim.Adam(trainable, lr=cfg["lr"])
    t0 = time.time()
    curve, kl_curve = [], []
    timer = costlib.StepTimer().start()
    tok_acc = []
    step_times = []
    gen_extra = cfg["gen_extra"]
    stop_ids = cfg["stop_ids"]

    for step in range(cfg["max_steps"]):
        if time.time() - t0 > cfg["time_budget_s"]:
            print(f"[{label}] time budget hit at step {step}", flush=True)
            break

        batch = [train_items[(step * cfg["B"] + i) % len(train_items)] for i in range(cfg["B"])]
        step_r, nz_groups, kl_acc, did_backward = [], 0, [], False
        step_tokens = 0
        opt.zero_grad(set_to_none=True)

        for q, gold in batch:
            prompt = build_prompt(tok, q)
            pids = tok(prompt, return_tensors="pt").input_ids[0].to(dev)
            with torch.no_grad():
                gen = model.generate(
                    pids[None],
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    num_return_sequences=cfg["K"],
                    max_new_tokens=cfg["max_new"],
                    **gen_extra,
                )
            comps = [trim_completion(gen[k, pids.numel():], stop_ids)[0] for k in range(cfg["K"])]
            texts = [tok.decode(c, skip_special_tokens=True) for c in comps]
            rs = torch.tensor([reward_of(t, gold) for t in texts], dtype=torch.float32)
            step_r.append(rs.mean().item())
            if rs.std() > 1e-6:
                nz_groups += 1
            adv = (rs - rs.mean()) / (rs.std() + 1e-4)

            for k in range(cfg["K"]):
                step_tokens += pids.numel() + comps[k].numel()
                lp, kl = comp_logp_and_kl(model, names, pids, comps[k], dev)
                kl_acc.append(kl.item())
                pg = -adv[k].to(dev).detach() * lp if adv[k].abs() >= 1e-6 else 0.0 * lp
                ((pg + cfg["beta_kl"] * kl) / (cfg["B"] * cfg["K"])).backward()
                did_backward = True

        if did_backward:
            opt.step()
            if clamp_o is not None:
                with torch.no_grad():
                    trainable[0].clamp_(-clamp_o, clamp_o)

        mr = sum(step_r) / len(step_r)
        mkl = sum(kl_acc) / len(kl_acc) if kl_acc else 0.0
        curve.append(mr)
        kl_curve.append(mkl)
        tok_acc.append(step_tokens)
        timer.tick()
        step_times.append(timer.per_step[-1])
        tline = telem_fn(trainable) if telem_fn else ""
        save_root = cfg.get("save_root")
        if save_root:
            write_curve_files(save_root, label, curve, kl_curve, step_times, tok_acc)
            append_jsonl(
                os.path.join(save_root, "progress.jsonl"),
                {
                    "event": "train_step",
                    "variant": label,
                    "step": step,
                    "reward": mr,
                    "kl": mkl,
                    "nz_groups": nz_groups,
                    "batch": cfg["B"],
                    "step_s": timer.per_step[-1],
                    "elapsed_s": time.time() - t0,
                    "tokens": step_tokens,
                    "telemetry": tline,
                },
            )
            save_every = max(1, int(cfg.get("save_every", 20)))
            kind = cfg.get("kind", "adapter")
            if kind == "map" or step < 5 or (step + 1) % save_every == 0 or step + 1 == cfg["max_steps"]:
                save_adapter_checkpoint(
                    save_root,
                    label,
                    kind,
                    trainable,
                    cfg.get("adapter_meta", {}),
                    step=step,
                    final=False,
                )
                if kind == "map" and ((step + 1) % save_every == 0 or step + 1 == cfg["max_steps"]):
                    meta = dict(cfg.get("adapter_meta", {}))
                    meta["partial_step"] = step
                    emit_map_recovery(save_root, f"{label}-step{step:04d}", trainable, meta, curve, kl_curve)
        print_every = max(1, int(cfg.get("print_every", 20)))
        if step < 5 or step % print_every == 0:
            print(
                f"[{label}] step {step:3d}  reward={mr:.3f}  nz={nz_groups}/{cfg['B']}  "
                f"KL={mkl:.4f}  {tline}  step_s={timer.per_step[-1]:.1f}  "
                f"elapsed={time.time()-t0:.0f}s",
                flush=True,
            )

    tokens_per_step = int(sum(tok_acc) / len(tok_acc)) if tok_acc else 0
    return curve, kl_curve, timer, tokens_per_step


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
        out.append(f"  pred={pred!r} gold={gold!r} {'OK' if (pred == gold and gold) else 'WRONG'}")
    return "\n".join(out)


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def adapter_sanity(model, tok, names, sample_item, dev, result_root, lora_r):
    """Check adapter identity, perturbation, and restore before training."""
    q, _ = sample_item
    ids = tok(build_prompt(tok, q), return_tensors="pt").input_ids.to(dev)

    def last_logits():
        with torch.no_grad():
            return model(ids).logits[:, -1, :].float().detach()

    model.eval()
    base = last_logits()
    rows = []

    originals = [getattr(*get_parent(model, n)) for n in names]
    params, total_out = install_direct_map(model, names, 2048)
    ident = (last_logits() - base).abs().max().item()
    with torch.no_grad():
        params[0].fill_(0.01)
    changed = (last_logits() - base).abs().max().item()
    restore(model, names, originals)
    restored = (last_logits() - base).abs().max().item()
    rows.append({
        "adapter": "Map-G2048",
        "identity_max_abs": ident,
        "perturb_max_abs": changed,
        "restore_max_abs": restored,
        "trainable_params": int(params[0].numel()),
        "total_out": int(total_out),
        "pass": ident < 1e-4 and changed > 1e-7 and restored < 1e-4,
    })

    originals = [getattr(*get_parent(model, n)) for n in names]
    params = install_lora(model, names, lora_r)
    ident = (last_logits() - base).abs().max().item()
    with torch.no_grad():
        if len(params) < 2:
            raise RuntimeError("LoRA install returned too few parameters")
        params[1].fill_(0.01)
    changed = (last_logits() - base).abs().max().item()
    restore(model, names, originals)
    restored = (last_logits() - base).abs().max().item()
    rows.append({
        "adapter": f"LoRA-r{lora_r}",
        "identity_max_abs": ident,
        "perturb_max_abs": changed,
        "restore_max_abs": restored,
        "trainable_params": int(sum(p.numel() for p in params)),
        "pass": ident < 1e-4 and changed > 1e-7 and restored < 1e-4,
    })

    write_json(os.path.join(result_root, "adapter_sanity.json"), rows)
    print("ADAPTER_SANITY " + json.dumps(rows, separators=(",", ":")), flush=True)
    if not all(r["pass"] for r in rows):
        raise RuntimeError("adapter sanity failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="01-ai/Yi-1.5-9B-Chat")
    ap.add_argument("--out", default="results/9b-math500/results.txt")
    ap.add_argument("--cost-out", default="results/9b-math500/cost-table.md")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--attn-impl", default="", help="optional HF attn_implementation, e.g. sdpa")
    ap.add_argument("--hardware-label", default="Colab")
    ap.add_argument("--model-label", default="9B")
    ap.add_argument("--max-steps", type=int, default=350)
    ap.add_argument("--time-budget-s", type=float, default=6 * 3600)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-new-eval", type=int, default=512)
    ap.add_argument("--eval-batch", type=int, default=1)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--g-sweep", default="256,2048")
    ap.add_argument("--lr-o", type=float, default=0.005)
    ap.add_argument("--o-clamp", type=float, default=0.10)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-lr-sweep", default="1e-4,1e-3")
    ap.add_argument("--beta-kl", type=float, default=0.05)
    ap.add_argument("--n-cases", type=int, default=3)
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--baseline-json", default="", help="reuse an existing baseline.json instead of re-evaluating")
    ap.add_argument("--skip-baseline-eval", action="store_true")
    ap.add_argument("--skip-final-eval", action="store_true")
    ap.add_argument("--no-final-checkpoint", action="store_true")
    ap.add_argument("--print-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--min-train-level", type=int, default=3)
    ap.add_argument("--max-train-level", type=int, default=5)
    ap.add_argument("--train-selection", choices=["head", "stride"], default="stride")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    args = ap.parse_args()

    dev = args.device
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    g_sweep = parse_ints(args.g_sweep)
    lora_lr_sweep = parse_floats(args.lora_lr_sweep)
    cfg_base = dict(
        B=args.B,
        K=args.K,
        max_steps=args.max_steps,
        time_budget_s=args.time_budget_s,
        max_new=args.max_new,
        beta_kl=args.beta_kl,
        print_every=args.print_every,
    )
    result_root = os.path.dirname(args.out) or "."
    ensure_dirs(result_root)
    os.makedirs(os.path.join(result_root, "map_params"), exist_ok=True)

    print(f"device={dev} dtype={dt} model={args.model}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(
            f"cuda_device={torch.cuda.get_device_name(0)} "
            f"total_mem_gb={props.total_memory/1024**3:.2f}",
            flush=True,
        )

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    gen_extra = generation_kwargs(tok)
    stop_ids = stop_token_ids(tok)
    cfg_base["gen_extra"] = gen_extra
    cfg_base["stop_ids"] = stop_ids
    print(f"stop_token_ids={stop_ids} gen_extra={gen_extra}", flush=True)
    load_kwargs = dict(torch_dtype=dt, trust_remote_code=True, low_cpu_mem_usage=True)
    if args.attn_impl:
        load_kwargs["attn_implementation"] = args.attn_impl
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).to(dev)
    model.requires_grad_(False)
    if not args.no_grad_ckpt:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    nlayers = num_layers_of(model)
    base_params = sum(p.numel() for p in model.parameters())
    print(
        f"num_hidden_layers={nlayers} base_params={base_params:,} "
        f"grad_ckpt={not args.no_grad_ckpt}",
        flush=True,
    )
    print(
        f"config: B={args.B} K={args.K} MAX_STEPS={args.max_steps} "
        f"MAX_NEW={args.max_new} MAX_NEW_EVAL={args.max_new_eval} N_EVAL={args.n_eval} "
        f"LR_O={args.lr_o} O_CLAMP={args.o_clamp} LORA_R={args.lora_r} "
        f"G_SWEEP={g_sweep} LORA_LR_SWEEP={lora_lr_sweep}",
        flush=True,
    )
    run_config = dict(
        model=args.model,
        device=str(dev),
        dtype=str(dt),
        hardware_label=args.hardware_label,
        model_label=args.model_label,
        B=args.B,
        K=args.K,
        max_steps=args.max_steps,
        time_budget_s=args.time_budget_s,
        max_new=args.max_new,
        max_new_eval=args.max_new_eval,
        eval_batch=args.eval_batch,
        n_eval=args.n_eval,
        print_every=args.print_every,
        save_every=args.save_every,
        skip_baseline_eval=args.skip_baseline_eval,
        skip_final_eval=args.skip_final_eval,
        no_final_checkpoint=args.no_final_checkpoint,
        min_train_level=args.min_train_level,
        max_train_level=args.max_train_level,
        train_selection=args.train_selection,
        stop_token_ids=stop_ids,
        beta_kl=args.beta_kl,
        lr_o=args.lr_o,
        o_clamp=args.o_clamp,
        lora_r=args.lora_r,
        g_sweep=g_sweep,
        lora_lr_sweep=lora_lr_sweep,
        cost_target_reward=COST_TARGET_REWARD,
        grad_ckpt=not args.no_grad_ckpt,
    )
    write_json(os.path.join(result_root, "run_config.json"), run_config)

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    all_items = [(r["problem"], gold_answer(r)) for r in ds]
    eval_items = all_items[:args.n_eval]
    train_records = [r for r in ds][args.n_eval:] if len(ds) > args.n_eval else [r for r in ds]
    train_records = [
        r for r in train_records
        if args.min_train_level <= int(r.get("level") or 5) <= args.max_train_level
    ]
    if not train_records:
        raise RuntimeError("empty train pool after level filtering")
    train_records.sort(key=lambda r: int(r.get("level") or 5))
    train_pool = [(r["problem"], gold_answer(r)) for r in train_records]
    needed = args.B * args.max_steps + 64
    if args.train_selection == "stride":
        train_items = [train_pool[int((i * len(train_pool)) / needed) % len(train_pool)] for i in range(needed)]
    else:
        train_items = [train_pool[i % len(train_pool)] for i in range(needed)]
    level_hist = {}
    for r in train_records:
        level = str(int(r.get("level") or 5))
        level_hist[level] = level_hist.get(level, 0) + 1
    print(
        f"train pool: {len(train_pool)} problems levels={level_hist} selection={args.train_selection}",
        flush=True,
    )

    names = target_modules(model)
    print(f"target *_proj linears: {len(names)}", flush=True)
    if not names:
        raise RuntimeError(
            "No target projection linears found. This runner needs Llama-style "
            "module names under model.layers.* ending in _proj."
        )
    write_json(
        os.path.join(result_root, "target_modules.json"),
        {"count": len(names), "names": names, "num_hidden_layers": nlayers, "base_params": base_params},
    )
    adapter_sanity(model, tok, names, train_items[0], dev, result_root, args.lora_r)

    if args.skip_baseline_eval:
        k_base, n_base, acc_base, ci_base, cases_base = 0, 0, 0.0, (0.0, 0.0), []
        baseline_payload = dict(acc=acc_base, k=k_base, n=n_base, ci=list(ci_base), cases=[])
        print("[baseline] skipped by --skip-baseline-eval", flush=True)
    elif args.baseline_json:
        with open(args.baseline_json) as f:
            baseline_payload = json.load(f)
        k_base, n_base = int(baseline_payload["k"]), int(baseline_payload["n"])
        acc_base = float(baseline_payload["acc"])
        ci_base = tuple(baseline_payload["ci"])
        cases_base = [
            (c["problem"], c["model"], c["pred"], c["gold"])
            for c in baseline_payload.get("cases", [])
        ]
        print(
            f"[baseline] loaded {args.baseline_json}: acc={acc_base:.4f} ({k_base}/{n_base}) "
            f"CI [{ci_base[0]:.3f},{ci_base[1]:.3f}]",
            flush=True,
        )
    else:
        k_base, n_base, cases_base = evaluate(
            model, tok, eval_items, dev, "base", args.eval_batch, args.max_new_eval,
            args.n_cases, gen_extra, stop_ids
        )
        acc_base = k_base / n_base
        ci_base = wilson_ci(k_base, n_base)
        print(
            f"[baseline] MATH-500 acc = {acc_base:.4f} ({k_base}/{n_base}) "
            f"CI [{ci_base[0]:.3f},{ci_base[1]:.3f}]",
            flush=True,
        )
        baseline_payload = dict(
            acc=acc_base,
            k=k_base,
            n=n_base,
            ci=list(ci_base),
            cases=[dict(problem=q, model=t, pred=p, gold=g) for q, t, p, g in cases_base],
        )
    write_json(os.path.join(result_root, "baseline.json"), baseline_payload)
    save_cases(result_root, "baseline", cases_base)
    append_jsonl(os.path.join(result_root, "progress.jsonl"), {"event": "baseline_done", **baseline_payload})
    if acc_base >= 0.85:
        print("STOP: baseline is at ceiling for this task.", flush=True)
        return
    if args.baseline_only:
        print("[baseline-only] stopping after baseline.", flush=True)
        return

    results = {}
    cost_records = []

    for G in g_sweep:
        key = f"Map-G{G}"
        print(f"\n{'=' * 50}\nDIRECT MAP G={G}\n{'=' * 50}", flush=True)
        o_orig = [getattr(*get_parent(model, n)) for n in names]
        costlib.reset_peak_vram(dev)
        params, total_out = install_direct_map(model, names, G)
        n_par = sum(p.numel() for p in params)
        print(f"[{key}] trainable params = {n_par} total_out={total_out}", flush=True)
        adapter_meta = dict(
            model=args.model,
            model_label=args.model_label,
            hardware_label=args.hardware_label,
            n_par=n_par,
            total_out=total_out,
            alpha_mod=ALPHA_MOD,
            lr=args.lr_o,
            o_clamp=args.o_clamp,
            target_modules=len(names),
            base_params=base_params,
            run_config=run_config,
        )
        cfg = dict(
            cfg_base,
            lr=args.lr_o,
            save_root=result_root,
            save_every=args.save_every,
            kind="map",
            adapter_meta=adapter_meta,
        )
        curve, kl_curve, timer, tps = train_grpo(
            model, tok, names, train_items, params, cfg, dev, key, o_telem, args.o_clamp
        )
        cost_rec = costlib.cost_record(key, params, base_params, curve, timer, dev, tps, COST_TARGET_REWARD)
        cost_records.append(cost_rec)
        final_mao = params[0].detach().abs().mean().item()
        final_mg = (1.0 + ALPHA_MOD * params[0].detach()).abs().max().item()
        if not args.no_final_checkpoint:
            save_adapter_checkpoint(result_root, key, "map", params, adapter_meta, step=len(curve) - 1, final=True)
        emit_map_recovery(result_root, key, params, adapter_meta, curve, kl_curve)
        if args.skip_final_eval:
            k_g, n_g, cases_g = 0, 0, []
        else:
            k_g, n_g, cases_g = evaluate(
                model, tok, eval_items, dev, key, args.eval_batch, args.max_new_eval,
                args.n_cases, gen_extra, stop_ids
            )
        restore(model, names, o_orig)
        cleanup_cuda()
        acc_g = k_g / n_g if n_g else 0.0
        ci_g = wilson_ci(k_g, n_g) if n_g else (0.0, 0.0)
        fkl = sum(kl_curve[-5:]) / max(1, len(kl_curve[-5:])) if kl_curve else 0.0
        print(f"[{key}] acc={acc_g:.4f} ({k_g}/{n_g}) CI [{ci_g[0]:.3f},{ci_g[1]:.3f}] KL={fkl:.4f}", flush=True)
        results[key] = dict(
            kind="map",
            n_par=n_par,
            curve=curve,
            kl_curve=kl_curve,
            k=k_g,
            n=n_g,
            acc=acc_g,
            ci=ci_g,
            final_kl=fkl,
            cases=cases_g,
            mean_abs_o=final_mao,
            max_gate=final_mg,
        )
        save_cases(result_root, key, cases_g)
        write_json(os.path.join(result_root, "variant_summaries", f"{safe_key(key)}.json"), results[key] | {"cost_record": cost_rec})
        append_jsonl(
            os.path.join(result_root, "progress.jsonl"),
            {"event": "variant_done", "variant": key, "acc": acc_g, "k": k_g, "n": n_g, "ci": list(ci_g), "final_kl": fkl},
        )

    lora_variants = []
    for lr_lora in lora_lr_sweep:
        key = f"LoRA-r{args.lora_r}-lr{lr_lora:g}"
        print(f"\n{'=' * 50}\nLoRA r={args.lora_r} lr={lr_lora:g}\n{'=' * 50}", flush=True)
        o_orig = [getattr(*get_parent(model, n)) for n in names]
        costlib.reset_peak_vram(dev)
        params = install_lora(model, names, args.lora_r)
        n_par = sum(p.numel() for p in params)
        print(f"[{key}] trainable params = {n_par}", flush=True)
        adapter_meta = dict(
            model=args.model,
            model_label=args.model_label,
            hardware_label=args.hardware_label,
            n_par=n_par,
            lora_r=args.lora_r,
            lr=lr_lora,
            target_modules=len(names),
            base_params=base_params,
            run_config=run_config,
        )
        cfg = dict(
            cfg_base,
            lr=lr_lora,
            save_root=result_root,
            save_every=args.save_every,
            kind="lora",
            adapter_meta=adapter_meta,
        )
        curve, kl_curve, timer, tps = train_grpo(
            model, tok, names, train_items, params, cfg, dev, key, lora_telem
        )
        cost_rec = costlib.cost_record(key, params, base_params, curve, timer, dev, tps, COST_TARGET_REWARD)
        cost_records.append(cost_rec)
        if not args.no_final_checkpoint:
            save_adapter_checkpoint(result_root, key, "lora", params, adapter_meta, step=len(curve) - 1, final=True)
        if args.skip_final_eval:
            k_l, n_l, cases_l = 0, 0, []
        else:
            k_l, n_l, cases_l = evaluate(
                model, tok, eval_items, dev, key, args.eval_batch, args.max_new_eval,
                args.n_cases, gen_extra, stop_ids
            )
        restore(model, names, o_orig)
        cleanup_cuda()
        acc_l = k_l / n_l if n_l else 0.0
        ci_l = wilson_ci(k_l, n_l) if n_l else (0.0, 0.0)
        fkl = sum(kl_curve[-5:]) / max(1, len(kl_curve[-5:])) if kl_curve else 0.0
        print(f"[{key}] acc={acc_l:.4f} ({k_l}/{n_l}) CI [{ci_l[0]:.3f},{ci_l[1]:.3f}] KL={fkl:.4f}", flush=True)
        results[key] = dict(
            kind="lora",
            lr=lr_lora,
            n_par=n_par,
            curve=curve,
            kl_curve=kl_curve,
            k=k_l,
            n=n_l,
            acc=acc_l,
            ci=ci_l,
            final_kl=fkl,
            cases=cases_l,
        )
        save_cases(result_root, key, cases_l)
        write_json(os.path.join(result_root, "variant_summaries", f"{safe_key(key)}.json"), results[key] | {"cost_record": cost_rec})
        append_jsonl(
            os.path.join(result_root, "progress.jsonl"),
            {"event": "variant_done", "variant": key, "acc": acc_l, "k": k_l, "n": n_l, "ci": list(ci_l), "final_kl": fkl},
        )
        lora_variants.append(key)

    def lora_score(k):
        r = results[k]
        s2t = costlib.steps_to_target(r["curve"], COST_TARGET_REWARD)
        return (-r["acc"], s2t if s2t else 10**9, -r["final_kl"])

    best_lora_key = min(lora_variants, key=lora_score) if lora_variants else None
    if best_lora_key is not None:
        results[best_lora_key]["is_best_lora"] = True
        print(f"\n[best-LoRA] -> {best_lora_key} (acc={results[best_lora_key]['acc']:.4f})", flush=True)
    else:
        print("\n[best-LoRA] skipped: no LoRA variants in this chunk", flush=True)

    order = [f"Map-G{G}" for G in g_sweep] + lora_variants
    train_summary = {
        "baseline": baseline_payload,
        "order": order,
        "skip_baseline_eval": bool(args.skip_baseline_eval),
        "skip_final_eval": bool(args.skip_final_eval),
        "no_final_checkpoint": bool(args.no_final_checkpoint),
        "variants": {
            key: {
                "kind": results[key]["kind"],
                "reward_curve": [float(x) for x in results[key]["curve"]],
                "kl_curve": [float(x) for x in results[key]["kl_curve"]],
                "final_kl": float(results[key]["final_kl"]),
                "acc": float(results[key]["acc"]),
                "n": int(results[key]["n"]),
                "steps_to_target": costlib.steps_to_target(results[key]["curve"], COST_TARGET_REWARD),
                "mean_abs_o": results[key].get("mean_abs_o"),
                "max_gate": results[key].get("max_gate"),
            }
            for key in order
        },
    }
    write_json(os.path.join(result_root, "train_summary.json"), train_summary)
    print("TRAIN_SUMMARY_JSON " + json.dumps(train_summary, separators=(",", ":")), flush=True)

    def overlap(a, b):
        return a[0] <= b[1] and b[0] <= a[1]

    lines = [
        "=" * 78,
        f"MATH-500 - modulation vs LoRA on frozen {args.model_label} ({args.hardware_label})",
        "=" * 78,
        f"model: {args.model}",
        f"num_hidden_layers={nlayers} target linears={len(names)} base_params={base_params:,}",
        f"config: B={args.B} K={args.K} MAX_STEPS={args.max_steps} MAX_NEW={args.max_new} "
        f"MAX_NEW_EVAL={args.max_new_eval} N_EVAL={args.n_eval} LR_o={args.lr_o} "
        f"O_CLAMP={args.o_clamp} LORA_R={args.lora_r} LoRA_lr_sweep={lora_lr_sweep} "
        f"best_LoRA={best_lora_key}",
        f"baseline: {acc_base:.4f} ({k_base}/{n_base}) CI [{ci_base[0]:.3f}, {ci_base[1]:.3f}]",
        "",
        f"MATH-500 greedy acc (n={args.n_eval}), Wilson 95% CI:",
    ]
    for key in order:
        r = results[key]
        mark = "  ** BEST LoRA **" if r.get("is_best_lora") else ""
        lines.append(
            f"  {key:<22s} params={r['n_par']:<10d}: acc={r['acc']:.4f} ({r['k']}/{r['n']}) "
            f"CI [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] KL={r['final_kl']:.4f}{mark}"
        )
    lines.append("")
    for key in order:
        r = results[key]
        cl = r["ci"][0] > ci_base[1]
        ob = overlap(r["ci"], ci_base)
        v = "CLEARS baseline" if cl else "OVERLAP" if ob else "BELOW"
        lines.append(f"  {key} vs baseline: {r['acc']-acc_base:+.4f} -> {v}")
    lines.append("")
    for key in order:
        r = results[key]
        if r["kind"] == "map":
            lev = "RISES" if r["final_kl"] > 0.05 else ("partial" if r["final_kl"] > 0.01 else "~0")
            coh = "IN-BAND" if r["mean_abs_o"] <= args.o_clamp + 1e-3 else "OUT-OF-BAND"
            lines.append(
                f"  {key}: KL={r['final_kl']:.4f} -> {lev} "
                f"mean|o|={r['mean_abs_o']:.4f} max_gate={r['max_gate']:.3f} -> {coh}"
            )
        else:
            lines.append(f"  {key}: KL={r['final_kl']:.4f} (LoRA)")
    lines.append("")
    for key in order:
        r = results[key]
        lines.append(f"mean_reward {key} ({len(r['curve'])} steps): " + " ".join(f"{x:.2f}" for x in r["curve"]))
        lines.append(f"mean_KL     {key} ({len(r['kl_curve'])} steps): " + " ".join(f"{x:.3f}" for x in r["kl_curve"]))
    lines.append("\nDECODED CASES (baseline):\n" + fmt_cases(cases_base))
    for key in order:
        lines.append(f"\nDECODED CASES ({key}):\n" + fmt_cases(results[key]["cases"]))

    report = "\n".join(lines)
    print("\n" + report, flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(f"wrote {args.out}", flush=True)

    for rec in cost_records:
        if rec["variant"] == best_lora_key:
            rec["variant"] = rec["variant"] + "  ** BEST LoRA **"
    tps_val = int(cost_records[0]["base_flops_step"] / (6.0 * base_params)) if base_params and cost_records else 0
    meta = dict(
        label=f"{args.model_label} MATH-500 on {args.hardware_label}",
        model=args.model,
        base_params=base_params,
        device=dev,
        target_reward=COST_TARGET_REWARD,
        max_steps=args.max_steps,
        tokens_per_step=tps_val,
    )
    cost_table = costlib.render_cost_table(cost_records, meta)
    os.makedirs(os.path.dirname(args.cost_out) or ".", exist_ok=True)
    with open(args.cost_out, "w") as f:
        f.write(cost_table + "\n")
    print(f"wrote {args.cost_out}", flush=True)

    payload = dict(
        model=args.model,
        device=str(dev),
        dtype=str(dt),
        hardware_label=args.hardware_label,
        model_label=args.model_label,
        num_hidden_layers=nlayers,
        target_proj_linears=len(names),
        base_params=base_params,
        config=dict(
            B=args.B,
            K=args.K,
            max_steps=args.max_steps,
            time_budget_s=args.time_budget_s,
            max_new=args.max_new,
            max_new_eval=args.max_new_eval,
            n_eval=args.n_eval,
            save_every=args.save_every,
            skip_baseline_eval=args.skip_baseline_eval,
            skip_final_eval=args.skip_final_eval,
            no_final_checkpoint=args.no_final_checkpoint,
            min_train_level=args.min_train_level,
            max_train_level=args.max_train_level,
            train_selection=args.train_selection,
            stop_token_ids=stop_ids,
            beta_kl=args.beta_kl,
            lr_o=args.lr_o,
            o_clamp=args.o_clamp,
            lora_r=args.lora_r,
            g_sweep=g_sweep,
            lora_lr_sweep=lora_lr_sweep,
            cost_target_reward=COST_TARGET_REWARD,
            grad_ckpt=not args.no_grad_ckpt,
        ),
        baseline=dict(
            acc=acc_base,
            k=k_base,
            n=n_base,
            ci=list(ci_base),
            cases=[dict(problem=q, model=t, pred=p, gold=g) for q, t, p, g in cases_base],
        ),
        best_lora_key=best_lora_key,
        order=order,
        cost_records=cost_records,
        variants={},
    )
    for key in order:
        r = results[key]
        s2t = costlib.steps_to_target(r["curve"], COST_TARGET_REWARD)
        payload["variants"][key] = dict(
            kind=r["kind"],
            lr=r.get("lr"),
            n_par=r["n_par"],
            acc=r["acc"],
            k=r["k"],
            n=r["n"],
            ci=list(r["ci"]),
            final_kl=r["final_kl"],
            steps_to_target=s2t,
            is_best_lora=bool(r.get("is_best_lora")),
            mean_abs_o=r.get("mean_abs_o"),
            max_gate=r.get("max_gate"),
            reward_curve=r["curve"],
            kl_curve=r["kl_curve"],
            cases=[dict(problem=q, model=t, pred=p, gold=g) for q, t, p, g in r["cases"]],
        )
    json_out = os.path.join(os.path.dirname(args.out) or ".", "results.json")
    with open(json_out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {json_out}", flush=True)


if __name__ == "__main__":
    main()
