"""Performance sweep for Map active-bank GRPO.

Loads the 9B model once, builds/reuses one active bank, then tests a small set of
Map-only training configurations. This is for wall-clock engineering, not final
accuracy.
"""

import argparse
import json
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import experiments.math500_active_grpo_9b as A  # noqa: E402
from src.adapters import get_parent, install_direct_map, restore, target_modules, ALPHA_MOD  # noqa: E402
from src.generation_utils import generation_kwargs, stop_token_ids  # noqa: E402


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def parse_list(s, typ):
    return [typ(x.strip()) for x in s.split(",") if x.strip()]


def reset_cuda_peak():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def run_map_config(model, tok, names, active, args, cfg, dev):
    originals = [getattr(*get_parent(model, n)) for n in names]
    params, total_out = install_direct_map(model, names, 2048)
    opt = torch.optim.Adam(params, lr=cfg["lr_o"])
    events, skipped = [], 0
    update, attempt = 0, 0
    t_start = time.time()
    reset_cuda_peak()
    try:
        while update < cfg["updates"] and attempt < cfg["max_attempts"]:
            t0 = time.time()
            batch_items = [
                active[(attempt * cfg["train_batch"] + i) % len(active)]
                for i in range(cfg["train_batch"])
            ]
            groups = A.sample_many(
                model,
                tok,
                batch_items,
                dev,
                args.K,
                cfg["max_new"],
                args.temperature,
                args.top_p,
                args.gen_extra,
                args.stop_ids,
            )
            selected, selected_adv, group_events = [], [], []
            for item, pids, samples in groups:
                corrects = [s["correct"] for s in samples]
                shaped = A.shaped_rewards(samples, args.format_weight, args.overlong_penalty)
                if max(shaped) == min(shaped):
                    skipped += 1
                    continue
                rs = torch.tensor(shaped, dtype=torch.float32)
                adv = (rs - rs.mean()) / (rs.std() + 1e-4)
                for sample, adv_i in zip(samples, adv):
                    selected.append((pids, sample))
                    selected_adv.append(adv_i)
                group_events.append({
                    "correct_mean": sum(corrects) / len(corrects),
                    "shaped_mean": sum(shaped) / len(shaped),
                })
            if not selected:
                attempt += 1
                continue

            opt.zero_grad(set_to_none=True)
            prompt_batch = [pids for pids, _ in selected]
            comps = [sample["comp"] for _, sample in selected]
            logps, kls = A.batched_logp_and_kl(
                model,
                names,
                prompt_batch,
                comps,
                args.gen_extra["pad_token_id"],
                cfg["beta_kl"],
            )
            adv_tensor = torch.stack(selected_adv).to(dev).detach()
            loss = (-(adv_tensor * logps) + cfg["beta_kl"] * kls).mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                params[0].clamp_(-args.o_clamp, args.o_clamp)
            step_s = time.time() - t0
            tokens = sum(p.numel() + c.numel() for p, c in zip(prompt_batch, comps))
            events.append({
                "update": update,
                "attempt": attempt,
                "groups": len(groups),
                "active_groups": len(group_events),
                "correct_mean": sum(e["correct_mean"] for e in group_events) / len(group_events),
                "shaped_mean": sum(e["shaped_mean"] for e in group_events) / len(group_events),
                "kl": float(kls.detach().mean().item()),
                "step_s": step_s,
                "tokens": int(tokens),
            })
            print(
                f"[cfg {cfg['name']}] update={update} groups={len(group_events)}/{len(groups)} "
                f"step_s={step_s:.2f} tok_s={tokens/max(1e-9, step_s):.1f} "
                f"correct={events[-1]['correct_mean']:.3f}",
                flush=True,
            )
            update += 1
            attempt += 1
        total_s = sum(e["step_s"] for e in events)
        total_tokens = sum(e["tokens"] for e in events)
        result = {
            "ok": update == cfg["updates"],
            "name": cfg["name"],
            **cfg,
            "updates_done": update,
            "skipped_groups": skipped,
            "elapsed_s": time.time() - t_start,
            "mean_step_s": total_s / max(1, len(events)),
            "tokens_per_s": total_tokens / max(1e-9, total_s),
            "peak_alloc_gb": peak_gb(),
            "best_correct_mean": max([e["correct_mean"] for e in events], default=0.0),
            "final_correct_mean": events[-1]["correct_mean"] if events else 0.0,
            "events": events,
            "mean_abs_o": float(params[0].detach().abs().mean().item()),
            "max_gate": float((1.0 + ALPHA_MOD * params[0].detach()).abs().max().item()),
            "total_out": int(total_out),
        }
    except torch.cuda.OutOfMemoryError as e:
        result = {
            "ok": False,
            "name": cfg["name"],
            **cfg,
            "error": "oom",
            "message": str(e).splitlines()[0],
            "updates_done": update,
            "elapsed_s": time.time() - t_start,
            "peak_alloc_gb": peak_gb(),
            "events": events,
        }
        print(f"[cfg {cfg['name']}] OOM after updates={update}: {result['message']}", flush=True)
    finally:
        restore(model, names, originals)
        del params, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="01-ai/Yi-1.5-9B-Chat")
    ap.add_argument("--out", default="results/9b-math500/perf_sweep_map.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--candidate-n", type=int, default=30)
    ap.add_argument("--probe-k", type=int, default=8)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--updates", type=int, default=3)
    ap.add_argument("--max-attempts", type=int, default=12)
    ap.add_argument("--max-new-list", default="256,512")
    ap.add_argument("--train-batch-list", default="1,2,4")
    ap.add_argument("--beta-kl-list", default="0,0.05")
    ap.add_argument("--lr-o", type=float, default=0.005)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--format-weight", type=float, default=0.1)
    ap.add_argument("--overlong-penalty", type=float, default=0.2)
    ap.add_argument("--o-clamp", type=float, default=0.10)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = args.device
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"gpu={torch.cuda.get_device_name(0)} mem_gb={props.total_memory/1024**3:.2f}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    args.gen_extra = generation_kwargs(tok)
    args.stop_ids = stop_token_ids(tok)

    load_kwargs = dict(torch_dtype=dt, trust_remote_code=True, low_cpu_mem_usage=True)
    if args.attn_impl:
        load_kwargs["attn_implementation"] = args.attn_impl
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).to(dev)
    model.requires_grad_(False)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    bank = A.build_active_bank(model, tok, ds, argparse.Namespace(
        active_bank_json="",
        n_eval=200,
        min_level=3,
        max_level=5,
        candidate_n=args.candidate_n,
        probe_k=args.probe_k,
        max_new=512,
        temperature=args.temperature,
        top_p=args.top_p,
    ), dev, args.gen_extra, args.stop_ids)
    if len(bank["active"]) < 4:
        raise RuntimeError(f"not enough active prompts: {len(bank['active'])}")

    names = target_modules(model)
    configs = []
    for max_new in parse_list(args.max_new_list, int):
        for train_batch in parse_list(args.train_batch_list, int):
            for beta_kl in parse_list(args.beta_kl_list, float):
                configs.append({
                    "name": f"new{max_new}_b{train_batch}_kl{beta_kl:g}",
                    "max_new": max_new,
                    "train_batch": train_batch,
                    "beta_kl": beta_kl,
                    "updates": args.updates,
                    "max_attempts": args.max_attempts,
                    "lr_o": args.lr_o,
                })
    results = [run_map_config(model, tok, names, bank["active"], args, cfg, dev) for cfg in configs]
    payload = {"bank_summary": bank["summary"], "results": results}
    A.write_json(args.out, payload)
    print("PERF_SWEEP_JSON " + json.dumps({
        "bank_summary": bank["summary"],
        "results": [
            {k: r.get(k) for k in [
                "name", "ok", "max_new", "train_batch", "beta_kl", "updates_done",
                "mean_step_s", "tokens_per_s", "peak_alloc_gb", "best_correct_mean",
                "final_correct_mean", "error",
            ]}
            for r in results
        ],
    }, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
