"""Reward-signal probe for the 9B MATH-500 GRPO experiment.

This script samples completions only. It does not train. Its job is to decide
whether the online RL loop has enough reward variance to justify a training run.
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

from src.generation_utils import generation_kwargs, stop_token_ids, trim_completion  # noqa: E402
from src.math_scorer import extract_answer, extract_last_braced, gold_answer, reward_of  # noqa: E402


SYS = (
    "Solve the math problem. Reason briefly. The final line must contain only "
    "\\boxed{...} with the final answer inside the braces. Stop immediately after "
    "the closing brace."
)


def parse_ints(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_prompt(tok, q):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"System: {SYS}\n\nUser: {q}\n\nAssistant:"


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="01-ai/Yi-1.5-9B-Chat")
    ap.add_argument("--out", default="results/9b-math500/reward-probe.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--probe-n", type=int, default=50)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--k-list", default="8,16")
    ap.add_argument("--max-new-list", default="256,512")
    ap.add_argument("--min-level", type=int, default=3)
    ap.add_argument("--max-level", type=int, default=5)
    ap.add_argument("--selection", choices=["head", "stride"], default="stride")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--sample-cases", type=int, default=3)
    args = ap.parse_args()

    torch.manual_seed(0)
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    dev = args.device
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1024**3)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    gen_extra = generation_kwargs(tok)
    stop_ids = stop_token_ids(tok)
    print(f"stop_token_ids={stop_ids} gen_extra={gen_extra}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dt,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(dev)
    model.eval()

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    train_records = [r for r in ds][args.n_eval:]
    train_records = [
        r for r in train_records
        if args.min_level <= int(r.get("level") or 5) <= args.max_level
    ]
    if not train_records:
        raise RuntimeError("empty probe pool after level filtering")
    train_records.sort(key=lambda r: int(r.get("level") or 5))
    if args.selection == "stride" and len(train_records) > args.probe_n:
        stride = len(train_records) / args.probe_n
        train_records = [train_records[min(len(train_records) - 1, int(i * stride))] for i in range(args.probe_n)]
    items = [(r["problem"], gold_answer(r), int(r.get("level") or 5)) for r in train_records[: args.probe_n]]
    level_hist = {}
    for _, _, level in items:
        level_hist[str(level)] = level_hist.get(str(level), 0) + 1
    print(f"probe pool size={len(train_records)} selected={len(items)} levels={level_hist}", flush=True)

    all_cfg = []
    for K in parse_ints(args.k_list):
        for max_new in parse_ints(args.max_new_list):
            all_cfg.append((K, max_new))

    payload = {
        "model": args.model,
        "probe_n": args.probe_n,
        "n_eval": args.n_eval,
        "min_level": args.min_level,
        "max_level": args.max_level,
        "selection": args.selection,
        "level_hist": level_hist,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "configs": [],
    }

    for K, max_new in all_cfg:
        t0 = time.time()
        rows = []
        total = correct = boxed = extracted = long_outputs = stopped_outputs = variance = 0
        decoded_cases = []
        print(f"\n=== probe K={K} max_new={max_new} ===", flush=True)
        for idx, (q, gold, level) in enumerate(items):
            prompt = build_prompt(tok, q)
            ids = tok(prompt, return_tensors="pt").input_ids[0].to(dev)
            with torch.no_grad():
                gen = model.generate(
                    ids[None],
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=K,
                    max_new_tokens=max_new,
                    **gen_extra,
                )
            rewards, preds, lengths, boxed_flags, stopped_flags = [], [], [], [], []
            for k in range(K):
                comp_raw = gen[k, ids.numel():]
                comp, stopped = trim_completion(comp_raw, stop_ids)
                text = tok.decode(comp, skip_special_tokens=True)
                pred = extract_answer(text)
                has_boxed = extract_last_braced(text, "\\boxed{") is not None
                rew = reward_of(text, gold)
                rewards.append(float(rew))
                preds.append(pred)
                boxed_flags.append(bool(has_boxed))
                stopped_flags.append(bool(stopped))
                lengths.append(int(comp.numel()))
                total += 1
                correct += int(rew > 0)
                extracted += int(pred is not None and pred != "")
                boxed += int(has_boxed)
                stopped_outputs += int(stopped)
                long_outputs += int((not stopped) and comp.numel() >= max_new)
                if len(decoded_cases) < args.sample_cases:
                    decoded_cases.append({
                        "problem": q,
                        "gold": gold,
                        "completion": text[:1500],
                        "pred": pred,
                        "has_boxed": bool(has_boxed),
                        "stopped": bool(stopped),
                        "reward": float(rew),
                        "length": int(comp.numel()),
                    })
            has_var = max(rewards) != min(rewards)
            variance += int(has_var)
            row = {
                "idx": idx,
                "level": level,
                "gold": gold,
                "rewards": rewards,
                "preds": preds,
                "lengths": lengths,
                "has_boxed": boxed_flags,
                "stopped": stopped_flags,
                "has_variance": has_var,
                "num_correct": int(sum(rewards)),
            }
            rows.append(row)
            if idx < 5 or (idx + 1) % 10 == 0:
                print(
                    f"[{idx+1}/{len(items)}] correct={sum(rewards):.0f}/{K} "
                    f"var={has_var} boxed={sum(boxed_flags)}/{K} "
                    f"extract={sum(bool(p) for p in preds)}/{K} "
                    f"avg_len={sum(lengths)/len(lengths):.1f}",
                    flush=True,
                )
        summary = {
            "K": K,
            "max_new": max_new,
            "sample_correct_rate": correct / max(1, total),
            "boxed_rate": boxed / max(1, total),
            "extract_rate": extracted / max(1, total),
            "stopped_rate": stopped_outputs / max(1, total),
            "long_output_rate": long_outputs / max(1, total),
            "variance_prompt_rate": variance / max(1, len(items)),
            "elapsed_s": time.time() - t0,
            "go": (
                variance / max(1, len(items)) >= 0.20
                and 0.02 <= correct / max(1, total) <= 0.40
                and boxed / max(1, total) >= 0.90
                and long_outputs / max(1, total) < 0.10
            ),
        }
        print("summary", summary, flush=True)
        payload["configs"].append({"summary": summary, "rows": rows, "decoded_cases": decoded_cases})
        write_json(args.out, payload)

    write_json(args.out, payload)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
