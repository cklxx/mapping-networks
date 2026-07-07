"""Inspect Qwen3.5 MATH completions for prompt/scorer failures."""

import argparse
import json
import os
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.math500_active_grpo_9b import build_prompt  # noqa: E402
from src.generation_utils import generation_kwargs, stop_token_ids, trim_completion  # noqa: E402
from src.math_scorer import extract_answer, extract_last_braced, gold_answer, reward_of  # noqa: E402


PROMPTS = {
    "answer_only": "Return only the final answer in \\boxed{...}. No explanation. No text after the box.",
    "answer_only_strict": "Output exactly one line: \\boxed{final answer}. Do not explain.",
    "qwen_xml": "Answer the math problem. Put the final answer inside <answer>...</answer>. No extra text.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--out", default="results/9b-math500/qwen35-case-probe.json")
    ap.add_argument("--prompt-modes", default="answer_only,answer_only_strict,qwen_xml")
    ap.add_argument("--indices", default="203,226,261,304,339")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = args.device
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    gen_extra = generation_kwargs(tok)
    stop_ids = stop_token_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(dev)
    model.eval()

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    payload = {
        "model": args.model,
        "K": args.K,
        "max_new": args.max_new,
        "prompt_modes": args.prompt_modes.split(","),
        "cases": [],
    }
    for mode in payload["prompt_modes"]:
        system_prompt = PROMPTS[mode]
        template_kwargs = {"enable_thinking": False}
        suffix = "/no_think"
        for idx in [int(x) for x in args.indices.split(",") if x.strip()]:
            row = ds[idx]
            q, gold = row["problem"], gold_answer(row)
            prompt = build_prompt(tok, q, suffix, template_kwargs, system_prompt)
            ids = tok(prompt, return_tensors="pt").input_ids[0].to(dev)
            with torch.no_grad():
                gen = model.generate(
                    ids[None],
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=args.K,
                    max_new_tokens=args.max_new,
                    **gen_extra,
                )
            samples = []
            for k in range(args.K):
                comp_raw = gen[k, ids.numel():]
                comp, stopped = trim_completion(comp_raw, stop_ids)
                text = tok.decode(comp, skip_special_tokens=True)
                samples.append({
                    "k": k,
                    "text": text,
                    "pred": extract_answer(text),
                    "boxed": extract_last_braced(text, "\\boxed{") is not None,
                    "answer_tag": ("<answer>" in text and "</answer>" in text),
                    "reward": reward_of(text, gold),
                    "stopped": stopped,
                    "length": int(comp.numel()),
                })
            rec = {
                "mode": mode,
                "dataset_idx": idx,
                "level": int(row.get("level") or 5),
                "problem": q,
                "gold": gold,
                "prompt_tail": prompt[-500:],
                "samples": samples,
            }
            payload["cases"].append(rec)
            print(
                f"{mode} idx={idx} gold={gold} "
                f"correct={sum(s['reward'] for s in samples):.0f}/{args.K} "
                f"boxed={sum(s['boxed'] for s in samples)}/{args.K} "
                f"stopped={sum(s['stopped'] for s in samples)}/{args.K}",
                flush=True,
            )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
