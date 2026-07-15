"""Active-prompt-bank GRPO gate for the 9B MATH-500 experiment.

This is a training-signal gate, not the final benchmark. It fixes the previous
failure mode where a global reward probe passed but the exact online training
schedule still produced zero-variance groups.
"""

import argparse
import csv
import gc
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy
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
    restore,
    target_modules,
)
from src.generation_utils import generation_kwargs, stop_token_ids, trim_completion  # noqa: E402
from src.math_scorer import extract_answer, extract_last_braced, gold_answer, reward_of  # noqa: E402


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

SYS = (
    "Solve the math problem. Reason briefly. The final line must contain only "
    "\\boxed{...} with the final answer inside the braces. Stop immediately after "
    "the closing brace."
)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def parse_json_obj(s):
    if not s:
        return {}
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("expected JSON object")
    return obj


def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_reproducibility(path, args, dev):
    """Write a reproducibility.json capturing the exact environment + config."""
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).decode().strip()
    except Exception:
        git_commit = "unknown"
    env_masked = {}
    for k, v in sorted(os.environ.items()):
        if "TOKEN" in k or "KEY" in k or "SECRET" in k:
            env_masked[k] = "***MASKED***"
        else:
            env_masked[k] = v
    cuda_info = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        cuda_info = {
            "device_name": torch.cuda.get_device_name(0),
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
            "cuda_available": True,
        }
    else:
        cuda_info = {"cuda_available": False}
    import transformers
    payload = {
        "git_commit": git_commit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "args": vars(args),
        "env": env_masked,
        "cuda": cuda_info,
        "device": dev,
    }
    write_json(path, payload)


def build_prompt(tok, q, prompt_suffix="", chat_template_kwargs=None, system_prompt=None):
    if prompt_suffix:
        q = q.rstrip() + "\n" + prompt_suffix
    msgs = [{"role": "system", "content": system_prompt or SYS}, {"role": "user", "content": q}]
    if getattr(tok, "chat_template", None):
        kwargs = dict(chat_template_kwargs or {})
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kwargs)
    return f"System: {SYS}\n\nUser: {q}\n\nAssistant:"


def select_candidate_records(ds, n_eval, min_level, max_level, candidate_n):
    records = []
    for dataset_idx, row in enumerate(ds):
        if dataset_idx < n_eval:
            continue
        level = int(row.get("level") or 5)
        if min_level <= level <= max_level:
            records.append((dataset_idx, row))
    if not records:
        raise RuntimeError("empty candidate pool")
    records.sort(key=lambda x: int(x[1].get("level") or 5))
    if len(records) <= candidate_n:
        return records
    stride = len(records) / candidate_n
    return [records[min(len(records) - 1, int(i * stride))] for i in range(candidate_n)]


def sample_one(model, tok, q, gold, dev, K, max_new, temperature, top_p, gen_extra, stop_ids):
    prompt = build_prompt(
        tok,
        q,
        getattr(tok, "_mn_prompt_suffix", ""),
        getattr(tok, "_mn_chat_template_kwargs", None),
        getattr(tok, "_mn_system_prompt", None),
    )
    pids = tok(prompt, return_tensors="pt").input_ids[0].to(dev)
    with torch.no_grad():
        gen = model.generate(
            pids[None],
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=K,
            max_new_tokens=max_new,
            use_cache=False,
            **gen_extra,
        )
    gen_cpu = gen.detach().cpu()
    del gen
    torch.cuda.empty_cache()
    rows = []
    for k in range(K):
        comp_raw = gen_cpu[k, pids.numel():]
        comp, stopped = trim_completion(comp_raw, stop_ids)
        comp = comp.to(dev).clone()  # GPU tensor, independent of gen_cpu
        text = tok.decode(comp, skip_special_tokens=True)
        pred = extract_answer(text)
        has_boxed = extract_last_braced(text, "\\boxed{") is not None
        has_answer_tag = "<answer>" in text.lower() and "</answer>" in text.lower()
        correct = float(reward_of(text, gold))
        overlong = float((not stopped) and comp.numel() >= max_new)
        rows.append({
            "comp": comp,
            "comp_ids": [int(x) for x in comp.detach().cpu().tolist()],
            "text": text,
            "pred": pred,
            "correct": correct,
            "format": float(has_boxed or has_answer_tag),
            "overlong": overlong,
            "stopped": bool(stopped),
            "length": int(comp.numel()),
        })
    return pids, rows


def sample_many(model, tok, items, dev, K, max_new, temperature, top_p, gen_extra, stop_ids):
    prompts = [
            build_prompt(
                tok,
                item["problem"],
                getattr(tok, "_mn_prompt_suffix", ""),
                getattr(tok, "_mn_chat_template_kwargs", None),
                getattr(tok, "_mn_system_prompt", None),
            )
        for item in items
    ]
    prev_side = tok.padding_side
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
    tok.padding_side = prev_side
    with torch.no_grad():
        gen = model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=K,
            max_new_tokens=max_new,
            use_cache=False,
            **gen_extra,
        )
    # Move gen to CPU immediately so GPU memory is released before we extract views.
    gen_cpu = gen.detach().cpu()
    del gen
    torch.cuda.empty_cache()
    prompt_width = enc.input_ids.shape[1]
    groups = []
    for i, item in enumerate(items):
        pids = enc.input_ids[i][enc.attention_mask[i].bool()].to(dev)
        samples = []
        for k in range(K):
            row = i * K + k
            comp_raw = gen_cpu[row, prompt_width:]
            comp, stopped = trim_completion(comp_raw, stop_ids)
            comp = comp.to(dev).clone()  # GPU tensor, independent of gen_cpu
            text = tok.decode(comp, skip_special_tokens=True)
            pred = extract_answer(text)
            has_boxed = extract_last_braced(text, "\\boxed{") is not None
            has_answer_tag = "<answer>" in text.lower() and "</answer>" in text.lower()
            correct = float(reward_of(text, item["gold"]))
            overlong = float((not stopped) and comp.numel() >= max_new)
            samples.append({
                "comp": comp,
                "comp_ids": [int(x) for x in comp.detach().cpu().tolist()],
                "text": text,
                "pred": pred,
                "correct": correct,
                "format": float(has_boxed or has_answer_tag),
                "overlong": overlong,
                "stopped": bool(stopped),
                "length": int(comp.numel()),
            })
        groups.append((item, pids, samples))
    return groups


def comp_logp_and_kl(model, names, prompt_ids, comp_ids, dev):
    ids = torch.cat([prompt_ids, comp_ids], 0)[None].to(dev)
    tgt = ids[0, 1:]
    n_prompt = prompt_ids.numel()
    comp_start = max(0, n_prompt - 1)
    tgt_comp = tgt[comp_start:]

    with torch.no_grad(), base_forward(model, names):
        base_logits = model(ids, use_cache=False).logits[0, comp_start:-1].float()
        base_logp = torch.log_softmax(base_logits, -1)

    logits = model(ids, use_cache=False).logits[0, comp_start:-1].float()
    logp = torch.log_softmax(logits, -1)
    tok_lp = logp.gather(1, tgt_comp[:, None]).squeeze(1)
    p = logp.exp()
    kl = (p * (logp - base_logp)).sum(-1).mean()
    return tok_lp.sum(), kl


def batched_logp_and_kl(model, names, prompt_ids_list, comps, pad_id, beta_kl,
                        adv=None, micro_batch=0):
    """One policy forward/backward for all selected GRPO samples.

    If `adv` is given, returns the scalar GRPO loss (advantage-weighted logp
    plus KL penalty) so the forward graph is built and freed per chunk; the
    per-token logits [B,T,V] are never materialized for the whole batch at once.
    Otherwise returns (sum_lp, kl) per sample for logging.

    If beta_kl is zero, the base forward is skipped entirely. With KL enabled
    this is the minimum exact path: one policy forward plus one no-grad base
    forward per chunk for the B*K completions that survived zero-variance
    filtering. `micro_batch` chunks the forward to bound peak memory (0 = one
    chunk). Logits stay in the model dtype (bf16) instead of being upcast to
    float32, halving the logits memory footprint.
    """
    batch = len(comps)
    prompt_lens = [p.numel() for p in prompt_ids_list]
    comp_lens = [c.numel() for c in comps]
    max_len = max(p + c for p, c in zip(prompt_lens, comp_lens))
    dev = prompt_ids_list[0].device
    dtype = prompt_ids_list[0].dtype
    ids = torch.full((batch, max_len), int(pad_id), device=dev, dtype=dtype)
    attn = torch.zeros((batch, max_len), device=dev, dtype=torch.bool)
    mask = torch.zeros((batch, max_len - 1), device=dev, dtype=torch.bool)
    for i, (prompt_ids, comp) in enumerate(zip(prompt_ids_list, comps)):
        prompt_len = prompt_lens[i]
        clen = comp_lens[i]
        ids[i, :prompt_len] = prompt_ids
        ids[i, prompt_len:prompt_len + clen] = comp
        attn[i, :prompt_len + clen] = True
        mask[i, prompt_len - 1:prompt_len + clen - 1] = True

    chunk = batch if micro_batch <= 0 else min(batch, micro_batch)
    sum_lp_all = []
    kl_all = []
    # Accumulate the GRPO loss as a scalar sum of per-chunk means so each chunk's
    # forward graph is freed before the next chunk starts (prevents ~6GB/update growth
    # from concatenating graph-bearing tensors across all chunks).
    total_loss = None
    for s in range(0, batch, chunk):
        e = min(s + chunk, batch)
        ids_c, attn_c, mask_c = ids[s:e], attn[s:e], mask[s:e]
        target_c = ids_c[:, 1:]
        logits = model(ids_c, attention_mask=attn_c, use_cache=False).logits[:, :-1, :]
        logp = torch.log_softmax(logits, -1)
        tok_lp = logp.gather(2, target_c[:, :, None]).squeeze(2)
        sum_lp_c = (tok_lp * mask_c).sum(1)
        if beta_kl > 0:
            with torch.no_grad(), base_forward(model, names):
                base_logits = model(ids_c, attention_mask=attn_c, use_cache=False).logits[:, :-1, :]
                base_logp = torch.log_softmax(base_logits, -1)
            token_kl = (logp.exp() * (logp - base_logp)).sum(-1)
            denom = mask_c.sum(1).clamp_min(1)
            kl_c = (token_kl * mask_c).sum(1) / denom
        else:
            kl_c = torch.zeros(e - s, device=dev)
        sum_lp_all.append(sum_lp_c.detach())
        kl_all.append(kl_c.detach())
        if adv is not None:
            adv_c = adv[s:e]
            chunk_loss = (-(adv_c * sum_lp_c) + beta_kl * kl_c).mean()
            total_loss = chunk_loss if total_loss is None else total_loss + chunk_loss
        # Free this chunk's graph before the next forward pass.
        del logits, logp, tok_lp, sum_lp_c
        if beta_kl > 0:
            del base_logits, base_logp, token_kl, kl_c
        torch.cuda.empty_cache()
    sum_lp = torch.cat(sum_lp_all)
    kl = torch.cat(kl_all)
    if adv is not None:
        n_chunks = (batch + chunk - 1) // chunk
        return total_loss / n_chunks, sum_lp, kl
    return sum_lp, kl


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * torch.sqrt(torch.tensor(p * (1 - p) / n + z * z / (4 * n * n))).item() / denom
    return (center - half, center + half)


@torch.no_grad()
def evaluate(model, tok, items, dev, max_new_eval, eval_batch, gen_extra, stop_ids, label):
    model.eval()
    prev_side = tok.padding_side
    tok.padding_side = "left"
    correct, cases = 0, []
    t0 = time.time()
    done = 0
    for b0 in range(0, len(items), eval_batch):
        batch = items[b0:b0 + eval_batch]
        prompts = [
            build_prompt(
                tok,
                q,
                getattr(tok, "_mn_prompt_suffix", ""),
                getattr(tok, "_mn_chat_template_kwargs", None),
                getattr(tok, "_mn_system_prompt", None),
            )
            for q, _ in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
        out = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_eval,
            use_cache=False,
            **gen_extra,
        )
        gen = out[:, enc.input_ids.shape[1]:]
        texts = [
            tok.decode(trim_completion(row, stop_ids)[0], skip_special_tokens=True)
            for row in gen
        ]
        for (q, gold), text in zip(batch, texts):
            pred = extract_answer(text)
            ok = int(pred == gold and bool(gold))
            correct += ok
            if len(cases) < 3:
                cases.append({
                    "problem": q,
                    "gold": gold,
                    "pred": pred,
                    "ok": bool(ok),
                    "text": text[:1200],
                })
        done += len(batch)
        if done <= 5 or done % 25 == 0 or done == len(items):
            print(
                f"[eval {label}] {done}/{len(items)} acc={correct/max(1, done):.3f} "
                f"{(time.time()-t0)/max(1, done):.1f}s/q",
                flush=True,
            )
    tok.padding_side = prev_side
    elapsed = time.time() - t0
    return {
        "k": int(correct),
        "n": int(len(items)),
        "acc": correct / max(1, len(items)),
        "ci": list(wilson_ci(correct, len(items))),
        "elapsed_s": elapsed,
        "s_per_q": elapsed / max(1, len(items)),
        "cases": cases,
    }


def build_active_bank(model, tok, ds, args, dev, gen_extra, stop_ids):
    if args.active_bank_json:
        with open(args.active_bank_json) as f:
            bank = json.load(f)
        print(
            "ACTIVE_BANK_LOADED "
            + json.dumps(bank.get("summary", {}), separators=(",", ":")),
            flush=True,
        )
        return bank

    candidates = select_candidate_records(
        ds, args.n_eval, args.min_level, args.max_level, args.candidate_n
    )
    active, all_rows = [], []
    totals = dict(correct=0, boxed=0, long=0, stopped=0, total=0, variance=0)
    t0 = time.time()
    for i, (dataset_idx, row) in enumerate(candidates):
        q, gold, level = row["problem"], gold_answer(row), int(row.get("level") or 5)
        _, samples = sample_one(
            model, tok, q, gold, dev, args.probe_k, args.max_new,
            args.temperature, args.top_p, gen_extra, stop_ids,
        )
        corrects = [s["correct"] for s in samples]
        formats = [s["format"] for s in samples]
        overlongs = [s["overlong"] for s in samples]
        stopped = [s["stopped"] for s in samples]
        preds = [s["pred"] for s in samples]
        lengths = [s["length"] for s in samples]
        num_correct = int(sum(corrects))
        has_variance = max(corrects) != min(corrects)
        totals["correct"] += num_correct
        totals["boxed"] += int(sum(formats))
        totals["long"] += int(sum(overlongs))
        totals["stopped"] += int(sum(1 for x in stopped if x))
        totals["total"] += len(samples)
        totals["variance"] += int(has_variance)
        bank_row = {
            "bank_id": len(active),
            "probe_idx": i,
            "dataset_idx": int(dataset_idx),
            "level": level,
            "problem": q,
            "gold": gold,
            "num_correct": num_correct,
            "K": args.probe_k,
            "correct_rewards": corrects,
            "format_rewards": formats,
            "overlong": overlongs,
            "stopped": stopped,
            "preds": preds,
            "lengths": lengths,
            "samples": [
                {
                    "comp_ids": s["comp_ids"],
                    "text": s["text"],
                    "pred": s["pred"],
                    "correct": s["correct"],
                    "format": s["format"],
                    "overlong": s["overlong"],
                    "stopped": s["stopped"],
                    "length": s["length"],
                }
                for s in samples
            ],
            "has_correct_variance": bool(has_variance),
        }
        all_rows.append(bank_row)
        if 0 < num_correct < args.probe_k:
            active.append(dict(bank_row, bank_id=len(active)))
        if i < 5 or (i + 1) % 10 == 0:
            print(
                f"[bank {i+1}/{len(candidates)}] dataset_idx={dataset_idx} "
                f"level={level} correct={num_correct}/{args.probe_k} "
                f"active={0 < num_correct < args.probe_k}",
                flush=True,
            )
    total = max(1, totals["total"])
    summary = {
        "candidate_n": len(candidates),
        "active_n": len(active),
        "probe_k": args.probe_k,
        "max_new": args.max_new,
        "sample_correct_rate": totals["correct"] / total,
        "boxed_rate": totals["boxed"] / total,
        "stopped_rate": totals["stopped"] / total,
        "long_output_rate": totals["long"] / total,
        "variance_prompt_rate": totals["variance"] / max(1, len(candidates)),
        "elapsed_s": time.time() - t0,
    }
    bank = {
        "summary": summary,
        "active": active,
        "all_rows": all_rows,
        "selection": {
            "n_eval": args.n_eval,
            "min_level": args.min_level,
            "max_level": args.max_level,
            "candidate_n": args.candidate_n,
        },
    }
    return bank


def shaped_rewards(samples, format_weight, overlong_penalty):
    return [
        float(s["correct"] + format_weight * s["format"] - overlong_penalty * s["overlong"])
        for s in samples
    ]


def save_curve(root, key, events):
    os.makedirs(os.path.join(root, "curves"), exist_ok=True)
    path = os.path.join(root, "curves", f"{key}.csv")
    fields = [
        "update", "attempt", "bank_id", "dataset_idx", "correct_mean",
        "shaped_mean", "kl", "step_s", "tokens",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in events:
            w.writerow({k: e.get(k) for k in fields})
    write_json(os.path.join(root, "curves", f"{key}.json"), events)


def train_variant(model, tok, names, active, args, dev, kind, result_root, bank_builder=None):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    if kind == "map":
        key = "Map-G2048"
        originals = [getattr(*get_parent(model, n)) for n in names]
        params, total_out = install_direct_map(model, names, 2048)
        n_trainable = int(sum(p.numel() for p in params))
        opt = torch.optim.Adam(params, lr=args.lr_o)
        telem = lambda: {
            "mean_abs_o": float(params[0].detach().abs().mean().item()),
            "max_gate": float((1.0 + ALPHA_MOD * params[0].detach()).abs().max().item()),
            "total_out": int(total_out),
        }
    elif kind == "lora":
        key = f"LoRA-r{args.lora_r}-lr{args.lora_lr:g}"
        originals = [getattr(*get_parent(model, n)) for n in names]
        params = install_lora(model, names, args.lora_r)
        n_trainable = int(sum(p.numel() for p in params))
        opt = torch.optim.Adam(params, lr=args.lora_lr)
        telem = lambda: {
            "mean_abs_param": float(torch.stack([p.detach().abs().mean() for p in params]).mean().item())
        }
    else:
        raise ValueError(kind)

    events, skipped = [], []
    update = 0
    updates_since_refresh = 0
    t_start = time.time()
    gen_extra = args.gen_extra
    stop_ids = args.stop_ids
    for attempt in range(args.max_attempts):
        if update >= args.target_updates:
            break
        if args.time_budget_s > 0 and time.time() - t_start >= args.time_budget_s:
            print(f"[{key}] time budget hit at attempt={attempt} update={update}", flush=True)
            break

        # Bank refresh: rebuild the active bank every N updates to avoid staleness.
        if (
            args.bank_refresh_interval > 0
            and bank_builder is not None
            and updates_since_refresh >= args.bank_refresh_interval
        ):
            print(
                f"[{key}] refreshing active bank after {updates_since_refresh} updates "
                f"(interval={args.bank_refresh_interval})",
                flush=True,
            )
            new_bank = bank_builder()
            active = new_bank["active"]
            updates_since_refresh = 0
            refresh_path = os.path.join(
                result_root, f"active_bank_refresh_{key}_update{update}.json"
            )
            write_json(refresh_path, new_bank)
            refresh_event = {
                "event": "bank_refresh",
                "variant": key,
                "update": update,
                "attempt": attempt,
                "active_n": len(active),
                "bank_summary": new_bank.get("summary", {}),
                "path": refresh_path,
            }
            events.append(refresh_event)
            append_jsonl(os.path.join(result_root, "progress.jsonl"), refresh_event)
            # model.generate() re-enables use_cache; reset so training forwards don't
            # leak KV caches after the bank rebuild.
            model.config.use_cache = False
            gc.collect()
            torch.cuda.empty_cache()
            print(
                f"[{key}] bank refreshed: active_n={len(active)} -> {refresh_path}",
                flush=True,
            )

        t0 = time.time()
        gc.collect()
        torch.cuda.empty_cache()
        if update % 5 == 0:
            _ma = torch.cuda.memory_allocated() / 1024**3
            _mr = torch.cuda.memory_reserved() / 1024**3
            print(f"[{key}] mem_pre update={update} alloc={_ma:.2f}GB reserved={_mr:.2f}GB", flush=True)
        batch_items = [
            active[(attempt * args.train_batch + i) % len(active)]
            for i in range(args.train_batch)
        ]
        groups = sample_many(
            model, tok, batch_items, dev, args.K, args.max_new,
            args.temperature, args.top_p, gen_extra, stop_ids,
        )

        selected = []
        selected_adv = []
        group_events = []
        for item, pids, samples in groups:
            corrects = [s["correct"] for s in samples]
            shaped = shaped_rewards(samples, args.format_weight, args.overlong_penalty)
            correct_var = max(corrects) != min(corrects)
            shaped_var = max(shaped) != min(shaped)
            event_base = {
                "variant": key,
                "attempt": attempt,
                "bank_id": item["bank_id"],
                "dataset_idx": item["dataset_idx"],
                "level": item["level"],
                "gold": item["gold"],
                "correct_rewards": corrects,
                "shaped_rewards": shaped,
                "format_rewards": [s["format"] for s in samples],
                "overlong": [s["overlong"] for s in samples],
                "lengths": [s["length"] for s in samples],
                "preds": [s["pred"] for s in samples],
                "correct_var": bool(correct_var),
                "shaped_var": bool(shaped_var),
            }
            if not shaped_var:
                skipped.append(dict(event_base, event="skipped_zero_variance"))
                append_jsonl(os.path.join(result_root, "progress.jsonl"), skipped[-1])
                continue
            rs = torch.tensor(shaped, dtype=torch.float32)
            adv = (rs - rs.mean()) / (rs.std() + 1e-4)
            for sample, adv_i in zip(samples, adv):
                selected.append((pids, sample))
                selected_adv.append(adv_i)
            group_events.append(dict(
                event_base,
                correct_mean=sum(corrects) / len(corrects),
                shaped_mean=sum(shaped) / len(shaped),
            ))

        if not selected:
            print(f"[{key}] skip attempt={attempt} groups={len(groups)} all_zero_variance", flush=True)
            continue

        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        prompt_batch = [pids for pids, _ in selected]
        comps = [sample["comp"] for _, sample in selected]
        adv_tensor = torch.stack(selected_adv).to(dev).detach()
        loss, logps, kls = batched_logp_and_kl(
            model,
            names,
            prompt_batch,
            comps,
            args.gen_extra["pad_token_id"],
            args.beta_kl,
            adv=adv_tensor,
            micro_batch=args.micro_batch,
        )
        tokens = sum(pids.numel() + comp.numel() for pids, comp in zip(prompt_batch, comps))
        loss.backward()
        opt.step()
        if kind == "map":
            with torch.no_grad():
                params[0].clamp_(-args.o_clamp, args.o_clamp)
        kl_mean = float(kls.detach().mean().item())
        correct_mean = sum(e["correct_mean"] for e in group_events) / len(group_events)
        shaped_mean = sum(e["shaped_mean"] for e in group_events) / len(group_events)
        n_groups = len(groups)
        n_active = len(group_events)
        # free GPU tensors now that scalars are extracted
        del loss, logps, kls, adv_tensor, prompt_batch, comps, selected, groups
        gc.collect()
        torch.cuda.empty_cache()
        if update % 5 == 0:
            _ma = torch.cuda.memory_allocated() / 1024**3
            _mr = torch.cuda.memory_reserved() / 1024**3
            print(f"[{key}] mem_post update={update} alloc={_ma:.2f}GB reserved={_mr:.2f}GB", flush=True)
        e = dict(
            event="update",
            variant=key,
            attempt=attempt,
            update=update,
            groups=n_groups,
            active_groups=n_active,
            skipped_groups=n_groups - n_active,
            group_events=group_events,
            correct_mean=correct_mean,
            shaped_mean=shaped_mean,
            kl=kl_mean,
            step_s=time.time() - t0,
            elapsed_s=time.time() - t_start,
            tokens=tokens,
            telemetry=telem(),
        )
        events.append(e)
        append_jsonl(os.path.join(result_root, "progress.jsonl"), e)
        print(
            f"[{key}] update={update} attempt={attempt} groups={n_active}/{n_groups} "
            f"correct={correct_mean:.3f} shaped={shaped_mean:.3f} kl={kl_mean:.4f}",
            flush=True,
        )
        update += 1
        updates_since_refresh += 1
        if args.convergence_correct > 0 and e["correct_mean"] >= args.convergence_correct:
            print(
                f"[{key}] convergence target hit: correct_mean={e['correct_mean']:.3f} "
                f">= {args.convergence_correct:.3f}",
                flush=True,
            )
            break

    save_curve(result_root, key, events)
    eval_result = None
    if args.eval_after_train:
        eval_result = evaluate(
            model, tok, args.eval_items, dev, args.max_new_eval, args.eval_batch,
            args.gen_extra, args.stop_ids, key,
        )
    if kind == "map":
        payload = {
            "variant": key,
            "o": [float(x) for x in params[0].detach().float().cpu().tolist()],
            "events": events,
            "skipped": skipped,
        }
        os.makedirs(os.path.join(result_root, "map_params"), exist_ok=True)
        write_json(os.path.join(result_root, "map_params", "Map-G2048_active_o.json"), payload)
    checkpoint_bytes_est = n_trainable * (2 if args.dtype == "bf16" else 2)
    restore(model, names, originals)
    total_s = sum(e["step_s"] for e in events)
    total_tokens = sum(e["tokens"] for e in events)
    return {
        "variant": key,
        "kind": kind,
        "trainable_params": n_trainable,
        "checkpoint_bytes_est": checkpoint_bytes_est,
        "updates": len(events),
        "skipped": len(skipped),
        "target_updates": args.target_updates,
        "max_attempts": args.max_attempts,
        "time_budget_s": args.time_budget_s,
        "elapsed_train_s": time.time() - t_start,
        "correct_curve": [e["correct_mean"] for e in events],
        "shaped_curve": [e["shaped_mean"] for e in events],
        "kl_curve": [e["kl"] for e in events],
        "best_correct_mean": max([e["correct_mean"] for e in events], default=0.0),
        "final_correct_mean": events[-1]["correct_mean"] if events else 0.0,
        "best_shaped_mean": max([e["shaped_mean"] for e in events], default=0.0),
        "final_shaped_mean": events[-1]["shaped_mean"] if events else 0.0,
        "mean_step_s": total_s / max(1, len(events)),
        "tokens_per_s": total_tokens / max(1e-9, total_s),
        "peak_alloc_gb": (
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available()
            else 0.0
        ),
        "eval": eval_result,
        "events": events,
        "skipped_events": skipped,
        "pass": (
            len(events) >= args.target_updates
            and any(e["correct_mean"] > 0 for e in events)
            and (eval_result is None or eval_result["n"] > 0)
        ),
    }


def adapter_sanity(model, tok, names, active, dev, result_root, lora_r):
    q = active[0]["problem"]
    ids = tok(
        build_prompt(
            tok,
            q,
            getattr(tok, "_mn_prompt_suffix", ""),
            getattr(tok, "_mn_chat_template_kwargs", None),
            getattr(tok, "_mn_system_prompt", None),
        ),
        return_tensors="pt",
    ).input_ids.to(dev)

    def last_logits():
        with torch.no_grad():
            return model(ids).logits[:, -1, :].float().detach()

    base = last_logits()
    rows = []
    originals = [getattr(*get_parent(model, n)) for n in names]
    params, total_out = install_direct_map(model, names, 2048)
    ident = (last_logits() - base).abs().max().item()
    with torch.no_grad():
        params[0].fill_(0.01)
    perturb = (last_logits() - base).abs().max().item()
    restore(model, names, originals)
    restored = (last_logits() - base).abs().max().item()
    rows.append({
        "adapter": "Map-G2048",
        "identity_max_abs": ident,
        "perturb_max_abs": perturb,
        "restore_max_abs": restored,
        "trainable_params": int(params[0].numel()),
        "total_out": int(total_out),
        "pass": ident < 1e-4 and perturb > 1e-7 and restored < 1e-4,
    })

    originals = [getattr(*get_parent(model, n)) for n in names]
    params = install_lora(model, names, lora_r)
    ident = (last_logits() - base).abs().max().item()
    with torch.no_grad():
        params[1].fill_(0.01)
    perturb = (last_logits() - base).abs().max().item()
    restore(model, names, originals)
    restored = (last_logits() - base).abs().max().item()
    rows.append({
        "adapter": f"LoRA-r{lora_r}",
        "identity_max_abs": ident,
        "perturb_max_abs": perturb,
        "restore_max_abs": restored,
        "trainable_params": int(sum(p.numel() for p in params)),
        "pass": ident < 1e-4 and perturb > 1e-7 and restored < 1e-4,
    })
    write_json(os.path.join(result_root, "adapter_sanity.json"), rows)
    print("ADAPTER_SANITY " + json.dumps(rows, separators=(",", ":")), flush=True)
    if not all(r["pass"] for r in rows):
        raise RuntimeError("adapter sanity failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="01-ai/Yi-1.5-9B-Chat")
    ap.add_argument("--out-dir", default="results/9b-math500/active-grpo-gate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--deterministic", action="store_true",
                    help="Enable deterministic mode (cudnn.deterministic=True, cudnn.benchmark=False).")
    ap.add_argument("--bank-refresh-interval", type=int, default=0,
                    help="Rebuild active bank every N updates; 0 = no refresh (backward compatible).")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--min-level", type=int, default=3)
    ap.add_argument("--max-level", type=int, default=5)
    ap.add_argument("--candidate-n", type=int, default=50)
    ap.add_argument("--active-bank-json", default="")
    ap.add_argument("--require-bank-gate", action="store_true")
    ap.add_argument("--min-active-prompts", type=int, default=20)
    ap.add_argument("--min-boxed-rate", type=float, default=0.90)
    ap.add_argument("--max-long-rate", type=float, default=0.10)
    ap.add_argument("--probe-k", type=int, default=8)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--max-new-eval", type=int, default=512)
    ap.add_argument("--eval-n", type=int, default=0)
    ap.add_argument("--eval-batch", type=int, default=1)
    ap.add_argument("--eval-after-train", action="store_true")
    ap.add_argument("--skip-baseline-eval", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--target-updates", type=int, default=20)
    ap.add_argument("--max-attempts", type=int, default=60)
    ap.add_argument("--train-batch", type=int, default=1)
    ap.add_argument("--micro-batch", type=int, default=0,
                    help="logp/KL forward chunk size; 0 = whole batch at once. "
                         "Set smaller (e.g. 4) to cut peak [B,T,V] logits memory.")
    ap.add_argument("--time-budget-s", type=float, default=0.0)
    ap.add_argument("--convergence-correct", type=float, default=0.0)
    ap.add_argument("--format-weight", type=float, default=0.1)
    ap.add_argument("--overlong-penalty", type=float, default=0.2)
    ap.add_argument("--beta-kl", type=float, default=0.05)
    ap.add_argument("--lr-o", type=float, default=0.005)
    ap.add_argument("--o-clamp", type=float, default=0.10)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-lr", type=float, default=1e-4)
    ap.add_argument("--variants", default="map,lora")
    ap.add_argument("--attn-impl", default="")
    ap.add_argument("--last-n-layers", type=int, default=0)
    ap.add_argument("--target-subset", default="all", choices=["all", "attn", "mlp", "o", "down", "o_down"])
    ap.add_argument("--system-prompt", default="")
    ap.add_argument("--prompt-suffix", default="")
    ap.add_argument("--chat-template-kwargs", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.makedirs(args.out_dir, exist_ok=True)
    dev = args.device
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
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
    tok._mn_system_prompt = args.system_prompt or None
    tok._mn_prompt_suffix = args.prompt_suffix
    tok._mn_chat_template_kwargs = parse_json_obj(args.chat_template_kwargs)
    gen_extra = generation_kwargs(tok)
    stop_ids = stop_token_ids(tok)
    args.gen_extra = gen_extra
    args.stop_ids = stop_ids
    print(f"stop_token_ids={stop_ids} gen_extra={gen_extra}", flush=True)

    load_kwargs = dict(torch_dtype=dt, trust_remote_code=True, low_cpu_mem_usage=True)
    if args.attn_impl:
        load_kwargs["attn_implementation"] = args.attn_impl
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).to(dev)
    model.requires_grad_(False)
    model.config.use_cache = False  # generate() re-enables it per-call; default off for training
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    write_reproducibility(os.path.join(args.out_dir, "reproducibility.json"), args, dev)

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    eval_records = [r for r in ds][:args.n_eval]
    if args.eval_n > 0:
        eval_records = eval_records[:args.eval_n]
    args.eval_items = [(r["problem"], gold_answer(r)) for r in eval_records]

    bank = build_active_bank(model, tok, ds, args, dev, gen_extra, stop_ids)
    write_json(os.path.join(args.out_dir, "active_bank.json"), bank)
    print("ACTIVE_BANK_SUMMARY " + json.dumps(bank["summary"], separators=(",", ":")), flush=True)
    # model.generate() enables use_cache; reset so training forwards don't leak KV caches.
    model.config.use_cache = False
    gc.collect()
    torch.cuda.empty_cache()
    if not bank["active"]:
        raise RuntimeError("no active prompts found")
    if args.require_bank_gate:
        s = bank["summary"]
        if s["active_n"] < args.min_active_prompts:
            raise RuntimeError(f"active bank too small: {s['active_n']} < {args.min_active_prompts}")
        if s["boxed_rate"] < args.min_boxed_rate:
            raise RuntimeError(f"boxed_rate below gate: {s['boxed_rate']:.4f} < {args.min_boxed_rate:.4f}")
        if s["long_output_rate"] > args.max_long_rate:
            raise RuntimeError(f"long_output_rate above gate: {s['long_output_rate']:.4f} > {args.max_long_rate:.4f}")

    last_n_layers = None if args.last_n_layers <= 0 else args.last_n_layers
    names = target_modules(model, last_n_layers=last_n_layers, subset=args.target_subset)
    write_json(os.path.join(args.out_dir, "target_modules.json"), {"count": len(names), "names": names})
    if not names:
        raise RuntimeError("no target modules found")
    adapter_sanity(model, tok, names, bank["active"], dev, args.out_dir, args.lora_r)

    baseline_eval = None
    if args.eval_after_train and args.eval_items and not args.skip_baseline_eval:
        baseline_eval = evaluate(
            model, tok, args.eval_items, dev, args.max_new_eval, args.eval_batch,
            gen_extra, stop_ids, "baseline",
        )

    def bank_builder():
        # Always rebuild with the current (adapted) model; ignore any cached bank JSON
        # so refresh actually re-probes the candidate pool.
        saved = args.active_bank_json
        args.active_bank_json = ""
        try:
            return build_active_bank(model, tok, ds, args, dev, gen_extra, stop_ids)
        finally:
            args.active_bank_json = saved

    summaries = {}
    for variant in [x.strip() for x in args.variants.split(",") if x.strip()]:
        summaries[variant] = train_variant(
            model, tok, names, bank["active"], args, dev, variant, args.out_dir,
            bank_builder=bank_builder,
        )
    payload = {
        "model": args.model,
        "bank_summary": bank["summary"],
        "variants": summaries,
        "baseline_eval": baseline_eval,
        "config": {
            "K": args.K,
            "max_new": args.max_new,
            "max_new_eval": args.max_new_eval,
            "eval_n": len(args.eval_items),
            "target_updates": args.target_updates,
            "max_attempts": args.max_attempts,
            "train_batch": args.train_batch,
            "time_budget_s": args.time_budget_s,
            "convergence_correct": args.convergence_correct,
            "format_weight": args.format_weight,
            "overlong_penalty": args.overlong_penalty,
            "beta_kl": args.beta_kl,
        },
    }
    write_json(os.path.join(args.out_dir, "active_train_summary.json"), payload)
    print("ACTIVE_TRAIN_SUMMARY_JSON " + json.dumps({
        "bank_summary": bank["summary"],
        "baseline_eval": baseline_eval,
        "variants": {
            k: {
                "updates": v["updates"],
                "skipped": v["skipped"],
                "target_updates": v["target_updates"],
                "max_attempts": v["max_attempts"],
                "time_budget_s": v["time_budget_s"],
                "elapsed_train_s": v["elapsed_train_s"],
                "pass": v["pass"],
                "trainable_params": v["trainable_params"],
                "checkpoint_bytes_est": v["checkpoint_bytes_est"],
                "mean_step_s": v["mean_step_s"],
                "tokens_per_s": v["tokens_per_s"],
                "peak_alloc_gb": v["peak_alloc_gb"],
                "eval": v["eval"],
                "best_correct_mean": v["best_correct_mean"],
                "final_correct_mean": v["final_correct_mean"],
                "best_shaped_mean": v["best_shaped_mean"],
                "final_shaped_mean": v["final_shaped_mean"],
                "correct_curve": v["correct_curve"],
                "shaped_curve": v["shaped_curve"],
                "kl_curve": v["kl_curve"],
            }
            for k, v in summaries.items()
        },
    }, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
