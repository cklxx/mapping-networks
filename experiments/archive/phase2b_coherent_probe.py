"""[ARCHIVED — negative result, confound-closing probe] Phase 2b follow-up: trace GSM8K
acc vs o-magnitude on the RL-LEARNED direction.

PURPOSE: separate "the small-param direction doesn't help" from "the end-state was
over-driven into degradation". The main phase2b run evals only the END-STATE o (driven to
mean|o|~0.067 / KL~0.34, into the coherence-degradation zone). This probe re-trains G=256
capped at 12 steps (lands the learned o mid-band, coherent), then evals at scales s of that
learned direction. If NO scale beats baseline, "no productive lift" is airtight.

RESULT (see phase2b_coherent_probe.txt): as o moves along the RL-learned direction away
from identity, accuracy declines MONOTONICALLY (0.420 -> 0.410 -> 0.390 -> ... -> 0.245) and
never clears the baseline CI upward at ANY scale, INCLUDING fully-coherent low-s points.
CONCLUSION: the negative main-run result is NOT an over-driven-end-state artifact — the
learned direction has no productive component anywhere on its ray. A genuine
capacity/function-class limit on 0.8B/GSM8K, not a tuning/confound artifact. (The probe
process was killed externally during the s=0.75 eval; the recovered s=0.00/0.25/0.50 rows +
the main-run end-state fully trace the monotone decline.)

REFACTOR (vs the original repro script): imports the REFACTORED local phase2b (which itself
imports the adapter math from src/adapters.py); the hardcoded /host paths are --model-path /
--out / --o-save flags. Logic preserved.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import phase2b as P  # noqa: E402 — the refactored sibling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3.5-0.8B",
                    help="local dir / HF id / snapshot glob (was a hardcoded pod path)")
    ap.add_argument("--out", default="phase2b_coherent_probe.txt")
    ap.add_argument("--o-save", default="o_G256_final.pt")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    snap = P.resolve_model(args.model_path)
    print("coherent-probe: re-train G=256, then sweep acc vs o-scale on the learned direction", flush=True)
    tok = P.AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
    model = P.AutoModelForCausalLM.from_pretrained(snap, dtype=dt, trust_remote_code=True).to(dev)
    model.requires_grad_(False)
    ds = P.load_dataset("openai/gsm8k", "main")
    train_items = [(r["question"], P.gold_of(r["answer"])) for r in ds["train"].select(range(P.B * P.MAX_STEPS + 64))]
    eval_items = [(r["question"], P.gold_of(r["answer"])) for r in ds["test"].select(range(P.N_EVAL))]
    names = P.target_modules(model)

    kb, nb, _ = P.evaluate(model, tok, eval_items, dev, label="base")
    ci_b = P.wilson_ci(kb, nb)
    print(f"[baseline] {kb}/{nb} = {kb/nb:.4f}  CI [{ci_b[0]:.3f},{ci_b[1]:.3f}]", flush=True)

    # re-train G=256 capped at 12 steps so the learned o lands mid-band (coherent).
    P.MAX_STEPS = 12
    P.TIME_BUDGET_S = 30 * 60
    o_orig = [getattr(*P.get_parent(model, n)) for n in names]
    params, _ = P.install_direct_map(model, names, 256)
    print(f"[train] G=256 params={params[0].numel()} cap={P.MAX_STEPS} steps (coherent-band learned o)", flush=True)
    P.train_grpo(model, tok, names, train_items, params, P.LR_O, dev, "probe-G256")
    o_final = params[0].detach().clone()
    torch.save(o_final, args.o_save)
    print(f"[train done] final mean|o|={o_final.abs().mean():.4f} max|o|={o_final.abs().max():.4f}", flush=True)

    lines = ["=" * 78, "PHASE 2b COHERENT PROBE — acc vs o-scale on the RL-learned G=256 direction", "=" * 78,
             f"baseline: {kb/nb:.4f} ({kb}/{nb})  CI [{ci_b[0]:.3f},{ci_b[1]:.3f}]",
             f"learned o: mean|o|={o_final.abs().mean():.4f}  max|o|={o_final.abs().max():.4f}  G=256",
             "", "scale s   mean|o|   max_gate   acc(n=200)   Wilson CI        vs baseline"]
    best = None
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        with torch.no_grad():
            params[0].copy_(o_final * s)
        collect = 3 if s in (0.25, 0.5) else 0
        k, n, cases = P.evaluate(model, tok, eval_items, dev, label=f"s={s}", collect_cases=collect)
        acc = k / n
        ci = P.wilson_ci(k, n)
        mo = (o_final * s).abs().mean().item()
        mg = (1 + (o_final * s)).abs().max().item()
        disj = "DISJOINT-ABOVE" if ci[0] > ci_b[1] else ("DISJOINT-BELOW" if ci[1] < ci_b[0] else "overlap")
        lines.append(f"  {s:.2f}    {mo:.4f}    {mg:.3f}     {acc:.4f}     [{ci[0]:.3f},{ci[1]:.3f}]   {acc-kb/nb:+.4f} {disj}")
        print(lines[-1], flush=True)
        if best is None or acc > best[1]:
            best = (s, acc, ci, cases)
    restore = P.restore
    restore(model, names, o_orig)
    lines.append("")
    lines.append(f"BEST scale s={best[0]} acc={best[1]:.4f} CI [{best[2][0]:.3f},{best[2][1]:.3f}]")
    above = best[2][0] > ci_b[1]
    lines.append(f"VERDICT: best-coherent-point {'CLEARS' if above else 'does NOT clear'} baseline CI upward.")
    lines.append("")
    if best[3]:
        lines.append(f"DECODED CASES at best scale s={best[0]}:")
        lines.append(P.fmt_cases(best[3]))
    lines.append("=" * 78)
    report = "\n".join(lines)
    print("\n" + report, flush=True)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
