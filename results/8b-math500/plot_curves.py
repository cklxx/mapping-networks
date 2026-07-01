"""Figures for 8B MATH-500 modulation-vs-LoRA result. Data-driven from results.json."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.join(HERE, "results.json")

C = {"g2048": "#d1495b", "g256": "#edae49", "lora": "#30638e", "base": "#9aa0a6"}
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 140,
    "axes.spines.top": False, "axes.spines.right": False,
})

def ema(x, a=0.5):
    if not x: return x
    o = [x[0]]
    for v in x[1:]: o.append(a * v + (1 - a) * o[-1])
    return o

def load():
    if not os.path.exists(JSON):
        raise SystemExit(f"missing {JSON}. Run the experiment first.")
    with open(JSON) as f: return json.load(f)

def variant_color(key, kind, is_best_lora):
    if "G2048" in key: return C["g2048"]
    if "G256" in key: return C["g256"]
    if kind == "lora": return C["lora"]
    return C["base"]

def short_label(key, v):
    n = v["n_par"]
    nstr = f"{n/1e6:.1f}M" if n >= 1e6 else f"{n}"
    if v["kind"] == "map":
        g = key.split("G")[-1]
        return f"Map-G{g} ({nstr} params)"
    tag = " ** best **" if v.get("is_best_lora") else ""
    lr = v.get("lr")
    lrs = f" lr={lr:g}" if lr is not None else ""
    return f"LoRA-r8{lrs} ({nstr} params){tag}"

def fig_training(d):
    variants = d["variants"]
    keys = [k for k in d["order"] if variants[k]["kind"] == "map"]
    best_lora = d.get("best_lora_key")
    if best_lora: keys.append(best_lora)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for k in keys:
        v = variants[k]
        col = variant_color(k, v["kind"], v.get("is_best_lora"))
        mk = "o" if "G2048" in k else "s" if "G256" in k else "^"
        ax[0].plot(v["kl_curve"], color=col, lw=2.2 if "G2048" in k else 2.0,
                   marker=mk, ms=3, label=short_label(k, v))
    ax[0].axhline(0.05, ls="--", c="gray", lw=1, alpha=0.6)
    ax[0].text(0.3, 0.052, "leverage threshold ~0.05", fontsize=8, color="gray")
    ax[0].set_title("Policy movement: KL(pi||base) per RL step")
    ax[0].set_xlabel("GRPO step"); ax[0].set_ylabel("mean KL")
    ax[0].legend(fontsize=8)

    for k in keys:
        v = variants[k]
        col = variant_color(k, v["kind"], v.get("is_best_lora"))
        ax[1].plot(ema(v["reward_curve"]), color=col, lw=2.2 if "G2048" in k else 2.0,
                   label=short_label(k, v))
    ax[1].set_title("Reward (correct-rate) per step, EMA-smoothed")
    ax[1].set_xlabel("GRPO step"); ax[1].set_ylabel("mean reward")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(HERE, "fig_training_curves.png")
    plt.savefig(out); plt.close(fig)
    print("saved", out)

def fig_accuracy(d):
    base = d["baseline"]
    order = d["order"]
    variants = d["variants"]
    labels, accs, los, his, cols = [], [], [], [], []
    labels.append(f"baseline\n(0)"); accs.append(base["acc"])
    los.append(base["ci"][0]); his.append(base["ci"][1]); cols.append(C["base"])
    for k in order:
        v = variants[k]
        n = v["n_par"]
        nstr = f"{n/1e6:.1f}M" if n >= 1e6 else f"{n}"
        if v["kind"] == "map":
            short = f"Map-G{k.split('G')[-1]}\n({nstr})"
        else:
            tag = "\n** best **" if v.get("is_best_lora") else ""
            short = f"LoRA\nlr={v.get('lr'):g}\n({nstr}){tag}"
        labels.append(short); accs.append(v["acc"])
        los.append(v["ci"][0]); his.append(v["ci"][1])
        cols.append(variant_color(k, v["kind"], v.get("is_best_lora")))

    err = [[a - l for a, l in zip(accs, los)], [h - a for a, h in zip(accs, his)]]
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(labels)), 4.4))
    ax.bar(labels, accs, color=cols, yerr=err, capsize=5, width=0.62)
    ax.axhline(base["acc"], ls="--", c=C["base"], lw=1, alpha=0.8)
    ax.axhspan(base["ci"][0], base["ci"][1], color=C["base"], alpha=0.13)
    for i, a in enumerate(accs):
        ax.text(i, his[i] + 0.012, f"{a:.1%}", ha="center", fontsize=8.5, weight="bold")
    ax.set_title(f"MATH-500 accuracy (n={base['n']}, Wilson 95% CI) — frozen Qwen3-8B (A100)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, max(his) + 0.10)
    ax.grid(axis="y")
    plt.tight_layout()
    out = os.path.join(HERE, "fig_accuracy.png")
    plt.savefig(out); plt.close(fig)
    print("saved", out)

def fig_cost(d):
    recs = {r["variant"].replace("  ** BEST LoRA **", ""): r for r in d["cost_records"]}
    variants = d["variants"]
    map_keys = [k for k in d["order"] if variants[k]["kind"] == "map"]
    best_lora = d["best_lora_key"]
    trio = map_keys + [best_lora]
    def rec_of(k): return recs.get(k)
    labels, cols, params, vram, s2t, gpuh = [], [], [], [], [], []
    max_steps = d["config"]["max_steps"]
    for k in trio:
        v = variants[k]; r = rec_of(k)
        if r is None: continue
        n = v["n_par"]
        nstr = f"{n/1e6:.1f}M" if n >= 1e6 else f"{n}"
        if v["kind"] == "map":
            short = f"Map-G{k.split('G')[-1]}\n({nstr})"
        else:
            short = f"LoRA-r8\nbest lr={v.get('lr'):g}\n({nstr})"
        labels.append(short)
        cols.append(variant_color(k, v["kind"], v.get("is_best_lora")))
        params.append(r["trainable_params"])
        vram.append(r["peak_vram_bytes"] / 1024**3)
        st = r["steps_to_target"]
        s2t.append(st if st is not None else max_steps)
        gpuh.append(r["gpu_hours"])

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))
    a = ax[0][0]
    bars = a.bar(labels, params, color=cols, width=0.6)
    a.set_yscale("log"); a.set_title("Trainable parameters (log scale)")
    a.set_ylabel("params")
    for b, p in zip(bars, params):
        a.text(b.get_x() + b.get_width() / 2, p * 1.25, f"{p:,}", ha="center", fontsize=8, weight="bold")
    if len(params) >= 2:
        mod_min = min(params[:-1]) if len(params) > 1 else params[0]
        lora_p = params[-1]
        if mod_min:
            a.text(0.5, 0.92, f"LoRA / smallest-modulation = {lora_p/mod_min:,.0f}x",
                   transform=a.transAxes, ha="center", fontsize=8.5, color=C["g2048"], weight="bold")
    a = ax[0][1]
    if any(v > 0 for v in vram):
        bars = a.bar(labels, vram, color=cols, width=0.6)
        for b, vv in zip(bars, vram):
            a.text(b.get_x() + b.get_width() / 2, vv + max(vram) * 0.01,
                   f"{vv:.2f}", ha="center", fontsize=8, weight="bold")
    a.set_title("Peak VRAM"); a.set_ylabel("GB")
    a = ax[1][0]
    bars = a.bar(labels, s2t, color=cols, width=0.6)
    for b, s, k in zip(bars, s2t, trio):
        st = rec_of(k)["steps_to_target"]
        txt = f"{s}" if st is not None else f"{s}+ (n.c.)"
        a.text(b.get_x() + b.get_width() / 2, s + max(s2t) * 0.01, txt, ha="center", fontsize=8, weight="bold")
    a.set_title("Steps-to-target"); a.set_ylabel(f"steps to reward >= {d['config']['cost_target_reward']:.2f}")
    a.text(0.5, 0.92, "n.c. = did not converge", transform=a.transAxes, ha="center", fontsize=8, color="gray")
    a = ax[1][1]
    bars = a.bar(labels, gpuh, color=cols, width=0.6)
    for b, g in zip(bars, gpuh):
        a.text(b.get_x() + b.get_width() / 2, g + max(gpuh) * 0.01, f"{g:.2e}", ha="center", fontsize=8, weight="bold")
    a.set_title("GPU-hours to convergence"); a.set_ylabel("GPU-hours")
    a.text(0.5, 0.92, "compute/step = equal -> GPU-hours track steps-to-target",
           transform=a.transAxes, ha="center", fontsize=8.5, color=C["g2048"], weight="bold")
    fig.suptitle("Cost: modulation vs best-LoRA on frozen Qwen3-8B / MATH-500 (A100)",
                 fontsize=12.5, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(HERE, "fig_cost.png")
    plt.savefig(out); plt.close(fig)
    print("saved", out)

def main():
    d = load()
    fig_training(d)
    fig_accuracy(d)
    fig_cost(d)
    print("all figures written to", HERE)

if __name__ == "__main__":
    main()
