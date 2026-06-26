"""Cost instrumentation: the centerpiece. Measures, per adapter VARIANT, exactly
what makes one fine-tuning method cheaper than another in GPU-hours.

The cost story has FOUR axes, and they do NOT move together:

  1. trainable params   — exact count (modulation: G; LoRA: r*(in+out) per target).
  2. peak VRAM          — torch.cuda.max_memory_allocated() across the train loop,
                          after reset_peak_memory_stats(). Dominated by the frozen
                          base's activations/grads-through-base; the adapter's own
                          optimizer state (Adam m,v over the trainable params) is the
                          part that differs, and it scales with param count.
  3. compute / step     — FLOPs(fwd+bwd). MECHANISM (made explicit in the report):
                          the frozen base's forward+backward dominates the step;
                          the adapter's own FLOPs (LoRA's 2 extra matmuls vs the
                          modulation's element-wise per-channel scale) are a rounding
                          error against it. So FLOPs/step is ~EQUAL across adapters,
                          and the GPU-hour cost is NOT decided here.
  4. steps-to-target    — the number of optimizer steps to reach a fixed reward (or KL)
                          threshold. THIS is what decides GPU-hours, because (3) is ~equal
                          across adapters: gpu_hours ~= steps_to_target * wall_per_step.

So the headline question this module answers is exactly: "is the modulation cheaper in
GPU-hours, or only in adapter SIZE + optimizer VRAM?" — size (axis 1) is a certain ~10^4x
win; optimizer VRAM (axis 2) is a smaller win that grows with model size; compute/step
(axis 3) is ~equal; and GPU-hours (axes 3x4) is decided by steps-to-target, measured by
the run, not asserted a priori.

This file is the SINGLE SOURCE of the cost hooks. cost_benchmark.py drives the variant
sweep; experiments/math500_rl.py imports the same hooks so the validated runner emits the
identical numbers. No bespoke per-experiment accounting.
"""
import time

import torch


# ---------------------------------------------------------------------------
# trainable params (axis 1)
# ---------------------------------------------------------------------------
def trainable_param_count(params) -> int:
    """Exact trainable-parameter count for a list of nn.Parameter."""
    return int(sum(p.numel() for p in params))


# ---------------------------------------------------------------------------
# peak VRAM (axis 2)
# ---------------------------------------------------------------------------
def reset_peak_vram(device) -> None:
    """Reset CUDA peak-memory tracking before a measured train loop. No-op off CUDA."""
    if isinstance(device, str):
        is_cuda = device.startswith("cuda")
    else:
        is_cuda = getattr(device, "type", "") == "cuda"
    if is_cuda and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_vram_bytes(device) -> int:
    """Peak allocated bytes since the last reset. 0 off CUDA (CPU/MPS have no equivalent
    counter; the cost table marks it n/a there)."""
    if isinstance(device, str):
        is_cuda = device.startswith("cuda")
    else:
        is_cuda = getattr(device, "type", "") == "cuda"
    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
        return int(torch.cuda.max_memory_allocated())
    return 0


def fmt_vram(nbytes: int) -> str:
    return "n/a" if nbytes <= 0 else f"{nbytes / 1024**3:.2f} GB"


# ---------------------------------------------------------------------------
# FLOPs / step estimate (axis 3) — the MECHANISM, made explicit
# ---------------------------------------------------------------------------
def estimate_step_flops(n_base_params: int, tokens_per_step: int, n_adapter_params: int = 0):
    """Crude but variant-COMPARABLE FLOPs/step (fwd+bwd) estimate.

    Standard transformer rule of thumb: ~2*N FLOPs/token for a forward over N params,
    and backward ~2x the forward, so ~6*N FLOPs/token end-to-end. tokens_per_step is the
    total (prompt+completion) token count the step's forwards see.

    The point of returning BOTH the base term and the adapter term separately is to make
    the mechanism visible: base_flops is the same for every variant (same frozen base,
    same tokens); adapter_flops (LoRA's extra matmuls vs the modulation's element-wise
    scale) is microscopic against it. The caller prints the ratio so the reader sees
    compute/step is ~equal -> GPU-hours hinge on steps-to-target, not on this number.

    Returns (base_flops, adapter_flops, total_flops).
    """
    base_flops = 6.0 * n_base_params * tokens_per_step
    # adapter params take part in fwd+bwd too; same 6x rule, vastly smaller N.
    adapter_flops = 6.0 * n_adapter_params * tokens_per_step
    return base_flops, adapter_flops, base_flops + adapter_flops


def fmt_flops(f: float) -> str:
    for unit, scale in (("P", 1e15), ("T", 1e12), ("G", 1e9), ("M", 1e6)):
        if f >= scale:
            return f"{f / scale:.2f} {unit}FLOP"
    return f"{f:.0f} FLOP"


# ---------------------------------------------------------------------------
# steps-to-target + wall-clock (axes 4 + wall) — the GPU-hour drivers
# ---------------------------------------------------------------------------
def steps_to_target(reward_curve, target: float, smooth: int = 3):
    """First step index whose `smooth`-step trailing mean of the reward curve reaches
    `target`. Returns None if never reached (the variant did not converge in budget).

    Trailing-mean smoothing keeps a single lucky GRPO batch from declaring premature
    convergence — the convergence number must reflect a held direction, not noise."""
    if not reward_curve:
        return None
    for i in range(len(reward_curve)):
        lo = max(0, i - smooth + 1)
        window = reward_curve[lo:i + 1]
        if sum(window) / len(window) >= target:
            return i + 1  # 1-indexed step count
    return None


class StepTimer:
    """Accumulates per-step wall-clock so the runner reports mean per-step + total without
    bespoke timing in every experiment."""

    def __init__(self):
        self.t0 = None
        self.per_step = []

    def start(self):
        self.t0 = time.time()
        return self

    def tick(self):
        now = time.time()
        if self.t0 is not None:
            self.per_step.append(now - self.t0)
        self.t0 = now

    @property
    def total_s(self) -> float:
        return float(sum(self.per_step))

    @property
    def mean_step_s(self) -> float:
        return float(sum(self.per_step) / len(self.per_step)) if self.per_step else 0.0


# ---------------------------------------------------------------------------
# the per-variant cost record + GPU-hours rollup
# ---------------------------------------------------------------------------
def gpu_hours(steps_to_target_val, mean_step_s: float, total_steps: int) -> float:
    """GPU-hours to convergence. If the target was reached, charge only the steps it took
    (steps_to_target * mean_step_s); else charge the full budget run (an upper bound, the
    variant did not converge). This is the axis-3 x axis-4 product the headline turns on."""
    n = steps_to_target_val if steps_to_target_val is not None else total_steps
    return float(n * mean_step_s / 3600.0)


def cost_record(variant, trainable, base_params, reward_curve, timer, device,
                tokens_per_step, target_reward):
    """Assemble one variant's full cost row from the live training artifacts."""
    n_par = trainable_param_count(trainable)
    s2t = steps_to_target(reward_curve, target_reward)
    base_f, adapter_f, total_f = estimate_step_flops(base_params, tokens_per_step, n_par)
    return {
        "variant": variant,
        "trainable_params": n_par,
        "peak_vram_bytes": peak_vram_bytes(device),
        "steps_to_target": s2t,
        "total_steps": len(reward_curve),
        "mean_step_s": timer.mean_step_s,
        "total_s": timer.total_s,
        "base_flops_step": base_f,
        "adapter_flops_step": adapter_f,
        "total_flops_step": total_f,
        "gpu_hours": gpu_hours(s2t, timer.mean_step_s, len(reward_curve)),
        "final_reward": reward_curve[-1] if reward_curve else 0.0,
    }


# ---------------------------------------------------------------------------
# the COST TABLE renderer (results/cost-table.md)
# ---------------------------------------------------------------------------
def _adapter_param_share(rec):
    """adapter FLOPs as a fraction of total — the 'mechanism is visible' number."""
    if rec["total_flops_step"] <= 0:
        return 0.0
    return rec["adapter_flops_step"] / rec["total_flops_step"]


def render_cost_table(records, meta, pending_rows=None):
    """Render the markdown COST TABLE. `records` = list of cost_record dicts (measured).
    `pending_rows` = list of dicts for the not-yet-run GPU variants (a-priori predictions).

    Columns: variant | trainable params | peak VRAM | steps-to-target | wall-clock |
             FLOPs/step (est) | GPU-hours.
    """
    L = []
    L.append("# Cost table — modulation vs LoRA fine-tuning")
    L.append("")
    L.append(f"- **run**: {meta.get('label', '?')}  ")
    L.append(f"- **base model**: {meta.get('model', '?')} "
             f"({meta.get('base_params', 0):,} params)  ")
    L.append(f"- **device**: {meta.get('device', '?')}  ")
    L.append(f"- **target reward** (steps-to-target threshold): "
             f"{meta.get('target_reward', 0):.2f}  ")
    L.append(f"- **step budget**: {meta.get('max_steps', '?')}  "
             f"**tokens/step (est)**: {meta.get('tokens_per_step', 0):,}")
    L.append("")
    L.append("## The headline question")
    L.append("")
    L.append("**Is the modulation cheaper in GPU-hours, or only in adapter SIZE + "
             "optimizer VRAM?**")
    L.append("")
    L.append("A-priori prediction (before reading the numbers):")
    L.append("")
    L.append("| cost axis | prediction | certainty |")
    L.append("|---|---|---|")
    L.append("| adapter **size** (trainable params) | modulation ~10^4x smaller "
             "(G=2048 vs LoRA ~16.5M) | **certain** — it is arithmetic |")
    L.append("| **optimizer VRAM** (Adam m,v over trainable) | modulation smaller; the "
             "gap *grows* with model size | likely |")
    L.append("| **compute / step** (FLOPs fwd+bwd) | ~EQUAL — the frozen base dominates; "
             "LoRA's 2 matmuls and the gate's element-wise scale are both negligible | "
             "**certain** — see the FLOPs-share column |")
    L.append("| **GPU-hours** (compute/step x steps-to-target) | decided by "
             "**steps-to-target**, since compute/step is ~equal | **TBD by the run** |")
    L.append("")
    L.append("The mechanism, stated plainly: the *adapter FLOPs share* column below is "
             "`6·N_adapter / 6·(N_base + N_adapter)` — it goes to ~0% as the base grows (on "
             "the 4B base, <0.5% for both adapters; on a tiny smoke base it is larger, since "
             "the base itself is tiny). Once the base dominates, compute/step is ≈equal "
             "across adapters, so the GPU-hour cost reduces to **steps-to-target x "
             "wall-per-step**. The only way the modulation wins on GPU-hours is by "
             "*converging in fewer steps*, not by doing less work per step. The size and "
             "optimizer-VRAM wins are real but separate.")
    L.append("")
    L.append("## Cost table")
    L.append("")
    L.append("| variant | trainable params | peak VRAM | steps-to-target | "
             "wall-clock (mean/step · total) | FLOPs/step (est) | adapter FLOPs share | "
             "GPU-hours |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in records:
        s2t = "—" if r["steps_to_target"] is None else f"{r['steps_to_target']} / {r['total_steps']}"
        share = _adapter_param_share(r)
        share_s = f"{share*100:.2e}%" if 0 < share < 1e-3 else f"{share*100:.3f}%"
        L.append(
            f"| {r['variant']} "
            f"| {r['trainable_params']:,} "
            f"| {fmt_vram(r['peak_vram_bytes'])} "
            f"| {s2t} "
            f"| {r['mean_step_s']:.2f}s · {r['total_s']:.1f}s "
            f"| {fmt_flops(r['total_flops_step'])} "
            f"| {share_s} "
            f"| {r['gpu_hours']:.2e} |"
        )
    if pending_rows:
        for r in pending_rows:
            L.append(
                f"| {r['variant']} "
                f"| {r.get('trainable_params', '?')} "
                f"| {r.get('peak_vram', 'PENDING')} "
                f"| {r.get('steps_to_target', 'PENDING')} "
                f"| {r.get('wall', 'PENDING')} "
                f"| {r.get('flops', 'PENDING')} "
                f"| {r.get('share', '~0%')} "
                f"| {r.get('gpu_hours', 'PENDING')} |"
            )
    L.append("")
    L.append("steps-to-target column reads `reached / total`; `—` = the trailing-mean "
             "reward never crossed the target inside the step budget (no convergence → "
             "GPU-hours charges the full budget as an upper bound).")
    L.append("")
    return "\n".join(L)
