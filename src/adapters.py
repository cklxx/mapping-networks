"""Weight-modulation adapters: a per-group multiplicative gate vs a LoRA baseline.

Both wrap a FROZEN `nn.Linear` in place and train only the adapter params. The base
weight is never mutated; `restore()` puts the original module back.

Modulation (the method):
    W'[c, :] = W[c, :] * (1 + alpha * o[group(c)])
  A single trainable vector `o` of length G is SHARED across every wrapped linear.
  `group(c)` is a FIXED partition of the concatenated output channels into G groups,
  so the trainable param count is G (e.g. 256, 2048) — independent of the model's
  width or depth. `o` inits to zeros, so the adapter starts as an exact identity.

LoRA (the baseline):
    W' x = W x + (alpha / r) * B (A x),  A ~ N(0, 1/in_f), B = 0  (identity at init)
  Trainable params = r * (in_f + out_f) per wrapped linear — orders of magnitude more.

This file is the single source of truth for the adapter math; experiments import it.
"""
import math
import re

import torch
import torch.nn as nn

# Modulation gate strength. 1.0 = honest direct gate (gate = 1 + o). The coherent
# operating band is enforced by clamping |o| in the training loop, not by shrinking alpha.
ALPHA_MOD = 1.0


class DirectMapLinear(nn.Module):
    """W'[c, :] = W[c, :] * (1 + alpha * o[group(c)]).

    `gather_idx` maps THIS linear's local output channels (length out_f) to their group
    index in the shared `o`. Trains `o` only; `base` stays frozen.
    """

    def __init__(self, base: nn.Linear, o_param: nn.Parameter, gather_idx: torch.Tensor):
        super().__init__()
        self.base, self.o, self.gather_idx = base, o_param, gather_idx

    def forward(self, x):
        o = self.o.to(x.dtype)
        gate = 1.0 + ALPHA_MOD * o[self.gather_idx]  # per-output-channel scale
        out = torch.nn.functional.linear(x, self.base.weight.to(x.dtype), None)
        out = out * gate
        if self.base.bias is not None:
            out = out + self.base.bias.to(x.dtype)
        return out


class LoRALinear(nn.Module):
    """W' x = W x + (alpha / r) * B (A x). B inits to zeros -> identity at init."""

    def __init__(self, base: nn.Linear, r: int, alpha: int = 16):
        super().__init__()
        self.base = base
        in_f, out_f = base.weight.shape[1], base.weight.shape[0]
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.randn(r, in_f, device=dev, dtype=dt) * (1.0 / math.sqrt(in_f)))
        self.B = nn.Parameter(torch.zeros(out_f, r, device=dev, dtype=dt))
        self.scale = alpha / r

    def forward(self, x):
        base_out = torch.nn.functional.linear(
            x,
            self.base.weight.to(x.dtype),
            None if self.base.bias is None else self.base.bias.to(x.dtype),
        )
        lora = torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)
        )
        return base_out + self.scale * lora


# ---------------------------------------------------------------------------
# module discovery + (un)installation
# ---------------------------------------------------------------------------
def num_layers_of(model) -> int:
    """Decoder depth from config (generic across HF causal-LM configs)."""
    cfg = model.config
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    mx = -1
    for name, _ in model.named_modules():
        m = re.search(r"\.layers\.(\d+)\.", name)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1


def target_modules(model, last_n_layers: int | None = None, subset: str = "all") -> list[str]:
    """Names of every decoder `*_proj` linear. `last_n_layers` restricts to the top
    block (None = all layers)."""
    allowed = {
        "all": None,
        "attn": ("q_proj", "k_proj", "v_proj", "o_proj", "out_proj"),
        "mlp": ("gate_proj", "up_proj", "down_proj"),
        "o": ("o_proj", "out_proj"),
        "down": ("down_proj",),
        "o_down": ("o_proj", "out_proj", "down_proj"),
    }
    if subset not in allowed:
        raise ValueError(f"unknown target subset: {subset}")
    suffixes = allowed[subset]
    nlayers = num_layers_of(model)
    floor = 0 if last_n_layers is None else max(0, nlayers - last_n_layers)
    out = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        m = re.search(r"\.layers\.(\d+)\.", name)
        if not m or int(m.group(1)) < floor:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if name.endswith("_proj") and (suffixes is None or leaf in suffixes):
            out.append(name)
    return out


def get_parent(model, dotted: str):
    parent = model
    *path, leaf = dotted.split(".")
    for p in path:
        parent = getattr(parent, p)
    return parent, leaf


def install_direct_map(model, names: list[str], G: int):
    """Install the shared modulation gate over `names`. Returns ([o], total_out_channels).

    `o` inits to zeros (exact identity). The group partition is a FIXED, uniform split of
    the concatenated output-channel axis: group(c) = c * G // total_out.
    """
    outs = [getattr(*get_parent(model, n)).weight.shape[0] for n in names]
    total_out = sum(outs)
    dev = getattr(*get_parent(model, names[0])).weight.device
    dt = getattr(*get_parent(model, names[0])).weight.dtype
    o = nn.Parameter(torch.zeros(G, device=dev, dtype=dt))
    global_idx = (torch.arange(total_out, device=dev) * G // total_out).long()
    off = 0
    for n, out_f in zip(names, outs):
        parent, leaf = get_parent(model, n)
        gidx = global_idx[off : off + out_f].clone()
        setattr(parent, leaf, DirectMapLinear(getattr(parent, leaf), o, gidx).to(dev))
        off += out_f
    return [o], total_out


def install_lora(model, names: list[str], r: int):
    """Install an independent LoRA on each of `names`. Returns the trainable param list."""
    params = []
    for n in names:
        parent, leaf = get_parent(model, n)
        dev = getattr(parent, leaf).weight.device
        mod = LoRALinear(getattr(parent, leaf), r).to(dev)
        setattr(parent, leaf, mod)
        params += [mod.A, mod.B]
    return params


def restore(model, names: list[str], originals: list[nn.Module]):
    """Put the original (frozen) modules back, undoing an install_*."""
    for n, orig in zip(names, originals):
        parent, leaf = get_parent(model, n)
        setattr(parent, leaf, orig)


class base_forward:
    """Context manager: swap adapters OUT so a forward runs on the frozen base
    (used to compute the KL(policy || base) reference)."""

    def __init__(self, model, names: list[str]):
        self.model, self.names = model, names

    def __enter__(self):
        self.cur = [getattr(*get_parent(self.model, n)) for n in self.names]
        self.orig = [m.base for m in self.cur]  # each adapter wraps .base = frozen Linear
        restore(self.model, self.names, self.orig)
        return self

    def __exit__(self, *a):
        restore(self.model, self.names, self.cur)
