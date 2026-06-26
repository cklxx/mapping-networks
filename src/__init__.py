"""Mapping-network weight-modulation: reusable adapters + MATH scorer.

`adapters`     — DirectMapLinear (the modulation gate) + LoRALinear (baseline) + install/restore.
`math_scorer`  — MATH-500 \\boxed{} extraction + math-equivalence reward.
"""
from . import adapters, math_scorer  # noqa: F401
