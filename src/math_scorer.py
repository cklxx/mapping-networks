"""MATH-500 \\boxed{} answer extraction + math-equivalence scoring.

Ported faithfully from ARLE's `scripts/arle_capability_eval.py`. Kept verbatim so the
RL reward and the greedy-eval verdict use the SAME extractor the rest of the toolchain
uses — no bespoke scoring that could quietly inflate or deflate a variant.

KNOWN CAVEAT (load-bearing for honest numbers): MATH-500 gold strings occasionally carry
a transcription artifact (e.g. a tuple `(3, \\frac{\\pi}{2})` whose comma was dropped to
`(3\\frac{\\pi}{2})`). This adds a few points of extractor noise to ABSOLUTE accuracy, but
it hits every variant equally, so it does not move the cross-variant deltas.
"""
import re


def extract_last_braced(text: str, marker: str):
    """Depth-aware brace matcher: the LAST balanced {...} after `marker`
    (handles nested braces, e.g. \\boxed{\\frac{a}{b}})."""
    last = None
    start = 0
    while True:
        pos = text.find(marker, start)
        if pos < 0:
            return last
        i = pos + len(marker)
        depth = 1
        out = []
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = "".join(out).strip()
                    if candidate:
                        last = candidate
                    break
            out.append(ch)
            i += 1
        start = pos + len(marker)


def normalize_answer(answer: str) -> str:
    """Strip \\boxed/\\text/\\left/\\right/spacing macros, $, commas, whitespace;
    \\dfrac/\\tfrac -> \\frac; lower-case."""
    s = answer.strip()
    boxed = extract_last_braced(s, "\\boxed{")
    if boxed is not None:
        s = boxed
    s = s.replace("\\$", "").strip("$")
    for old, new in (
        ("\\left", ""),
        ("\\right", ""),
        ("\\!", ""),
        ("\\,", ""),
        ("\\;", ""),
        ("\\:", ""),
        ("\\dfrac", "\\frac"),
        ("\\tfrac", "\\frac"),
    ):
        s = s.replace(old, new)
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = s.replace(",", "")
    s = re.sub(r"\s+", "", s)
    s = s.replace("^\\circ", "").replace("\\circ", "").replace("°", "")
    s = s.rstrip(".")
    return s.lower()


def gold_answer(example: dict) -> str:
    """Normalized gold from a MATH-500 record (`answer` field, else \\boxed in `solution`)."""
    raw = str(example.get("answer") or "")
    if raw:
        return normalize_answer(raw)
    boxed = extract_last_braced(str(example.get("solution") or ""), "\\boxed{")
    return normalize_answer(boxed or "")


def extract_answer(text: str):
    """Last \\boxed{} wins; else #### marker; else 'final answer is/answer is/therefore';
    else the last non-empty line."""
    boxed = extract_last_braced(text, "\\boxed{")
    if boxed:
        return normalize_answer(boxed)
    answer_tags = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if answer_tags:
        return normalize_answer(answer_tags[-1])
    m = re.search(r"####\s*(.+)", text)
    if m:
        return normalize_answer(m.group(1).splitlines()[0])
    for pattern in (
        r"final answer is\s*:?\s*(.+)",
        r"answer is\s*:?\s*(.+)",
        r"therefore\s*,?\s*(.+)",
    ):
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return normalize_answer(matches[-1].splitlines()[0])
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    return normalize_answer(lines[-1])


def reward_of(text: str, gold: str) -> float:
    """1.0 iff the extracted answer math-matches a non-empty gold."""
    return 1.0 if (extract_answer(text) == gold and gold) else 0.0
