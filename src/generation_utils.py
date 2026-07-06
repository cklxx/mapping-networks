"""Small generation helpers shared by the 9B experiment scripts."""


def stop_token_ids(tok):
    """Return unique token ids that should terminate chat generation."""
    ids = []

    def add(token_id):
        if isinstance(token_id, (list, tuple)):
            for x in token_id:
                add(x)
            return
        if token_id is None:
            return
        try:
            token_id = int(token_id)
        except (TypeError, ValueError):
            return
        if token_id >= 0 and token_id not in ids:
            ids.append(token_id)

    add(getattr(tok, "eos_token_id", None))
    unk = getattr(tok, "unk_token_id", None)
    for token in ("<|im_end|>", "</s>", "<|endoftext|>"):
        try:
            tid = tok.convert_tokens_to_ids(token)
        except Exception:
            continue
        if tid != unk:
            add(tid)
    return ids


def generation_kwargs(tok):
    kwargs = {
        "pad_token_id": tok.pad_token_id
        if tok.pad_token_id is not None
        else tok.eos_token_id,
    }
    stops = stop_token_ids(tok)
    if stops:
        kwargs["eos_token_id"] = stops[0] if len(stops) == 1 else stops
    return kwargs


def trim_completion(comp, stops):
    """Remove padding after the first stop token.

    Returns (trimmed_tensor, stopped). The stop token itself is kept so the
    policy still receives credit for ending the answer, but any padded tail is
    excluded from length, KL, and reward decoding.
    """
    if not stops:
        return comp, False
    stop_set = set(int(x) for x in stops)
    for i, tid in enumerate(comp.detach().cpu().tolist()):
        if int(tid) in stop_set:
            return comp[: i + 1], True
    return comp, False
