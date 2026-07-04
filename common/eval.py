"""
Shared quality-eval helpers so every recipe measures "how much did it cost?"
on a held-out set with the same method, instead of each recipe inventing its
own proxy (or skipping quality entirely).

    from common.eval import lm_perplexity, classification_accuracy, max_abs_diff
"""
from __future__ import annotations


def lm_perplexity(model, batches, device, ctx=None) -> float:
    """Perplexity of a causal LM over held-out batches (not used for training).
    `ctx` is an optional context manager (e.g. torch.autocast) applied per
    batch, so eval matches the precision a variant would actually be served at.
    """
    import contextlib
    import math
    import torch

    ctx = ctx or contextlib.nullcontext()
    was_training = model.training
    model.eval()
    nlls = []
    with torch.no_grad():
        for x in batches:
            x = x.to(device)
            with ctx:
                nlls.append(model(input_ids=x, labels=x).loss.item())
    if was_training:
        model.train()
    return math.exp(sum(nlls) / len(nlls))


def classification_accuracy(model, batches, device) -> float:
    """Accuracy over (input_ids, attention_mask, labels) batches."""
    import torch

    was_training = model.training
    model.eval()
    hit = tot = 0
    with torch.no_grad():
        for ids, mask, y in batches:
            logits = model(input_ids=ids.to(device),
                           attention_mask=mask.to(device)).logits
            hit += (logits.argmax(-1).cpu() == y).sum().item()
            tot += len(y)
    if was_training:
        model.train()
    return hit / tot


def max_abs_diff(a, b) -> float:
    """Largest elementwise |a - b|, as a plain float. For proving two code
    paths that are supposed to compute 'the same math' actually do."""
    return (a.detach().float() - b.detach().float()).abs().max().item()
