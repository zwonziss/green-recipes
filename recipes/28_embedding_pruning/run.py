"""
Recipe 28 — Shrink the output vocabulary to your actual domain.

A general-purpose LM's final layer projects every hidden state onto its
FULL vocabulary (tens of thousands of tokens) to get next-token logits --
one of the few parts of a forward pass whose cost scales directly with
vocab size. Deploy that model for one narrow domain (this recipe uses
Shakespeare's text as a stand-in for "your actual production traffic") and
most of that vocabulary is dead weight: tokens the domain never uses, whose
logits get computed and thrown away on every single token, every request.

This recipe measures which tokens the domain corpus actually uses, builds a
SMALLER output head containing only those rows (an exact index_select, not
an approximation), and swaps it in -- input embeddings are left untouched,
since the model still needs to be able to READ any token, only the OUTPUT
projection shrinks.

    python recipes/28_embedding_pruning/run.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.greenmeter import (  # noqa: E402
    GreenMeter, aggregate_trials, compare, print_receipt, save_result,
)
from common.data import get_tinyshakespeare, lm_batches, pick_device, set_seed, tokenize_all  # noqa: E402
from common.eval import max_abs_diff  # noqa: E402

RECIPE = "28_embedding_pruning"
MODEL = "HuggingFaceTB/SmolLM2-135M"


def build_pruned_head(old_head, used_ids):
    import torch
    import torch.nn as nn
    has_bias = old_head.bias is not None
    pruned = nn.Linear(old_head.in_features, len(used_ids), bias=has_bias)
    idx = torch.tensor(used_ids, dtype=torch.long, device=old_head.weight.device)
    with torch.no_grad():
        pruned.weight.copy_(old_head.weight.index_select(0, idx))
        if has_bias:
            pruned.bias.copy_(old_head.bias.index_select(0, idx))
    return pruned.to(old_head.weight.device).to(old_head.weight.dtype)


def check_equivalent(model, used_ids, batch, device):
    """The pruned head is an exact index_select of the full head's rows -- its
    logits over the RETAINED tokens must match the full head's, not just be close."""
    import torch
    old_head = model.get_output_embeddings()
    with torch.no_grad():
        full_logits = model(input_ids=batch).logits
    idx = torch.tensor(used_ids, dtype=torch.long, device=device)
    full_at_used = full_logits.index_select(-1, idx)

    pruned_head = build_pruned_head(old_head, used_ids)
    model.set_output_embeddings(pruned_head)
    with torch.no_grad():
        pruned_logits = model(input_ids=batch).logits
    diff = max_abs_diff(full_at_used, pruned_logits)
    model.set_output_embeddings(old_head)  # restore before the "full_vocab_head" timing below

    print(f"Part 0 — numerical check: max |full_head[..., used_ids] - pruned_head| "
          f"logits = {diff:.2e} (should be tiny float noise -- same projection, "
          f"restricted to a subset of output rows).")
    return pruned_head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain-chars", type=int, default=200_000,
                    help="how much of the domain corpus to scan for used tokens")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--n-batches", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=3, help="trials per variant to aggregate")
    args = ap.parse_args()

    device = pick_device(args.device)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
    model.eval()

    vocab_total = model.get_output_embeddings().out_features
    text = get_tinyshakespeare()[:args.domain_chars]
    used_ids = sorted(set(tokenize_all(text, tok)))
    print(f"Domain corpus uses {len(used_ids)} of {vocab_total} vocabulary tokens "
          f"({len(used_ids) / vocab_total * 100:.1f}%).")

    batches = lm_batches(text, tok, args.seq_len, args.batch_size, args.n_batches)
    pruned_head = check_equivalent(model, used_ids, batches[0].to(device), device)

    out_dir = Path(__file__).parent
    results = []
    for label in ("full_vocab_head", "pruned_vocab_head"):
        if label == "pruned_vocab_head":
            model.set_output_embeddings(pruned_head)
        trials = []
        for t in range(args.repeats):
            if args.repeats > 1:
                print(f" -- trial {t + 1}/{args.repeats} [{label}] --")
            with GreenMeter(label) as m:
                with torch.no_grad():
                    for b in batches:
                        model(input_ids=b.to(device))
            m.add(vocab_used=len(used_ids), vocab_total=vocab_total,
                 prune_ratio=round(1 - len(used_ids) / vocab_total, 4))
            trials.append(m.result)
        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, out_dir / "results", is_baseline=(label == "full_vocab_head"))

    compare(results, extra_keys=("vocab_used", "vocab_total", "prune_ratio"))
    print("Same transformer body, same attention/FFN cost -- only the final "
          "projection to vocabulary logits shrank. The bigger that projection's "
          "share of total FLOPs (small models, big vocabularies), the more this "
          "recipe is worth doing; see the Honesty box for what it costs you.")


if __name__ == "__main__":
    main()
