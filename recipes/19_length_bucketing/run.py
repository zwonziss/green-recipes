"""
Recipe 19 — Length-bucketed batching: padding is wasted compute.

A batch's tensor width is fixed to its longest sequence -- every shorter
sequence in that batch pads out to that width, and the model still runs
full attention/FFN math over those pad positions (masked out of the result,
not free to compute). Batch random-length examples together and the average
pad fraction is high; sort by length first so each batch groups similar
lengths, and the pad fraction collapses.

Part 0 proves this changes nothing about the answer: the same example
produces the same logits whether it's batched randomly or by length bucket
-- attention masking makes padding invisible to the math, only to the bill.

    python recipes/19_length_bucketing/run.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.greenmeter import (  # noqa: E402
    GreenMeter, aggregate_trials, compare, print_receipt, save_result,
)
from common.data import pick_device, set_seed  # noqa: E402
from common.eval import max_abs_diff  # noqa: E402

RECIPE = "19_length_bucketing"
MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


def load_examples(tok, n, max_len=128, seed=42):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2", split="train").shuffle(seed=seed).select(range(n))
    ids = [tok(s, truncation=True, max_length=max_len)["input_ids"] for s in ds["sentence"]]
    return ids, list(ds["label"])


def make_batches(ids, labels, batch_size, tok, bucketed, seed=0):
    import torch
    order = list(range(len(ids)))
    if bucketed:
        order.sort(key=lambda i: len(ids[i]))
    # unbucketed: keep the dataset's own (already-shuffled) order -- this is
    # exactly what "just batch requests as they arrive" looks like in practice.
    batches = []
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        seqs = [ids[j] for j in idx]
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), width), dtype=torch.long)
        for r, s in enumerate(seqs):
            input_ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
            attn[r, :len(s)] = 1
        batches.append((input_ids, attn, torch.tensor([labels[j] for j in idx]), idx,
                        sum(len(s) for s in seqs), len(seqs) * width))
    return batches


def check_equivalent(model, ids, tok, device):
    """The same example, alone vs. padded inside a batch, must score identically --
    padding only changes what's computed, never the answer for a real token."""
    import torch
    example = ids[0]
    alone = torch.tensor([example], dtype=torch.long)
    alone_mask = torch.ones_like(alone)
    padded = torch.full((1, len(example) + 20), tok.pad_token_id, dtype=torch.long)
    padded[0, :len(example)] = torch.tensor(example, dtype=torch.long)
    padded_mask = torch.zeros_like(padded)
    padded_mask[0, :len(example)] = 1
    with torch.no_grad():
        logits_alone = model(input_ids=alone.to(device), attention_mask=alone_mask.to(device)).logits
        logits_padded = model(input_ids=padded.to(device), attention_mask=padded_mask.to(device)).logits
    diff = max_abs_diff(logits_alone, logits_padded)
    print(f"Part 0 — numerical check: max |unpadded - padded(+20 pad tokens)| logits "
          f"for the same example = {diff:.2e} (should be tiny float noise).")


def run_pass(model, batches, device):
    import torch
    correct = total = 0
    real_tokens = padded_tokens = 0
    with torch.no_grad():
        for input_ids, attn, labels, idx, real, padded in batches:
            logits = model(input_ids=input_ids.to(device), attention_mask=attn.to(device)).logits
            correct += (logits.argmax(-1).cpu() == labels).sum().item()
            total += len(labels)
            real_tokens += real
            padded_tokens += padded
    return correct / total, 1 - real_tokens / padded_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-examples", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=3, help="trials per variant to aggregate")
    args = ap.parse_args()

    device = pick_device(args.device)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)
    model.eval()

    ids, labels = load_examples(tok, args.n_examples)
    check_equivalent(model, ids, tok, device)

    out_dir = Path(__file__).parent
    results = []
    for label_name, bucketed in (("random_batching", False), ("length_bucketed", True)):
        print(f"\n### {label_name} ###")
        trials = []
        for t in range(args.repeats):
            if args.repeats > 1:
                print(f" -- trial {t + 1}/{args.repeats} --")
            batches = make_batches(ids, labels, args.batch_size, tok, bucketed)
            with GreenMeter(label_name) as m:
                acc, pad_ratio = run_pass(model, batches, device)
            m.add(val_accuracy=round(acc, 4), padding_ratio=round(pad_ratio, 4),
                 n_examples=args.n_examples)
            trials.append(m.result)
        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, out_dir / "results", is_baseline=(label_name == "random_batching"))

    compare(results, extra_keys=("val_accuracy", "padding_ratio", "n_examples"))
    print("Accuracy should match (same math per example, only the padding differs). "
          "The padding_ratio and energy/time columns are where the story lives: "
          "sorting by length before batching pads far less, for free.")


if __name__ == "__main__":
    main()
