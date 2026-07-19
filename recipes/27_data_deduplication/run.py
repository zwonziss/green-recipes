"""
Recipe 27 — Duplicate data wastes training compute.

A scraped or aggregated corpus routinely contains exact and near-exact
duplicates -- mirrors, syndicated content, templated boilerplate. Training
on it "as is" spends real gradient steps re-teaching the model things it
already learned from the first copy. This recipe builds a synthetically
duplicate-laden corpus (every example tripled, then shuffled -- exact
duplicates, the simplest and most verifiable case), dedups it with a plain
hash set, and trains the SAME number of gradient steps on both. Energy
cost is (by construction) about the same either way -- the point isn't a
wall-clock win, it's getting more validation accuracy for the same power bill.

    python recipes/27_data_deduplication/run.py
"""
import argparse
import gc
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.greenmeter import (  # noqa: E402
    GreenMeter, aggregate_trials, compare, print_receipt, save_result,
)
from common.data import set_seed, pick_device  # noqa: E402
from common.eval import classification_accuracy  # noqa: E402

RECIPE = "27_data_deduplication"
STUDENT = "google/bert_uncased_L-2_H-128_A-2"  # BERT-Tiny, 4.4M params


def build_corpora(n_unique, dup_factor, seed=42):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2", split="train").shuffle(seed=seed).select(range(n_unique))
    unique_pairs = list(zip(ds["sentence"], ds["label"]))
    assert len(set(t for t, _ in unique_pairs)) == n_unique, \
        "sampled 'unique' pool already contains duplicate sentences -- pick a smaller n_unique or a different seed"

    dup_pool = unique_pairs * dup_factor
    random.Random(seed).shuffle(dup_pool)  # scatter the copies, not adjacent blocks -- mimics a
                                            # real crawl where a document and its mirror land far apart

    seen = set()
    dedup_pairs = []
    for t, l in dup_pool:
        if t not in seen:
            seen.add(t)
            dedup_pairs.append((t, l))
    assert len(dedup_pairs) == n_unique, "dedup did not recover exactly the original unique count"
    return dup_pool, dedup_pairs


def make_val_batches(tok, batch_size=64, max_len=64):
    import torch
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2", split="validation")  # real held-out set, untouched by dup/dedup
    batches = []
    for i in range(0, len(ds), batch_size):
        rows = ds[i:i + batch_size]
        enc = tok(rows["sentence"], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
        batches.append((enc["input_ids"], enc["attention_mask"], torch.tensor(rows["label"])))
    return batches


def make_step_batches(pairs, tok, batch_size, n_steps, seed, max_len=64):
    """Cycles through `pairs` for exactly n_steps batches, reshuffling on every
    wraparound -- the same helper serves a single clean pass (n_steps * batch_size
    <= len(pairs)) and many repeated passes over a smaller deduped pool."""
    import torch
    rng = random.Random(seed)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    batches, pos = [], 0
    for _ in range(n_steps):
        if pos + batch_size > len(order):
            rng.shuffle(order)
            pos = 0
        idx = order[pos:pos + batch_size]
        pos += batch_size
        texts = [pairs[i][0] for i in idx]
        labels = [pairs[i][1] for i in idx]
        enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        batches.append((enc["input_ids"], enc["attention_mask"], torch.tensor(labels)))
    return batches


def train_and_eval(label, pairs, n_steps, tok, val_batches, device, batch_size, lr, seed):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification

    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(STUDENT, num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    batches = make_step_batches(pairs, tok, batch_size, n_steps, seed=seed)

    model.train()
    with GreenMeter(label) as m:
        for i, (ids, mask, y) in enumerate(batches):
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            loss = F.cross_entropy(model(input_ids=ids, attention_mask=mask).logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if (i + 1) % 20 == 0:
                print(f"  [{label}] seed={seed} step {i + 1:>3}/{n_steps}  loss {loss.item():.3f}")

    acc = classification_accuracy(model, val_batches, device)
    m.add(val_accuracy=round(acc, 4), n_steps=n_steps)
    del model, opt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    time.sleep(1)
    return m.result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-unique", type=int, default=1200)
    ap.add_argument("--dup-factor", type=int, default=3,
                    help="how many times each unique example is repeated in the raw corpus")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seeds", default="42,43,44",
                    help="comma list of seeds -- each drives a fresh init + data order")
    args = ap.parse_args()

    device = pick_device(args.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDENT)
    seeds = [int(s) for s in args.seeds.split(",")]

    dup_pool, dedup_pairs = build_corpora(args.n_unique, args.dup_factor)
    n_steps = len(dup_pool) // args.batch_size
    removed = len(dup_pool) - len(dedup_pairs)
    print(f"Part 0 — numerical check: raw corpus has {len(dup_pool)} examples, "
          f"{removed} of them ({removed / len(dup_pool) * 100:.0f}%) exact duplicates. "
          f"Deduping recovers exactly {len(dedup_pairs)} unique examples "
          f"(matches --n-unique={args.n_unique}: {len(dedup_pairs) == args.n_unique}).")
    print(f"Both variants train for the SAME {n_steps} gradient steps -- "
          f"raw_duplicated sees each unique example ~{args.dup_factor}x in one pass; "
          f"deduped cycles its smaller pool ~{n_steps * args.batch_size / len(dedup_pairs):.1f}x "
          f"to fill the same step budget.")

    val_batches = make_val_batches(tok)
    out_dir = Path(__file__).parent
    results = []
    for label, pairs in (("raw_duplicated", dup_pool), ("deduped", dedup_pairs)):
        print(f"\n### {label} across seeds {seeds} ###")
        trials = []
        for seed in seeds:
            trials.append(train_and_eval(label, pairs, n_steps, tok, val_batches, device,
                                         args.batch_size, args.lr, seed))
        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, out_dir / "results", is_baseline=(label == "raw_duplicated"))

    compare(results, extra_keys=("val_accuracy", "n_steps"))
    print("Energy/time should land close together -- same steps, same batch size, "
          "same model. Watch val_accuracy against the */~ marker: with the SAME "
          "compute bill, does spending it on non-duplicated signal actually win?")


if __name__ == "__main__":
    main()
