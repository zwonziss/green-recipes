"""
Recipe 05 — Pruning: deleting weights (and busting a myth).

Takes an already-trained SST-2 classifier, zeroes out 30/50/70/90% of its
Linear weights by magnitude, and measures accuracy, latency, energy and
gzip'd checkpoint size at each level. The forward pass over the validation
set is repeated `--repeats` times (accuracy is deterministic here -- no
dropout, no pruning randomness after the weights are fixed -- but latency
and energy are not, so only those should move). Spoiler: latency will NOT drop.

    python recipes/05_pruning/run.py
"""
import argparse
import gc
import gzip
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.greenmeter import (  # noqa: E402
    GreenMeter, aggregate_trials, compare, print_receipt, save_result,
)
from common.data import pick_device, set_seed  # noqa: E402

RECIPE = "05_pruning"
MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


def make_val_batches(tok, batch_size=64, max_len=128):
    import torch
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2", split="validation")
    batches = []
    for i in range(0, len(ds), batch_size):
        rows = ds[i : i + batch_size]
        enc = tok(rows["sentence"], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
        batches.append((enc["input_ids"], enc["attention_mask"],
                        torch.tensor(rows["label"])))
    return batches


def gzip_size_mb(model) -> float:
    import torch
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(gzip.compress(buf.getvalue(), compresslevel=6)) / 2**20


def sparsity_of(model) -> float:
    import torch.nn as nn
    zeros = total = 0
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            zeros += (mod.weight == 0).sum().item()
            total += mod.weight.numel()
    return zeros / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amounts", default="0,0.3,0.5,0.7,0.9")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=3, help="forward-pass trials to aggregate")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.nn.utils import prune
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = pick_device(args.device)
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL)
    val_b = make_val_batches(tok)
    out_dir = Path(__file__).parent
    results = []

    for amount in [float(a) for a in args.amounts.split(",")]:
        label = f"sparsity_{int(amount * 100)}pct"
        print(f"\n### {label} ###")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)
        model.eval()

        if amount > 0:
            params = [(m, "weight") for m in model.modules()
                      if isinstance(m, nn.Linear)]
            prune.global_unstructured(params, pruning_method=prune.L1Unstructured,
                                      amount=amount)
            for m, name in params:
                prune.remove(m, name)          # bake masks into the weights

        # warmup
        ids, mask, _ = val_b[0]
        with torch.no_grad():
            model(input_ids=ids.to(device), attention_mask=mask.to(device))

        # deterministic given fixed pruned weights -- compute once, not per repeat
        true_sparsity = round(sparsity_of(model), 3)
        gzip_ckpt_mb = round(gzip_size_mb(model), 1)

        trials = []
        for t in range(args.repeats):
            if args.repeats > 1:
                print(f"  -- trial {t + 1}/{args.repeats} --")
            hit = tot = 0
            with GreenMeter(label) as m, torch.no_grad():
                for ids, mask, y in val_b:
                    logits = model(input_ids=ids.to(device),
                                   attention_mask=mask.to(device)).logits
                    hit += (logits.argmax(-1).cpu() == y).sum().item()
                    tot += len(y)
            m.add(val_accuracy=round(hit / tot, 4), true_sparsity=true_sparsity,
                  gzip_ckpt_mb=gzip_ckpt_mb,
                  ms_per_example=round(m.result["wall_s"] / tot * 1000, 2))
            trials.append(m.result)

        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, out_dir / "results", is_baseline=(amount == 0))

        del model
        gc.collect()
        torch.cuda.empty_cache() if device == "cuda" else None
        time.sleep(1.5)

    compare(results, extra_keys=("val_accuracy", "true_sparsity",
                                 "gzip_ckpt_mb", "ms_per_example"))
    print("Accuracy survives surprisingly deep cuts — but check ms_per_example: "
          "dense GPU kernels multiply zeros at full speed. Sparsity saves bytes, "
          "not (automatically) time. For real speed you need structured pruning, "
          "2:4 semi-structured sparsity, or sparse runtimes.")


if __name__ == "__main__":
    main()
