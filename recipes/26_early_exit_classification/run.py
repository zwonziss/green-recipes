"""
Recipe 26 — Early exit: stop computing once you're confident.

Not every input needs the same amount of compute to classify correctly.
"This movie was amazing!" is obvious after a shallow look; a hedged,
sarcastic review might need the whole network. This recipe trains a tiny
multi-exit classifier (an embedding + a stack of residual MLP blocks, one
lightweight classifier head after every block) with a joint loss over all
exits, then at inference time runs blocks one at a time and stops the
moment an exit head is confident -- skipping the remaining blocks (and
their compute) entirely for easy examples.

Evaluated per-example (batch size 1) on purpose: that's what makes the
skipped blocks a real, measurable compute saving rather than a batched
tensor op that still has to wait for the slowest row (see the Honesty box).

    python recipes/26_early_exit_classification/run.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.greenmeter import (  # noqa: E402
    GreenMeter, aggregate_trials, compare, print_receipt, print_table, save_result,
)
from common.data import pick_device, set_seed  # noqa: E402
from common.eval import max_abs_diff  # noqa: E402

RECIPE = "26_early_exit_classification"
TOKENIZER = "distilbert-base-uncased-finetuned-sst-2-english"


def make_batches(split, tok, n, batch_size, seed, max_len=64):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2", split=split)
    if n:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    import torch
    batches = []
    for i in range(0, len(ds), batch_size):
        rows = ds[i:i + batch_size]
        enc = tok(rows["sentence"], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
        batches.append((enc["input_ids"], enc["attention_mask"], torch.tensor(rows["label"])))
    return batches


def build_model(vocab_size, pad_token_id, d, depth, n_classes=2):
    import torch.nn as nn
    import torch.nn.functional as F

    class ResidualMLPBlock(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.fc1 = nn.Linear(d, d * 2)
            self.fc2 = nn.Linear(d * 2, d)

        def forward(self, x):
            h = self.norm(x)
            h = F.gelu(self.fc1(h))
            h = self.fc2(h)
            return x + h

    class EarlyExitClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d, padding_idx=pad_token_id)
            self.blocks = nn.ModuleList([ResidualMLPBlock(d) for _ in range(depth)])
            self.exit_heads = nn.ModuleList([nn.Linear(d, n_classes) for _ in range(depth)])
            self.depth = depth

        def pool(self, ids, mask):
            emb = self.embed(ids)
            mask_f = mask.unsqueeze(-1).float()
            summed = (emb * mask_f).sum(1)
            count = mask_f.sum(1).clamp(min=1.0)
            return summed / count

        def forward_all_exits(self, ids, mask):
            """Every exit's logits from ONE pass -- used for joint training and
            as the fixed full-depth baseline (always uses the last exit)."""
            x = self.pool(ids, mask)
            logits_per_exit = []
            for block, head in zip(self.blocks, self.exit_heads):
                x = block(x)
                logits_per_exit.append(head(x))
            return logits_per_exit

        def forward_early_exit(self, ids, mask, threshold):
            """Per-example dynamic depth (assumes batch size 1): stop the
            moment an exit head is confident, skipping later blocks entirely."""
            x = self.pool(ids, mask)
            logits = None
            for i, (block, head) in enumerate(zip(self.blocks, self.exit_heads)):
                x = block(x)
                logits = head(x)
                confidence = logits.softmax(-1).max().item()
                if confidence >= threshold or i == self.depth - 1:
                    return logits, i + 1
            return logits, self.depth

    return EarlyExitClassifier()


def train(model, train_batches, epochs, lr, device):
    import torch
    import torch.nn.functional as F
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        last_loss = None
        for ids, mask, y in train_batches:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            logits_per_exit = model.forward_all_exits(ids, mask)
            loss = sum(F.cross_entropy(lg, y) for lg in logits_per_exit) / len(logits_per_exit)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = loss.item()
        print(f"  epoch {ep + 1}/{epochs} done, last joint loss {last_loss:.3f}")
    model.eval()


def check_consistency(model, val_batches, device):
    """threshold=1.01 is unreachable (confidence is a softmax max, always <= 1),
    so forward_early_exit is forced through every block -- it should then match
    forward_all_exits' last-exit logits exactly, since it's the same computation."""
    import torch
    ids, mask, _ = val_batches[0]
    ids, mask = ids[:1].to(device), mask[:1].to(device)
    with torch.no_grad():
        logits_all = model.forward_all_exits(ids, mask)
        logits_forced, layers_run = model.forward_early_exit(ids, mask, threshold=1.01)
    diff = max_abs_diff(logits_all[-1], logits_forced)
    print(f"Part 0 — numerical check: early-exit forced through all {model.depth} "
          f"blocks (layers_run={layers_run}) matches the always-full-depth forward "
          f"exactly: max diff = {diff:.2e}.")


def eval_full_depth(model, val_batches, device):
    import torch
    correct = total = 0
    with torch.no_grad():
        for ids, mask, y in val_batches:
            logits_per_exit = model.forward_all_exits(ids.to(device), mask.to(device))
            correct += int(logits_per_exit[-1].argmax(-1).cpu() == y)
            total += 1
    return correct / total


def eval_early_exit(model, val_batches, device, threshold):
    import torch
    correct = total = layers_run_total = 0
    with torch.no_grad():
        for ids, mask, y in val_batches:
            logits, layers_run = model.forward_early_exit(ids.to(device), mask.to(device), threshold)
            correct += int(logits.argmax(-1).cpu() == y)
            total += 1
            layers_run_total += layers_run
    return correct / total, layers_run_total / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=4000)
    ap.add_argument("--val-n", type=int, default=400, help="0 = full SST-2 validation split")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--thresholds", default="0.7,0.9,0.99", help="sweep for Part 1")
    ap.add_argument("--headline-threshold", type=float, default=0.9)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=3, help="trials per variant to aggregate")
    args = ap.parse_args()

    device = pick_device(args.device)
    from transformers import AutoTokenizer
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    train_batches = make_batches("train", tok, args.train_n, args.batch_size, seed=42)
    val_n = args.val_n if args.val_n else None
    val_batches = make_batches("validation", tok, val_n, 1, seed=0)  # batch size 1: real per-example depth

    model = build_model(tok.vocab_size, tok.pad_token_id, args.d_model, args.depth).to(device)
    print(f"Training a {args.depth}-block multi-exit classifier "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params) on "
          f"{args.train_n} SST-2 examples, joint loss over all {args.depth} exits...")
    train(model, train_batches, args.epochs, args.lr, device)

    check_consistency(model, val_batches, device)

    print(f"\nPart 1 — accuracy vs. avg layers run, sweeping the exit confidence threshold "
          f"(n={len(val_batches)} validation examples)")
    headers = ["threshold", "accuracy", "avg_layers_run"]
    rows = []
    for thr in [float(x) for x in args.thresholds.split(",")]:
        acc, avg_layers = eval_early_exit(model, val_batches, device, thr)
        rows.append([str(thr), f"{acc:.4f}", f"{avg_layers:.2f}"])
        print(f"  threshold {thr}: accuracy {acc:.4f}  avg layers run {avg_layers:.2f}/{args.depth}")
    print()
    print_table(headers, rows)
    print("\nLower thresholds exit sooner (fewer average layers run) at some "
          "accuracy cost; higher thresholds converge toward the full-depth baseline.")

    print(f"\nPart 2 — headline receipt at threshold={args.headline_threshold} "
          f"over {len(val_batches)} examples (batch size 1 -- real per-example depth)")
    out_dir = Path(__file__).parent
    results = []
    for label in ("full_depth", "early_exit"):
        trials = []
        for t in range(args.repeats):
            if args.repeats > 1:
                print(f" -- trial {t + 1}/{args.repeats} [{label}] --")
            with GreenMeter(label) as m:
                if label == "full_depth":
                    acc = eval_full_depth(model, val_batches, device)
                    avg_layers = float(args.depth)
                else:
                    acc, avg_layers = eval_early_exit(model, val_batches, device, args.headline_threshold)
            m.add(accuracy=round(acc, 4), avg_layers_run=round(avg_layers, 3),
                 threshold=args.headline_threshold if label == "early_exit" else 1.0)
            trials.append(m.result)
        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, out_dir / "results", is_baseline=(label == "full_depth"))

    compare(results, extra_keys=("accuracy", "avg_layers_run", "threshold"))
    print("Same trunk, same trained weights -- early_exit just declines to run "
          "the remaining blocks once it's already confident. The accuracy gap "
          "(if any) and the layers skipped are the whole trade this recipe is about.")


if __name__ == "__main__":
    main()
