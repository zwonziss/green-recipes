"""
Recipe 15 — DataLoader efficiency: a GPU waiting on the CPU still draws power.

Builds a deliberately CPU-heavy Dataset (it re-tokenizes a fresh text chunk
on every __getitem__, instead of reusing a precomputed cache like every other
recipe in this cookbook) and trains the same steps through it with three
DataLoader configs: no workers, a couple of pinned workers, and pinned
workers with prefetching. Idle GPU time between batches still burns power --
this recipe measures whether fixing the loader actually gets some of it back.

    python recipes/15_dataloader_efficiency/run.py
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
from common.data import get_tinyshakespeare, pick_device, set_seed  # noqa: E402

RECIPE = "15_dataloader_efficiency"
MODEL = "HuggingFaceTB/SmolLM2-135M"
CHUNK_CHARS = 6000  # deliberately large -> real, visible per-item CPU cost


class TokenizingDataset:
    """Re-tokenizes a fresh text slice per item -- the realistic 'preprocess
    on the fly' case, unlike common.data.lm_batches' precomputed tensors."""

    def __init__(self, text, tokenizer, seq_len, n_items):
        self.text, self.tok, self.seq_len, self.n_items = text, tokenizer, seq_len, n_items

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        start = random.Random(idx).randrange(0, len(self.text) - CHUNK_CHARS)
        chunk = self.text[start:start + CHUNK_CHARS]
        enc = self.tok(chunk, truncation=True, max_length=self.seq_len,
                       padding="max_length", return_tensors="pt")
        return enc["input_ids"][0]


def run_variant(label, loader_kwargs, dataset, steps, lr, device):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM

    set_seed(42)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    loader = DataLoader(dataset, **loader_kwargs)
    it = iter(loader)
    t0 = time.perf_counter()
    first_batch = next(it)                 # includes worker spin-up cost
    startup_s = time.perf_counter() - t0

    def step(x):
        x = x.to(device)
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.item()

    with GreenMeter(label) as m:
        losses = [step(first_batch)]
        for _ in range(steps - 1):
            losses.append(step(next(it)))

    seq_len = first_batch.shape[-1]
    tokens = steps * loader_kwargs["batch_size"] * seq_len
    m.add(tokens_per_s=int(tokens / m.result["wall_s"]),
          loader_startup_s=round(startup_s, 3),
          num_workers=loader_kwargs.get("num_workers", 0))

    del model, opt, loader, it
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    time.sleep(1)
    return m.result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=196)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=3, help="trials per config to aggregate")
    args = ap.parse_args()

    device = pick_device(args.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = get_tinyshakespeare()
    dataset = TokenizingDataset(text, tok, args.seq_len, n_items=args.steps * args.batch_size * 3)
    pin = device == "cuda"

    plan = [
        ("workers0_nopin", dict(batch_size=args.batch_size, num_workers=0,
                                pin_memory=False, shuffle=False)),
        ("workers2_pin", dict(batch_size=args.batch_size, num_workers=2, pin_memory=pin,
                              shuffle=False, persistent_workers=True)),
        ("workers4_pin_prefetch", dict(batch_size=args.batch_size, num_workers=4,
                                       pin_memory=pin, shuffle=False,
                                       persistent_workers=True, prefetch_factor=4)),
    ]

    out_dir = Path(__file__).parent
    results = []
    for label, kwargs in plan:
        print(f"\n### {label} ###")
        trials = []
        for t in range(args.repeats):
            if args.repeats > 1:
                print(f" -- trial {t + 1}/{args.repeats} --")
            trials.append(run_variant(label, kwargs, dataset, args.steps, args.lr, device))
        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, out_dir / "results", is_baseline=(label == "workers0_nopin"))

    compare(results, extra_keys=("tokens_per_s", "loader_startup_s", "num_workers"))
    print("Same model, same steps, same data distribution -- only the loader "
          "config changed. Watch gpu_avg_power_w alongside wall_s: a lower "
          "average power with a LONGER wall time is the GPU idling on data, "
          "not doing less work.")


if __name__ == "__main__":
    main()
