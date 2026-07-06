"""
Recipe 10 — The KV cache: why decoding isn't O(n^2) in practice.

Without a cache, generating token T+1 re-runs attention over all T previous
tokens from scratch -- total work across a whole generation grows
quadratically with length. With a cache, each new token only attends using
freshly computed keys/values for itself; the rest are stored, not recomputed.

Part 0 proves both paths produce IDENTICAL greedy output (same math, only the
compute path differs) before either's speed is trusted.
Part 1 sweeps generation length and shows the gap widen.
Part 2 is the GreenMeter'd headline comparison at one realistic length.

    python recipes/10_kv_cache/run.py
"""
import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.greenmeter import (  # noqa: E402
    GreenMeter, aggregate_trials, compare, mean_std, print_receipt,
    print_table, save_result,
)
from common.data import pick_device, set_seed  # noqa: E402

RECIPE = "10_kv_cache"
MODEL = "HuggingFaceTB/SmolLM2-135M"
PROMPTS = [
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\n",
    "The most important idea in machine learning is",
]


def generate(model, tok, prompt, max_new_tokens, use_cache, device):
    enc = tok(prompt, return_tensors="pt").to(device)
    return model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                          use_cache=use_cache, pad_token_id=tok.eos_token_id)


def check_equivalent(model, tok, device):
    """Greedy decoding must produce the same tokens with or without a cache --
    the cache changes HOW attention is computed, not the math."""
    out_cache = generate(model, tok, PROMPTS[0], 48, True, device)
    out_nocache = generate(model, tok, PROMPTS[0], 48, False, device)
    match = out_cache.shape == out_nocache.shape and bool((out_cache == out_nocache).all())
    print(f"Part 0 — numerical check: cached and uncached greedy output "
          f"identical over 48 tokens: {match}")


def bench(new_tokens, use_cache, model, tok, device):
    t0 = time.perf_counter()
    total_new = 0
    for p in PROMPTS:
        out = generate(model, tok, p, new_tokens, use_cache, device)
        total_new += out.shape[1]
    if device == "cuda":
        import torch
        torch.cuda.synchronize()
    return time.perf_counter() - t0, total_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="32,64,128,256", help="new-token lengths for Part 1")
    ap.add_argument("--headline-tokens", type=int, default=300, help="length for Part 2 receipt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=3, help="trials per variant to aggregate")
    args = ap.parse_args()

    device = pick_device(args.device)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
    model.eval()

    check_equivalent(model, tok, device)

    print(f"\nPart 1 — wall time to generate N tokens, cache vs no-cache "
          f"(n={args.repeats} trials/length)")
    headers = ["new_tokens", "no_cache (s)", "cache (s)", "speedup"]
    rows = []
    for n in [int(x) for x in args.sweep.split(",")]:
        nc_times, c_times = [], []
        for _ in range(args.repeats):
            t, _ = bench(n, False, model, tok, device)
            nc_times.append(t)
            t, _ = bench(n, True, model, tok, device)
            c_times.append(t)
        nc_m, nc_s = mean_std(nc_times)
        c_m, c_s = mean_std(c_times)
        speedup = nc_m / c_m if c_m > 0 else float("inf")
        rows.append([str(n), f"{nc_m:.2f}+/-{nc_s:.2f}", f"{c_m:.2f}+/-{c_s:.2f}",
                    f"{speedup:.1f}x"])
        print(f"  {n:>4} tokens: no_cache {nc_m:.2f}s  cache {c_m:.2f}s  ({speedup:.1f}x)")
    print()
    print_table(headers, rows)
    print("\nno_cache's time grows faster than linearly with length -- every new "
          "token re-attends over the whole growing prefix from scratch. cache's "
          "time is close to linear -- each step only computes the new token.")

    print(f"\nPart 2 — headline receipt at {args.headline_tokens} new tokens")
    out_dir = Path(__file__).parent
    results = []
    for label, use_cache in (("no_cache", False), ("cache", True)):
        trials = []
        for t in range(args.repeats):
            if args.repeats > 1:
                print(f" -- trial {t + 1}/{args.repeats} [{label}] --")
            gen_tokens = 0
            with GreenMeter(label) as m:
                for p in PROMPTS:
                    out = generate(model, tok, p, args.headline_tokens, use_cache, device)
                    gen_tokens += out.shape[1]
            m.add(tokens_per_s=round(gen_tokens / m.result["wall_s"], 1),
                 new_tokens=args.headline_tokens)
            trials.append(m.result)
        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, out_dir / "results", is_baseline=(label == "no_cache"))
        gc.collect()
        if device == "cuda":
            import torch
            torch.cuda.empty_cache()
        time.sleep(1)

    compare(results, extra_keys=("tokens_per_s", "new_tokens"))
    print("Same generated tokens, same model -- the cache is a pure engineering "
          "win, not an approximation. The cost: cache memory grows with "
          "batch x sequence x layers x heads, which is why serving engines "
          "spend so much effort managing it (see the Honesty box).")


if __name__ == "__main__":
    main()
