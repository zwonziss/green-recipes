"""
Recipe 18 — Prefix caching: stop re-reading the system prompt every request.

Many real workloads send many requests that share one long prefix (a system
prompt, few-shot examples, a tool schema) plus a short, different suffix each
time. Recomputing that shared prefix's attention keys/values from scratch on
every request wastes work identical to the last request's. This recipe
computes the prefix's KV cache once per "cached" run and reuses a
(deep-copied) snapshot of it per request, verifying the math matches a
from-scratch forward pass first.

Prefix and suffix token ids are concatenated, never re-tokenized as one
joined string -- a BPE tokenizer can merge differently across that seam,
which would silently make the two paths compute over different tokens.

    python recipes/18_prefix_caching/run.py
"""
import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.greenmeter import (  # noqa: E402
    GreenMeter, aggregate_trials, compare, mean_std, print_receipt,
    print_table, save_result,
)
from common.data import pick_device, set_seed  # noqa: E402
from common.eval import max_abs_diff  # noqa: E402

RECIPE = "18_prefix_caching"
MODEL = "HuggingFaceTB/SmolLM2-135M"

SYSTEM_PROMPT = (
    "You are a careful, concise assistant embedded in a customer support tool. "
    "Always answer in at most two sentences. Never invent policy details you "
    "were not given. If a request is ambiguous, ask exactly one clarifying "
    "question instead of guessing. Follow the house style guide: no exclamation "
    "marks, no emoji, no filler phrases like 'great question'. When a customer "
    "reports an error, first restate the error in your own words, then propose "
    "the single most likely fix. When you are not confident, say so explicitly "
    "rather than presenting a guess as fact. Treat every prior turn in this "
    "conversation as authoritative context that must not be contradicted.\n\n"
)
USER_TURNS = [f"User: Question #{i}, please respond.\nAssistant:" for i in range(100)]


def clone_cache(past_key_values):
    return copy.deepcopy(past_key_values)


def tokenize_suffixes(tok, suffixes, device):
    return [tok(s, return_tensors="pt").input_ids.to(device) for s in suffixes]


def run_baseline(model, prefix_ids, suffix_id_list, device):
    """Every request re-reads prefix + suffix from scratch -- token ids are
    concatenated (never re-tokenized as one string) so both paths process
    the exact same tokens."""
    import torch
    for suffix_ids in suffix_id_list:
        full_ids = torch.cat([prefix_ids, suffix_ids], dim=1)
        mask = torch.ones_like(full_ids)
        with torch.no_grad():
            model(input_ids=full_ids, attention_mask=mask, use_cache=False)


def run_cached(model, prefix_ids, suffix_id_list, device):
    """Pays for the prefix once per call (a realistic 'cached' cost -- not
    hidden outside the timed block), then reuses a cloned snapshot per request."""
    import torch
    with torch.no_grad():
        base_cache = model(input_ids=prefix_ids, use_cache=True).past_key_values
    prefix_len = prefix_ids.shape[1]
    for suffix_ids in suffix_id_list:
        suffix_len = suffix_ids.shape[1]
        mask = torch.ones((1, prefix_len + suffix_len), dtype=torch.long, device=device)
        # position_ids must continue from the cached prefix's length -- a bare
        # forward call (unlike generate()) won't infer this on its own, and
        # defaulting to 0..suffix_len-1 would silently feed the model wrong
        # rotary positions for every suffix token.
        positions = torch.arange(prefix_len, prefix_len + suffix_len, device=device).unsqueeze(0)
        with torch.no_grad():
            model(input_ids=suffix_ids, past_key_values=clone_cache(base_cache),
                 attention_mask=mask, position_ids=positions, use_cache=True)


def check_equivalent(model, prefix_ids, suffix_id_list, device):
    import torch
    suffix_ids = suffix_id_list[0]
    suffix_len = suffix_ids.shape[1]
    full_ids = torch.cat([prefix_ids, suffix_ids], dim=1)
    with torch.no_grad():
        base_logits = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids),
                            use_cache=False).logits[:, -suffix_len:, :]
        base_cache = model(input_ids=prefix_ids, use_cache=True).past_key_values
        prefix_len = prefix_ids.shape[1]
        mask = torch.ones((1, prefix_len + suffix_len), dtype=torch.long, device=device)
        positions = torch.arange(prefix_len, prefix_len + suffix_len, device=device).unsqueeze(0)
        cached_logits = model(input_ids=suffix_ids, past_key_values=clone_cache(base_cache),
                              attention_mask=mask, position_ids=positions, use_cache=True).logits
    diff = max_abs_diff(base_logits, cached_logits)
    print(f"Part 0 — numerical check: max |from_scratch - cached_prefix| logits "
          f"over the suffix = {diff:.2e} (should be tiny float noise).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="1,10,50", help="request counts for Part 1")
    ap.add_argument("--headline-requests", type=int, default=50, help="count for Part 2 receipt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=3, help="trials per variant to aggregate")
    args = ap.parse_args()

    device = pick_device(args.device)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL)
    # fp32 on purpose: SmolLM2 checkpoints default-load in bf16, whose ~3
    # significant digits make the Part 0 logit diff look alarmingly large
    # (~0.5) even when the two paths compute the identical thing -- fp32
    # keeps the "should be tiny float noise" check meaningful.
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(device)
    model.eval()

    prefix_ids = tok(SYSTEM_PROMPT, return_tensors="pt").input_ids.to(device)
    all_suffix_ids = tokenize_suffixes(tok, USER_TURNS, device)
    print(f"Shared prefix: {prefix_ids.shape[1]} tokens")

    check_equivalent(model, prefix_ids, all_suffix_ids, device)

    print(f"\nPart 1 — wall time over N requests, no cache vs prefix cache "
          f"(n={args.repeats} trials/count)")
    headers = ["n_requests", "no_cache (s)", "cached (s)", "speedup"]
    rows = []
    for n in [int(x) for x in args.sweep.split(",")]:
        suffix_ids_n = all_suffix_ids[:n]
        nc_times, c_times = [], []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            run_baseline(model, prefix_ids, suffix_ids_n, device)
            nc_times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            run_cached(model, prefix_ids, suffix_ids_n, device)
            c_times.append(time.perf_counter() - t0)
        nc_m, nc_s = mean_std(nc_times)
        c_m, c_s = mean_std(c_times)
        speedup = nc_m / c_m if c_m > 0 else float("inf")
        rows.append([str(n), f"{nc_m:.3f}+/-{nc_s:.3f}", f"{c_m:.3f}+/-{c_s:.3f}", f"{speedup:.1f}x"])
        print(f"  {n:>3} requests: no_cache {nc_m:.3f}s  cached {c_m:.3f}s  ({speedup:.1f}x)")
    print()
    print_table(headers, rows)
    print("\nAt n=1 the one-time prefix cost is paid by both sides, so they "
          "should cost about the same; as n grows, no_cache re-pays the prefix "
          "every request while cached pays it once and amortizes it away.")

    print(f"\nPart 2 — headline receipt at {args.headline_requests} requests")
    n = args.headline_requests
    suffix_ids_n = all_suffix_ids[:n]
    out_dir = Path(__file__).parent
    results = []
    for label in ("no_prefix_cache", "prefix_cache"):
        trials = []
        for t in range(args.repeats):
            if args.repeats > 1:
                print(f" -- trial {t + 1}/{args.repeats} [{label}] --")
            with GreenMeter(label) as m:
                if label == "no_prefix_cache":
                    run_baseline(model, prefix_ids, suffix_ids_n, device)
                else:
                    run_cached(model, prefix_ids, suffix_ids_n, device)
            m.add(ms_per_request=round(m.result["wall_s"] / n * 1000, 3), n_requests=n)
            trials.append(m.result)
        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, out_dir / "results", is_baseline=(label == "no_prefix_cache"))

    compare(results, extra_keys=("ms_per_request", "n_requests"))
    print("Same shared system prompt, same per-request answers -- the cache just "
          "stops paying for the part of the input that never changed. Benefit "
          "scales with prefix_len / suffix_len: a long shared prompt and a short "
          "per-request suffix is exactly this recipe's best case.")


if __name__ == "__main__":
    main()
