"""
Recipe 25 — Exact-match response caching: stop re-answering the same question.

Real traffic repeats. A handful of canonical questions (FAQs, common
requests, retries) usually account for a disproportionate share of total
volume -- a classic long-tail/Zipf-ish distribution. A plain dict keyed by
the exact request string turns every repeat into a lookup instead of a full
generation. This is a different kind of reuse than recipe 10 (within one
generation) or recipe 18 (a shared prefix across otherwise-different
requests): here the ENTIRE request, and therefore the entire response, is
identical to one already served.

Part 0 checks the invariant this relies on: under greedy decoding, two
independent fresh calls for the same prompt produce the identical output --
exactly what a cache hit hands back instead of recomputing.

    python recipes/25_response_caching/run.py
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.greenmeter import (  # noqa: E402
    GreenMeter, aggregate_trials, compare, print_receipt, save_result,
)
from common.data import pick_device, set_seed  # noqa: E402

RECIPE = "25_response_caching"
MODEL = "HuggingFaceTB/SmolLM2-135M"
CANONICAL_PROMPTS = [f"Frequently asked question #{i}: please respond.\n" for i in range(20)]


def build_traffic(n_requests, seed=42):
    """A Zipf-ish skew: question #0 is asked far more often than question #19 --
    the same long-tail shape real request logs tend to show."""
    weights = [1.0 / (i + 1) for i in range(len(CANONICAL_PROMPTS))]
    rng = random.Random(seed)
    return rng.choices(CANONICAL_PROMPTS, weights=weights, k=n_requests)


def check_determinism(model, tok, device, max_new_tokens):
    enc = tok(CANONICAL_PROMPTS[0], return_tensors="pt").to(device)
    out_a = model.generate(**enc, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
                           do_sample=False, pad_token_id=tok.eos_token_id)
    out_b = model.generate(**enc, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
                           do_sample=False, pad_token_id=tok.eos_token_id)
    match = bool((out_a == out_b).all())
    print(f"Part 0 — numerical check: two independent fresh calls for the same "
          f"prompt (greedy decoding) produce identical output: {match} -- exactly "
          f"what a cache hit hands back instead of recomputing.")


def run_uncached(model, tok, traffic, max_new_tokens, device):
    for p in traffic:
        enc = tok(p, return_tensors="pt").to(device)
        model.generate(**enc, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
                       do_sample=False, pad_token_id=tok.eos_token_id)
    return len(traffic)


def run_cached(model, tok, traffic, max_new_tokens, device):
    cache = {}
    generate_calls = 0
    for p in traffic:
        if p in cache:
            continue  # cache hit -- skip generate() entirely
        enc = tok(p, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        cache[p] = out
        generate_calls += 1
    return generate_calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-requests", type=int, default=150)
    ap.add_argument("--new-tokens", type=int, default=24)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=3, help="trials per variant to aggregate")
    args = ap.parse_args()

    device = pick_device(args.device)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
    model.eval()

    check_determinism(model, tok, device, args.new_tokens)

    traffic = build_traffic(args.n_requests)
    unique = len(set(traffic))
    print(f"\nSimulated traffic: {len(traffic)} requests, {unique} distinct questions "
          f"({len(traffic) - unique} repeats, {(len(traffic) - unique) / len(traffic) * 100:.0f}% "
          f"of volume).")

    out_dir = Path(__file__).parent
    results = []
    for label in ("no_response_cache", "response_cache"):
        trials = []
        for t in range(args.repeats):
            if args.repeats > 1:
                print(f" -- trial {t + 1}/{args.repeats} [{label}] --")
            with GreenMeter(label) as m:
                if label == "no_response_cache":
                    calls = run_uncached(model, tok, traffic, args.new_tokens, device)
                else:
                    calls = run_cached(model, tok, traffic, args.new_tokens, device)
            hit_rate = 1 - calls / len(traffic)
            m.add(generate_calls=calls, hit_rate=round(hit_rate, 4), n_requests=len(traffic))
            trials.append(m.result)
        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, out_dir / "results", is_baseline=(label == "no_response_cache"))

    compare(results, extra_keys=("generate_calls", "hit_rate", "n_requests"))
    print("no_response_cache pays for a full generation on every request, repeats "
          "included. response_cache pays once per DISTINCT question and returns "
          "the stored answer on every repeat -- the win is entirely a function "
          "of how much your traffic actually repeats (see the Honesty box).")


if __name__ == "__main__":
    main()
