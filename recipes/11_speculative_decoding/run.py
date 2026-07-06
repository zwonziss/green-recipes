"""
Recipe 11 — Speculative decoding: let a tiny model guess, let the big one nod.

A small "draft" model proposes several tokens greedily; the big "target"
model verifies all of them in ONE forward pass and accepts the matching
prefix, falling back to its own token the instant they disagree. With
greedy decoding on both models this is provably equivalent to plain
target-only greedy decoding -- proven here, not assumed -- while often
needing far fewer expensive target forward passes.

Uses transformers' built-in `assistant_model=` API (the correct, tested
implementation of this scheme) rather than hand-rolled cache surgery.

    python recipes/11_speculative_decoding/run.py
"""
import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.greenmeter import (  # noqa: E402
    GreenMeter, aggregate_trials, compare, print_receipt, save_result,
)
from common.data import pick_device, set_seed  # noqa: E402

RECIPE = "11_speculative_decoding"
TARGET = "HuggingFaceTB/SmolLM2-360M"
DRAFT = "HuggingFaceTB/SmolLM2-135M"
PROMPTS = [
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\n",
    "The most important idea in machine learning is",
    "KING RICHARD III:\nNow is the winter of",
]


def check_equivalent(target, draft, tok, device, new_tokens=48):
    enc = tok(PROMPTS[0], return_tensors="pt").to(device)
    out_plain = target.generate(**enc, max_new_tokens=new_tokens, do_sample=False,
                                pad_token_id=tok.eos_token_id)
    out_spec = target.generate(**enc, max_new_tokens=new_tokens, do_sample=False,
                               assistant_model=draft, pad_token_id=tok.eos_token_id)
    match = out_plain.shape == out_spec.shape and bool((out_plain == out_spec).all())
    print(f"Part 0 — numerical check: target-only greedy and speculative greedy "
          f"produce identical output over {new_tokens} tokens: {match}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-tokens", type=int, default=200)
    ap.add_argument("--draft-tokens", type=int, default=5,
                    help="candidate tokens proposed per round (num_assistant_tokens)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=3, help="trials per variant to aggregate")
    args = ap.parse_args()

    device = pick_device(args.device)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(42)
    tok = AutoTokenizer.from_pretrained(TARGET)
    tok_draft = AutoTokenizer.from_pretrained(DRAFT)
    assert tok.vocab_size == tok_draft.vocab_size, \
        "draft and target must share a tokenizer for speculative decoding"

    dtype = torch.float16 if device == "cuda" else torch.float32
    target = AutoModelForCausalLM.from_pretrained(TARGET, torch_dtype=dtype).to(device)
    draft = AutoModelForCausalLM.from_pretrained(DRAFT, torch_dtype=dtype).to(device)
    target.eval()
    draft.eval()
    t_params = sum(p.numel() for p in target.parameters()) / 1e6
    d_params = sum(p.numel() for p in draft.parameters()) / 1e6
    print(f"target: {TARGET} ({t_params:.0f}M)   draft: {DRAFT} ({d_params:.0f}M)")

    check_equivalent(target, draft, tok, device)

    out_dir = Path(__file__).parent
    plan = [("target_only", {}),
            ("speculative", dict(assistant_model=draft, num_assistant_tokens=args.draft_tokens))]

    results = []
    for label, gen_kwargs in plan:
        print(f"\n### {label} ###")
        trials = []
        for t in range(args.repeats):
            if args.repeats > 1:
                print(f" -- trial {t + 1}/{args.repeats} --")
            gen_tokens = 0
            with GreenMeter(label) as m:
                for p in PROMPTS:
                    enc = tok(p, return_tensors="pt").to(device)
                    out = target.generate(**enc, max_new_tokens=args.new_tokens,
                                          do_sample=False, pad_token_id=tok.eos_token_id,
                                          **gen_kwargs)
                    gen_tokens += out.shape[1] - enc["input_ids"].shape[1]
            m.add(tokens_per_s=round(gen_tokens / m.result["wall_s"], 1))
            trials.append(m.result)
        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, out_dir / "results", is_baseline=(label == "target_only"))
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        time.sleep(1)

    compare(results, extra_keys=("tokens_per_s",))
    print("Same target model, same greedy output, fewer expensive target-model "
          "calls -- the draft's compute is the price paid for that. Watch whether "
          "tokens_per_s clears the * bar: speedup lives or dies on how often the "
          "draft actually guesses right (see the Honesty box).")


if __name__ == "__main__":
    main()
