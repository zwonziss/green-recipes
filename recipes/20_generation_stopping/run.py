"""
Recipe 20 — Stop generating when you're done.

A common anti-pattern: `max_new_tokens=256` with nothing telling the model
it's allowed to stop sooner, so every response pays for 256 tokens whether
the real answer took 8 or 200. This recipe forces full-length generation
(`min_new_tokens == max_new_tokens`, the "just in case" setting that
silently disables early stopping) against generation that's allowed to stop
the moment it emits a natural boundary (a newline), via transformers'
built-in `stop_strings=`.

Part 0 proves this isn't a different (cheaper, worse) model: under greedy
decoding, the stopped output is an exact token-for-token PREFIX of what the
forced-length run produces for the same prompt -- stopping early changes
nothing about the tokens already generated, it just declines to generate more.

    python recipes/20_generation_stopping/run.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.greenmeter import (  # noqa: E402
    GreenMeter, aggregate_trials, compare, print_receipt,
    print_table, save_result,
)
from common.data import pick_device, set_seed  # noqa: E402

RECIPE = "20_generation_stopping"
MODEL = "HuggingFaceTB/SmolLM2-135M"
STOP_STR = "\n"
PROMPTS = [
    "Q: What is the capital of France?\nA:",
    "Q: What color is the sky on a clear day?\nA:",
    "Q: How many days are in a week?\nA:",
    "Q: What is 2 plus 2?\nA:",
    "Q: Name a common house pet.\nA:",
    "Q: What do bees make?\nA:",
    "Q: What is the opposite of hot?\nA:",
    "Q: What season comes after winter?\nA:",
]


def generate_forced(model, tok, prompt, max_new_tokens, device):
    enc = tok(prompt, return_tensors="pt").to(device)
    return model.generate(**enc, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
                          do_sample=False, pad_token_id=tok.eos_token_id)


def generate_stopped(model, tok, prompt, max_new_tokens, device):
    enc = tok(prompt, return_tensors="pt").to(device)
    return model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                          stop_strings=STOP_STR, tokenizer=tok, pad_token_id=tok.eos_token_id)


def check_prefix(model, tok, device, max_new_tokens):
    """Greedy decoding is deterministic -- stopping early must not change any
    token that WAS generated, only decline to generate more."""
    prompt = PROMPTS[0]
    forced = generate_forced(model, tok, prompt, max_new_tokens, device)
    stopped = generate_stopped(model, tok, prompt, max_new_tokens, device)
    prompt_len = tok(prompt, return_tensors="pt").input_ids.shape[1]
    forced_new = forced[:, prompt_len:]
    stopped_new = stopped[:, prompt_len:]
    n = stopped_new.shape[1]
    is_prefix = bool((forced_new[:, :n] == stopped_new).all())
    print(f"Part 0 — numerical check: stopped output ({n} tokens) is an exact "
          f"prefix of the forced {max_new_tokens}-token output: {is_prefix}.")


def count_new_tokens(out, prompt_len):
    return out.shape[1] - prompt_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=200,
                    help="the 'just in case' fixed length forced generation pays for every time")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=3, help="trials per variant to aggregate")
    args = ap.parse_args()

    device = pick_device(args.device)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
    model.eval()

    check_prefix(model, tok, device, args.max_new_tokens)

    print(f"\nPart 1 — new tokens generated per prompt, forced vs stop-on-newline "
          f"(max_new_tokens={args.max_new_tokens})")
    headers = ["prompt #", "forced (tokens)", "stopped (tokens)", "saved"]
    rows = []
    for i, p in enumerate(PROMPTS):
        prompt_len = tok(p, return_tensors="pt").input_ids.shape[1]
        forced = generate_forced(model, tok, p, args.max_new_tokens, device)
        stopped = generate_stopped(model, tok, p, args.max_new_tokens, device)
        f_n = count_new_tokens(forced, prompt_len)
        s_n = count_new_tokens(stopped, prompt_len)
        rows.append([str(i), str(f_n), str(s_n), f"{(1 - s_n / f_n) * 100:.0f}%"])
    print()
    print_table(headers, rows)

    print(f"\nPart 2 — headline receipt over {len(PROMPTS)} prompts")
    out_dir = Path(__file__).parent
    results = []
    for label, gen_fn in (("forced_full_length", generate_forced), ("stop_on_newline", generate_stopped)):
        trials = []
        for t in range(args.repeats):
            if args.repeats > 1:
                print(f" -- trial {t + 1}/{args.repeats} [{label}] --")
            total_new = 0
            with GreenMeter(label) as m:
                for p in PROMPTS:
                    prompt_len = tok(p, return_tensors="pt").input_ids.shape[1]
                    out = gen_fn(model, tok, p, args.max_new_tokens, device)
                    total_new += count_new_tokens(out, prompt_len)
            m.add(avg_new_tokens=round(total_new / len(PROMPTS), 1), n_prompts=len(PROMPTS))
            trials.append(m.result)
        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, out_dir / "results", is_baseline=(label == "forced_full_length"))

    compare(results, extra_keys=("avg_new_tokens", "n_prompts"))
    print("Same model, same prompts, same greedy math (Part 0) -- the only "
          "difference is whether generation is allowed to notice it's done. "
          "Forcing a fixed length doesn't buy anything except a bigger energy bill.")


if __name__ == "__main__":
    main()
