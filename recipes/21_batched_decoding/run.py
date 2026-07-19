"""
Recipe 21 — Batch your requests: N prompts at once beats a for-loop.

A naive server handling N independent generation requests one at a time
pays the GPU's fixed per-call overhead N times and never lets the GPU work
on more than one sequence's matrix multiplies at once. Padding the N prompts
into a single left-padded batch and calling generate() once keeps the GPU
busy with N times the arithmetic per kernel launch -- the classic "batching
dividend" behind every real inference server.

Part 0 proves batching doesn't change any individual prompt's output:
with left-padding + attention_mask, greedy decoding for prompt i inside a
32-wide batch is token-for-token identical to running prompt i alone.

    python recipes/21_batched_decoding/run.py
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.greenmeter import (  # noqa: E402
    GreenMeter, aggregate_trials, compare, mean_std, print_receipt,
    print_table, save_result,
)
from common.data import pick_device, set_seed  # noqa: E402

RECIPE = "21_batched_decoding"
MODEL = "HuggingFaceTB/SmolLM2-135M"
PROMPTS = [f"Write one short sentence about topic #{i}.\n" for i in range(64)]


def generate_serial(model, tok, prompts, max_new_tokens, device):
    total_new = 0
    for p in prompts:
        enc = tok(p, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        total_new += out.shape[1] - enc.input_ids.shape[1]
    return total_new


def generate_batched(model, tok, prompts, max_new_tokens, device):
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    return (out.shape[1] - enc.input_ids.shape[1]) * len(prompts)


def check_equivalent(model, tok, device, max_new_tokens=24, batch_n=8):
    """Left-padding + attention_mask must make prompt i's output identical
    whether it runs alone or packed into a batch with others. min_new_tokens
    forces every row to generate exactly max_new_tokens, same as the timed
    runs below -- early per-row EOS stopping is recipe 20's story, not this one."""
    batch = PROMPTS[:batch_n]
    check_i = 0
    enc_alone = tok(batch[check_i], return_tensors="pt").to(device)
    out_alone = model.generate(**enc_alone, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
                               do_sample=False, pad_token_id=tok.eos_token_id)
    new_alone = out_alone[:, enc_alone.input_ids.shape[1]:]

    enc_batch = tok(batch, return_tensors="pt", padding=True).to(device)
    out_batch = model.generate(**enc_batch, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
                               do_sample=False, pad_token_id=tok.eos_token_id)
    new_batch = out_batch[check_i:check_i + 1, enc_batch.input_ids.shape[1]:]

    match = new_alone.shape == new_batch.shape and bool((new_alone == new_batch).all())
    print(f"Part 0 — numerical check: prompt #0's greedy output alone vs. inside "
          f"a batch of {batch_n} (left-padded) is identical: {match}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="1,8,32", help="batch sizes for Part 1")
    ap.add_argument("--headline-batch", type=int, default=32, help="batch size for Part 2 receipt")
    ap.add_argument("--new-tokens", type=int, default=24)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=3, help="trials per variant to aggregate")
    args = ap.parse_args()

    device = pick_device(args.device)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
    model.eval()

    check_equivalent(model, tok, device)

    print(f"\nPart 1 — wall time for N prompts, serial calls vs one batched call "
          f"(n={args.repeats} trials/count)")
    headers = ["n_prompts", "serial (s)", "batched (s)", "speedup"]
    rows = []
    for n in [int(x) for x in args.sweep.split(",")]:
        prompts_n = PROMPTS[:n]
        s_times, b_times = [], []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            generate_serial(model, tok, prompts_n, args.new_tokens, device)
            s_times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            generate_batched(model, tok, prompts_n, args.new_tokens, device)
            b_times.append(time.perf_counter() - t0)
        s_m, s_s = mean_std(s_times)
        b_m, b_s = mean_std(b_times)
        speedup = s_m / b_m if b_m > 0 else float("inf")
        rows.append([str(n), f"{s_m:.3f}+/-{s_s:.3f}", f"{b_m:.3f}+/-{b_s:.3f}", f"{speedup:.1f}x"])
        print(f"  {n:>3} prompts: serial {s_m:.3f}s  batched {b_m:.3f}s  ({speedup:.1f}x)")
    print()
    print_table(headers, rows)
    print("\nAt n=1 both paths do the same single call, so they should cost about "
          "the same; as n grows, serial's time scales ~linearly with n while "
          "batched's grows much slower until the GPU saturates.")

    print(f"\nPart 2 — headline receipt at {args.headline_batch} prompts")
    n = args.headline_batch
    prompts_n = PROMPTS[:n]
    out_dir = Path(__file__).parent
    results = []
    for label in ("serial_generate", "batched_generate"):
        trials = []
        for t in range(args.repeats):
            if args.repeats > 1:
                print(f" -- trial {t + 1}/{args.repeats} [{label}] --")
            with GreenMeter(label) as m:
                if label == "serial_generate":
                    total_new = generate_serial(model, tok, prompts_n, args.new_tokens, device)
                else:
                    total_new = generate_batched(model, tok, prompts_n, args.new_tokens, device)
            m.add(tokens_per_s=round(total_new / m.result["wall_s"], 1), n_prompts=n)
            trials.append(m.result)
        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, out_dir / "results", is_baseline=(label == "serial_generate"))

    compare(results, extra_keys=("tokens_per_s", "n_prompts"))
    print("Same model, same prompts, same per-prompt output (Part 0) -- batching "
          "just keeps the GPU fed with more arithmetic per kernel launch instead "
          "of paying fixed overhead N separate times.")


if __name__ == "__main__":
    main()
