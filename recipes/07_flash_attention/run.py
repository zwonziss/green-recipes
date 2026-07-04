"""
Recipe 07 — Flash attention: the O(n^2) memory wall, and the kernel that broke it.

Part 0: a single deterministic forward checks naive attention and SDPA
actually agree numerically, before either's speed is trusted.
Part 1: microbenchmark, fwd+bwd, across sequence lengths, `--repeats` trials
per shape, until the naive version hits the memory wall (OOM recorded as data).
Part 2: the same idea on a real model -- HF `attn_implementation` "eager" vs
"sdpa", repeated, with a held-out perplexity check so "faster" isn't quietly "worse".

    python recipes/07_flash_attention/run.py
"""
import argparse
import gc
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.greenmeter import (  # noqa: E402
    GreenMeter, aggregate_trials, compare, mean_std, print_receipt,
    print_table, save_result,
)
from common.data import get_tinyshakespeare, lm_batches, pick_device, set_seed  # noqa: E402
from common.eval import lm_perplexity, max_abs_diff  # noqa: E402

RECIPE = "07_flash_attention"
MODEL = "HuggingFaceTB/SmolLM2-135M"


def naive_attention(q, k, v):
    """The textbook version: materializes the full (T x T) score matrix."""
    import torch
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    t = q.size(-2)
    mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=q.device), 1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def sdpa_attention(q, k, v):
    import torch.nn.functional as F
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def check_equivalent(seq_len=512, batch=2, heads=4, dim=64):
    """Proves naive and sdpa compute the same math before comparing their speed."""
    import torch
    set_seed(0)
    shape = (batch, heads, seq_len, dim)
    q = torch.randn(*shape, device="cuda", dtype=torch.float32)
    k = torch.randn(*shape, device="cuda", dtype=torch.float32)
    v = torch.randn(*shape, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        out_naive = naive_attention(q, k, v)
        out_sdpa = sdpa_attention(q, k, v)
    diff = max_abs_diff(out_naive, out_sdpa)
    print(f"Part 0 — numerical check (fp32, seq={seq_len}): "
          f"max |naive - sdpa| = {diff:.2e} (should be tiny float noise).")


def bench(fn, seq_len, batch=8, heads=8, dim=64, iters=8):
    """One trial: (mean ms/iter, peak MB) for fwd+bwd, or (None, None) on OOM."""
    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        shape = (batch, heads, seq_len, dim)
        q = torch.randn(*shape, device="cuda", dtype=torch.float16, requires_grad=True)
        k = torch.randn(*shape, device="cuda", dtype=torch.float16, requires_grad=True)
        v = torch.randn(*shape, device="cuda", dtype=torch.float16, requires_grad=True)
        for _ in range(3):                                   # warmup
            fn(q, k, v).sum().backward()
            q.grad = k.grad = v.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(q, k, v).sum().backward()
            q.grad = k.grad = v.grad = None
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1000
        mb = torch.cuda.max_memory_allocated() / 2**20
        return ms, mb
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return None, None


def bench_repeated(fn, seq_len, repeats, **kw):
    """Repeat bench() and aggregate; bails on the first OOM (retrying an
    out-of-memory shape produces no new information)."""
    ms_vals, mb_vals = [], []
    for _ in range(repeats):
        ms, mb = bench(fn, seq_len, **kw)
        if ms is None:
            return None, None, None, None
        ms_vals.append(ms)
        mb_vals.append(mb)
    ms_m, ms_s = mean_std(ms_vals)
    mb_m, mb_s = mean_std(mb_vals)
    return ms_m, ms_s, mb_m, mb_s


def _cell(mean, std, width, fmt="{:.1f}"):
    if mean is None:
        return "OOM".rjust(width)
    s = fmt.format(mean) + (f"+/-{fmt.format(std)}" if std else "")
    return s.rjust(width)


def real_model_part(device, steps, warmup, repeats, eval_batches):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    batches = lm_batches(get_tinyshakespeare(), tok, seq_len=1024,
                         batch_size=4, n_batches=steps + warmup)
    held_out = lm_batches(get_tinyshakespeare(), tok, seq_len=1024,
                          batch_size=4, n_batches=eval_batches, seed=999)
    results = []
    for impl in ("eager", "sdpa"):
        print(f"\n### real model, attn_implementation='{impl}' ###")
        trials = []
        for t in range(repeats):
            if repeats > 1:
                print(f" -- trial {t + 1}/{repeats} --")
            set_seed(42)  # same seed every trial, see module docstring
            model = AutoModelForCausalLM.from_pretrained(
                MODEL, attn_implementation=impl).to(device)
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
            scaler = torch.amp.GradScaler(device)
            autocast = lambda: torch.autocast(device_type=device, dtype=torch.float16)

            def step(x):
                x = x.to(device)
                with autocast():
                    loss = model(input_ids=x, labels=x).loss
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                return loss.item()

            for x in batches[:warmup]:
                step(x)
            with GreenMeter(f"model_{impl}") as m:
                losses = [step(x) for x in batches[warmup:]]
            ppl = lm_perplexity(model, held_out, device, ctx=autocast())
            tokens = sum(b.numel() for b in batches[warmup:])
            m.add(final_loss=sum(losses[-5:]) / 5, eval_ppl=round(ppl, 3),
                  tokens_per_s=int(tokens / m.result["wall_s"]))
            trials.append(m.result)
            del model, opt
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, Path(__file__).parent / "results",
                    is_baseline=(impl == "eager"))
    compare(results, extra_keys=("final_loss", "eval_ppl", "tokens_per_s"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="512,1024,2048,4096,8192")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=3, help="trials per shape/variant to aggregate")
    ap.add_argument("--eval-batches", type=int, default=4, help="held-out batches for Part 2 perplexity")
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()

    import torch
    device = pick_device("auto")
    if device != "cuda":
        raise SystemExit("This recipe needs a CUDA GPU.")
    cap = torch.cuda.get_device_capability()
    backend = "FlashAttention kernels" if cap >= (8, 0) else \
              "memory-efficient kernels (no flash on pre-Ampere, still a big win)"
    print(f"SDPA will dispatch to: {backend}\n")

    check_equivalent()

    print(f"\nPart 1 — fwd+bwd, batch=8, heads=8, head_dim=64, fp16, n={args.repeats} trials/shape")
    headers = ["seq_len", "naive ms", "naive MB", "sdpa ms", "sdpa MB"]
    rows = []
    for s in [int(x) for x in args.seqs.split(",")]:
        n_ms, n_ms_s, n_mb, n_mb_s = bench_repeated(naive_attention, s, args.repeats)
        f_ms, f_ms_s, f_mb, f_mb_s = bench_repeated(sdpa_attention, s, args.repeats)
        rows.append([str(s), _cell(n_ms, n_ms_s, 8), _cell(n_mb, n_mb_s, 9, "{:.0f}"),
                    _cell(f_ms, f_ms_s, 8), _cell(f_mb, f_mb_s, 9, "{:.0f}")])
        print(f"  seq {s:>5}: naive {rows[-1][1].strip()} ms / {rows[-1][2].strip()} MB   "
              f"sdpa {rows[-1][3].strip()} ms / {rows[-1][4].strip()} MB")
    print()
    print_table(headers, rows)
    print("\nnaive memory grows with T^2 (it stores the full attention matrix); "
          "sdpa computes the same math in tiles and never materializes it.")

    if not args.skip_model:
        print("\nPart 2 — the same effect on a real model (T=1024):")
        real_model_part(device, args.steps, args.warmup, args.repeats, args.eval_batches)


if __name__ == "__main__":
    main()
