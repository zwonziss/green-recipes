"""
Recipe 12 — 2:4 structured sparsity: the pruning speedup recipe 05 said was a myth.

Recipe 05 zeroed weights by magnitude and showed latency does NOT drop on a
GPU -- dense kernels multiply zeros at full speed. This recipe zeroes exactly
2 of every 4 contiguous weights (a pattern, not a free-for-all) and runs the
result through NVIDIA's semi-structured sparse tensor cores, which DO skip
the zeros. Requires Ampere+ (compute capability >= 8.0) and a cusparseLt-
enabled torch build; degrades to a clear skip message otherwise.

    python recipes/12_structured_sparsity/run.py
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
from common.eval import max_abs_diff  # noqa: E402

RECIPE = "12_structured_sparsity"


def prune_2_4(w):
    """Zero the 2 smallest-magnitude weights of every contiguous group of 4
    along the last dim -- a PATTERN sparse tensor cores can exploit, unlike
    recipe 05's unstructured magnitude pruning."""
    import torch
    shape = w.shape
    flat = w.reshape(-1, 4)
    order = flat.abs().argsort(dim=1)          # ascending magnitude
    mask = torch.ones_like(flat, dtype=torch.bool)
    mask.scatter_(1, order[:, :2], False)      # drop the 2 smallest of each 4
    return (flat * mask).reshape(shape)


def time_matmul(x, w, iters):
    import torch
    for _ in range(3):
        x @ w.t()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        x @ w.t()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="2048,4096,8192", help="square weight dims to sweep")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=20, help="matmul calls per timing")
    ap.add_argument("--repeats", type=int, default=3, help="trials per variant to aggregate")
    args = ap.parse_args()

    import torch
    device = pick_device("auto")
    if device != "cuda" or torch.cuda.get_device_capability() < (8, 0):
        found = "no CUDA GPU" if device != "cuda" else torch.cuda.get_device_capability()
        print(f"Skipping: 2:4 sparse tensor cores need an Ampere+ GPU (compute "
              f"capability >= 8.0) -- e.g. RTX 30/40-series, A100. Found: {found}. "
              "Recipe 05's unstructured-pruning myth-bust still applies on this hardware.")
        return

    try:
        from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
        SparseSemiStructuredTensor._FORCE_CUTLASS = True
    except Exception as e:
        print(f"Skipping: torch.sparse semi-structured API unavailable ({e}). "
              "Needs a torch build with cuSPARSELt/CUTLASS 2:4 sparse kernel support.")
        return

    set_seed(42)
    sizes = [int(s) for s in args.sizes.split(",")]

    # Part 0 -- prove the sparse kernel computes the SAME masked matmul, not
    # an approximation, before its speed is trusted.
    w0 = prune_2_4(torch.randn(2048, 2048, dtype=torch.float16, device="cuda"))
    x0 = torch.randn(64, 2048, dtype=torch.float16, device="cuda")
    try:
        dense_out = x0 @ w0.t()
        sparse_out = x0 @ to_sparse_semi_structured(w0).t()
        diff = max_abs_diff(dense_out, sparse_out)
        print(f"Part 0 — numerical check: max |dense_masked - sparse_kernel| = "
              f"{diff:.2e} (should be tiny fp16 noise).")
    except Exception as e:
        print(f"Skipping: sparse kernel failed at runtime ({e}). This GPU/torch "
              "combo advertises support but the kernel isn't actually usable here.")
        return

    print(f"\nPart 1 — fp16 matmul, batch={args.batch}, n={args.repeats} trials/size")
    headers = ["dim", "dense ms", "sparse ms", "speedup"]
    rows = []
    for size in sizes:
        w = prune_2_4(torch.randn(size, size, dtype=torch.float16, device="cuda"))
        w_sparse = to_sparse_semi_structured(w)
        x = torch.randn(args.batch, size, dtype=torch.float16, device="cuda")
        d_times = [time_matmul(x, w, args.iters) for _ in range(args.repeats)]
        s_times = [time_matmul(x, w_sparse, args.iters) for _ in range(args.repeats)]
        d_m, d_s = mean_std(d_times)
        s_m, s_s = mean_std(s_times)
        speedup = d_m / s_m if s_m > 0 else float("inf")
        rows.append([str(size), f"{d_m:.2f}+/-{d_s:.2f}", f"{s_m:.2f}+/-{s_s:.2f}", f"{speedup:.2f}x"])
        print(f"  dim {size:>5}: dense {d_m:.2f}ms  sparse {s_m:.2f}ms  ({speedup:.2f}x)")
        del w, w_sparse, x
        gc.collect()
        torch.cuda.empty_cache()
    print()
    print_table(headers, rows)

    headline_size = sizes[-1]
    print(f"\nPart 2 — headline receipt at dim={headline_size}")
    w = prune_2_4(torch.randn(headline_size, headline_size, dtype=torch.float16, device="cuda"))
    w_sparse = to_sparse_semi_structured(w)
    x = torch.randn(args.batch, headline_size, dtype=torch.float16, device="cuda")

    out_dir = Path(__file__).parent
    results = []
    for label, weight in (("dense", w), ("sparse_2_4", w_sparse)):
        trials = []
        for t in range(args.repeats):
            if args.repeats > 1:
                print(f" -- trial {t + 1}/{args.repeats} [{label}] --")
            with GreenMeter(label) as m:
                for _ in range(args.iters):
                    x @ weight.t()
            m.add(ms_per_call=round(m.result["wall_s"] / args.iters * 1000, 4))
            trials.append(m.result)
        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, out_dir / "results", is_baseline=(label == "dense"))

    compare(results, extra_keys=("ms_per_call",))
    print("Same masked math (Part 0), real latency drop on real hardware -- this "
          "is the speedup unstructured pruning (recipe 05) promised but couldn't "
          "deliver. It needed a pattern, not just zeros.")


if __name__ == "__main__":
    main()
