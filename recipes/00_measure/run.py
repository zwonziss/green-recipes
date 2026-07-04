"""
Recipe 00 — How we measure everything.

Checks your environment, then burns the GPU for a few seconds with big
matmuls and prints the first "energy receipt". Every other recipe uses
exactly this instrument, so if this works, everything works.

Runs the burn `--repeats` times so you see the instrument's own noise floor
(mean +/- std) before trusting any single-shot number in a later recipe.

    python recipes/00_measure/run.py --seconds 8 --repeats 3
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.greenmeter import (  # noqa: E402
    GreenMeter, aggregate_trials, env_report, print_receipt, save_result,
)

RECIPE = "00_measure"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0, help="how long to burn, per trial")
    ap.add_argument("--size", type=int, default=4096, help="matmul size")
    ap.add_argument("--repeats", type=int, default=3, help="trials to aggregate")
    args = ap.parse_args()

    env_report()

    import torch
    if not torch.cuda.is_available():
        print("No GPU -> demonstrating the CPU fallback (time only).")
        trials = []
        for _ in range(args.repeats):
            with GreenMeter("cpu_demo") as m:
                time.sleep(1.0)
            trials.append(m.result)
        print_receipt(aggregate_trials(trials))
        return

    dev = "cuda"
    a = torch.randn(args.size, args.size, device=dev, dtype=torch.float16)
    b = torch.randn(args.size, args.size, device=dev, dtype=torch.float16)
    for _ in range(3):                      # warmup: cuBLAS autotune, context init
        (a @ b).sum().item()

    n_flop_per_mm = 2 * args.size**3
    trials = []
    for t in range(args.repeats):
        iters = 0
        with GreenMeter(f"matmul_{args.size}") as m:
            t_end = time.perf_counter() + args.seconds
            while time.perf_counter() < t_end:
                c = a @ b
                iters += 1
            c.sum().item()                  # force sync inside the metered block
        tflops = n_flop_per_mm * iters / m.result["wall_s"] / 1e12
        m.add(iterations=iters, achieved_tflops=round(tflops, 1))
        print(f"  trial {t + 1}/{args.repeats}: {tflops:.1f} TFLOPS, "
              f"{m.result.get('gpu_energy_wh', 0):.4f} Wh")
        trials.append(m.result)

    r = aggregate_trials(trials)
    print_receipt(r)
    save_result(r, RECIPE, Path(__file__).parent / "results", is_baseline=True)

    if args.repeats > 1:
        spread = r.get("gpu_energy_wh_std", 0) / max(r.get("gpu_energy_wh", 1), 1e-9) * 100
        print(f"Run-to-run spread on energy: ~{spread:.0f}% of the mean. "
              "That's the noise floor every single-run number in recipes 01-08 "
              "is competing against -- which is why they support --repeats too.")

    print("If you saw watts and Wh above, the lab is open. Go run recipe 01.")


if __name__ == "__main__":
    main()
