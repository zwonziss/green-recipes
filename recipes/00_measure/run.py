"""
Recipe 00 — How we measure everything.

Checks your environment, then burns the CPU or GPU for a few seconds with
big matmuls and prints the first "energy receipt". Every other recipe uses
exactly this instrument, so if this works, everything works. On CPU this
is real work (not a sleep placeholder), so it's also what
`tools/energy_regression.py` runs in CI to catch performance regressions
without needing a GPU runner.

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
from common.data import pick_device  # noqa: E402

RECIPE = "00_measure"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0, help="how long to burn, per trial")
    ap.add_argument("--size", type=int, default=4096, help="matmul size on GPU")
    ap.add_argument("--cpu-size", type=int, default=512,
                    help="matmul size on CPU (much slower per FLOP than a GPU)")
    ap.add_argument("--repeats", type=int, default=3, help="trials to aggregate")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--results-dir", default=None,
                    help="override the results/ dir (e.g. for CI regression checks)")
    args = ap.parse_args()

    env_report()

    import torch
    device = pick_device(args.device)
    size = args.size if device == "cuda" else args.cpu_size
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device != "cuda":
        print(f"Burning CPU matmuls at size {size} (fp32) -- no CUDA/energy backend here "
              "means only wall time + TFLOPS are guaranteed; energy shows up only if a "
              "power backend (e.g. Linux RAPL) was detected above.")

    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    for _ in range(3):                      # warmup: kernel autotune / cache warmup
        (a @ b).sum().item()

    n_flop_per_mm = 2 * size**3
    trials = []
    for t in range(args.repeats):
        iters = 0
        with GreenMeter(f"matmul_{size}") as m:
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
    results_dir = Path(args.results_dir) if args.results_dir else Path(__file__).parent / "results"
    save_result(r, RECIPE, results_dir, is_baseline=True)

    if args.repeats > 1 and r.get("gpu_energy_wh"):
        spread = r.get("gpu_energy_wh_std", 0) / max(r.get("gpu_energy_wh", 1), 1e-9) * 100
        print(f"Run-to-run spread on energy: ~{spread:.0f}% of the mean. "
              "That's the noise floor every single-run number in recipes 01-08 "
              "is competing against -- which is why they support --repeats too.")

    print("If you saw watts and Wh above, the lab is open. Go run recipe 01.")


if __name__ == "__main__":
    main()
