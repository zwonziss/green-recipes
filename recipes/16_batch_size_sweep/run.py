"""
Recipe 16 — The batch-size sweep: biggest isn't always most efficient.

Sweeps inference batch size on a fixed-length forward pass until the GPU
OOMs (recorded as data, not a crash), tracking both tokens/sec AND Wh per
1,000 tokens at every point. The batch size that maximizes throughput and
the batch size that minimizes energy per token are not always the same
one -- this recipe finds both instead of assuming "bigger batch = better".

    python recipes/16_batch_size_sweep/run.py
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

RECIPE = "16_batch_size_sweep"
MODEL = "HuggingFaceTB/SmolLM2-360M"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--iters", type=int, default=10, help="forward passes per trial")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=3, help="trials per batch size to aggregate")
    args = ap.parse_args()

    device = pick_device(args.device)
    import torch
    from transformers import AutoModelForCausalLM

    set_seed(42)
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
    model.eval()
    vocab = model.config.vocab_size
    out_dir = Path(__file__).parent
    sizes = [int(s) for s in args.sizes.split(",")]

    results = []
    for bs in sizes:
        label = f"batch_{bs}"
        print(f"\n### {label} ###")
        x = torch.randint(0, vocab, (bs, args.seq_len), device=device)
        try:
            with torch.no_grad():
                for _ in range(2):                     # warmup, outside the meter
                    model(input_ids=x)
        except RuntimeError as e:
            print(f"  {label}: OOM on warmup ({e}) -- stopping the sweep here.")
            results.append({"label": label, "wall_s": None, "extra": {"batch_size": bs, "oom": True}})
            break

        trials, oom = [], False
        for t in range(args.repeats):
            if args.repeats > 1:
                print(f" -- trial {t + 1}/{args.repeats} --")
            try:
                with GreenMeter(label) as m, torch.no_grad():
                    for _ in range(args.iters):
                        model(input_ids=x)
            except RuntimeError as e:
                print(f"  {label}: OOM mid-run ({e}) -- stopping the sweep here.")
                oom = True
                break
            tokens = args.iters * bs * args.seq_len
            m.add(tokens_per_s=round(tokens / m.result["wall_s"], 1), batch_size=bs)
            if m.result.get("gpu_energy_wh"):
                m.add(wh_per_1k_tok=round(m.result["gpu_energy_wh"] / tokens * 1000, 5))
            trials.append(m.result)

        del x
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        if oom:
            results.append({"label": label, "wall_s": None, "extra": {"batch_size": bs, "oom": True}})
            break
        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, out_dir / "results", is_baseline=(bs == sizes[0]))
        time.sleep(1)

    ok = [r for r in results if r.get("wall_s") is not None]
    compare(ok, extra_keys=("tokens_per_s", "wh_per_1k_tok", "batch_size"))

    if ok:
        max_tp = max(ok, key=lambda r: r["extra"]["tokens_per_s"])
        with_energy = [r for r in ok if "wh_per_1k_tok" in r.get("extra", {})]
        if with_energy:
            best_eff = min(with_energy, key=lambda r: r["extra"]["wh_per_1k_tok"])
            same = max_tp["extra"]["batch_size"] == best_eff["extra"]["batch_size"]
            print(f"\nMax throughput: batch={max_tp['extra']['batch_size']} "
                  f"({max_tp['extra']['tokens_per_s']} tok/s). Best energy efficiency: "
                  f"batch={best_eff['extra']['batch_size']} "
                  f"({best_eff['extra']['wh_per_1k_tok']} Wh/1k tok). " +
                  ("Same batch size here -- push to the memory ceiling." if same else
                   "DIFFERENT batch sizes -- the most efficient point isn't the "
                   "biggest batch that happens to fit."))
        else:
            print(f"\nMax throughput: batch={max_tp['extra']['batch_size']} "
                  f"({max_tp['extra']['tokens_per_s']} tok/s). No power backend detected "
                  "here, so the Wh/1k-token efficiency comparison isn't available "
                  "-- see recipe 00's environment check.")


if __name__ == "__main__":
    main()
