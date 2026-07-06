"""
Recipe 14 — Energy-aware hyperparameter search: stop training losers.

A naive grid search trains every candidate config to completion before
picking the best. An ASHA-lite successive-halving search checkpoints all
configs after a short "rung" of steps, keeps only the top half by held-out
perplexity, and only lets survivors keep training -- spending less total
energy to find the same (or a near-identical) best config.

    python recipes/14_energy_aware_hpo/run.py
"""
import argparse
import gc
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.greenmeter import (  # noqa: E402
    GCO2_PER_KWH, GreenMeter, aggregate_trials, compare, print_receipt, save_result,
)
from common.data import get_tinyshakespeare, lm_batches, pick_device, set_seed  # noqa: E402
from common.eval import lm_perplexity  # noqa: E402

RECIPE = "14_energy_aware_hpo"
MODEL = "HuggingFaceTB/SmolLM2-135M"


def train_steps(model, opt, batches, device):
    for x in batches:
        x = x.to(device)
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def to_result(label, energy_wh, wall_s, best_ppl, best_lr, n_units):
    return {"label": label, "wall_s": round(wall_s, 2),
           "gpu_energy_wh": round(energy_wh, 4),
           "co2_g": round(energy_wh / 1000 * GCO2_PER_KWH, 3),
           "extra": {"best_val_ppl": round(best_ppl, 3), "best_lr": f"{best_lr:g}",
                    "rung_units_trained": n_units}}


def run_naive(configs, batches, eval_batches, device):
    import torch
    from transformers import AutoModelForCausalLM
    energy = wall = 0.0
    best_ppl, best_lr = math.inf, None
    for lr in configs:
        set_seed(42)
        model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        with GreenMeter(f"naive_lr{lr:g}") as m:
            train_steps(model, opt, batches, device)
        ppl = lm_perplexity(model, eval_batches, device)
        energy += m.result.get("gpu_energy_wh", 0.0)
        wall += m.result["wall_s"]
        print(f"  naive lr={lr:g}: val_ppl={ppl:.2f} (trained all {len(batches)} steps)")
        if ppl < best_ppl:
            best_ppl, best_lr = ppl, lr
        del model, opt
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        time.sleep(1)
    return to_result("naive_full", energy, wall, best_ppl, best_lr, len(configs))


def run_asha(configs, rung_batches, eval_batches, device, n_rungs, keep_frac):
    import torch
    from transformers import AutoModelForCausalLM
    energy = wall = 0.0
    survivors = list(configs)
    states = {lr: None for lr in configs}
    ppl_by_lr = {}
    n_units = 0
    for rung in range(n_rungs):
        for lr in survivors:
            set_seed(42)
            model = AutoModelForCausalLM.from_pretrained(MODEL)
            if states[lr] is not None:
                model.load_state_dict(states[lr])
            model.to(device)
            model.train()
            # NOTE: fresh optimizer each rung (see Honesty box) -- real ASHA
            # implementations checkpoint optimizer state too.
            opt = torch.optim.AdamW(model.parameters(), lr=lr)
            with GreenMeter(f"asha_r{rung}_lr{lr:g}") as m:
                train_steps(model, opt, rung_batches[rung], device)
            ppl = lm_perplexity(model, eval_batches, device)
            ppl_by_lr[lr] = ppl
            energy += m.result.get("gpu_energy_wh", 0.0)
            wall += m.result["wall_s"]
            n_units += 1
            print(f"  ASHA rung {rung} lr={lr:g}: val_ppl={ppl:.2f}")
            states[lr] = {k: v.cpu() for k, v in model.state_dict().items()}
            del model, opt
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            time.sleep(1)
        if rung < n_rungs - 1:
            keep_n = max(1, math.ceil(len(survivors) * keep_frac))
            survivors = sorted(survivors, key=lambda lr: ppl_by_lr[lr])[:keep_n]
            print(f"  ASHA: pruned to top {keep_n} by val_ppl -> {survivors}")
    best_lr = min(ppl_by_lr, key=ppl_by_lr.get)
    return to_result("asha_lite", energy, wall, ppl_by_lr[best_lr], best_lr, n_units)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="1e-3,5e-4,2e-4,1e-4,5e-5,1e-5",
                    help="candidate learning rates")
    ap.add_argument("--steps-per-rung", type=int, default=6)
    ap.add_argument("--n-rungs", type=int, default=3)
    ap.add_argument("--keep-frac", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=3, help="full-search repeats to aggregate")
    ap.add_argument("--eval-batches", type=int, default=3, help="held-out batches for val ppl")
    args = ap.parse_args()

    device = pick_device(args.device)
    from transformers import AutoTokenizer
    configs = [float(x) for x in args.configs.split(",")]
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = get_tinyshakespeare()
    total_steps = args.n_rungs * args.steps_per_rung
    batches = lm_batches(text, tok, args.seq_len, args.batch_size, total_steps)
    rung_batches = [batches[r * args.steps_per_rung:(r + 1) * args.steps_per_rung]
                    for r in range(args.n_rungs)]
    eval_batches = lm_batches(text, tok, args.seq_len, batch_size=4,
                              n_batches=args.eval_batches, seed=999)
    out_dir = Path(__file__).parent

    naive_trials, asha_trials = [], []
    for t in range(args.repeats):
        if args.repeats > 1:
            print(f"\n=== repeat {t + 1}/{args.repeats} ===")
        print("\n--- naive: every config trained to completion ---")
        naive_trials.append(run_naive(configs, batches, eval_batches, device))
        print("\n--- ASHA-lite: prune the bottom half every rung ---")
        asha_trials.append(run_asha(configs, rung_batches, eval_batches, device,
                                    args.n_rungs, args.keep_frac))

    results = []
    for label, trials in (("naive_full", naive_trials), ("asha_lite", asha_trials)):
        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, out_dir / "results", is_baseline=(label == "naive_full"))

    compare(results, extra_keys=("best_val_ppl", "best_lr", "rung_units_trained"))
    e_naive, e_asha = results[0].get("gpu_energy_wh"), results[1].get("gpu_energy_wh")
    if e_naive:
        print(f"\nASHA-lite found a best (or near-best) config using "
              f"~{(1 - e_asha / e_naive) * 100:.0f}% less total training energy "
              "than training every candidate to completion -- by killing the "
              "losers early instead of proving they're losers to the bitter end.")


if __name__ == "__main__":
    main()
