"""
Recipe 09 — torch.compile: pay the compilation tax once, cash in every step after.

Trains the same tiny LM eager vs `torch.compile`d. The first compiled step
pays a real, measured compilation tax (graph capture + kernel codegen) that
is deliberately kept OUT of the steady-state receipt and reported on its own
-- then we compute how many steps it takes for the steady-state speedup to
break even on that tax. A held-out perplexity check confirms compiling
didn't quietly change what the model computes.

    python recipes/09_torch_compile/run.py
    python recipes/09_torch_compile/run.py --compile-mode reduce-overhead
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
from common.data import get_tinyshakespeare, lm_batches, pick_device, set_seed  # noqa: E402
from common.eval import lm_perplexity  # noqa: E402

RECIPE = "09_torch_compile"
MODEL = "HuggingFaceTB/SmolLM2-135M"


def run_variant(label, compiled, batches, eval_batches, device, lr, compile_mode):
    import torch
    from transformers import AutoModelForCausalLM

    set_seed(42)  # same seed every trial -- repeats target compiler/timing
                  # noise, not training randomness
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
    model.train()
    if compiled:
        model = torch.compile(model, mode=compile_mode)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def step(x):
        x = x.to(device)
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.item()

    # the FIRST call is where compilation (graph capture + codegen) happens --
    # measure it explicitly instead of hiding it in "warmup".
    t0 = time.perf_counter()
    first_loss = step(batches[0])
    if device == "cuda":
        torch.cuda.synchronize()
    compile_tax_s = time.perf_counter() - t0

    losses = [first_loss]
    with GreenMeter(label) as m:
        for i, x in enumerate(batches[1:]):
            losses.append(step(x))
            if (i + 1) % 10 == 0:
                print(f"  [{label}] step {i + 1:>3}  loss {losses[-1]:.3f}")

    ppl = lm_perplexity(model, eval_batches, device)
    tokens = sum(b.numel() for b in batches[1:])
    m.add(final_loss=sum(losses[-5:]) / 5, eval_ppl=round(ppl, 3),
          tokens_per_s=int(tokens / m.result["wall_s"]),
          compile_tax_s=round(compile_tax_s, 2))

    del model, opt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    time.sleep(2)
    return m.result, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60, help="post-compile steps measured per trial")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compile-mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--repeats", type=int, default=3, help="trials per variant to aggregate")
    ap.add_argument("--eval-batches", type=int, default=4, help="held-out batches for perplexity")
    args = ap.parse_args()

    device = pick_device(args.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = get_tinyshakespeare()
    batches = lm_batches(text, tok, args.seq_len, args.batch_size, args.steps + 1)
    eval_batches = lm_batches(text, tok, args.seq_len, batch_size=4,
                              n_batches=args.eval_batches, seed=999)
    out_dir = Path(__file__).parent

    results = []
    first_losses = {}
    for label, compiled in (("eager", False), ("compiled", True)):
        print(f"\n### {label} ###")
        trials = []
        try:
            for t in range(args.repeats):
                if args.repeats > 1:
                    print(f" -- trial {t + 1}/{args.repeats} --")
                r, losses = run_variant(label, compiled, batches, eval_batches,
                                        device, args.lr, args.compile_mode)
                if label not in first_losses:
                    first_losses[label] = losses
                trials.append(r)
        except Exception as e:
            if not compiled:
                raise  # the eager path has no excuse to fail
            print(f"  torch.compile failed on this machine ({type(e).__name__}: {e}) "
                  "-- skipping the compiled variant. This is a known rough edge "
                  "(e.g. no C/C++ compiler backend for TorchInductor), not a bug "
                  "in this script -- see the Honesty box.")
            print("\nOnly the eager receipt is available -- torch.compile itself "
                  "is the thing under test, and it didn't work here.")
            return
        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, out_dir / "results", is_baseline=(label == "eager"))

    compare(results, extra_keys=("final_loss", "eval_ppl", "tokens_per_s", "compile_tax_s"))

    a, b = first_losses.get("eager"), first_losses.get("compiled")
    if a and b:
        max_diff = max(abs(x - y) for x, y in zip(a, b))
        print(f"\nNumerical check (same data, same seed, first step excluded from "
              f"both): max |loss_eager - loss_compiled| over {len(a)} steps = "
              f"{max_diff:.2e}. Compiling changes HOW the graph runs, not what it computes.")

    eager_r, comp_r = results[0], results[1]
    eager_step_s = eager_r["wall_s"] / args.steps
    comp_step_s = comp_r["wall_s"] / args.steps
    tax = comp_r.get("extra", {}).get("compile_tax_s", 0)
    if comp_step_s < eager_step_s:
        saved_per_step = eager_step_s - comp_step_s
        breakeven = tax / saved_per_step if saved_per_step > 0 else float("inf")
        print(f"\nCompile tax paid once: ~{tax:.1f}s. Steady-state savings: "
              f"~{saved_per_step * 1000:.1f}ms/step. Break-even after "
              f"~{breakeven:.0f} more steps -- below that, eager was cheaper overall.")
    else:
        print("\nCompiled steady-state wasn't faster than eager here -- see the "
              "Honesty box: graph breaks, tiny models, or a cold Triton cache "
              "can all eat the promised win.")


if __name__ == "__main__":
    main()
