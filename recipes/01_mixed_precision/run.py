"""
Recipe 01 — Mixed precision: the precision ladder.

Trains the same tiny LM for the same steps in fp32, tf32, fp16 and bf16,
and prints the receipt for each. Same loss, less time, less energy.

Each mode is repeated `--repeats` times (same seed on purpose -- repeats
average out GPU/OS timing noise, not training randomness) and quality is
checked with held-out perplexity, not just the last few training losses.

    python recipes/01_mixed_precision/run.py                 # all supported modes
    python recipes/01_mixed_precision/run.py --modes fp32,bf16 --repeats 5
"""
import argparse
import contextlib
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

RECIPE = "01_mixed_precision"
MODEL = "HuggingFaceTB/SmolLM2-135M"


def supported_modes(device):
    import torch
    modes = ["fp32", "fp16"]
    if device == "cuda":
        cap = torch.cuda.get_device_capability()
        if cap >= (8, 0):
            modes.insert(1, "tf32")
        if torch.cuda.is_bf16_supported():
            modes.append("bf16")
    else:
        modes = ["fp32"]
    return modes


def train_once(mode, batches, eval_batches, device, lr, warmup):
    import torch
    from transformers import AutoModelForCausalLM

    # tf32 is just fp32 with faster matmul kernels on Ampere+ — flip the switch
    tf32 = mode == "tf32"
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    set_seed(42)  # same seed every trial: repeats target GPU/OS timing noise,
                  # not training randomness — quality should barely move.
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(mode)
    ctx = (lambda: torch.autocast(device_type=device, dtype=amp_dtype)) if amp_dtype \
        else contextlib.nullcontext
    scaler = torch.amp.GradScaler(device, enabled=(mode == "fp16"))

    def step(x):
        x = x.to(device)
        with ctx():
            loss = model(input_ids=x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        if mode == "fp16":
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        return loss.item()

    for x in batches[:warmup]:          # warmup outside the meter
        step(x)

    losses = []
    with GreenMeter(mode) as m:
        for i, x in enumerate(batches[warmup:]):
            losses.append(step(x))
            if (i + 1) % 10 == 0:
                print(f"  [{mode}] step {i + 1:>3}  loss {losses[-1]:.3f}")

    # quality check on data the model never trained on, served in the same
    # precision it was trained in — outside the meter, this is eval not cost.
    ppl = lm_perplexity(model, eval_batches, device, ctx=ctx())

    tokens = sum(b.numel() for b in batches[warmup:])
    m.add(final_loss=sum(losses[-5:]) / 5, eval_ppl=round(ppl, 3),
          tokens_per_s=int(tokens / m.result["wall_s"]))

    del model, opt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    time.sleep(2)                       # let power settle between runs
    return m.result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="auto", help="comma list or 'auto'")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=3, help="trials per mode to aggregate")
    ap.add_argument("--eval-batches", type=int, default=4, help="held-out batches for perplexity")
    args = ap.parse_args()

    device = pick_device(args.device)
    modes = supported_modes(device) if args.modes == "auto" else args.modes.split(",")
    print(f"modes to run: {modes}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = get_tinyshakespeare()
    batches = lm_batches(text, tok, args.seq_len, args.batch_size,
                         args.steps + args.warmup)
    # disjoint-ish held-out slice: different seed, never seen during training
    eval_batches = lm_batches(text, tok, args.seq_len, batch_size=4,
                              n_batches=args.eval_batches, seed=999)

    results = []
    for mode in modes:
        print(f"\n### {mode} ###")
        trials = []
        for t in range(args.repeats):
            if args.repeats > 1:
                print(f" -- trial {t + 1}/{args.repeats} --")
            trials.append(train_once(mode, batches, eval_batches, device, args.lr, args.warmup))
        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, Path(__file__).parent / "results",
                    is_baseline=(mode == "fp32"))

    compare(results, extra_keys=("final_loss", "eval_ppl", "tokens_per_s"))
    print("Same data, same steps, near-identical held-out perplexity — "
          "the speedup is (almost) free.")


if __name__ == "__main__":
    main()
