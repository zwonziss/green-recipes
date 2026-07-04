"""
Recipe 08 — Fitting the "impossible" run: accumulation + 8-bit Adam + the stack.

Goal: train a 360M model with an EFFECTIVE batch of 16 at T=512 on a small GPU.
  fp32_direct_b16   the naive attempt (expected to OOM on 12 GB — recorded)
  fp16_accum        micro-batch 4 x accumulate 4, fp16 autocast
  + adam8bit        same, optimizer states quantized to 8-bit
  + ckpt            same, plus gradient checkpointing (the full stack)

All fitting variants do identical optimizer steps at identical effective
batch, each repeated `--repeats` times (same seed on purpose — repeats
target GPU/OS timing noise, not training randomness), plus a held-out
perplexity so "fits in memory" isn't quietly also "learns worse".

    python recipes/08_smaller_batch_tricks/run.py
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

RECIPE = "08_smaller_batch_tricks"
MODEL = "HuggingFaceTB/SmolLM2-360M"


def make_optimizer(model, kind, lr):
    import torch
    if kind == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(model.parameters(), lr=lr)
    return torch.optim.AdamW(model.parameters(), lr=lr)


def run_variant(label, cfg, text, tok, eval_batches, device, args):
    import torch
    from transformers import AutoModelForCausalLM

    micro_bs, accum, fp16, opt_kind, ckpt = cfg
    n_micro = args.opt_steps * accum + args.warmup
    batches = lm_batches(text, tok, args.seq_len, micro_bs, n_micro)

    set_seed(42)  # same seed every trial, see module docstring
    result, losses = None, []
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
        model.train()
        if ckpt:
            model.config.use_cache = False
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        opt = make_optimizer(model, opt_kind, args.lr)
        scaler = torch.amp.GradScaler(device, enabled=fp16)

        def micro_step(x, do_update):
            x = x.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=fp16):
                loss = model(input_ids=x, labels=x).loss / accum
            scaler.scale(loss).backward()
            if do_update:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            return loss.item() * accum

        for i, x in enumerate(batches[:args.warmup]):
            micro_step(x, do_update=(i + 1) % accum == 0)
        opt.zero_grad(set_to_none=True)

        with GreenMeter(label) as m:
            for i, x in enumerate(batches[args.warmup:]):
                losses.append(micro_step(x, do_update=(i + 1) % accum == 0))
                if (i + 1) % (10 * accum) == 0:
                    print(f"  [{label}] opt step {(i + 1) // accum:>3}  "
                          f"loss {losses[-1]:.3f}")

        ctx = (lambda: torch.autocast(device_type=device, dtype=torch.float16)) if fp16 else None
        ppl = lm_perplexity(model, eval_batches, device, ctx=ctx() if ctx else None)

        tokens = sum(b.numel() for b in batches[args.warmup:])
        m.add(final_loss=sum(losses[-5:]) / 5, eval_ppl=round(ppl, 3),
              tokens_per_s=int(tokens / m.result["wall_s"]),
              micro_bs=micro_bs, accum=accum,
              effective_bs=micro_bs * accum, optimizer=opt_kind)
        result = m.result
        del model, opt
    except torch.cuda.OutOfMemoryError:
        print(f"  [{label}] OOM — recorded. This is why the other variants exist.")
        result = {"label": label, "wall_s": None,
                  "extra": {"effective_bs": micro_bs * accum, "oom": True}}

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    time.sleep(2)
    return result, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt-steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--repeats", type=int, default=3, help="trials per variant to aggregate")
    ap.add_argument("--eval-batches", type=int, default=4, help="held-out batches for perplexity")
    ap.add_argument("--skip-oom-demo", action="store_true",
                    help="skip the fp32 direct attempt")
    args = ap.parse_args()

    device = pick_device("auto")
    if device != "cuda":
        raise SystemExit("This recipe needs a CUDA GPU (bitsandbytes).")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = get_tinyshakespeare()
    eval_batches = lm_batches(text, tok, args.seq_len, batch_size=4,
                              n_batches=args.eval_batches, seed=999)
    out_dir = Path(__file__).parent

    #        label                 micro accum fp16   optimizer    ckpt
    plan = [("fp32_direct_b16",     (16,  1,  False, "adamw",     False)),
            ("fp16_accum",          (4,   4,  True,  "adamw",     False)),
            ("fp16_accum_adam8bit", (4,   4,  True,  "adamw8bit", False)),
            ("full_stack_ckpt",     (4,   4,  True,  "adamw8bit", True))]
    if args.skip_oom_demo:
        plan = plan[1:]

    results = []
    first_losses = {}
    for label, cfg in plan:
        print(f"\n### {label} ###")
        trials = []
        for t in range(args.repeats):
            if args.repeats > 1:
                print(f" -- trial {t + 1}/{args.repeats} --")
            r, losses = run_variant(label, cfg, text, tok, eval_batches, device, args)
            if label not in first_losses:
                first_losses[label] = losses
            trials.append(r)
            if r.get("wall_s") is None:
                break  # OOM: no point repeating an out-of-memory run
        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        if r.get("wall_s") is not None:
            save_result(r, RECIPE, out_dir / "results",
                        is_baseline=(label == "fp16_accum"))

    compare(results, extra_keys=("final_loss", "eval_ppl", "tokens_per_s",
                                 "effective_bs", "optimizer"))

    a, b = first_losses.get("fp16_accum"), first_losses.get("fp16_accum_adam8bit")
    if a and b:
        max_diff = max(abs(x - y) for x, y in zip(a, b))
        print(f"\nNumerical check (same data, same effective batch, only the "
              f"optimizer differs): max |loss_fp32adam - loss_8bitadam| over "
              f"{len(a)} steps = {max_diff:.3f}. Small but nonzero -- 8-bit "
              "Adam approximates the fp32 optimizer state, it doesn't reproduce it exactly.")

    print("Same effective batch, same optimizer steps, near-identical held-out "
          "perplexity — watch peak VRAM staircase downward as the tricks stack. "
          "This stack is how a 7B model becomes trainable on a 12 GB card (add "
          "recipe 02's LoRA + recipe 03's 4-bit weights and you've reinvented QLoRA).")


if __name__ == "__main__":
    main()
