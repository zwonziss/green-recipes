"""
Recipe 23 — CPU-offloaded activations: trade PCIe bandwidth for memory.

Recipe 06 freed VRAM by throwing activations away and recomputing them
during backward -- more FLOPs, same GPU the whole time. This recipe frees
VRAM a different way: torch.autograd.graph.save_on_cpu() keeps every saved
activation, but moves it to pinned host RAM during forward and copies it
back to the GPU during backward. No extra compute, a real PCIe transfer
cost instead -- a genuinely different trade than recipe 06's, worth
measuring on its own rather than assumed identical.

Meaningless without a CUDA GPU (there's no separate "host" to offload to
otherwise), so this recipe skips gracefully on CPU-only / MPS machines.

    python recipes/23_activation_offloading/run.py
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

RECIPE = "23_activation_offloading"
MODEL = "HuggingFaceTB/SmolLM2-360M"


def run_variant(label, batches, offload, device, lr, warmup):
    import torch
    from transformers import AutoModelForCausalLM

    set_seed(42)  # same seed every trial -- repeats target timing noise, not training randomness
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def step(x):
        x = x.to(device)
        if offload:
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                loss = model(input_ids=x, labels=x).loss
        else:
            loss = model(input_ids=x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.item()

    result, losses = None, []
    try:
        for x in batches[:warmup]:
            step(x)
        with GreenMeter(label) as m:
            for i, x in enumerate(batches[warmup:]):
                losses.append(step(x))
                if (i + 1) % 10 == 0:
                    print(f"  [{label}] step {i + 1:>3}  loss {losses[-1]:.3f}")
        tokens = sum(b.numel() for b in batches[warmup:])
        m.add(final_loss=sum(losses[-5:]) / 5,
              tokens_per_s=int(tokens / m.result["wall_s"]),
              batch_size=batches[0].shape[0])
        result = m.result
    except torch.cuda.OutOfMemoryError:
        print(f"  [{label}] OOM — this variant does not fit on your GPU.")
        result = {"label": label, "wall_s": None,
                  "extra": {"batch_size": batches[0].shape[0], "oom": True}}

    del model, opt
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    time.sleep(2)
    return result, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=3, help="trials per variant to aggregate")
    args = ap.parse_args()

    device = pick_device(args.device)
    if device != "cuda":
        print("This recipe demonstrates trading GPU VRAM for host RAM + PCIe "
              "transfer time via torch.autograd.graph.save_on_cpu() — there's no "
              "separate 'host' to offload to without a CUDA GPU. Skipping.")
        return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = get_tinyshakespeare()
    n = args.steps + args.warmup
    small = lm_batches(text, tok, args.seq_len, args.batch_size, n)
    big = lm_batches(text, tok, args.seq_len, args.batch_size * 2, n)

    out_dir = Path(__file__).parent
    plan = [("baseline_b8", small, False, True),
            ("offload_b8", small, True, False),
            ("offload_b16", big, True, False)]

    results = []
    first_losses = {}
    for label, batches, offload, is_base in plan:
        print(f"\n### {label} ###")
        trials = []
        for t in range(args.repeats):
            if args.repeats > 1:
                print(f" -- trial {t + 1}/{args.repeats} --")
            r, losses = run_variant(label, batches, offload, device, args.lr, args.warmup)
            if label not in first_losses:
                first_losses[label] = losses
            trials.append(r)
            if r.get("wall_s") is None:
                break  # OOM: no point repeating an out-of-memory run
        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        if r.get("wall_s") is not None:
            save_result(r, RECIPE, out_dir / "results", is_baseline=is_base)

    compare(results, extra_keys=("final_loss", "tokens_per_s", "batch_size"))

    if first_losses.get("baseline_b8") and first_losses.get("offload_b8"):
        a, b = first_losses["baseline_b8"], first_losses["offload_b8"]
        max_diff = max(abs(x - y) for x, y in zip(a, b))
        print(f"\nNumerical check (same batch size, same data, same seed): "
              f"max |loss_baseline - loss_offload| over {len(a)} steps = {max_diff:.2e}. "
              "Offloading moves tensors, it doesn't change the math.")

    print("Offloading alone: less VRAM, more wall time (PCIe transfer, not "
          "recomputation). Offloading + the bigger batch it enables often "
          "closes most of that gap back. Contrast with recipe 06: same VRAM "
          "goal, a completely different resource (bandwidth, not FLOPs) spent to get there.")


if __name__ == "__main__":
    main()
