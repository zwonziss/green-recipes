"""
Recipe 02 — LoRA vs full fine-tuning.

Fine-tunes the same model twice: every parameter vs. tiny low-rank adapters.
Compares trainable params, VRAM, time, energy, held-out perplexity — and
what lands on disk. Each condition is repeated `--repeats` times (same seed
on purpose — repeats target GPU/OS timing noise, not training randomness).

    python recipes/02_lora/run.py
    python recipes/02_lora/run.py --repeats 5
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

RECIPE = "02_lora"
MODEL = "HuggingFaceTB/SmolLM2-135M"


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20


def train(model, batches, eval_batches, device, lr, warmup, label, use_fp16=True):
    import torch
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    scaler = torch.amp.GradScaler(device, enabled=use_fp16 and device == "cuda")
    autocast_on = use_fp16 and device == "cuda"

    def step(x):
        x = x.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=autocast_on):
            loss = model(input_ids=x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return loss.item()

    for x in batches[:warmup]:
        step(x)

    losses = []
    with GreenMeter(label) as m:
        for i, x in enumerate(batches[warmup:]):
            losses.append(step(x))
            if (i + 1) % 10 == 0:
                print(f"  [{label}] step {i + 1:>3}  loss {losses[-1]:.3f}")

    # held-out quality check, served in the same precision it trained in
    ctx = (lambda: torch.autocast(device_type=device, dtype=torch.float16))
    ppl = lm_perplexity(model, eval_batches, device,
                        ctx=ctx() if autocast_on else None)

    m.add(final_loss=sum(losses[-5:]) / 5, eval_ppl=round(ppl, 3),
          trainable_params_m=round(count_trainable(model) / 1e6, 2))
    return m


def run_condition(label, build_fn, save_dirname, batches, eval_batches, device,
                  lr, warmup, repeats, out_dir, is_baseline):
    import torch
    trials = []
    disk_mb = None
    for t in range(repeats):
        if repeats > 1:
            print(f" -- trial {t + 1}/{repeats} --")
        set_seed(42)  # same seed every trial, see module docstring
        model = build_fn()
        m = train(model, batches, eval_batches, device, lr, warmup, label)
        if t == 0:  # save artifacts once; re-saving identical weights every
                    # trial is wasted disk I/O, not more information
            save_dir = out_dir / save_dirname
            model.save_pretrained(save_dir)
            disk_mb = round(dir_size_mb(save_dir), 1)
        m.add(disk_mb=disk_mb)
        trials.append(m.result)
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        time.sleep(2)

    r = aggregate_trials(trials)
    print_receipt(r)
    save_result(r, RECIPE, out_dir / "results", is_baseline=is_baseline)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=3, help="trials per condition to aggregate")
    ap.add_argument("--eval-batches", type=int, default=4, help="held-out batches for perplexity")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = get_tinyshakespeare()
    batches = lm_batches(text, tok, args.seq_len, args.batch_size, args.steps + args.warmup)
    eval_batches = lm_batches(text, tok, args.seq_len, batch_size=4,
                              n_batches=args.eval_batches, seed=999)
    out_dir = Path(__file__).parent
    results = []

    # --- condition 1: full fine-tuning --------------------------------------
    print("\n### full fine-tuning (every weight gets a gradient + Adam state) ###")
    results.append(run_condition(
        "full_ft", lambda: AutoModelForCausalLM.from_pretrained(MODEL).to(device),
        "out_full", batches, eval_batches, device, lr=5e-5, warmup=args.warmup,
        repeats=args.repeats, out_dir=out_dir, is_baseline=True))

    # --- condition 2: LoRA ---------------------------------------------------
    def build_lora():
        model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
        cfg = LoraConfig(task_type="CAUSAL_LM", r=args.rank, lora_alpha=args.rank * 2,
                         lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
        return model

    print(f"\n### LoRA (rank {args.rank}: base frozen, tiny adapters learn) ###")
    results.append(run_condition(
        f"lora_r{args.rank}", build_lora, "out_lora", batches, eval_batches, device,
        lr=2e-4, warmup=args.warmup, repeats=args.repeats, out_dir=out_dir,
        is_baseline=False))

    compare(results, extra_keys=("final_loss", "eval_ppl", "trainable_params_m", "disk_mb"))
    print("Adam keeps 2 extra copies of every TRAINABLE weight — freeze the base "
          "model and that VRAM (and the 270 MB checkpoint) mostly disappears, "
          "with near-identical held-out perplexity.")


if __name__ == "__main__":
    main()
