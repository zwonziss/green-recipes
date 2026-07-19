"""
Recipe 24 — Merge your LoRA adapter: zero inference overhead, not just tiny training cost.

Recipe 02 showed LoRA's training-time win: freeze the base model, train tiny
adapters, Adam's memory bill collapses. That says nothing about serving.
Left unmerged, every adapted layer computes base(x) + B(A(x)) on every
forward call -- a real, if small, extra pair of matmuls per layer, per
token, forever. peft's merge_and_unload() folds B@A*scale into the base
weight matrices once, offline, and hands back a plain dense model with no
adapter modules at all -- the training-time trick and the inference-time
cost are two different bills, addressed here on its own.

Part 0 proves merging is exact: base_W + BA is the same weight merge_and_unload
computes, so a merged model's logits match the unmerged adapter's up to float
rounding -- not an approximation.

    python recipes/24_lora_merging/run.py
"""
import argparse
import copy
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.greenmeter import (  # noqa: E402
    GreenMeter, aggregate_trials, compare, print_receipt, save_result,
)
from common.data import get_tinyshakespeare, lm_batches, pick_device, set_seed  # noqa: E402
from common.eval import max_abs_diff  # noqa: E402

RECIPE = "24_lora_merging"
MODEL = "HuggingFaceTB/SmolLM2-135M"
PROMPTS = [f"Write one short sentence about topic #{i}.\n" for i in range(20)]


def train_adapter(device, steps, warmup, batch_size, seq_len, rank, lr):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    set_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
    cfg = LoraConfig(task_type="CAUSAL_LM", r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(base, cfg)
    model.print_trainable_parameters()

    text = get_tinyshakespeare()
    batches = lm_batches(text, tok, seq_len, batch_size, steps + warmup)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    model.train()
    for i, x in enumerate(batches):
        x = x.to(device)
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (i + 1) % 5 == 0:
            print(f"  training adapter — step {i + 1:>2}/{len(batches)}  loss {loss.item():.3f}")
    model.eval()
    return model, tok


def check_equivalent(model, tok, device):
    """base_W + B@A*scale is exactly what merge_and_unload computes -- a merged
    model should answer identically to the still-adapted one, not approximately."""
    import torch
    enc = tok(PROMPTS[0], return_tensors="pt").to(device)
    with torch.no_grad():
        logits_adapter = model(input_ids=enc.input_ids).logits
    merged_copy = copy.deepcopy(model).merge_and_unload()
    with torch.no_grad():
        logits_merged = merged_copy(input_ids=enc.input_ids).logits
    diff = max_abs_diff(logits_adapter, logits_merged)
    print(f"Part 0 — numerical check: max |adapter - merged| logits for the same "
          f"input = {diff:.2e} (should be tiny float noise).")
    del merged_copy
    gc.collect()


def generate_all(model, tok, device, max_new_tokens):
    total_new = 0
    for p in PROMPTS:
        enc = tok(p, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        total_new += out.shape[1] - enc.input_ids.shape[1]
    return total_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=3, help="trials per variant to aggregate")
    args = ap.parse_args()

    device = pick_device(args.device)
    model, tok = train_adapter(device, args.steps, args.warmup, args.batch_size,
                               args.seq_len, args.rank, args.lr)

    check_equivalent(model, tok, device)

    out_dir = Path(__file__).parent
    results = []

    print(f"\n### lora_adapter: {len(PROMPTS)} prompts, adapter layers active ###")
    trials = []
    for t in range(args.repeats):
        if args.repeats > 1:
            print(f" -- trial {t + 1}/{args.repeats} --")
        with GreenMeter("lora_adapter") as m:
            total_new = generate_all(model, tok, device, args.new_tokens)
        m.add(tokens_per_s=round(total_new / m.result["wall_s"], 1), n_prompts=len(PROMPTS))
        trials.append(m.result)
    r = aggregate_trials(trials)
    print_receipt(r)
    results.append(r)
    save_result(r, RECIPE, out_dir / "results", is_baseline=True)

    print("\nMerging the adapter into the base weights (one-way, offline op)...")
    merged_model = model.merge_and_unload()
    merged_model.eval()

    print(f"\n### lora_merged: same {len(PROMPTS)} prompts, plain dense weights ###")
    trials = []
    for t in range(args.repeats):
        if args.repeats > 1:
            print(f" -- trial {t + 1}/{args.repeats} --")
        with GreenMeter("lora_merged") as m:
            total_new = generate_all(merged_model, tok, device, args.new_tokens)
        m.add(tokens_per_s=round(total_new / m.result["wall_s"], 1), n_prompts=len(PROMPTS))
        trials.append(m.result)
    r = aggregate_trials(trials)
    print_receipt(r)
    results.append(r)
    save_result(r, RECIPE, out_dir / "results", is_baseline=False)

    compare(results, extra_keys=("tokens_per_s", "n_prompts"))
    print("Same trained skill, same generated text (Part 0) -- merging just "
          "removes the per-layer adapter matmuls that unmerged LoRA pays on "
          "every single forward call. Only worth it when you're serving ONE "
          "fixed adapter (see the Honesty box for when it isn't).")


if __name__ == "__main__":
    main()
