"""
Recipe 03 — Quantized inference: smaller, cheaper... and how much dumber?

Loads the same 1.7B model in fp16, int8 and 4-bit NF4. For each: VRAM,
generation speed (repeated `--repeats` times, same loaded weights, to
average out timing noise), energy per 1k tokens, AND a perplexity probe
so the quality cost is a number, not a vibe. Requires a CUDA GPU (bitsandbytes).

    python recipes/03_quantization_inference/run.py
    python recipes/03_quantization_inference/run.py --modes fp16,nf4 --model HuggingFaceTB/SmolLM2-360M
"""
import argparse
import gc
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.greenmeter import (  # noqa: E402
    GreenMeter, aggregate_trials, compare, print_receipt, save_result,
)
from common.data import get_tinyshakespeare, lm_batches, set_seed  # noqa: E402

RECIPE = "03_quantization_inference"

PROMPTS = [
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\n",
    "The most important idea in machine learning is",
    "KING RICHARD III:\nNow is the winter of",
]


def load(model_name, mode):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    kw = {}
    if mode == "fp16":
        kw = dict(torch_dtype=torch.float16, device_map={"": 0})
    elif mode == "int8":
        kw = dict(quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                  device_map={"": 0})
    elif mode == "nf4":
        kw = dict(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True),
            device_map={"": 0})
    else:
        raise ValueError(mode)
    return AutoModelForCausalLM.from_pretrained(model_name, **kw)


def perplexity(model, eval_batches):
    import torch
    nlls = []
    with torch.no_grad():
        for x in eval_batches:
            x = x.to(model.device)
            nlls.append(model(input_ids=x, labels=x).loss.item())
    return math.exp(sum(nlls) / len(nlls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B")
    ap.add_argument("--modes", default="fp16,int8,nf4")
    ap.add_argument("--new-tokens", type=int, default=200)
    ap.add_argument("--ppl-chunks", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3, help="generation-loop trials to aggregate")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("This recipe needs a CUDA GPU (bitsandbytes).")
    from transformers import AutoTokenizer

    set_seed(42)
    tok = AutoTokenizer.from_pretrained(args.model)
    eval_batches = lm_batches(get_tinyshakespeare(), tok, seq_len=512,
                              batch_size=1, n_batches=args.ppl_chunks, seed=7)
    out_dir = Path(__file__).parent
    results = []

    for mode in args.modes.split(","):
        print(f"\n### {mode} ###")
        model = load(args.model, mode)
        model.eval()
        load_vram = torch.cuda.memory_allocated() / 2**20
        print(f"  weights on GPU: {load_vram:.0f} MB")

        # warmup generation (kernel compilation, cache allocs)
        enc = tok(PROMPTS[0], return_tensors="pt").to(model.device)
        model.generate(**enc, max_new_tokens=16, do_sample=False,
                       pad_token_id=tok.eos_token_id)

        trials = []
        for t in range(args.repeats):
            if args.repeats > 1:
                print(f"  -- trial {t + 1}/{args.repeats} --")
            gen_tokens = 0
            with GreenMeter(mode) as m:
                for p in PROMPTS:
                    enc = tok(p, return_tensors="pt").to(model.device)
                    out = model.generate(**enc, max_new_tokens=args.new_tokens,
                                         do_sample=False, pad_token_id=tok.eos_token_id)
                    gen_tokens += out.shape[1] - enc["input_ids"].shape[1]
            tps = gen_tokens / m.result["wall_s"]
            wh_per_1k = m.result.get("gpu_energy_wh", 0) / gen_tokens * 1000
            m.add(tokens_per_s=round(tps, 1), wh_per_1k_tok=round(wh_per_1k, 3))
            trials.append(m.result)

        r = aggregate_trials(trials)
        ppl = perplexity(model, eval_batches)  # deterministic (greedy forward) -> once
        r["extra"]["weights_vram_mb"] = round(load_vram, 0)
        r["extra"]["ppl"] = round(ppl, 3)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, out_dir / "results", is_baseline=(mode == "fp16"))

        del model
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)

    compare(results, extra_keys=("weights_vram_mb", "tokens_per_s",
                                 "wh_per_1k_tok", "ppl"))
    print("Watch the ppl column: that's the quality you paid for the memory you saved.")


if __name__ == "__main__":
    main()
