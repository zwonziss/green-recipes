"""
Recipe 22 — Quantize the cache itself: int8 KV cache for longer contexts.

Recipe 10 introduced the KV cache; recipe 18 reused it across requests. Both
still store it at full precision. But the cache is just a big pile of
tensors -- it can be quantized exactly like weights (recipe 03), trading a
little precision for a lot of memory. This recipe quantizes a real
generated cache to int8 (per-tensor scale), measures the memory it frees,
and checks how much the dequantized values actually drift from the
originals before feeding either version back into one more decoding step.

    python recipes/22_kv_cache_quantization/run.py
"""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.greenmeter import (  # noqa: E402
    GreenMeter, aggregate_trials, compare, print_receipt, save_result,
)
from common.data import get_tinyshakespeare, pick_device, set_seed  # noqa: E402
from common.eval import max_abs_diff  # noqa: E402

RECIPE = "22_kv_cache_quantization"
MODEL = "HuggingFaceTB/SmolLM2-135M"
PREFIX_TOKENS = 600


def to_legacy(past_key_values):
    """Cache objects (recent transformers) expose to_legacy_cache() for exactly
    this round-trip; older versions already return a plain tuple."""
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def from_legacy(legacy, like):
    if hasattr(like, "to_legacy_cache"):
        return type(like).from_legacy_cache(legacy)
    return legacy


def quantize_int8(t):
    import torch
    scale = t.abs().max().clamp(min=1e-8) / 127.0
    q = (t / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def quantize_cache(legacy_pkv):
    quantized, scales = [], []
    for k, v in legacy_pkv:
        qk, sk = quantize_int8(k.float())
        qv, sv = quantize_int8(v.float())
        quantized.append((qk, qv))
        scales.append((sk, sv))
    return tuple(quantized), scales


def dequantize_cache(quantized, scales, dtype):
    out = []
    for (qk, qv), (sk, sv) in zip(quantized, scales):
        out.append(((qk.float() * sk).to(dtype), (qv.float() * sv).to(dtype)))
    return tuple(out)


def cache_bytes(legacy_pkv, per_element_bytes=None):
    total = 0
    for k, v in legacy_pkv:
        eb_k = per_element_bytes if per_element_bytes is not None else k.element_size()
        eb_v = per_element_bytes if per_element_bytes is not None else v.element_size()
        total += k.numel() * eb_k + v.numel() * eb_v
    return total


def worst_abs_diff(legacy_a, legacy_b):
    return max(max(max_abs_diff(ak, bk), max_abs_diff(av, bv))
               for (ak, av), (bk, bv) in zip(legacy_a, legacy_b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-tokens", type=int, default=PREFIX_TOKENS)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=3, help="trials per variant to aggregate")
    args = ap.parse_args()

    device = pick_device(args.device)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL)
    # fp32 on purpose (see recipe 18): keeps the quantization drift attributable
    # to int8, not pre-mixed with bf16's own ~3-significant-digit rounding.
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(device)
    model.eval()

    text = get_tinyshakespeare()
    ids = tok(text[:8000], return_tensors="pt").input_ids[:, :args.prefix_tokens].to(device)
    prefix_len = ids.shape[1]
    print(f"Prefix: {prefix_len} tokens")

    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    orig_pkv_native = out.past_key_values
    legacy_pkv = to_legacy(orig_pkv_native)

    fp_bytes = cache_bytes(legacy_pkv)
    q_pkv, scales = quantize_cache(legacy_pkv)
    int8_bytes = cache_bytes(q_pkv, per_element_bytes=1)
    deq_legacy = dequantize_cache(q_pkv, scales, legacy_pkv[0][0].dtype)
    drift = worst_abs_diff(legacy_pkv, deq_legacy)

    print(f"Part 0 — numerical check: cache size {fp_bytes / 2**20:.2f} MB (as-loaded) "
          f"-> {int8_bytes / 2**20:.2f} MB (int8), a {fp_bytes / int8_bytes:.1f}x reduction. "
          f"Max |original - dequantized| over every K/V tensor = {drift:.2e} "
          f"(bounded by each tensor's own scale/127 quantization step).")

    next_id = out.logits[:, -1:].argmax(-1)
    position = torch.tensor([[prefix_len]], device=device)
    mask = torch.ones((1, prefix_len + 1), dtype=torch.long, device=device)
    deq_cache_native = from_legacy(deq_legacy, orig_pkv_native)

    out_dir = Path(__file__).parent
    results = []
    for label, cache in (("fp_cache", orig_pkv_native), ("int8_cache", deq_cache_native)):
        trials = []
        for t in range(args.repeats):
            if args.repeats > 1:
                print(f" -- trial {t + 1}/{args.repeats} [{label}] --")
            # forward() mutates past_key_values in place (appends the new
            # token) -- deep-copy per trial so repeats don't compound.
            with GreenMeter(label) as m:
                with torch.no_grad():
                    model(input_ids=next_id, past_key_values=copy.deepcopy(cache),
                         position_ids=position, attention_mask=mask, use_cache=True)
            m.add(cache_mb=round((fp_bytes if label == "fp_cache" else int8_bytes) / 2**20, 3),
                 max_abs_diff=round(drift, 6) if label == "int8_cache" else 0.0,
                 prefix_tokens=prefix_len)
            trials.append(m.result)
        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, out_dir / "results", is_baseline=(label == "fp_cache"))

    compare(results, extra_keys=("cache_mb", "max_abs_diff", "prefix_tokens"))
    print("The decode step itself costs about the same either way -- the win is "
          "memory, not speed. A cache that fits in less VRAM means a longer "
          "context, or more concurrent requests, in the same GPU.")


if __name__ == "__main__":
    main()
