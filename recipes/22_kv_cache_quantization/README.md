# 22 — Quantize the cache itself: int8 KV cache for longer contexts

**The lesson:** recipe 10's KV cache and recipe 18's prefix cache both store keys/values at full precision. The cache is just tensors — it can be quantized exactly like weights (recipe 03), trading a small, bounded precision loss for a large, real memory win. A cache that fits in less VRAM means a longer context, or more concurrent cached requests, in the same GPU.

## Run

```bash
python recipes/22_kv_cache_quantization/run.py --repeats 3
```

Generates a real KV cache from a 600-token prefix, quantizes every key/value tensor to int8 (per-tensor symmetric scale), and dequantizes back. Part 0 reports the exact size reduction and the worst-case dequantization error over every tensor in the cache. Part 2 is the `GreenMeter`'d headline: one more decode step using the original cache vs. one more decode step using the quantized-then-dequantized cache.

## What to expect (rough, not yet measured — run it and replace this)

Since the model loads at fp32 here (see How it works), the int8 cache should be about 4x smaller, and the max dequantization error should be small and bounded — each tensor's own `max_abs_value / 127` quantization step, typically a small fraction of the tensor's value range. The two GreenMeter receipts' wall time and energy should land close to each other: quantizing/dequantizing is real but cheap arithmetic relative to a full attention forward pass, so this is a memory story, not a speed story.

## How it works

After one forward pass with `use_cache=True`, `past_key_values` holds every layer's key/value tensors. Each tensor is quantized independently: `scale = max(|tensor|) / 127`, `q = round(tensor / scale)` clamped to int8's range — the same per-tensor affine scheme recipe 03 uses for weights. Dequantizing (`q * scale`) recovers an approximation, not the exact original — the size of that gap is exactly what Part 0 measures rather than asserts. The model loads at fp32 (as recipe 18 also does) so the reported drift is attributable to int8 quantization alone, not mixed in with bf16's own ~3-significant-digit rounding. `past_key_values` is converted to a plain tuple via `to_legacy_cache()` (and back via `from_legacy_cache()`) — the documented, version-safe round-trip for transformers' `Cache` objects — so this works whether your installed transformers returns a legacy tuple or a `Cache` instance.

## Honesty box

- **This recipe quantizes a cache snapshot once and reuses it for a single extra step** — a real serving system would need to quantize *every* new token's key/value as they're produced, and dequantize the whole cache on every attention read, adding a real (if small) per-step tax not fully amortized here the way a production implementation would.
- Per-tensor scales are the crudest possible scheme. Real int8-KV-cache implementations (e.g. in vLLM, TensorRT-LLM) typically use per-channel or per-head scales, which cost a little more bookkeeping but meaningfully reduce the worst-case error reported in Part 0.
- The bytes freed here are cache-only. The much larger weight matrices (recipe 03) and activation memory (recipe 06) are untouched — this stacks with those techniques, it doesn't replace them.
- Quantization error compounds: this recipe checks one dequantization round-trip. A long generation that re-quantizes the growing cache at every step accumulates rounding error across steps in a way this snapshot check doesn't capture — worth validating with an end-to-end perplexity check before trusting it over very long contexts.

## Go deeper

Hooper et al., *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization* (2024) · Liu et al., *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache* (2024).

## Video hook

"Same cache, a quarter of the bytes. Here's exactly how much the numbers actually moved."
