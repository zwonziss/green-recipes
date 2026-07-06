# 10 — The KV cache: why decoding isn't O(n²) in practice

**The lesson:** generating token T+1 only needs *new* work if you keep yesterday's attention keys/values around. Without a cache, every step re-derives the whole prefix from scratch — total generation cost grows worse than linearly with length. This recipe times both paths and proves they compute the identical thing first.

## Run

```bash
python recipes/10_kv_cache/run.py --repeats 3
```

Part 0: a deterministic 48-token greedy generation with `use_cache=True` vs `use_cache=False` must produce byte-identical output — proven, not assumed. Part 1: sweeps generation length and tabulates wall time for both, so the gap's *shape* (not just its size) is visible. Part 2: a `GreenMeter` receipt pair at one realistic length, saved and compared.

## What to expect (rough, not yet measured — run it and replace this)

Part 0 should print `True`. In Part 1, `cache` time should scale close to linearly with `new_tokens`; `no_cache` time should visibly bend upward — the speedup ratio in the last column should grow as length grows, not stay flat. At a few hundred tokens on SmolLM2-135M, expect cache to already be several times faster; the gap keeps widening for longer generations.

## How it works

Causal attention lets token *t* attend only to tokens `<= t`. Once a token's key/value vectors are computed they never change — so caching them means step *t+1* only computes one new query against `t` cached keys/values instead of recomputing all of them. `model.generate(..., use_cache=True)` (the HF default) does exactly this; `use_cache=False` forces a fresh forward pass over the whole growing sequence every step, on purpose, so you can see the cost of not caching.

## Honesty box

- This is single-request, single-GPU decoding. Real serving systems (vLLM, TGI, SGLang) spend most of their engineering on managing caches across *many concurrent* requests — paging cache blocks (PagedAttention), evicting cold ones, batching requests with different cache states. None of that is shown here.
- Cache memory scales with `batch x sequence_length x layers x heads x head_dim` — it is often the actual memory bottleneck in production serving, not the model weights (see recipe 03 for weight memory, this recipe for the *other* half of the VRAM bill).
- If you send many requests that share a long common prefix (a system prompt), you can go further than a single request's cache — see recipe 18 (prefix caching).
- The equivalence check in Part 0 relies on greedy (`do_sample=False`) decoding; with sampling, tiny floating-point differences in the compute path could in principle change which token gets sampled at a near-tie in probabilities. That's a sampling-implementation subtlety, not a cache correctness issue.

## Go deeper

Pope et al., *Efficiently Scaling Transformer Inference* (2022) · Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023, vLLM).

## Video hook

"I turned off one flag and watched generation grind to a crawl — here's the flag, and here's exactly why it matters more the longer you generate."
