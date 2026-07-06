# 18 — Prefix caching: stop re-reading the system prompt every request

**The lesson:** recipe 10 cached attention *within* one generation. Real traffic often shares a long, unchanging prefix *across* many separate requests — a system prompt, few-shot examples, a tool schema. Computing that shared prefix's KV cache once and reusing it turns "prefix + suffix" work per request into just "suffix" work after the first.

## Run

```bash
python recipes/18_prefix_caching/run.py --repeats 3
```

A fixed ~120-token system prompt plus up to 100 distinct short "user turns". Part 0 proves a from-scratch forward pass and a cached-prefix forward pass produce numerically identical logits over the suffix. Part 1 sweeps request count (1, 10, 50 by default) and tabulates wall time for both approaches. Part 2 is the `GreenMeter`'d headline receipt at `--headline-requests` (default 50).

## What to expect (rough, not yet measured — run it and replace this)

Part 0 should print a tiny (near-zero) float difference. In Part 1, at `n=1` the two approaches should cost about the same (the cached path still pays for the prefix once, just as the baseline does). As `n` grows, `no_cache`'s time should scale with `n x (prefix_len + suffix_len)` while `cached`'s scales close to `n x suffix_len` plus one fixed prefix cost — the speedup column should visibly grow with `n`, not stay flat.

## How it works

The system prompt's tokens produce the same keys/values on every request, since nothing before them ever changes. This recipe forwards the prefix once with `use_cache=True`, keeps the resulting `past_key_values`, and for each "request" deep-copies that cache and forwards only the new suffix tokens against it — the model attends over the (reused) prefix keys/values plus the (freshly computed) suffix ones, which is mathematically identical to attending over a freshly computed full sequence.

## Honesty box

- **Real serving engines go much further.** vLLM's automatic prefix caching and SGLang's RadixAttention manage *many* distinct, dynamically overlapping prefixes with LRU-style eviction across concurrent requests; this script hard-codes one fixed prefix in a single process — the minimal illustration of the underlying trick, not a production cache.
- Cloning the cache (`copy.deepcopy`) per request has its own real, measured cost — cheap relative to recomputing a nontrivial prefix from scratch, but not free, and it's included in the `cached` timings rather than hand-waved away.
- The cache itself costs VRAM proportional to `prefix_len x layers x heads x head_dim`; an unbounded number of distinct cached prefixes will eventually evict or OOM something — not addressed here.
- The benefit is entirely a function of `prefix_len / suffix_len`. A long shared system prompt with short user turns (chat assistants, agents with big tool schemas) benefits enormously; a short prefix followed by long generations gets little from this specific trick (recipe 10's within-generation cache matters more there).

## Go deeper

Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023, vLLM) · Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs* (2024, RadixAttention).

## Video hook

"I have one system prompt and a hundred different questions. Watch what happens when the model stops re-reading the instructions from scratch every single time."
