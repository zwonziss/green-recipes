# 25 — Exact-match response caching: stop re-answering the same question

**The lesson:** real traffic repeats, often in a long-tailed way — a handful of canonical questions (FAQs, common requests, retries, a popular button in your product) account for a disproportionate share of total volume. A plain dict keyed by the exact request string turns every repeat into a lookup instead of a full generation. This is a different reuse than recipe 10 (caching *within* one generation) or recipe 18 (a shared *prefix* across otherwise-different requests) — here the entire request, and therefore the entire response, is identical to one already served.

## Run

```bash
python recipes/25_response_caching/run.py --repeats 3
```

Simulates 150 requests drawn from 20 canonical questions with a Zipf-ish skew (question #0 asked far more often than question #19 — the same long-tail shape real request logs tend to show). Part 0 proves the invariant this relies on: two independent fresh calls for the same prompt, under greedy decoding, produce identical output. Then it replays the traffic with `no_response_cache` (every request generates fresh) and `response_cache` (a dict cache — only the first occurrence of each distinct question generates; every repeat is a lookup).

## What to expect (rough, not yet measured — run it and replace this)

Part 0 should print `True`. The printed traffic summary should show a large fraction of requests are repeats (a Zipf-ish draw over 20 items at 150 requests typically repeats heavily). `response_cache`'s `generate_calls` should equal the number of *distinct* questions actually drawn (≤ 20), versus `no_response_cache`'s fixed 150 — wall time and energy should drop by roughly the same ratio as the hit rate.

## How it works

```python
if p in cache:
    continue  # skip generate() entirely
out = model.generate(...)
cache[p] = out
```

Nothing more sophisticated than a dict. The entire benefit comes from the fact that a repeated request is, by definition, asking for work already done — the cheapest possible reuse.

## Honesty box

- **Exact match only.** Any whitespace, casing, or punctuation difference is a cache miss even if the question means the same thing. A fuzzier, semantic cache (embedding similarity, e.g. GPTCache-style) catches more repeats but introduces its own false-positive risk — a near-match isn't always an acceptable substitute answer. Out of scope here.
- **The entire benefit is a property of your traffic, not the model.** A workload with zero repeated requests gets zero benefit from this technique — unlike prefix caching (18) or the KV cache (10), which help even on entirely unique requests.
- **Only safe for deterministic serving (greedy / temperature 0), or when a stale/previous sample is genuinely acceptable.** Caching a `temperature > 0` sampled response would silently make outputs less diverse than the caller asked for — a real correctness footgun, not just a missed optimization.
- This dict grows forever. A real deployment needs bounded eviction (LRU, TTL, or a max size) sized to available memory — not shown here.

## Go deeper

Bang, *GPTCache: An Open-Source Semantic Cache for LLM Applications* (2023) — the production, fuzzy-match extension of this same idea · Zheng et al., *SGLang* (already cited in recipe 18) — RadixAttention subsumes exact-prefix reuse, of which this recipe's exact-request case is the limit where the whole request is the shared prefix.

## Video hook

"Twenty questions, a hundred and fifty requests. Watch how few of them actually reach the model."
