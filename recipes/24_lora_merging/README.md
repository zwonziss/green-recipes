# 24 — Merge your LoRA adapter: zero inference overhead, not just tiny training cost

**The lesson:** recipe 02 showed LoRA's training-time win — freeze the base model, train tiny adapters, Adam's memory bill collapses. That's a training story. Left unmerged, every adapted layer still computes `base(x) + B(A(x))` on *every* forward call, forever — a small but real extra cost paid on every request you ever serve. `merge_and_unload()` folds `B@A*scale` into the base weights once, offline, and hands back a plain dense model with zero adapter overhead.

## Run

```bash
python recipes/24_lora_merging/run.py --repeats 3
```

Trains a small rank-8 LoRA adapter on SmolLM2-135M for a handful of steps (just enough for non-trivial weights — this recipe is about serving cost, not training quality). Part 0 proves a merged copy of the model answers identically to the still-adapted one. Then it times generation over 20 prompts with the adapter active, merges the model, and times the same generation again with plain dense weights.

## What to expect (rough, not yet measured — run it and replace this)

Part 0's logit diff should be tiny (float noise) — merging is exact arithmetic, not an approximation. `lora_merged` should be at least as fast as `lora_adapter`, typically modestly faster: each adapted linear layer drops two small extra matmuls per forward call. Don't expect a dramatic speedup — for a single adapter on a small model, the adapter's own overhead is a small fraction of total compute; the point is that it's real and it's free to remove.

## How it works

A LoRA-adapted layer computes `y = xW^T + x(BA)^T \cdot scale`, holding `W` frozen and training only `B` and `A` (tiny, rank-`r` matrices). `merge_and_unload()` computes `W' = W + BA \cdot scale` once, writes it into the base layer, and returns the unwrapped model — from then on it's an ordinary dense linear layer computing `y = xW'^T` directly, mathematically identical to the adapted form for the same input, with no separate LoRA matmuls left to pay for.

## Honesty box

- **Merging is only a win when you're serving ONE fixed adapter.** If you need to swap between many different LoRA adapters per request (multi-tenant serving — a different fine-tune per customer, say), merging defeats the purpose: you'd need a separately merged full copy of the base model *per adapter*, multiplying storage by however many adapters you have. The unmerged form lets many adapters share one base model in memory — see S-LoRA below.
- `merge_and_unload()` is a one-way operation on the model object it's called on (this recipe works around that by deep-copying just for the Part 0 check, and merging the real model only once, after the adapter's own timing is already measured).
- This recipe measures generation cost, which reflects the adapter overhead once per generated token; a workload doing many short forward passes (classification-style, one pass per input) will see the same *relative* overhead, just over fewer total calls.
- The training-time cost recipe 02 addresses (optimizer memory) and the inference-time cost this recipe addresses (per-forward matmuls) are unrelated — a technique can win on one and be irrelevant to the other.

## Go deeper

Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021) · Sheng et al., *S-LoRA: Serving Thousands of Concurrent LoRA Adapters* (2023) — the production answer to "why you wouldn't always merge."

## Video hook

"Same fine-tune, same answers — I just folded the adapter back into the weights. Watch the per-token cost disappear."
