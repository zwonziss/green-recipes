# 21 — Batch your requests: N prompts at once beats a for-loop

**The lesson:** a for-loop over N independent generation requests pays the GPU's fixed per-call overhead N times and only ever gives it one sequence's worth of matrix multiplies to chew on at once. Pad the N prompts into a single left-padded batch and call `generate()` once, and the same GPU kernel launches process N sequences' arithmetic together — the "batching dividend" every real inference server (vLLM, TGI, TensorRT-LLM) is built around.

## Run

```bash
python recipes/21_batched_decoding/run.py --repeats 3
```

64 short, distinct prompts. Part 0 proves prompt #0's greedy output is identical whether it runs alone or packed into a left-padded batch of 8. Part 1 sweeps prompt count (1, 8, 32 by default) and tabulates wall time for serial calls vs. one batched call. Part 2 is the `GreenMeter`'d headline comparison at `--headline-batch` (default 32) prompts.

## What to expect (rough, not yet measured — run it and replace this)

Part 0 should print `True`. In Part 1, at `n=1` both paths issue the same single `generate()` call, so they should cost about the same; as `n` grows, `serial`'s time should scale close to linearly with `n` while `batched`'s grows much more slowly (until the GPU's compute is saturated, after which batching's advantage per additional prompt shrinks). The speedup column should visibly grow with `n`.

## How it works

`tok.padding_side = "left"` left-pads every prompt in the batch to the longest one's length, so all sequences' "next token to generate" lines up in the same column. `generate()` uses the `attention_mask` to both exclude pad positions from attention and derive correct `position_ids` for the real tokens — the same public path used everywhere else in this repo, no manual cache surgery needed (contrast recipe 18's KV-cache reuse, which does splice `past_key_values` by hand). Both variants force `min_new_tokens == max_new_tokens` so every row generates exactly the same number of tokens — isolating the batching question from early-stopping (recipe 20's territory).

## Honesty box

- This is **static batching**: every prompt in the batch waits for the slowest one to finish, and all rows generate the same fixed length. A request that would've finished early still burns compute matching the batch's pace. Continuous/in-flight batching (vLLM, TGI) solves this by pulling finished sequences out and inserting new ones mid-flight — not shown here.
- VRAM grows roughly linearly with batch size (and with sequence length) — batching trades memory for throughput up to whatever your GPU can hold (recipe 16's batch-size sweep territory), then OOMs.
- The speedup is bounded by how much idle GPU capacity a single sequence leaves on the table — a model/hardware pair that's already compute-bound at batch size 1 (a very large model on a small GPU) will see far less benefit than a small model on a big GPU, which is exactly this recipe's setup.
- Building the padded batch itself (tokenizing, padding to the max length) has a small real cost, included in the timed block here — not hand-waved outside it.

## Go deeper

Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI 2022) — introduces continuous batching · Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023, vLLM) — the production-grade version of "keep the GPU fed."

## Video hook

"Same 32 prompts, same model, same answers — one for-loop, one batch call. Watch the clock."
