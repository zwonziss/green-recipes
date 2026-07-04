# 06 — Gradient checkpointing: trade compute for memory (then win it back)

**The lesson:** activations, not weights, dominate training memory. Checkpointing throws most of them away and recomputes them during backward — ~30% more compute for a huge VRAM drop. The advanced move: spend that freed VRAM on a bigger batch and come out *ahead* on throughput.

## Run

```bash
python recipes/06_gradient_checkpointing/run.py --repeats 3
```

Three fp32 training variants of SmolLM2-360M at T=512: baseline B=8, checkpointed B=8, checkpointed B=16 — each retrained `--repeats` times (same seed — repeats target timing noise, not training randomness) and aggregated. Because checkpointing recomputes activations exactly rather than approximating them, the script also diffs baseline-B=8's and checkpointed-B=8's per-step losses (same data, same batch size) and reports the max difference as a numerical sanity check, not just a claim in prose.

## What to expect (rough, not yet measured — run it and replace this)

On a 12 GB card the fp32 baseline may simply OOM — the script catches it and records the 💥 as data. Where it fits (T4 16 GB): checkpointing should cut peak VRAM substantially while adding roughly 20–35% wall time at the same batch size (the `compare()` table should mark that slowdown `*`, not `~`). The B=16 checkpointed run often beats the baseline's tokens/sec despite recomputation. The baseline-vs-checkpoint loss diff should be tiny (float noise), confirming this is a memory/compute trade, not a different optimization.

## How it works

Normally every layer's activations are kept alive until backward needs them — memory grows with depth × batch × sequence. With checkpointing, only activations at layer boundaries are stored; everything in between is recomputed on the fly during backward (one extra forward pass, ~1/3 more FLOPs). In HF it's two lines:

```python
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
```

## Honesty box

- More FLOPs = more energy per step. Checkpointing is only "green" when it enables something (bigger batch, longer context, a model that otherwise wouldn't fit) — the third run is the honest accounting of that.
- `use_cache=False` matters: the KV cache is useless during training and fights with checkpointing.
- `use_reentrant=False` is the modern variant; the old reentrant one has sharp edges with frozen params (e.g. LoRA).

## Go deeper

Chen et al., *Training Deep Nets with Sublinear Memory Cost* (2016).

## Video hook

"My GPU crashed at batch 8. Two lines later it trains at batch 16, faster than before." Show the OOM traceback, then the receipts.
