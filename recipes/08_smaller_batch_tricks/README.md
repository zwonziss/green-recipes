# 08 — Fitting the "impossible" run: the small-GPU stack

**The lesson:** batch size is a *math* choice, not a *memory* choice. Gradient accumulation decouples them; 8-bit optimizer states and checkpointing then shrink the two biggest remaining memory hogs. Stacked, they let a small GPU train what "shouldn't fit" — with the same loss curve.

## Run

```bash
python recipes/08_smaller_batch_tricks/run.py --repeats 3
```

Four attempts at effective batch 16, T=512 on SmolLM2-360M, each repeated `--repeats` times and aggregated: naive fp32 direct (likely 💥 on 12 GB — recorded as data), fp16 + accumulation (4×4), + 8-bit AdamW, + gradient checkpointing. Quality is checked with held-out perplexity (`--eval-batches`), and the script diffs the fp32-Adam and 8-bit-Adam loss trajectories (same data, same effective batch) to put a number on how close the two optimizers actually track.

## What to expect (rough, not yet measured — run it and replace this)

The fitting variants should produce near-identical held-out perplexity — that's the headline; the fp32-vs-8-bit-Adam loss diff should be small but not exactly zero (8-bit Adam approximates the optimizer state, it doesn't reproduce it). Peak VRAM should staircase downward: 8-bit Adam alone removes roughly three-quarters of optimizer-state memory (fp32 Adam costs 8 bytes/param — for 360M that's ~2.9 GB just in moments). Accumulation costs no memory and almost no time; checkpointing costs some time (see recipe 06).

## How it works

- **Accumulation:** run `accum` micro-batches, dividing each loss by `accum`, calling `optimizer.step()` only every `accum`-th backward. Gradients sum in the `.grad` buffers — mathematically the big batch, memory of the small one.
- **8-bit Adam:** the two moment tensors are stored block-wise quantized to 8 bits and dequantized only inside the update. Weights and gradients stay full precision. One import, one class swap.

## Honesty box

- Accumulation is *not* perfectly identical to a real big batch when the model has batch-statistics layers (BatchNorm) — irrelevant for transformers (LayerNorm/RMSNorm), worth knowing elsewhere.
- 8-bit optimizers are remarkably robust in practice, but they are an approximation; the loss column in your results is the evidence, not my word.
- Wall-clock is similar or slightly slower than one big batch — you're trading a little launch overhead for feasibility.

## Go deeper

Dettmers et al., *8-bit Optimizers via Block-wise Quantization* (2021) · Dettmers et al., *QLoRA* (2023) — this stack plus recipes 02+03.

## Video hook

Open on the fp32 OOM traceback. "Three tricks later, the same GPU trains the same model at the same batch size. Here are the receipts."
