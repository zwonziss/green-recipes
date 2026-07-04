# 02 — LoRA vs full fine-tuning

**The lesson:** fine-tuning doesn't require touching every weight. Freeze the model, inject tiny low-rank adapters, and the optimizer's memory bill — and your checkpoint size — collapses.

## Run

```bash
python recipes/02_lora/run.py --rank 16 --repeats 3
```

Two conditions, same data and steps: full fine-tuning of SmolLM2-135M vs LoRA (r=16) on the attention projections. Each condition is retrained `--repeats` times (same seed — repeats target timing/energy noise, not training randomness) and aggregated; quality is a held-out perplexity (`--eval-batches`), not the raw training loss.

## What to expect (rough, not yet measured — run it and replace this)

Trainable parameters drop from ~135M to ~1–2M (≈1%). Optimizer-state VRAM shrinks from ~1 GB to ~10 MB, so peak VRAM falls noticeably even on this tiny model. Time per step is similar or slightly better. On disk: full checkpoint ~270 MB vs adapter ~5–10 MB. Held-out perplexity lands close to full FT for a short run like this.

## How it works

For a weight matrix `W`, LoRA learns `W + (α/r)·B·A` where `A` is `r×d` and `B` is `d×r` with rank `r ≪ d`. `W` stays frozen — no gradients, no Adam moments (which cost 8 bytes/param in fp32!). Only `A` and `B` train, and only they get saved. At inference you can merge `BA` into `W`, so there's **zero** extra latency.

## Honesty box

- On a 135M model the *absolute* savings are small; the point is the ratio. On a 7B model, full FT with Adam wants ~112 GB of GPU memory — LoRA is the difference between "impossible at home" and "fits on your 3060" (especially combined with recipe 03's quantization = QLoRA).
- The two runs use different learning rates (5e-5 vs 2e-4) because that's what each method actually needs — comparing at one shared lr would sandbag one of them.
- LoRA slightly restricts what the model can learn; for aggressive domain shifts, full FT can still win.

## Go deeper

Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021) · Dettmers et al., *QLoRA* (2023).

## Video hook

Hold up "the whole fine-tune" on screen: a 6 MB file. Then show the 270 MB alternative downloading… slowly.
