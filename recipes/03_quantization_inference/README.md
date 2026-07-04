# 03 — Quantized inference: smaller, cheaper… and how much dumber?

**The lesson:** weights don't need 16 bits at inference. 4-bit quantization cuts VRAM ~4× for a measurable-but-small quality cost — and this recipe puts a *number* on that cost instead of a vibe.

## Run

```bash
python recipes/03_quantization_inference/run.py --repeats 3
```

Loads SmolLM2-1.7B three ways — fp16, int8 (LLM.int8), 4-bit NF4 — and measures weights-VRAM, generation tokens/sec, **Wh per 1,000 tokens** (the generation loop is repeated `--repeats` times on the same loaded weights and aggregated into mean ± std), and perplexity on a fixed Shakespeare slice.

## What to expect (rough, not yet measured — run it and replace this)

Weights VRAM roughly: fp16 ~3.4 GB → int8 ~1.9 GB → nf4 ~1.2 GB. Perplexity should rise only slightly (order of 1–5% for nf4). **Plot twist:** int8 is often *slower* than fp16 — LLM.int8's outlier handling adds overhead — while nf4 usually lands near fp16 speed; check whether the `compare()` table marks that slowdown `*` (likely real) or `~` (could be noise) before repeating the claim. Smaller ≠ automatically faster.

## How it works

- Quantization stores each weight in fewer bits plus per-block scale factors; compute still happens in fp16 (weights are dequantized on the fly).
- **NF4** ("NormalFloat4") places its 16 levels where normally-distributed weights actually live, which is why 4 bits work as well as they do. Double quantization even quantizes the scale factors.
- Only `Linear` layers get quantized — embeddings and norms stay fp16, which is why VRAM doesn't shrink by exactly 4×.

## Honesty box

- Perplexity on Shakespeare is a narrow probe; degradation can be larger on reasoning-heavy tasks. It's still infinitely better than not measuring.
- bitsandbytes needs CUDA — this recipe won't run on CPU.
- This is *inference* quantization. Training on top of 4-bit weights = QLoRA (combine with recipe 02).

## Go deeper

Dettmers et al., *LLM.int8()* (2022) · Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023).

## Video hook

"I made the model 3× smaller — here's exactly how much dumber it got." Show the ppl column. People argue about this constantly with zero numbers.
