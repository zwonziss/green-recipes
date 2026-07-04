# 01 — Mixed precision: the precision ladder

**The lesson:** most training math doesn't need 32 bits. Dropping to 16-bit halves activation memory and unlocks tensor cores — roughly the same loss for a fraction of the time and energy.

## Run

```bash
python recipes/01_mixed_precision/run.py --repeats 3
```

Runs every mode your GPU supports: `fp32` → `tf32` → `fp16` → `bf16`, training SmolLM2-135M on tiny-shakespeare for identical steps. Each mode is retrained `--repeats` times (same seed on purpose — repeats target GPU/OS timing noise, not training randomness) and aggregated into one receipt; quality is checked with `--eval-batches` of **held-out** perplexity, not just the last few training losses.

## What to expect (rough, not yet measured — run it and replace this)

On an RTX-3060-class card, expect fp16/bf16 to be around 1.5–2.5× faster than fp32 with ~30–40% lower peak VRAM and proportionally less energy, at essentially the same held-out perplexity (the `compare()` table will mark real timing/energy deltas with `*` vs noise with `~`). tf32 lands in between and costs one line of code. On a Colab T4: no bf16, no tf32 (Turing) — the script detects this and skips them.

## How it works

- `torch.autocast` runs matmuls/convolutions in half precision while keeping loss-sensitive ops (softmax, norms) in fp32. Master weights stay fp32.
- **fp16** has a tiny exponent range → gradients can underflow to zero → `GradScaler` multiplies the loss up before backward and unscales before the optimizer step.
- **bf16** has fp32's exponent range (just less mantissa) → no scaler needed. This is why everyone who *can* use bf16, does.
- **tf32** isn't autocast at all — it's a hardware mode where fp32 matmuls internally use 10-bit mantissas: `torch.backends.cuda.matmul.allow_tf32 = True`.

## Honesty box

- fp16 occasionally destabilizes real trainings (loss spikes, NaNs). bf16 is the boring, safe choice on Ampere+.
- A 135M model can't show tensor-core gains at their full glory; on 1B+ models the gap widens.
- First run downloads the model (~270 MB) — that network transfer has a footprint too.
- Repeats reuse the same seed on purpose, so held-out perplexity barely moving across trials is expected, not evidence of a bug — it's timing and energy that should show the run-to-run spread.

## Go deeper

Micikevicius et al., *Mixed Precision Training* (2017) — the paper that made this standard.

## Video hook

Four progress bars racing on screen, four receipts at the end: "same model, same loss, 40% less CO2 — one line of code."
