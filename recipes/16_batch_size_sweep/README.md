# 16 — The batch-size sweep: biggest isn't always most efficient

**The lesson:** bigger batches amortize fixed overheads and usually raise tokens/sec — but "usually" isn't "always", and the batch size that maximizes throughput is not guaranteed to be the batch size that minimizes energy per token. This recipe finds both points instead of assuming the biggest batch that fits is the efficient choice.

## Run

```bash
python recipes/16_batch_size_sweep/run.py --repeats 3
```

Fixed-length (`--seq-len`, default 256) forward passes on SmolLM2-360M, batch size swept from 1 up through `--sizes` (default up to 128) until CUDA OOMs — recorded as data, sweep stops there. Every batch size is repeated `--repeats` times and aggregated; the script reports both the max-throughput batch size and the min-Wh/1k-token batch size at the end.

## What to expect (rough, not yet measured — run it and replace this)

`tokens_per_s` should climb steeply at first, then bend into diminishing returns as the GPU becomes compute- or memory-bandwidth-bound rather than launch-overhead-bound. `wh_per_1k_tok` typically improves (drops) alongside it for a while, then can flatten or even turn back up before the OOM ceiling — fixed per-call overhead is amortized quickly, and pushing further mostly just uses more memory for the same energy-per-token. Whether the two "best" batch sizes coincide is a real, hardware-specific answer this script measures rather than assumes.

## How it works

Small batches waste GPU capacity on per-kernel-launch and memory-access overhead relative to actual math; larger batches amortize that overhead across more work, which is why tokens/sec rises. But GPUs eventually hit a wall — memory bandwidth, kernel occupancy, or plain VRAM — where more batch stops buying proportionally more throughput. The efficiency curve (Wh per token) bends wherever the throughput curve does, just measured per unit of work instead of per unit of time.

## Honesty box

- **This tests forward-pass throughput, not full autoregressive generation.** Real serving also carries a growing KV cache per sequence in the batch (see recipe 10) — at large batch sizes, cache memory competes with activation memory for the same VRAM, a cost this static benchmark doesn't include.
- The OOM ceiling is specific to this model, sequence length, and GPU — it is not a universal number, and neither is the "best efficiency" batch size found here.
- Production servers use continuous/dynamic batching (vLLM, TGI) to get high throughput *and* low per-request latency simultaneously, by batching across requests that arrive at different times — this script's static, single-shot batch sweep is a simpler stand-in for that idea, not a replacement for it.
- If no power backend is detected on your machine (see recipe 00), only the throughput half of this story is available — the script says so rather than inventing an energy number.

## Go deeper

NVIDIA, *Inference Technical Overview* (batch size vs. GPU utilization) · Pope et al., *Efficiently Scaling Transformer Inference* (2022) for how batching interacts with memory-bound vs. compute-bound regimes.

## Video hook

"I kept doubling the batch size until the GPU gave up — the throughput chart and the energy-efficiency chart do NOT peak at the same point."
