# 07 — Flash attention: the O(n²) memory wall

**The lesson:** textbook attention materializes a T×T score matrix — memory explodes quadratically with sequence length. Fused kernels (FlashAttention / SDPA) compute *mathematically identical* attention in tiles that live in on-chip SRAM, so memory grows linearly and speed jumps.

## Run

```bash
python recipes/07_flash_attention/run.py --repeats 3
```

Part 0: a single deterministic forward pass checks that naive attention and SDPA actually agree numerically, before either's speed is trusted. Part 1: hand-written attention vs `F.scaled_dot_product_attention`, fwd+bwd, seq 512→8192, each shape run `--repeats` times and aggregated, until naive hits the wall (OOM recorded as a result, not a crash). Part 2: SmolLM2-135M trained with `attn_implementation="eager"` vs `"sdpa"`, also repeated, with a held-out perplexity check so "faster" isn't quietly also "worse".

## What to expect (rough, not yet measured — run it and replace this)

Part 0's max output difference should be tiny float noise (order 1e-3 or less in fp32) — that's the proof the two paths compute the same math. Naive memory should visibly quadruple every time the sequence doubles, then OOM outright around seq 8192 on a 12 GB card while SDPA cruises through. At long sequences expect several-× speedups; the real-model gap at T=1024 is smaller but free, with near-identical held-out perplexity between `eager` and `sdpa`.

## How it works

The trick is *tiling + online softmax*: process K/V in blocks, keep running max and running sum, and never write the T×T matrix to slow GPU DRAM. Attention is memory-bandwidth-bound, so avoiding those reads/writes is where both the speed and the memory win come from. The output is exact — this is not an approximation.

## Honesty box

- Your GPU matters: FlashAttention kernels need Ampere+ (RTX 3060 ✓). A Colab T4 falls back to SDPA's memory-efficient backend — no flash, but still a massive win over naive; the script tells you which path you're on.
- Nobody ships the naive version in real code; part 1 exists to make the *reason* for fused kernels visceral.
- SDPA is one keyword argument in HF. There is no excuse not to use it.

## Go deeper

Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (2022).

## Video hook

A memory-usage bar chart growing 1×, 4×, 16×… then bursting off-screen — next to SDPA's flat little bars. "Same math. One of them respects physics."
