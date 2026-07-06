# 12 — 2:4 structured sparsity: the pruning speedup recipe 05 said was a myth

**The lesson:** recipe 05 proved unstructured pruning saves bytes but not GPU time — dense kernels multiply zeros at full speed. This recipe zeroes exactly 2 of every 4 contiguous weights (a *pattern*, not free-for-all sparsity) and runs it through NVIDIA's semi-structured sparse tensor cores, which actually skip the zeros.

## Run

```bash
python recipes/12_structured_sparsity/run.py --repeats 3
```

**Needs an Ampere+ GPU** (RTX 30/40-series, A100 — compute capability ≥ 8.0) with a cuSPARSELt/CUTLASS-enabled torch build. On anything else (including the Colab T4 this cookbook otherwise targets) it prints a clear skip message and exits — that's the honest outcome, not a bug. Part 0 proves the sparse-kernel output matches a dense matmul on the same masked weights; Part 1 sweeps matrix size; Part 2 is the `GreenMeter`'d headline comparison at the largest size.

## What to expect (rough, not yet measured — run it and replace this)

On supported hardware, expect a real speedup that grows with matrix size — small matmuls may show little or no gain (kernel-launch overhead dominates), while large ones (the `--sizes` default goes up to 8192×8192) should show a measurable, real (`*`-marked) drop in ms/call. NVIDIA's own claims are up to ~2x on the largest supported shapes; this recipe measures your actual hardware rather than repeating that number.

## How it works

"2:4" means: in every group of 4 contiguous weights, exactly 2 must be zero. That fixed, predictable pattern lets Ampere+ tensor cores physically skip half the multiply-adds — unlike arbitrary (unstructured) sparsity, where the zeros are unpredictably scattered and a dense kernel can't exploit them at all. `torch.sparse.to_sparse_semi_structured()` packs a 2:4-masked tensor into the compressed format the hardware expects; the matmul then dispatches to the actual sparse kernel instead of a dense one.

## Honesty box

- **This benchmarks the kernel, not a trained model.** Weights here are random, then masked to fit the 2:4 pattern — no training happened, so there's no accuracy number to report. In practice you'd prune-then-fine-tune (or train with structured sparsity from the start) to recover accuracy at this pattern; that's a separate, much longer recipe of its own.
- **Small matrices may show no benefit or even a slowdown** — fixed kernel-launch/format-conversion overhead has to be amortized by enough FLOPs to be worth it.
- Requires both the right GPU generation AND a torch build with working cuSPARSELt/CUTLASS support — some CI runners, ROCm builds, and CPU-only installs will hit the graceful skip path in this script.
- 2:4 is a *fixed* ratio (50% sparse). It cannot be tuned to 30% or 70% like recipe 05's unstructured pruning — you get exactly this pattern or you get dense.

## Go deeper

Mishra et al. (NVIDIA), *Accelerating Sparse Deep Neural Networks* (2021) · [PyTorch semi-structured sparsity tutorial](https://pytorch.org/tutorials/prototype/semi_structured_sparse.html).

## Video hook

"Recipe 5 proved deleting half a model's weights doesn't make it faster. Here's the one rule that makes that untrue."
