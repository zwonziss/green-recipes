# 05 — Pruning: deleting weights (and busting a myth)

**The lesson:** networks are massively over-parameterized — you can zero out half the weights of a trained model with barely any accuracy loss. **The myth:** that this makes inference faster. On a GPU, it doesn't.

## Run

```bash
python recipes/05_pruning/run.py --repeats 3
```

Global L1 magnitude pruning on DistilBERT-SST2's Linear layers at 0/30/50/70/90% sparsity. Per level: validation accuracy, ms/example, energy, and gzip'd checkpoint size (zeros compress beautifully — that part is real). The forward pass is repeated `--repeats` times and aggregated; accuracy is deterministic at a given sparsity (no dropout, no randomness once the weights are fixed) so only timing and energy should show spread.

## What to expect (rough, not yet measured — run it and replace this)

Accuracy typically holds near baseline through ~50%, degrades gently around 70%, and falls off a cliff by 90%. `ms_per_example` stays **flat across every row** — the punchline, and the `compare()` table's `*`/`~` markers on that column should mostly read `~` (no real change) rather than `*`. The gzip'd checkpoint, meanwhile, shrinks steadily with sparsity.

## How it works

`torch.nn.utils.prune.global_unstructured` ranks *all* Linear weights by |magnitude| across the whole network and masks the smallest ones (global beats per-layer: some layers can afford to lose more). `prune.remove` then bakes masks into the tensors. The zeros are still stored, moved, and multiplied as ordinary fp32 numbers — which is exactly why nothing gets faster.

## Honesty box

- No fine-tuning after pruning here — that's deliberate, to show raw robustness. Prune-then-finetune (or iterate) recovers accuracy at much higher sparsities.
- To convert sparsity into actual speed you need **structure**: remove whole neurons/heads (the matrices physically shrink), use NVIDIA's 2:4 semi-structured sparsity (Ampere+ tensor cores, ~2× on supported ops), or run on a sparse-aware engine.
- The gzip trick is a fair proxy for "download/storage footprint", not for runtime memory.

## Go deeper

Han et al., *Learning both Weights and Connections* (2015) · Frankle & Carbin, *The Lottery Ticket Hypothesis* (2018).

## Video hook

"I deleted 70% of this model and it barely noticed — but here's the graph nobody shows you." Cut to the perfectly flat latency line.
