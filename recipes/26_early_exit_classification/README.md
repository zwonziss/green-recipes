# 26 — Early exit: stop computing once you're confident

**The lesson:** not every input needs the same amount of compute. "This movie was amazing!" is obvious after a shallow look; a hedged or sarcastic review might need the whole network. A multi-exit classifier — a lightweight prediction head after every block, trained jointly — lets easy examples stop early and skip the remaining blocks' compute entirely, while hard examples still get the full depth.

## Run

```bash
python recipes/26_early_exit_classification/run.py --repeats 3
```

Trains a small custom classifier from scratch: a token embedding, mean-pooled, feeding a stack of 6 residual MLP blocks, each followed by its own linear classifier head, jointly trained on SST-2 (a lightweight, fully self-contained "depth" stack — not a splice into a pretrained transformer's internals, so the mechanism stays inspectable end to end). Part 0 forces the early-exit path through every block (an unreachable confidence threshold) and checks it matches the always-full-depth forward exactly. Part 1 sweeps the exit confidence threshold and tabulates accuracy vs. average layers run. Part 2 is the `GreenMeter`'d headline at `--headline-threshold` (default 0.9), evaluated **per example (batch size 1)** so the layers skipped are a real, measured compute saving.

## What to expect (rough, not yet measured — run it and replace this)

Part 0 should print a tiny (near-zero) diff. In Part 1, lower thresholds should show a lower average-layers-run (more examples exit after block 1 or 2) at some accuracy cost; higher thresholds should climb back toward the 6-layer baseline's accuracy as fewer examples dare to exit early. At the headline threshold, expect `early_exit`'s `avg_layers_run` meaningfully below 6, accuracy within a percentage point or two of `full_depth`, and wall time / energy dropping roughly in proportion to the layers actually skipped.

## How it works

Every block updates a single pooled vector; every block has its own exit head. Training minimizes the *average* cross-entropy loss across all 6 exits at once (a standard multi-exit training recipe), so every exit head — not just the last — learns to classify from whatever depth of representation it sees. At inference, `forward_early_exit` runs one block, checks that block's exit head's softmax confidence, and returns immediately if it clears the threshold — the remaining blocks are never called, not just masked out.

## Honesty box

- **Per-example (batch size 1) evaluation is what makes the compute savings real.** Batching examples with different exit points together would force the whole batch to wait for its slowest (deepest-running) member, erasing the saving — real systems that want both batched throughput *and* adaptive depth (PABEE, CALM) need specialized batching strategies this recipe doesn't implement.
- The confidence threshold is a hand-tuned knob trading accuracy for compute — this recipe sweeps a few values, but a real deployment should tune it against a held-out set for your actual accuracy budget, and re-validate if the input distribution shifts.
- This is a small custom classifier built to keep the exit mechanism fully inspectable, not a demonstration on a real pretrained transformer's internal layers (which is where DeeBERT/PABEE apply the same idea in production). The lesson — skip computation once you're confident — transfers; the specific numbers won't.
- Joint multi-exit training is a known tension in the literature: a shared trunk optimized for *every* exit at once can make the earliest exits slightly worse than training that same shallow depth alone would. Not explored further here.

## Go deeper

Xin et al., *DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference* (ACL 2020) · Zhou et al., *BERT Loses Patience: Fast and Robust Inference with Early Exit* (PABEE, NeurIPS 2020) · Schuster et al., *Confident Adaptive Language Modeling* (CALM, NeurIPS 2022) — applies the same idea token-by-token during generation.

## Video hook

"Same six-layer network, same weights — watch how many layers it actually bothers running once it's sure of the answer."
