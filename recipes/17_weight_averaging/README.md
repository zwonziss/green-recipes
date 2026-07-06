# 17 — Stochastic Weight Averaging: an ensemble's accuracy bump, one run's energy bill

**The lesson:** averaging a model's own last few checkpoints (SWA) recovers a meaningful slice of what a real 2-model ensemble buys you — using the SAME training run's energy instead of a second one. This recipe measures both price tags, not just both accuracies.

## Run

```bash
python recipes/17_weight_averaging/run.py
```

Trains BERT-tiny (4.4M params) on an SST-2 subset. For each seed pair `(seed, seed+1)` in `--seeds`: run A trains normally, but its last `--swa-frac` (default 30%) of steps also update a running `AveragedModel` at a constant `SWALR` learning rate — one physical training run, two checkpoints (`baseline_final`, `swa_averaged`) that share the *same* measured energy. Run B is an independent second training (different seed) whose final checkpoint is ensembled with run A's via averaged softmax probabilities, reported as `ensemble_2x` with energy = run A + run B combined.

## What to expect (rough, not yet measured — run it and replace this)

`swa_averaged` should beat or match `baseline_final`'s validation accuracy by a small amount, at *identical* `wall_s`/`gpu_energy_wh` (they're the same run — the compare table's delta on those columns should read exactly 0%). `ensemble_2x` should do at least as well as `swa_averaged`, often better, but at roughly double the energy of either single run. Whether SWA's accuracy-per-energy-spent beats the ensemble's is the number worth actually looking at, not assuming.

## How it works

`torch.optim.swa_utils.AveragedModel` maintains a running (equal-weighted) average of a model's parameters, updated after each step once averaging starts; `SWALR` holds the learning rate at a constant (rather than decaying) value during that window, which is what makes the averaged checkpoints diverse enough for averaging to help instead of just blurring one point. Averaging several "nearby" minima this way approximates part of what an ensemble of independently-trained models gets from averaging genuinely *different* minima — cheaper, because it reuses steps you were already paying for.

## Honesty box

- This model has no BatchNorm (transformers use LayerNorm, which carries no running statistics), so `torch.optim.swa_utils.update_bn` — normally a required last step for SWA — is correctly skipped here. **If you apply this to a CNN with BatchNorm, skipping `update_bn` will make SWA look artificially worse than it is.**
- This is the cheapest possible ensemble (2 models). Larger ensembles (5–10 models) typically do better still, but their energy cost scales linearly with count — SWA's cost does not scale with "how many checkpoints" the way a real ensemble's does.
- SWA's benefit depends on the LR schedule during the averaging window; naively averaging checkpoints from a normally-decaying schedule (no `SWALR`) helps less than what's measured here.
- Model and dataset are intentionally tiny for a fast repo demo — the accuracy deltas here are illustrative, not benchmark numbers.

## Go deeper

Izmailov et al., *Averaging Weights Leads to Wider Optima and Better Generalization* (2018) · PyTorch docs, *Stochastic Weight Averaging*.

## Video hook

"Same training run. Same energy bill. I just averaged the last few checkpoints instead of throwing them away — here's how close that gets to training a whole second model."
