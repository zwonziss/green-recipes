# 14 — Energy-aware hyperparameter search: stop training losers

**The lesson:** a naive grid search trains every candidate to completion before comparing them — most of that compute is spent proving mediocre configs are mediocre. An ASHA-lite successive-halving search checks in early, kills the bottom half, and only lets survivors keep training.

## Run

```bash
python recipes/14_energy_aware_hpo/run.py --repeats 3
```

Six learning-rate candidates for SmolLM2-135M, `--n-rungs` rungs of `--steps-per-rung` steps each (default 3×6=18 steps max per config). **naive_full** trains all six to the full step count. **asha_lite** trains all six for rung 1, keeps the top half by held-out perplexity, trains those for rung 2, keeps the top half again, and finishes only the survivor(s). Both strategies are repeated `--repeats` times (fresh models, same seed/data — repeats target timing/energy noise) and their *total* energy across every config and rung is summed and aggregated.

## What to expect (rough, not yet measured — run it and replace this)

`asha_lite` should find the same or a very close best learning rate to `naive_full` (bad learning rates tend to look bad early, not just late) while training roughly 11 "config-rung" units of work total instead of naive's 18 — order of 30–40% less total energy for a similar `best_val_ppl`. Watch whether that energy delta clears the `*` bar in `compare()`'s table, and whether `best_val_ppl` stays within noise of each other — the whole point is losing (almost) nothing on the outcome.

## How it works

"ASHA" (Asynchronous Successive Halving Algorithm) trains every candidate for one short "rung", ranks them on a cheap proxy of final quality (here: held-out perplexity, not training loss), discards the worse half, and repeats with progressively longer rungs for the shrinking survivor set. The bet: a config that's clearly worse after 1/3 of training is very unlikely to become the best by the end, so paying to find out for certain is usually wasted energy.

## Honesty box

- **This is ASHA-*lite*.** Between rungs it checkpoints and reloads model *weights* but starts each rung with a **fresh optimizer** (Adam momentum reset to zero) rather than also checkpointing optimizer state, which real ASHA implementations do. This is a simplification for a ~150-line script, not a claim that this matches production hyperparameter-search libraries (Ray Tune, Optuna) exactly.
- Successive halving can occasionally prune a config that would have overtaken the leader later (a slow starter) — that's a real, known risk of the technique, not something this script protects against. The savings are a bet, not a guarantee, and the bet is usually — not always — right.
- Candidates and rung schedule here are small and fixed for a quick demo; real hyperparameter searches often sweep many more dimensions (batch size, warmup, weight decay) where the energy savings compound further.
- Perplexity here comes from a handful of held-out batches — a cheap, noisy proxy by design (that's the point: rung decisions must be cheap, or the search itself becomes the energy cost).

## Go deeper

Li et al., *A System for Massively Parallel Hyperparameter Tuning* (ASHA, 2020) · Jamieson & Talwalkar, *Non-stochastic Best Arm Identification and Hyperparameter Optimization* (2016).

## Video hook

"Six configs walk into a training run. Three get cut after round one. Here's how much energy that saved — and whether the 'winner' actually stayed the winner."
