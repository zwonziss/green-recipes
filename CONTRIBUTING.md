# Contributing a recipe

The whole value of this repo is its consistency. A recipe is accepted when it satisfies **the contract**:

1. **One technique, one file.** A single `run.py`, ideally under ~180 lines, readable top to bottom without visiting other recipes. Only `common/` may be imported.
2. **Runs on a free Colab T4** in under ~10 minutes with default arguments, and degrades gracefully (skip with a message, don't crash) on missing capabilities like bf16.
3. **Has a baseline.** Every technique is measured *against* the plain version, with `GreenMeter`, and ends with `compare(...)`.
4. **Saves receipts.** Call `save_result(...)` for each variant so `tools/build_table.py` picks it up. Mark exactly one variant `is_baseline=True`.
5. **README with an Honesty box.** Sections: the one-line lesson, Run, What to expect, How it works (≤10 lines), **Honesty box** (where the technique fails, misleads, or gets oversold — mandatory), Go deeper (1–2 papers), Video hook.
6. **No fabricated numbers.** "What to expect" gives rough ranges clearly labeled as expectations; real numbers only enter via committed `results/*.json`.

Style: standard library `argparse`, no config frameworks, seeds fixed with `common.data.set_seed(42)`, warmup steps outside the metered block, `gc.collect()` + `empty_cache()` + a 2 s cooldown between variants. Support `--repeats` (default 3) and aggregate with `common.greenmeter.aggregate_trials()` so `compare()` reports mean ± std, not one noisy number. If the technique makes a quality claim, back it with a held-out metric (`common.eval.lm_perplexity` / `classification_accuracy`) computed outside the metered block, not the raw training loss.

Open an issue first if you're unsure whether an idea fits — "it makes the resource cost of a common practice visible and measurable" is the test.
