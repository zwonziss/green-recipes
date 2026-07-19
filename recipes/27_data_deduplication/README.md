# 27 — Duplicate data wastes training compute

**The lesson:** a scraped or aggregated corpus routinely contains exact and near-exact duplicates — mirrors, syndicated content, templated boilerplate. Training on it as-is spends real gradient steps re-teaching the model something it already learned from the first copy. At a *fixed* compute budget, deduplicating first means every step spends its budget on content the model hasn't already seen.

## Run

```bash
python recipes/27_data_deduplication/run.py --seeds 42,43,44
```

Builds a synthetic "raw scraped" corpus: 1200 unique SST-2 sentences, each tripled and shuffled together (exact duplicates — the simplest, fully verifiable case). Part 0 hash-dedups it and confirms it recovers exactly the original 1200 unique examples. Both `raw_duplicated` and `deduped` then train **the same number of gradient steps** (one full pass over the tripled corpus) — `deduped` cycles its smaller, duplicate-free pool to fill the same step budget — and both are scored on the real, untouched SST-2 validation split.

## What to expect (rough, not yet measured — run it and replace this)

Part 0 should report ~67% of the raw corpus as duplicates (2 of every 3 copies) and recover exactly 1200 unique examples. `wall_s` / `gpu_energy_wh` should land close together between the two conditions — same steps, same batch size, same model, so the energy bill is essentially identical by construction. `val_accuracy` is where the story should show up: `deduped`, seeing ~3 real epochs of unique signal in the same step budget, should reach equal-or-better validation accuracy than `raw_duplicated`, which spent a third of its steps re-training on content already seen twice more.

## How it works

`raw_duplicated` = 1200 unique (sentence, label) pairs repeated 3x, shuffled once so the copies scatter through the corpus rather than sitting in adjacent blocks (mimicking a real crawl, where a document and its mirror land far apart, not next to each other). `deduped` = the same corpus after an order-preserving hash-set pass keyed on exact sentence text. Both train for `len(raw_duplicated) // batch_size` steps — `deduped` simply wraps around and reshuffles its smaller pool to keep training that long, so both conditions spend the identical energy/time budget; only what fills each step differs.

## Honesty box

- **This uses exact string-hash duplicates for a clean, fully verifiable demo.** Real-world near-duplicates (boilerplate with a few words changed, syndicated articles, templated pages) need fuzzier detection — MinHash/LSH or embedding-similarity clustering — that this simple hash pass will completely miss.
- The synthetic 3x-uniform duplication here is more extreme and more evenly spread than most real corpora, where duplication is long-tailed (a few documents copied thousands of times, most never repeated). See Lee et al. below for measured duplication rates in real web-scale corpora.
- **Energy/time cost is essentially unchanged between the two conditions by design** — the win is entirely "a better outcome for the same power bill," not a wall-clock speedup. A corpus with genuinely no duplicates gets zero benefit from this technique (the same caveat recipe 25's response cache carries: no repetition, no win).
- At this small scale (BERT-Tiny, a few real epochs) the memorization-vs-generalization effect of duplicates is more muted than at LLM-pretraining scale, where deduplication has been shown to also reduce verbatim memorization/regurgitation of training text, not just improve downstream accuracy.

## Go deeper

Lee et al., *Deduplicating Training Data Makes Language Models Better* (ACL 2022) · Broder, *On the Resemblance and Containment of Documents* (1997) — the MinHash technique real dedup pipelines (CCNet, GPT-3's fuzzy dedup) build on.

## Video hook

"Same model, same number of training steps, same energy bill. One corpus had a third of its content copy-pasted. Watch the validation accuracy."
