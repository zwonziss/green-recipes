# 19 — Length-bucketed batching: padding is wasted compute

**The lesson:** a batch's width is set by its longest sequence, and the model computes full attention/FFN math over every padded position in that batch — masked out of the result, not free to run. Batch examples in the order they happen to arrive and short sentences get padded out to whatever long sentence they were unlucky enough to share a batch with. Sort by length first and every batch groups similar lengths, so the padding fraction collapses.

## Run

```bash
python recipes/19_length_bucketing/run.py --repeats 3
```

Loads 4000 SST-2 training sentences (variable length), tokenizes them, and runs a forward pass over all of them in two groupings: `random_batching` (the dataset's own shuffled order — "just batch requests as they arrive") and `length_bucketed` (sorted by token length before batching). Part 0 proves one example scores identically whether it's alone or padded inside a batch. The headline receipt reports wall time, energy, accuracy, and the measured padding ratio for both.

## What to expect (rough, not yet measured — run it and replace this)

Accuracy should be identical (or differ only by float noise) between the two groupings — padding doesn't change any individual example's answer. The `padding_ratio` column should be visibly lower for `length_bucketed` (padded tokens as a fraction of total tokens processed), and wall time / energy should drop by roughly the same fraction, since the wasted padded positions are exactly what's being cut.

## How it works

Each batch's tensor width is `max(len(s) for s in batch)`. Random batching mixes short and long sentences together, so the "waste" (padding) in a batch is bounded by how different its members' lengths are. Sorting the whole dataset by length before slicing into batches means every batch's members are already close in length, so `max(len(s))` is barely bigger than most members — the padding shrinks to nearly zero without changing a single computed answer, because `attention_mask` already made pad positions invisible to the result.

## Honesty box

- This only works **offline / in a batch-inference or batch-training job**, where you have the next N examples' lengths available upfront to sort. A live single-request server (recipe 21's territory) doesn't get to choose who it batches with.
- The benefit scales with how skewed your length distribution is and how big your batches are — a dataset where every example is nearly the same length gets almost nothing from this; a dataset with a long tail of short and long examples benefits the most.
- Bucketing by length changes which examples land in the same batch, which can subtly change batch-norm-style statistics or gradient noise if used during *training* (not an issue here — this recipe measures inference only).
- Real training pipelines that bucket by length still need to reshuffle *which bucket* comes next each epoch, or the model sees "all short examples, then all long examples" as a suspicious curriculum — bucket the batches, then shuffle the batch order.

## Go deeper

Khomenko et al., *Accelerating Recurrent Neural Network Training via Two Stage Classes and Parallelization* — an early reference for length-bucketed batching in seq2seq training; most modern training frameworks (`transformers`' `LengthGroupedSampler`, NVIDIA's `bucketing` in NeMo) implement exactly this idea by default.

## Video hook

"Same sentences, same model, same answers — I just sorted them by length before batching. Watch the padding column."
