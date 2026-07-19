# 28 — Shrink the output vocabulary to your actual domain

**The lesson:** a general-purpose LM's final layer projects every hidden state onto its *full* vocabulary — tens of thousands of tokens — to get next-token logits. It's one of the few parts of a forward pass whose cost scales directly with vocab size. Deploy that model for one narrow domain and most of that vocabulary is dead weight: tokens the domain never uses, whose logits get computed and thrown away on every single generated token, every request.

## Run

```bash
python recipes/28_embedding_pruning/run.py --repeats 3
```

Scans a domain corpus (tinyshakespeare, standing in for "your actual production traffic") to find which vocabulary tokens it actually uses, then builds a smaller output head (`lm_head`) containing only those rows — an exact `index_select`, not an approximation. Part 0 proves the pruned head's logits match the full head's logits restricted to the same rows. The headline receipt times a forward pass over domain batches with the full head vs. the pruned one.

## What to expect (rough, not yet measured — run it and replace this)

Part 0's logit diff should be tiny (float noise). The printed "domain corpus uses X% of vocabulary tokens" is usually a small fraction — a few percent — of a general-purpose tokenizer's vocabulary for any single narrow domain. `pruned_vocab_head` should show measurably lower time/energy than `full_vocab_head`, though the *whole-model* speedup will be smaller than the raw vocab reduction ratio, since attention and feed-forward layers (untouched here) still cost what they always cost — see How it works and the Honesty box.

## How it works

`model.get_output_embeddings()` returns the model's final linear projection (`lm_head`, shape `[hidden_size, vocab_size]`). `used_ids = sorted(set(tokenize(domain_corpus)))` finds which of those `vocab_size` rows the domain ever asks for. A new, smaller `nn.Linear` is built by `index_select`-ing just those rows, and swapped in via `model.set_output_embeddings(...)` — a documented, architecture-agnostic `PreTrainedModel` method, so this works without touching any model-specific internals. **Only the output projection shrinks** — input embeddings are left fully intact, since the model still needs to be able to *read* any token (a comment, a rare name, a typo) even if it will only ever *emit* tokens from the retained set.

## Honesty box

- **This makes the model strictly unable to ever output any token outside the retained set.** If production traffic later needs a token missed by your domain snapshot (a number, a code fragment, another language, unusual punctuation), generation for that token fails or silently misbehaves — no fallback/UNK strategy is shown here. Validate retained-vocabulary coverage against real traffic, with a safety margin, before relying on this in production.
- The end-to-end speedup is bounded by how much of total FLOPs the output head represents. For this 135M-param model at a modest sequence length, the head is a meaningful slice; for longer sequences or deeper models, attention and feed-forward compute (which scale with sequence length and depth) dominate and this trick's share of the total savings shrinks.
- If input and output embeddings are tied (`config.tie_word_embeddings`), this recipe deliberately leaves the tie alone and only swaps the *output* module — the input embedding table stays the original, full-size one. A model with separately-learned input/output embeddings could prune both, roughly doubling the parameters saved.
- This prunes against ONE fixed offline corpus snapshot. A production system needs a versioned, monitored process for updating the retained vocabulary as domain language drifts — not addressed here.

## Go deeper

Grave et al., *Efficient softmax approximation for GPUs* (2017) — adaptive softmax, a closely related idea (partition the vocabulary by frequency to cut the output-projection cost) built into the training loop rather than applied post hoc · Devlin, *Sharp Models on Dull Hardware: Fast and Accurate Neural Machine Translation Decoding on the CPU* (2017) — vocabulary selection for fast NMT decoding in production.

## Video hook

"This model can technically say fifty thousand different tokens. My deployment only ever needs about a thousand of them. Watch what happens when I stop paying for the other forty-nine thousand."
