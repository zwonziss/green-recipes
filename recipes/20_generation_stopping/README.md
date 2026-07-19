# 20 — Stop generating when you're done

**The lesson:** `max_new_tokens=256` is a ceiling, not a target — but code that also sets `min_new_tokens` (or the legacy `min_length`) to the same value, "just in case," silently disables early stopping and forces every single response to pay for the full ceiling. Letting generation notice it's finished (EOS, or a stop string) costs nothing and routinely cuts the token bill by most of that ceiling.

## Run

```bash
python recipes/20_generation_stopping/run.py --repeats 3
```

Eight short Q&A-style prompts, generated two ways: `forced_full_length` (`min_new_tokens == max_new_tokens`, generation cannot stop early) and `stop_on_newline` (`stop_strings="\n"`, generation stops the moment it emits a newline or EOS). Part 0 proves the stopped output is an exact token-for-token prefix of the forced one under greedy decoding. Part 1 tabulates tokens generated per prompt for both. Part 2 is the `GreenMeter`'d headline over all 8 prompts.

## What to expect (rough, not yet measured — run it and replace this)

Part 0 should print `True`. In Part 1, `forced` should show `--max-new-tokens` (default 200) for every prompt; `stopped` should show far fewer — how few depends on how quickly this small base model reaches a newline after a short Q&A-style answer, but even a modest early stop (tens of tokens instead of 200) should show up clearly as a percentage saved. Part 2's `avg_new_tokens` and energy/time columns should track that same ratio.

## How it works

`generate()`'s decoding loop checks a stopping condition after every new token: has EOS been produced, or (with `stop_strings=`) has the decoded text so far ended in one of the given strings? Setting `min_new_tokens` equal to `max_new_tokens` suppresses that check entirely until the ceiling is hit — a real, common footgun (usually added to avoid empty/too-short outputs during development, then never revisited). Removing it and giving the model a real stop signal (`stop_strings=`, requires passing `tokenizer=` so `generate()` can decode incrementally to check) lets each response cost only what it actually needed.

## Honesty box

- The stop condition here (`"\n"`) is a convenience for short, single-line answers. A task that legitimately needs long, multi-paragraph output would truncate wrongly with this same stop string — the fix is to pick a stop condition that matches your task's real completion signal, not to remove stopping altogether.
- `stop_strings` matches on **decoded text**, not token IDs — a very short or generic stop string (a single common punctuation mark, say) can trigger mid-thought on unrelated content. Check your actual outputs, don't just trust the token-count savings.
- This recipe generates one prompt at a time. Combine this with batched generation (recipe 21) and you hit a real complication: different sequences in a batch finish at different steps, and a naive batched loop still pays for the slowest one — continuous-batching serving engines (vLLM, TGI) solve that; this recipe isolates the stopping-criterion lesson on its own.
- Forcing a minimum length is sometimes a *real* requirement (e.g. guaranteeing a model doesn't just emit EOS immediately) — the fix for that failure mode is a smarter stopping condition or a quality check, not disabling early stopping wholesale for every request.

## Go deeper

The `transformers` `generate()` docs on `StoppingCriteria` and `stop_strings` — this recipe uses the built-in string-matching path rather than a hand-rolled `StoppingCriteria` subclass, since it covers the common case with no extra code.

## Video hook

"Same prompt, same model, same greedy math — one of these was told it's allowed to stop talking. Watch the token counter."
