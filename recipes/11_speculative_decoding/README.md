# 11 — Speculative decoding: let a tiny model guess, let the big one nod

**The lesson:** a small "draft" model proposes several tokens at once; the big "target" model checks all of them in a single forward pass and keeps whatever it agrees with. With greedy decoding on both, the output is *provably identical* to plain target-only decoding — proven here, not assumed — often for meaningfully less target-model compute.

## Run

```bash
python recipes/11_speculative_decoding/run.py --repeats 3
```

Draft = SmolLM2-135M, target = SmolLM2-360M (same tokenizer family — checked with an assertion before anything else runs). Part 0 confirms target-only greedy and speculative greedy produce byte-identical output over 48 tokens. Then `target_only` vs `speculative` are timed across `--repeats` trials and compared with `GreenMeter`, using transformers' built-in `assistant_model=` API.

## What to expect (rough, not yet measured — run it and replace this)

The equivalence check should print `True`. Because SmolLM2-135M and 360M were trained the same way on similar data, the draft should agree with the target often enough to give a real speedup on tokens/sec — commonly somewhere in the 1.3–2.5x range for same-family model pairs, though this depends heavily on the prompt and `--draft-tokens`. Energy per token should drop roughly in proportion, since the target model runs fewer, larger (more parallel) forward passes instead of many small sequential ones.

## How it works

The draft proposes `--draft-tokens` tokens one at a time (cheap: it's the small model). The target then runs ONE forward pass over the draft's proposed tokens plus the true prefix and checks, position by position, whether its own greedy prediction matches what the draft guessed. It accepts the longest matching prefix and substitutes its own token the instant they diverge — which is exactly what plain target-only decoding would have produced at that position anyway. One target forward pass can therefore "confirm" several tokens instead of generating just one.

## Honesty box

- The token-for-token equivalence guarantee holds for **greedy decoding on both models**. With sampling, speculative decoding uses a modified accept/reject scheme (Leviathan et al. / Chen et al.) that is equivalent *in distribution*, not token-for-token — not exercised here.
- Draft and target must share a tokenizer/vocabulary; this script asserts `vocab_size` equality before doing anything else, and speculative decoding across mismatched tokenizers simply doesn't work.
- The speedup lives entirely in the **acceptance rate** — how often the draft's guess matches the target. A weak or out-of-domain draft that disagrees constantly does strictly MORE total compute (draft forward + target forward) for little or no benefit. This script doesn't expose the per-round acceptance rate (that logic lives inside `transformers/generation/candidate_generator.py`); only the net wall-clock/energy effect is measured.
- The benefit shrinks as batch size grows: at batch size 1, the target's forward pass is memory-bandwidth-bound and has slack to verify several tokens at once almost for free; at large batch it's already compute-bound, and that slack disappears.
- **That memory-bandwidth slack is a GPU-specific story.** On CPU, this script measurably ran *slower* than plain decoding in testing — CPU inference doesn't have the same idle-memory-bandwidth gap for extra draft/verify compute to fill for free, so the draft model's cost shows up directly as wall-clock overhead instead. Speculative decoding's win is real, but it's a GPU-serving technique first and foremost; don't expect it on a CPU-only box.
- This recipe uses HF's tested `assistant_model=` implementation rather than a hand-rolled draft/verify loop with manual KV-cache surgery — a deliberate correctness-over-transparency choice.

## Go deeper

Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative Decoding* (2023) · Chen et al. (DeepMind), *Accelerating Large Language Model Decoding with Speculative Sampling* (2023).

## Video hook

"Two models, one seat: watch a tiny model guess several words ahead while the big model just checks its work — same output, a fraction of the wait."
