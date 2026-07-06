# 09 — torch.compile: pay the compilation tax once, cash in every step after

**The lesson:** `torch.compile` fuses your model into optimized kernels the first time it actually runs — that first call is slower, not faster. This recipe measures that tax explicitly instead of hiding it in "warmup", then computes how many steps it takes for the steady-state speedup to pay it back.

## Run

```bash
python recipes/09_torch_compile/run.py --repeats 3
```

Trains SmolLM2-135M eager vs `torch.compile`d, same data/seed, `--repeats` trials aggregated. The very first step of each compiled trial is timed on its own (`compile_tax_s`) and excluded from the steady-state receipt; a held-out perplexity check and a per-step loss diff confirm compiling didn't change the math.

## What to expect (rough, not yet measured — run it and replace this)

The compile tax is typically several seconds to tens of seconds depending on GPU/driver/Triton cache state. Steady-state tokens/sec should be higher for `compiled` on CUDA (commonly 10–30% for a model this size, sometimes more), which the script converts into a break-even step count. On CPU, or on a cold Triton cache, the tax can dominate and never pay back within a short run — that's a real result, not a bug.

## How it works

Eager PyTorch dispatches one kernel per op. `torch.compile` traces the model into an FX graph, lets TorchInductor fuse ops and generate specialized (often Triton) kernels, then caches that compiled artifact — subsequent calls with the same input shapes reuse it directly. The first call pays tracing + codegen + (on CUDA) kernel autotuning; every call after is "free" until something forces a recompile.

## Honesty box

- **Shape changes trigger recompiles.** A different batch size or sequence length recompiles from scratch — this recipe deliberately uses fixed shapes throughout to isolate the one-time tax from repeat taxes you'd pay with variable-length batches.
- **Graph breaks silently fall back to eager** for the broken region (`.item()` calls, data-dependent control flow, unsupported ops). The speedup you measure is only as good as how much of the model actually stayed compiled — this recipe doesn't inspect that, `torch._dynamo.explain(model)` does.
- **Tiny models and short runs are the worst case for this technique** — which describes most recipes in this cookbook. The break-even print at the end is the honest version of "is it worth it here", and the answer can legitimately be no.
- **Windows + CUDA Triton support is newer and less battle-tested** than Linux; if compilation fails or silently no-ops on Windows, that's a known rough edge, not a bug in this script.
- `mode="max-autotune"` searches harder for fast kernels at compile time — expect a much bigger tax for (sometimes) a bigger steady-state win; not the default here because it can blow past a Colab-friendly runtime.

## Go deeper

PyTorch, *torch.compile* documentation and *TorchDynamo/TorchInductor* design docs · Bradbury et al.-style tracing compiler background via the PyTorch 2.0 paper (*PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation*, 2024).

## Video hook

"One line of code, and training got faster — except the first step got SLOWER. Here's the exact line where you break even."
