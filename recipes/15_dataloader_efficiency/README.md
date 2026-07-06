# 15 — DataLoader efficiency: a GPU waiting on the CPU still draws power

**The lesson:** if preprocessing can't keep up with the GPU, the GPU sits idle between batches — and idle GPUs still pull real power. `num_workers`, `pin_memory`, and prefetching move that preprocessing off the critical path; this recipe manufactures a genuinely CPU-bound workload to measure when that fix actually pays off.

## Run

```bash
python recipes/15_dataloader_efficiency/run.py --repeats 3
```

A custom `Dataset` re-tokenizes a fresh 6,000-character text chunk on every `__getitem__` (deliberately expensive, unlike every other recipe here, which pre-tokenizes once into RAM via `common.data.lm_batches`). Same SmolLM2-135M training steps run through three loader configs: `workers0_nopin` (PyTorch's naive default), `workers2_pin`, `workers4_pin_prefetch` — each repeated `--repeats` times.

## What to expect (rough, not yet measured — run it and replace this)

`workers0_nopin` should be the slowest and show the lowest `gpu_avg_power_w` — the tell-tale sign of a GPU idling while one CPU process tokenizes. `workers2_pin` and especially `workers4_pin_prefetch` should raise `tokens_per_s` and `gpu_avg_power_w` together (more time actually computing, less time waiting) even though every variant does *identical* model work. `loader_startup_s` (time to the first batch) is usually higher for the multi-worker configs — that's the process-spawn cost, paid once.

## How it works

- `num_workers>0` moves `__getitem__` calls to separate processes so the next batch can be prepared while the GPU is still busy with the current one.
- `pin_memory=True` allocates batches in page-locked host memory, which makes the host→device copy faster and can overlap with compute — it only matters when the destination is a CUDA GPU.
- `prefetch_factor` controls how many batches each worker prepares ahead of time, giving more slack to absorb per-item timing variance.
- `persistent_workers=True` avoids re-spawning the worker pool on every epoch — this benchmark runs one continuous pass, so that specific benefit isn't exercised here (see Honesty box).

## Honesty box

- **This bottleneck is manufactured on purpose.** Every *other* recipe in this cookbook uses `common.data.lm_batches`, which tokenizes once into RAM specifically to avoid this cost — because for tiny, already-cached datasets, spinning up worker processes buys nothing and only adds overhead. `num_workers` only helps when preprocessing is genuinely the bottleneck.
- **On Windows, multiprocessing workers use `spawn`**, which re-imports this script in every worker process — noticeably higher startup cost than Linux's `fork`. For short runs or tiny datasets, `num_workers>0` can measurably lose to `num_workers=0` on Windows. That's a real, measured result here, not a bug.
- `pin_memory` has no effect on CPU-only runs; this script only enables it when a CUDA device is in use.
- `persistent_workers`' real benefit (skipping worker re-spawn between epochs) isn't tested by this single continuous pass — it matters more in a training loop with many short epochs.
- More workers than CPU cores (or than the dataset can profitably parallelize) can make things *worse* via contention — this script's `--steps`/`--batch-size` defaults are tuned for a demo, not for finding your machine's actual optimum.

## Go deeper

PyTorch docs, *torch.utils.data — DataLoader* (the "Multi-process data loading" and "Memory Pinning" sections) · PyTorch performance tuning guide.

## Video hook

"Same model. Same data. Same steps. I only changed how the data gets fetched — watch the power meter drop because the GPU is finally busy the whole time."
