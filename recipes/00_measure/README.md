# 00 — How we measure everything

**The lesson:** you cannot optimize what you don't measure. This recipe checks your setup and prints the repo's first *energy receipt*.

## Run

```bash
python recipes/00_measure/run.py --seconds 8 --repeats 3
```

## What you should see

An environment report (GPU, bf16/tf32/flash support, NVML power access), then the burn repeated `--repeats` times and aggregated into a receipt like:

```
  RECEIPT: matmul_4096  (mean +/- std, n=3)
  wall time     : 8.0033 s
  peak VRAM     : 96.1 MB
  GPU energy    : 0.34 Wh +/- 0.012 Wh  (avg 153 W, peak 167 W)
  est. CO2      : 0.153 g +/- 0.005 g  @ 450 gCO2/kWh
  in real life  : ~= 0.03 phone charges, 2 min of an LED bulb
```

followed by a one-line summary of the run-to-run spread on energy — that spread *is* the noise floor every single-run number in recipes 01–08 is competing against, which is why they all support `--repeats` too.

## How the measurement works

A background thread samples GPU power over NVML every 100 ms and integrates it into joules. Peak VRAM comes from PyTorch's allocator (`max_memory_allocated`, reset at block start). CO2 = energy × grid intensity (default 450 gCO2/kWh; override with `GML_GCO2_PER_KWH=400`).

## Honesty box

- Energy is **GPU-board only** — CPU, RAM and PSU losses are not counted. Treat absolute numbers as a lower bound; treat *relative* comparisons between variants as fair.
- Idle draw is included. A GPU sitting there doing nothing still pulls 10–30 W.
- 100 ms sampling means very short runs (<2 s) are noisy. Recipes run long enough for this not to matter.
- A single run can't tell you if a difference is real or just thermal/clock jitter — `--repeats` (default 3 across the book) reports mean ± std instead of one number, and later recipes' `compare()` tables flag deltas as `*` (95% CIs don't overlap the baseline) or `~` (they do, could be noise). It's a cheap heuristic, not a real hypothesis test.

## Video hook

"I measured how much CO2 one matrix multiplication costs" — show the receipt live, explain the sampling thread in 30 seconds.
