# 🌱 The Efficient AI Cookbook

**Runnable recipes for training and serving neural networks with less energy, less memory, and less CO2 — every claim backed by a receipt.**

A [Green Mind Labs](https://greenmindlabs.org) project.

---

Training GPT-3 consumed an estimated 1,287 MWh and ~552 tonnes of CO2e (Patterson et al., 2021) — and inference at scale now costs far more than training ever did. Most writing about this problem stops at the alarm. This repo is the other half: **the techniques that actually cut the bill, each one demonstrated in a single ~150-line script you can run on a free Colab T4 or a 12 GB consumer GPU, printing its own energy receipt.**

```
  RECEIPT: bf16
  wall time     : 41.3 s
  peak VRAM     : 2311 MB
  GPU energy    : 1.62 Wh  (avg 141 W, peak 158 W)
  est. CO2      : 0.73 g  @ 450 gCO2/kWh
```

No frameworks to learn, no config files, no hidden magic. One technique per recipe, a baseline to compare against, and honest notes about when the technique *doesn't* help.

## Quickstart

```bash
git clone https://github.com/green-mind-labs/efficient-ai
cd efficient-ai
pip install -r requirements.txt        # on Colab, torch is already there
python recipes/00_measure/run.py       # environment check + your first receipt
python recipes/01_mixed_precision/run.py
```

Or bake everything (~30–60 min on an RTX 3060): `python tools/run_all.py`

On Colab: `Runtime → Change runtime type → T4 GPU`, then run the block above in a cell prefixed with `!`.

## The recipes

| # | Recipe | The lesson in one line |
|---|--------|------------------------|
| 00 | [How we measure](recipes/00_measure/) | You can't optimize what you don't measure — meet the energy receipt. |
| 01 | [Mixed precision](recipes/01_mixed_precision/) | Most training math doesn't need 32 bits; fp16/bf16 is a near-free 2×. |
| 02 | [LoRA](recipes/02_lora/) | Freeze the model, train tiny adapters — the optimizer's memory bill collapses. |
| 03 | [Quantized inference](recipes/03_quantization_inference/) | 4-bit weights cut VRAM ~4× — and we put a *number* on the quality cost. |
| 04 | [Knowledge distillation](recipes/04_knowledge_distillation/) | A 15× smaller student keeps the skill — pay once, save on every request forever. |
| 05 | [Pruning](recipes/05_pruning/) | Delete 50% of the weights, keep the accuracy — but the speedup is a myth (measured). |
| 06 | [Gradient checkpointing](recipes/06_gradient_checkpointing/) | Trade compute for memory, then spend the savings on throughput. |
| 07 | [Flash / SDPA attention](recipes/07_flash_attention/) | Watch O(n²) attention hit the memory wall, then watch one kernel walk through it. |
| 08 | [The small-GPU stack](recipes/08_smaller_batch_tricks/) | Accumulation + 8-bit Adam + checkpointing: train what "shouldn't fit". |

Every recipe follows the same contract: **one `run.py`, runs in minutes on a T4, prints receipts + a comparison table, saves JSON results, and its README has an Honesty box** listing where the technique fails or misleads.

## Results

Numbers below are generated from real runs committed to `recipes/*/results/` — never typed by hand. After running recipes on your hardware:

```bash
python tools/build_table.py
```

<!-- GML-RESULTS:START -->
_No results yet. Run any recipe (start with `python recipes/00_measure/run.py`), then run `python tools/build_table.py` to fill this section with your own measurements._
<!-- GML-RESULTS:END -->

## How measurement works (and its limits)

A background thread samples GPU board power via NVML every 100 ms and integrates it to energy; peak VRAM comes from PyTorch's allocator; CO2 = energy × grid intensity (default 450 gCO2/kWh, override with `GML_GCO2_PER_KWH`). Full details in [`common/greenmeter.py`](common/greenmeter.py).

Be honest about what this is: **GPU-board energy only** (no CPU/RAM/PSU losses, no cooling, no embodied hardware carbon), including idle draw, on tiny models. Absolute numbers are lower bounds; *relative* comparisons between variants on the same machine are fair — and the relative story is the lesson.

One run can't tell a real effect from GPU clock/thermal jitter, so every recipe supports `--repeats` (default 3): variants are retrained/re-run and aggregated into mean ± std, and comparison tables flag deltas as `*` (95% CIs don't overlap the baseline — likely real) or `~` (they overlap — could be noise). That's a cheap heuristic, not a real hypothesis test, but it beats reading a story into single-run noise. Several recipes also pair the efficiency numbers with a held-out quality metric (perplexity or validation accuracy) instead of a raw training loss, so "cheaper" always comes with "how much worse, if at all."

## Hardware notes

| Capability | Colab T4 (Turing, 16 GB) | RTX 3060 (Ampere, 12 GB) |
|---|---|---|
| fp16 + GradScaler | ✅ | ✅ |
| bf16 | ❌ (auto-skipped) | ✅ |
| tf32 | ❌ (auto-skipped) | ✅ |
| Flash SDPA | ❌ → memory-efficient fallback | ✅ |
| bitsandbytes int8 / nf4 / 8-bit Adam | ✅ | ✅ |

Recipes detect capabilities and skip what your card can't do, so both machines produce complete, comparable tables.

## Why tiny models?

Because the *mechanisms* are identical at every scale: Adam's 8-bytes-per-parameter tax, activation memory scaling, the T×T attention matrix. A 135M model makes them measurable in minutes on hardware students actually own — and the ratios you measure here are the same ratios that decide whether a 7B model trains on your GPU (recipes 02+03+08 stacked = QLoRA).

## Roadmap

`torch.compile` · KV-cache tricks & speculative decoding · CPU/edge inference (ONNX, GGUF) · structured / 2:4 sparsity done right · a Colab notebook that runs the whole book · energy-aware hyperparameter search.

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The bar: one file, one technique, one honest receipt.

## License & citation

MIT. If this repo helped your work or your teaching, a link back to Green Mind Labs helps the mission: making AI's resource costs visible and fixable.
