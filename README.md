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
| 09 | [torch.compile](recipes/09_torch_compile/) | Pay the compilation tax once, cash in every step after — if you stick around long enough. |
| 10 | [The KV cache](recipes/10_kv_cache/) | Why generation isn't O(n²) in practice: cache the keys/values, don't recompute them. |
| 11 | [Speculative decoding](recipes/11_speculative_decoding/) | A tiny model guesses several tokens ahead; the big model just checks its work. |
| 12 | [2:4 structured sparsity](recipes/12_structured_sparsity/) | The pruning speedup recipe 05 called a myth — real, if the zeros form a pattern. |
| 13 | [CPU / edge inference (ONNX)](recipes/13_onnx_edge_inference/) | No GPU? Export + quantize for a CPU box, not a data center. |
| 14 | [Energy-aware hyperparameter search](recipes/14_energy_aware_hpo/) | Stop training the losers — prune bad configs before they finish. |
| 15 | [DataLoader efficiency](recipes/15_dataloader_efficiency/) | An idle GPU waiting on the CPU still draws power. |
| 16 | [The batch-size sweep](recipes/16_batch_size_sweep/) | The most efficient batch size and the fastest one aren't always the same one. |
| 17 | [Stochastic Weight Averaging](recipes/17_weight_averaging/) | An ensemble's accuracy bump for one training run's energy bill. |
| 18 | [Prefix caching](recipes/18_prefix_caching/) | Stop re-reading the system prompt on every single request. |

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

A background thread samples power every 100 ms and integrates it to energy; peak VRAM comes from PyTorch's allocator; CO2 = energy × grid intensity (default 450 gCO2/kWh, override with `GML_GCO2_PER_KWH`). Full details in [`common/greenmeter.py`](common/greenmeter.py).

Be honest about what this is: **board/package energy only** (no PSU losses, no cooling, no embodied hardware carbon), including idle draw, on tiny models. Absolute numbers are lower bounds; *relative* comparisons between variants on the same machine are fair — and the relative story is the lesson.

One run can't tell a real effect from clock/thermal jitter, so every recipe supports `--repeats` (default 3): variants are retrained/re-run and aggregated into mean ± std, and comparison tables flag deltas as `*` (95% CIs don't overlap the baseline — likely real) or `~` (they overlap — could be noise). That's a cheap heuristic, not a real hypothesis test, but it beats reading a story into single-run noise. Several recipes also pair the efficiency numbers with a held-out quality metric (perplexity or validation accuracy) instead of a raw training loss, so "cheaper" always comes with "how much worse, if at all."

**Power backends:** `GreenMeter` auto-detects, in order, NVIDIA (NVML) → AMD ROCm (pyrsmi) → Apple Silicon (`powermetrics`) → Linux CPU package power (RAPL via `/sys/class/powercap`), and falls back to wall-time-only if none are available. **Only the NVIDIA/NVML path has been run on real hardware by this repo.** ROCm, Apple, and RAPL are implemented against each tool's public docs/interfaces but are unverified — `env_report()` (recipe 00) tells you which backend it actually found and marks non-NVML ones `[EXPERIMENTAL]`. If you have that hardware, please run recipe 00 and open an issue/PR with what you saw, good or bad.

## Hardware notes

| Capability | Colab T4 (Turing, 16 GB) | RTX 3060 (Ampere, 12 GB) | Apple Silicon | AMD (ROCm) | Linux CPU-only |
|---|---|---|---|---|---|
| fp16 + GradScaler | ✅ | ✅ | partial (MPS) | ✅ | ❌ |
| bf16 | ❌ (auto-skipped) | ✅ | partial | depends on card | ❌ |
| tf32 | ❌ (auto-skipped) | ✅ | n/a | n/a | n/a |
| Flash SDPA | ❌ → memory-efficient fallback | ✅ | ✅ (via SDPA) | depends on card | memory-efficient fallback |
| bitsandbytes int8 / nf4 / 8-bit Adam | ✅ | ✅ | ❌ (CUDA-only today) | ❌ (CUDA-only today) | ❌ |
| 2:4 sparse tensor cores | ❌ (Turing, pre-Ampere — auto-skipped) | ✅ | ❌ | ❌ | ❌ |
| Power/energy receipts | NVML ✅ | NVML ✅ | `powermetrics`, **experimental**, needs sudo | pyrsmi, **experimental** | RAPL, **experimental**, needs sysfs read access |

Recipes detect capabilities and skip what your card can't do, so every row above produces complete, comparable tables for the metrics it *can* measure — recipes needing bitsandbytes (03, 08) still require a CUDA GPU regardless of power backend, recipe 12's 2:4 sparse kernels need Ampere+ specifically (T4 auto-skips), and recipe 13 (ONNX/CPU) deliberately runs on CPU only, GPU or not.

## Green CI/CD tooling

Two developer-facing tools, independent of the recipes:

- **`tools/green_lint.py`** — a heuristic static linter (pure `ast`, no GPU needed) that scans Python for energy-heavy patterns and points at the recipe that addresses each one: hand-rolled attention with no SDPA in the file (→ recipe 07), a training loop with no autocast (→ recipe 01), `Adam`/`AdamW` with no 8-bit variant (→ recipe 08), or a full fine-tune with no LoRA/checkpointing (→ recipes 02/06). It's pattern-matching, not type/dataflow analysis — expect false positives and negatives; findings are prompts to look, not verdicts. `python tools/green_lint.py [paths...] [--format github] [--strict]`.
- **`tools/energy_regression.py`** — diffs two saved result JSONs (e.g. base branch vs. PR branch) and flags lower-is-better metrics that regressed past a threshold. Doesn't run anything itself; feed it two `save_result(...)` outputs.
- **`.github/workflows/green-ci.yml`** — wires both into a PR check: a lint pass, and an energy-regression pass that runs `recipes/00_measure` on both branches. **Read the caveat in that file** — GitHub-hosted runners have no GPU, so by default this only catches CPU wall-time/TFLOPS regressions; the workflow explains how to point it at a self-hosted GPU runner for real energy/CO2 regression testing. Both jobs annotate the PR's Job Summary and don't fail the build by default (`--strict` on either tool makes it a hard gate instead).

## Why tiny models?

Because the *mechanisms* are identical at every scale: Adam's 8-bytes-per-parameter tax, activation memory scaling, the T×T attention matrix. A 135M model makes them measurable in minutes on hardware students actually own — and the ratios you measure here are the same ratios that decide whether a 7B model trains on your GPU (recipes 02+03+08 stacked = QLoRA).

## Roadmap

Real-hardware validation of the ROCm/Apple/RAPL power backends (see Hardware notes) · real-time grid carbon intensity (e.g. the UK Carbon Intensity API, keyless and free — deferred so this doesn't ship as a US/EU-only or paywalled feature) · GGUF export for llama.cpp-style edge deployment (recipe 13 covers ONNX; GGUF is still open) · a Colab notebook that runs the whole book.

`torch.compile` (09), KV-cache tricks (10) & speculative decoding (11), structured 2:4 sparsity (12), CPU/edge inference via ONNX (13), and energy-aware hyperparameter search (14) are now covered — see the recipe table above.

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The bar: one file, one technique, one honest receipt.

## License & citation

MIT. If this repo helped your work or your teaching, a link back to Green Mind Labs helps the mission: making AI's resource costs visible and fixable.
