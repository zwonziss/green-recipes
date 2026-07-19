# 23 — CPU-offloaded activations: trade PCIe bandwidth for memory, not compute

**The lesson:** recipe 06 freed VRAM by throwing activations away and recomputing them during backward — more FLOPs, same GPU, the whole time. `torch.autograd.graph.save_on_cpu()` frees VRAM a different way: it keeps every saved activation, but parks it in pinned host RAM during forward and copies it back during backward. No extra compute — a real PCIe transfer cost instead. Same VRAM goal, a completely different resource spent to get there.

## Run

```bash
python recipes/23_activation_offloading/run.py --repeats 3
```

Requires a CUDA GPU (there's no separate "host" to offload activations *to* on CPU-only or MPS machines — the script prints a note and exits cleanly there). Three fp32 training variants of SmolLM2-360M at T=512: baseline B=8, offloaded B=8, offloaded B=16 — each retrained `--repeats` times (same seed — repeats target timing noise, not training randomness). Because offloading moves tensors rather than approximating anything, the script also diffs baseline-B=8's and offload-B=8's per-step losses and reports the max difference as a numerical check.

## What to expect (rough, not yet measured — run it and replace this)

Offloaded B=8 should show a meaningful VRAM drop and *some* added wall time (a PCIe round trip per saved tensor) — how much depends heavily on your PCIe generation/lane count and host RAM speed, far more hardware-sensitive than recipe 06's recompute cost. The B=16 offloaded run, spending the freed VRAM on a bigger batch, should claw back much of that slowdown in tokens/sec. The baseline-vs-offload loss diff should be tiny (float noise) — offloading doesn't change the optimization at all.

## How it works

Normally every layer's activations stay resident on the GPU until backward needs them. `save_on_cpu(pin_memory=True)` installs a pair of *saved-tensor hooks* (`torch.autograd.graph`'s public hook API): on the way into autograd's saved-tensor storage, each tensor is copied to pinned CPU memory; on the way out (during backward), it's copied back to the GPU. The forward pass's actual compute is untouched — only where the intermediate results *live* changes.

```python
with torch.autograd.graph.save_on_cpu(pin_memory=True):
    loss = model(input_ids=x, labels=x).loss
```

## Honesty box

- **Contrast with recipe 06 directly:** checkpointing spends extra FLOPs (recompute) to save memory; offloading spends PCIe bandwidth and host RAM (no extra FLOPs) to save the same memory. The wall-time result here is far more a function of *your* hardware's host↔GPU bandwidth than recipe 06's, which is mostly a function of the model's FLOP profile.
- Pinned memory is page-locked host RAM — it can't be swapped, and allocating a lot of it can slow down or destabilize a RAM-constrained machine. This isn't free real estate, it's a different budget.
- `save_on_cpu` offloads indiscriminately — every saved tensor, no per-layer control. Production systems doing this at scale (DeepSpeed's ZeRO-Offload, Colossal-AI) are far more selective and pipeline the transfers to overlap with compute; this recipe uses the plain PyTorch primitive to isolate the underlying mechanism.
- Like checkpointing, offloading alone at the *same* batch size is usually a pure regression in wall time — the win only shows up once you reinvest the freed VRAM into something (a bigger batch, here).

## Go deeper

Rajbhandari et al., *ZeRO-Offload: Democratizing Billion-Scale Model Training* (2021) · PyTorch docs on `torch.autograd.graph.saved_tensors_hooks` (the general mechanism `save_on_cpu` is built on).

## Video hook

"Recipe 06 traded compute for memory. This one trades bandwidth for memory instead — same GPU, same model, a completely different bill."
