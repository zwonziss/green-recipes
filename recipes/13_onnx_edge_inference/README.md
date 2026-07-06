# 13 — CPU / edge inference: ONNX export + dynamic int8 quantization

**The lesson:** not every deployment target has a GPU to spare. Export to ONNX and quantize weights to int8 dynamically, and a CPU-only box can serve a classifier meaningfully faster and smaller — for a small, measured accuracy cost.

## Run

```bash
python recipes/13_onnx_edge_inference/run.py --repeats 3
```

Exports DistilBERT-SST2 to ONNX, quantizes it with ONNX Runtime's dynamic int8 quantizer, and compares three single-example (batch=1) CPU paths: PyTorch eager, ONNX Runtime fp32, ONNX Runtime int8 — latency, on-disk model size, and validation accuracy over `--max-eval` examples (default 200).

## What to expect (rough, not yet measured — run it and replace this)

ONNX Runtime fp32 is typically somewhat faster than PyTorch eager on CPU purely from graph-level optimizations (operator fusion, better memory planning) with *zero* accuracy change. The int8 dynamic-quantized model should be noticeably smaller on disk (weights drop from 4 bytes/param to ~1) and usually faster still, at a small (order of ~0-1 point) validation accuracy cost. Whether the extra latency win is `*` (real) or `~` (noise) depends on how compute- vs overhead-bound this small model already is at batch=1.

## How it works

`torch.onnx.export` traces the model into a static computation graph; ONNX Runtime then applies its own graph optimizations (fusing ops PyTorch's eager mode runs one at a time) independent of quantization. `quantize_dynamic` additionally converts the weights of `MatMul`/`Gemm`-like ops to int8 *ahead of time*; activations are quantized on the fly per batch at runtime — hence "dynamic". Only Linear-like ops are touched, same as bitsandbytes' int8 in recipe 03, just via a different toolchain aimed at CPU serving instead of GPU.

## Honesty box

- **Read the energy numbers carefully.** `GreenMeter` always measures *whatever GPU is physically present* on the machine, regardless of which device your tensors actually run on — it has no way to know this recipe deliberately forced everything onto CPU. If you run this on a machine that also has a CUDA GPU, the energy/CO2 columns reflect that GPU's near-idle board draw over the wall-clock duration, **not real CPU package power**, and should be ignored. This recipe's energy story is only meaningful on a genuinely CPU-only machine, where `GreenMeter` falls back to Linux RAPL (experimental) or wall-time-only (Windows/macOS without `powermetrics`).
- Dynamic quantization only touches weights; activations stay in float and get quantized per-call. This is the "free lunch, mostly" option — static/QAT int8 quantizes activations ahead of time too and usually goes faster still, at more setup cost and more accuracy risk. Not shown here.
- This is single-stream (batch=1) latency, the tail case ONNX Runtime targets for edge devices — it says nothing about batched server throughput.
- ONNX export bakes in a *fixed* `max_length` padding here for simplicity; a real deployment would tune padding/truncation to its actual traffic shape rather than padding every input to 128 tokens.

## Go deeper

ONNX Runtime docs, *Quantize ONNX models* · Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* (2018).

## Video hook

"No GPU? No problem — three lines of ONNX export and a quantization call later, this classifier ships on a laptop CPU."
