"""
Recipe 13 — CPU / edge inference: ONNX export + dynamic int8 quantization.

Not every deployment target has a GPU. Exports a DistilBERT-SST2 classifier
to ONNX, quantizes its Linear-layer weights to int8 with ONNX Runtime's
dynamic quantizer, and compares PyTorch-eager-CPU vs ONNX-Runtime-fp32 vs
ONNX-Runtime-int8: single-example latency, disk footprint, and validation
accuracy -- all forced onto CPU regardless of whether this machine also has
a GPU (see the Honesty box for why that matters for the energy numbers).

    python recipes/13_onnx_edge_inference/run.py
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.greenmeter import (  # noqa: E402
    GreenMeter, aggregate_trials, compare, print_receipt, save_result,
)
from common.data import pick_device, set_seed  # noqa: E402

RECIPE = "13_onnx_edge_inference"
MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LEN = 128


def export_onnx(model, tok, path):
    import torch

    class LogitsOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    dummy = tok("dummy input", padding="max_length", truncation=True,
               max_length=MAX_LEN, return_tensors="pt")
    torch.onnx.export(
        LogitsOnly(model).eval(), (dummy["input_ids"], dummy["attention_mask"]), str(path),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"},
                      "logits": {0: "batch"}},
        opset_version=17,
    )


def make_val_examples(tok, max_n):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2", split="validation")
    if max_n:
        ds = ds.select(range(min(max_n, len(ds))))
    examples = []
    for row in ds:
        enc = tok(row["sentence"], padding="max_length", truncation=True,
                  max_length=MAX_LEN, return_tensors="np")
        # keep only the two inputs the ONNX graph was traced with -- some
        # tokenizers (e.g. DistilBERT's) also emit token_type_ids, which
        # would be an unrecognized input to both the exported graph and to
        # DistilBert*'s forward() signature (it doesn't accept that kwarg).
        examples.append(({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
                         row["label"]))
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=200, help="validation examples to score")
    ap.add_argument("--repeats", type=int, default=3, help="trials per variant to aggregate")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = pick_device("cpu")  # edge/CPU deployment is the whole point here
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
    examples = make_val_examples(tok, args.max_eval)
    examples_pt = [({"input_ids": torch.from_numpy(e["input_ids"]),
                    "attention_mask": torch.from_numpy(e["attention_mask"])}, y)
                  for e, y in examples]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        onnx_fp32, onnx_int8 = tmp / "model_fp32.onnx", tmp / "model_int8.onnx"
        export_onnx(model, tok, onnx_fp32)
        try:
            import onnxruntime as ort
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError:
            print("Skipping: `onnx` / `onnxruntime` not installed "
                  "(pip install onnx onnxruntime), then rerun.")
            return
        quantize_dynamic(str(onnx_fp32), str(onnx_int8), weight_type=QuantType.QInt8)
        sess_fp32 = ort.InferenceSession(str(onnx_fp32), providers=["CPUExecutionProvider"])
        sess_int8 = ort.InferenceSession(str(onnx_int8), providers=["CPUExecutionProvider"])

        torch_ckpt = tmp / "model_fp32.pt"
        torch.save(model.state_dict(), torch_ckpt)
        disk_mb = {"pytorch_eager_cpu": torch_ckpt.stat().st_size / 2**20,
                  "onnxruntime_fp32": onnx_fp32.stat().st_size / 2**20,
                  "onnxruntime_int8": onnx_int8.stat().st_size / 2**20}

        # warmup: one example through every path
        with torch.no_grad():
            model(**examples_pt[0][0])
        for sess in (sess_fp32, sess_int8):
            sess.run(None, {k: v for k, v in examples[0][0].items()})

        out_dir = Path(__file__).parent
        variants = [("pytorch_eager_cpu", None), ("onnxruntime_fp32", sess_fp32),
                   ("onnxruntime_int8", sess_int8)]
        results = []
        for label, sess in variants:
            print(f"\n### {label} ###")
            trials = []
            for t in range(args.repeats):
                if args.repeats > 1:
                    print(f" -- trial {t + 1}/{args.repeats} --")
                hit = 0
                with GreenMeter(label) as m:
                    if sess is None:
                        with torch.no_grad():
                            for enc, y in examples_pt:
                                logits = model(**enc).logits
                                hit += int(logits.argmax(-1).item() == y)
                    else:
                        for enc, y in examples:
                            logits = sess.run(None, dict(enc))[0]
                            hit += int(logits.argmax(-1).item() == y)
                n = len(examples)
                m.add(val_accuracy=round(hit / n, 4), disk_mb=round(disk_mb[label], 2),
                     ms_per_example=round(m.result["wall_s"] / n * 1000, 3))
                trials.append(m.result)
            r = aggregate_trials(trials)
            print_receipt(r)
            results.append(r)
            save_result(r, RECIPE, out_dir / "results", is_baseline=(label == "pytorch_eager_cpu"))

    compare(results, extra_keys=("val_accuracy", "ms_per_example", "disk_mb"))
    print("Same predictions (modulo tiny quantization noise), same CPU -- watch "
          "disk_mb and ms_per_example both drop while val_accuracy barely moves. "
          "This is the deployment path for anyone without a GPU to spare.")


if __name__ == "__main__":
    main()
