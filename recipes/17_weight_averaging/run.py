"""
Recipe 17 — Stochastic Weight Averaging: an ensemble's accuracy bump for
(close to) one training run's energy.

Trains a BERT-tiny SST-2 classifier once, averaging the last `--swa-frac` of
its checkpoints (SWA) alongside keeping the plain final checkpoint (baseline)
-- same training run, same energy, two accuracy outcomes. A second,
independent training run is then used to build a real 2-model ensemble, so
the "expensive" alternative's energy price tag (~2x) is measured, not assumed.

    python recipes/17_weight_averaging/run.py
"""
import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.greenmeter import (  # noqa: E402
    GreenMeter, aggregate_trials, compare, print_receipt, save_result,
)
from common.data import pick_device, set_seed  # noqa: E402
from common.eval import classification_accuracy  # noqa: E402

RECIPE = "17_weight_averaging"
STUDENT = "google/bert_uncased_L-2_H-128_A-2"  # BERT-tiny, 4.4M -- fast enough for repeats


def make_batches(split, tok, n, batch_size, seed, max_len=128):
    import torch
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2", split=split)
    if n:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    batches = []
    for i in range(0, len(ds), batch_size):
        rows = ds[i : i + batch_size]
        enc = tok(rows["sentence"], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
        batches.append((enc["input_ids"], enc["attention_mask"],
                        torch.tensor(rows["label"])))
    return batches


def train_run(seed, train_b, val_b, device, epochs, lr, swa_frac):
    import torch
    import torch.nn.functional as F
    from torch.optim.swa_utils import AveragedModel, SWALR
    from transformers import AutoModelForSequenceClassification

    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(STUDENT, num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = epochs * len(train_b)
    swa_start_step = int(total_steps * (1 - swa_frac))
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(opt, swa_lr=lr * 0.5)

    label = f"train_seed{seed}"
    step = 0
    with GreenMeter(label) as m:
        for ep in range(epochs):
            model.train()
            for ids, mask, y in train_b:
                ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                loss = F.cross_entropy(model(input_ids=ids, attention_mask=mask).logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if step >= swa_start_step:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()
                step += 1
            print(f"  [{label}] epoch {ep + 1} done, last loss {loss.item():.3f}")

    acc_final = classification_accuracy(model, val_b, device)
    acc_swa = classification_accuracy(swa_model, val_b, device)
    m.add(val_accuracy_final=round(acc_final, 4), val_accuracy_swa=round(acc_swa, 4))
    return m.result, model, swa_model


def ensemble_accuracy(model_a, model_b, val_b, device):
    import torch
    import torch.nn.functional as F
    was_a, was_b = model_a.training, model_b.training
    model_a.eval()
    model_b.eval()
    hit = tot = 0
    with torch.no_grad():
        for ids, mask, y in val_b:
            ids, mask = ids.to(device), mask.to(device)
            pa = F.softmax(model_a(input_ids=ids, attention_mask=mask).logits, dim=-1)
            pb = F.softmax(model_b(input_ids=ids, attention_mask=mask).logits, dim=-1)
            pred = ((pa + pb) / 2).argmax(-1).cpu()
            hit += (pred == y).sum().item()
            tot += len(y)
    if was_a:
        model_a.train()
    if was_b:
        model_b.train()
    return hit / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--swa-frac", type=float, default=0.3,
                    help="fraction of the SAME run's final steps spent averaging")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seeds", default="42,142,242",
                    help="each seed drives an independent (seed, seed+1) training pair")
    args = ap.parse_args()

    import torch
    device = pick_device(args.device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDENT)
    val_b = make_batches("validation", tok, None, args.batch_size, seed=0)
    out_dir = Path(__file__).parent

    baseline_trials, swa_trials, ensemble_trials = [], [], []
    for seed in [int(s) for s in args.seeds.split(",")]:
        print(f"\n=== seed pair ({seed}, {seed + 1}) ===")
        train_a = make_batches("train", tok, args.train_n, args.batch_size, seed=seed)
        r_a, model_a, swa_a = train_run(seed, train_a, val_b, device, args.epochs, args.lr, args.swa_frac)

        train_b = make_batches("train", tok, args.train_n, args.batch_size, seed=seed + 1)
        r_b, model_b, _ = train_run(seed + 1, train_b, val_b, device, args.epochs, args.lr, args.swa_frac)

        ens_acc = ensemble_accuracy(model_a, model_b, val_b, device)
        baseline_trials.append({**r_a, "label": "baseline_final",
                                "extra": {"val_accuracy": r_a["extra"]["val_accuracy_final"]}})
        swa_trials.append({**r_a, "label": "swa_averaged",
                           "extra": {"val_accuracy": r_a["extra"]["val_accuracy_swa"]}})
        ens = {"label": "ensemble_2x", "wall_s": round(r_a["wall_s"] + r_b["wall_s"], 2),
              "extra": {"val_accuracy": round(ens_acc, 4)}}
        if "gpu_energy_wh" in r_a and "gpu_energy_wh" in r_b:
            ens["gpu_energy_wh"] = round(r_a["gpu_energy_wh"] + r_b["gpu_energy_wh"], 4)
            ens["co2_g"] = round(r_a.get("co2_g", 0) + r_b.get("co2_g", 0), 3)
        ensemble_trials.append(ens)

        del model_a, model_b, swa_a
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        time.sleep(1)

    results = []
    for label, trials in (("baseline_final", baseline_trials), ("swa_averaged", swa_trials),
                          ("ensemble_2x", ensemble_trials)):
        r = aggregate_trials(trials)
        print_receipt(r)
        results.append(r)
        save_result(r, RECIPE, out_dir / "results", is_baseline=(label == "baseline_final"))

    compare(results, extra_keys=("val_accuracy",))
    print("swa_averaged spends the SAME training energy as baseline_final (it's "
          "a different checkpoint from the identical run) -- any accuracy gain "
          "there is close to free. ensemble_2x needed a full second training "
          "run to get its gain -- roughly double the energy. Compare the "
          "accuracy deltas against those very different price tags.")


if __name__ == "__main__":
    main()
