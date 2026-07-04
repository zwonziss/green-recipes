"""
Recipe 04 — Knowledge distillation: a 15x smaller model that keeps the skill.

Trains a 4.4M-param student on SST-2 twice: (a) hard labels only,
(b) hard labels + the teacher's soft probabilities (Hinton-style KD).
Each condition is trained across `--seeds` (init + data order both vary
per seed) so the accuracy comparison is a mean +/- std across independent
runs, not one lucky (or unlucky) seed. Then measures what actually matters
for the planet: serving cost.

    python recipes/04_knowledge_distillation/run.py
    python recipes/04_knowledge_distillation/run.py --seeds 42,43,44,45,46
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
from common.data import set_seed, pick_device  # noqa: E402
from common.eval import classification_accuracy  # noqa: E402

RECIPE = "04_knowledge_distillation"
TEACHER = "distilbert-base-uncased-finetuned-sst-2-english"  # 67M, already trained
STUDENT = "google/bert_uncased_L-2_H-128_A-2"                # BERT-Tiny, 4.4M


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


def train_student(use_kd, teacher, train_b, val_b, device, epochs, lr, T, alpha, seed):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification

    set_seed(seed)  # same seed drives both init and (via train_b) data order
    student = AutoModelForSequenceClassification.from_pretrained(
        STUDENT, num_labels=2).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    label = "kd_soft_labels" if use_kd else "hard_labels_only"

    with GreenMeter(label) as m:
        for ep in range(epochs):
            student.train()
            for ids, mask, y in train_b:
                ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                s_logits = student(input_ids=ids, attention_mask=mask).logits
                loss = F.cross_entropy(s_logits, y)
                if use_kd:
                    with torch.no_grad():
                        t_logits = teacher(input_ids=ids, attention_mask=mask).logits
                    kl = F.kl_div(F.log_softmax(s_logits / T, -1),
                                  F.softmax(t_logits / T, -1),
                                  reduction="batchmean") * T * T
                    loss = alpha * loss + (1 - alpha) * kl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            print(f"  [{label}] seed={seed} epoch {ep + 1} done, last loss {loss.item():.3f}")

    acc = classification_accuracy(student, val_b, device)
    m.add(val_accuracy=round(acc, 4),
          params_m=round(sum(p.numel() for p in student.parameters()) / 1e6, 1))
    return m.result, student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--temperature", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seeds", default="42,43,44",
                    help="comma list of seeds -- each drives a fresh init + data order")
    ap.add_argument("--serving-repeats", type=int, default=3,
                    help="trials for the final serving-cost timing")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    seeds = [int(s) for s in args.seeds.split(",")]
    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(TEACHER)  # same WordPiece vocab as BERT-Tiny
    val_b = make_batches("validation", tok, None, args.batch_size, seed=0)  # fixed across all seeds

    teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    out_dir = Path(__file__).parent
    results = []
    student = None
    for use_kd in (False, True):
        print(f"\n### {'KD soft labels' if use_kd else 'hard labels only'} "
              f"across seeds {seeds} ###")
        trials = []
        for seed in seeds:
            train_b = make_batches("train", tok, args.train_n, args.batch_size, seed=seed)
            r, student = train_student(use_kd, teacher, train_b, val_b, device,
                                       args.epochs, args.lr, args.temperature,
                                       args.alpha, seed=seed)
            trials.append(r)
            gc.collect()
            torch.cuda.empty_cache() if device == "cuda" else None
            time.sleep(1)
        agg = aggregate_trials(trials)
        print_receipt(agg)
        results.append(agg)
        save_result(agg, RECIPE, out_dir / "results", is_baseline=not use_kd)

    # --- serving cost: teacher vs (kd, last seed's) student over the val set ---
    print("\n### deployment cost: one pass over the validation set ###")
    serving = []
    for name, mdl in (("serve_teacher_67M", teacher), ("serve_student_4.4M", student)):
        trials = []
        for t in range(args.serving_repeats):
            with GreenMeter(name) as m:
                acc = classification_accuracy(mdl, val_b, device)
            m.add(val_accuracy=round(acc, 4))
            trials.append(m.result)
            time.sleep(1)
        r = aggregate_trials(trials)
        print_receipt(r)
        serving.append(r)
        save_result(r, RECIPE, out_dir / "results", is_baseline=(name == "serve_teacher_67M"))

    print(f"\n--- training the student across seeds {seeds}: hard labels vs KD ---")
    compare(results, extra_keys=("val_accuracy", "params_m"))
    print("--- serving: teacher vs distilled student ---")
    compare(serving, extra_keys=("val_accuracy",))
    print("Watch whether KD's accuracy edge survives the * / ~ marker above -- "
          "with one seed it's easy to mistake luck for a real improvement. "
          "Either way, distillation is paid once at training time; the serving "
          "savings repeat on every single request, forever.")


if __name__ == "__main__":
    main()
