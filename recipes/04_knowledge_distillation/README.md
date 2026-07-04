# 04 — Knowledge distillation: shrink the model, keep the skill

**The lesson:** a big model's *probabilities* contain more information than the dataset's hard labels. Train a tiny student to imitate them and it beats the same student trained on labels alone — then serves every future request at a fraction of the cost.

## Run

```bash
python recipes/04_knowledge_distillation/run.py --seeds 42,43,44
```

Teacher: DistilBERT fine-tuned on SST-2 (67M, already trained). Student: BERT-Tiny (4.4M) trained across `--seeds` (each seed drives both the init *and* the data order) — once per seed on hard labels, once with the teacher's soft targets (T=3, α=0.5) — so the accuracy comparison is a mean ± std over independent runs, not one lucky seed. Finishes with a head-to-head serving-cost measurement (also repeated, via `--serving-repeats`).

## What to expect (rough, not yet measured — run it and replace this)

Teacher sits around ~91% on SST-2 dev. Hard-labels-only student: roughly high-70s to low-80s. KD student: typically a couple of points higher **at identical training cost** — but on a model and dataset this small, seed-to-seed accuracy swings can be a few points too, so check whether the `compare()` table marks KD's edge `*` (95% CIs don't overlap — likely real) or `~` (overlap — could be seed luck) before believing a single-seed win. Serving pass: the 15× smaller student should be several times faster and cheaper per example.

## How it works

Loss = `α·CE(student, labels) + (1−α)·T²·KL(student/T ‖ teacher/T)`. The temperature `T` softens both distributions so the student can see *dark knowledge* — e.g. that "not bad at all" is 0.9 positive, not 1.0. The `T²` factor keeps gradient magnitudes comparable as T changes.

## Honesty box

- KD's cost includes running the teacher on every batch — that's inside the receipt here, not hidden. The bet is: pay once at training, save on every inference afterwards.
- SST-2 is an easy task; gaps between methods are a few points. On harder tasks (or LM distillation) both the gains and the engineering effort grow.
- Both models share BERT's WordPiece vocab, which lets one tokenizer feed both. Distilling across tokenizers is a genuinely harder problem.
- BERT-Tiny on 6k examples is seed-sensitive — a single run can make KD look like a bigger (or smaller) win than it is. That sensitivity is itself the reason this recipe trains across multiple seeds instead of one.

## Go deeper

Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network* (2015) · Sanh et al., *DistilBERT* (2019).

## Video hook

"This 4-million-parameter model learned from a teacher — and here's the report card." Two accuracy bars, then the serving receipts side by side.
