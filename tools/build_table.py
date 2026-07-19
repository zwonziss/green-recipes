"""
Rebuilds the results section of the root README from recipes/*/results/*.json.

    python tools/build_table.py            # rewrite README.md in place
    python tools/build_table.py --dry-run  # print the markdown instead
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- GML-RESULTS:START -->"
END = "<!-- GML-RESULTS:END -->"

CORE = [("wall_s", "time (s)"), ("peak_vram_mb", "VRAM (MB)"),
        ("gpu_energy_wh", "energy (Wh)"), ("co2_g", "CO2 (g)")]
EXTRA_ORDER = ["final_loss", "tokens_per_s", "val_accuracy", "ppl",
               "wh_per_1k_tok", "weights_vram_mb", "true_sparsity",
               "gzip_ckpt_mb", "ms_per_example", "trainable_params_m",
               "disk_mb", "effective_bs", "achieved_tflops",
               "compile_tax_s", "new_tokens", "ms_per_call", "best_val_ppl",
               "best_lr", "rung_units_trained", "loader_startup_s",
               "num_workers", "batch_size", "ms_per_request", "n_requests",
               "padding_ratio", "n_examples", "avg_new_tokens", "n_prompts",
               "cache_mb", "max_abs_diff", "prefix_tokens", "generate_calls",
               "hit_rate", "accuracy", "avg_layers_run", "threshold",
               "n_steps", "vocab_used", "vocab_total", "prune_ratio"]


def fmt(v, std=None):
    if v is None:
        return "—"
    s = f"{v:g}" if isinstance(v, float) else str(v)
    if std:
        s += f" (sd {std:g})"
    return s


def delta(v, base):
    if (isinstance(v, (int, float)) and isinstance(base, (int, float)) and base):
        return f" ({(v - base) / base * 100:+.0f}%)"
    return ""


def recipe_table(entries):
    entries.sort(key=lambda e: (not e.get("is_baseline", False), e.get("label", "")))
    base = entries[0]
    extra_keys = [k for k in EXTRA_ORDER
                  if any(k in e.get("extra", {}) for e in entries)]
    variant_header = "variant (n trials)" if any(e.get("n_trials", 1) > 1 for e in entries) else "variant"
    headers = ([variant_header] + [h for _, h in CORE]
               + extra_keys + ["measured (UTC)"])
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for e in entries:
        n = e.get("n_trials", 1)
        name = f"{e.get('label', '?')} (n={n})" if n > 1 else e.get("label", "?")
        row = [name]
        for k, _ in CORE:
            cell = fmt(e.get(k), e.get(f"{k}_std"))
            if e is not base:
                cell += delta(e.get(k), base.get(k))
            row.append(cell)
        extra = e.get("extra", {})
        row += [fmt(extra.get(k), extra.get(f"{k}_std")) for k in extra_keys]
        row.append(str(e.get("timestamp", ""))[:10])
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build() -> str:
    recipes = {}
    for f in sorted(ROOT.glob("recipes/*/results/*.json")):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"skipping unreadable {f}")
            continue
        recipes.setdefault(data.get("recipe", f.parent.parent.name), []).append(data)

    if not recipes:
        return ("_No results yet. Run any recipe (start with "
                "`python recipes/00_measure/run.py`), then run "
                "`python tools/build_table.py` to fill this section with your "
                "own measurements._")

    parts = []
    for name in sorted(recipes):
        entries = recipes[name]
        device = next((e.get("device") for e in entries if e.get("device")), "unknown GPU")
        parts.append(f"### {name} — {device}\n\n{recipe_table(entries)}")
    parts.append("_Deltas are relative to each recipe's baseline row. "
                 "Regenerate anytime with `python tools/build_table.py`._")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    section = build()
    if args.dry_run:
        print(section)
        return

    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"markers {START} / {END} not found in README.md")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(head + START + "\n" + section + "\n" + END + tail,
                      encoding="utf-8")
    print(f"README.md results section rebuilt "
          f"({sum(1 for _ in ROOT.glob('recipes/*/results/*.json'))} result files).")


if __name__ == "__main__":
    main()
