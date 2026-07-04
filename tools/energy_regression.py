"""
tools/energy_regression.py — diff two saved GreenMeter result JSONs (e.g. one
from the PR branch, one from the base branch) and flag regressions on
lower-is-better metrics.

Doesn't run anything itself and doesn't know about git -- it expects the
caller (a CI workflow, or you locally) to have already produced two result
JSONs with `save_result(...)` from the same recipe on the two code versions
you want to compare. See .github/workflows/green-ci.yml for how CI wires
this up (CPU-only on GitHub-hosted runners by default -- see that file for
how to point it at a self-hosted GPU runner instead).

    python recipes/00_measure/run.py --results-dir /tmp/pr   (on the PR branch)
    python recipes/00_measure/run.py --results-dir /tmp/base (on the base branch)
    python tools/energy_regression.py \\
        --pr-result /tmp/pr/matmul_512.json \\
        --base-result /tmp/base/matmul_512.json \\
        --threshold-pct 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# (metric key, display name, lower is better)
_METRICS = [
    ("wall_s", "wall time (s)", True),
    ("peak_vram_mb", "peak VRAM (MB)", True),
    ("gpu_energy_wh", "energy (Wh)", True),
    ("co2_g", "CO2 (g)", True),
]


def load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"result file not found: {path}")
    return json.loads(p.read_text())


def compare(base: dict, pr: dict, metric_keys, threshold_pct: float):
    """Returns a list of rows: (name, base_val, pr_val, pct_change, is_regression|None)."""
    rows = []
    active = [(k, h, lib) for k, h, lib in _METRICS if k in metric_keys]
    for key, name, lower_is_better in active:
        b, p = base.get(key), pr.get(key)
        if not isinstance(b, (int, float)) or not isinstance(p, (int, float)) or not b:
            rows.append((name, b, p, None, None))
            continue
        pct = (p - b) / b * 100
        is_regression = lower_is_better and pct > threshold_pct
        rows.append((name, b, p, pct, is_regression))
    return rows


def render_text(rows, base_label, pr_label, threshold_pct) -> str:
    out = [f"energy_regression: {base_label} (base) vs {pr_label} (PR), "
          f"threshold {threshold_pct:+.0f}%\n"]
    for name, b, p, pct, is_reg in rows:
        if pct is None:
            out.append(f"  {name:<16}: skipped (missing or zero baseline)")
            continue
        flag = " <-- REGRESSION" if is_reg else ""
        out.append(f"  {name:<16}: {b:g} -> {p:g}  ({pct:+.1f}%){flag}")
    n_reg = sum(1 for *_, is_reg in rows if is_reg)
    out.append(f"\n{n_reg} regression(s) over the {threshold_pct:g}% threshold.")
    return "\n".join(out)


def render_github(rows, base_label, pr_label, threshold_pct) -> str:
    out = [f"### energy_regression: `{base_label}` (base) vs `{pr_label}` (PR)\n",
          f"Threshold: flag if a lower-is-better metric grows more than "
          f"{threshold_pct:g}%. **CPU-only runners can't see GPU energy/CO2** -- "
          "those rows will show `skipped` unless this ran on a GPU runner.\n",
          "| metric | base | PR | delta | verdict |",
          "|---|---|---|---|---|"]
    for name, b, p, pct, is_reg in rows:
        if pct is None:
            out.append(f"| {name} | {b if b is not None else 'n/a'} | "
                       f"{p if p is not None else 'n/a'} | n/a | skipped |")
            continue
        verdict = "REGRESSION" if is_reg else "ok"
        out.append(f"| {name} | {b:g} | {p:g} | {pct:+.1f}% | {verdict} |")
    for name, b, p, pct, is_reg in rows:
        if is_reg:
            out.append(f"\n::warning::energy_regression: {name} grew {pct:+.1f}% "
                       f"(> {threshold_pct:g}% threshold): {b:g} -> {p:g}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr-result", required=True, help="result JSON from the PR/candidate branch")
    ap.add_argument("--base-result", required=True, help="result JSON from the base branch")
    ap.add_argument("--metrics", default="wall_s,peak_vram_mb,gpu_energy_wh,co2_g",
                    help="comma list of metric keys to compare")
    ap.add_argument("--threshold-pct", type=float, default=5.0,
                    help="flag a lower-is-better metric that grew more than this %%")
    ap.add_argument("--format", choices=["text", "github"], default="text")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any regression is found (default: exit 0, informational)")
    args = ap.parse_args()

    base = load(args.base_result)
    pr = load(args.pr_result)
    metric_keys = set(args.metrics.split(","))
    rows = compare(base, pr, metric_keys, args.threshold_pct)

    base_label = base.get("label", args.base_result)
    pr_label = pr.get("label", args.pr_result)
    if args.format == "github":
        print(render_github(rows, base_label, pr_label, args.threshold_pct))
    else:
        print(render_text(rows, base_label, pr_label, args.threshold_pct))

    if args.strict and any(is_reg for *_, is_reg in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
