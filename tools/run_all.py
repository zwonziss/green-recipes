"""
Bake the whole cookbook: run every recipe with default args, then rebuild
the README results table. Expect ~30-60 min on an RTX-3060-class GPU
(most of it is recipes 03 and 06-08).

    python tools/run_all.py
    python tools/run_all.py --only 01,02,05
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of recipe numbers, e.g. 01,05")
    args = ap.parse_args()

    recipes = sorted(p for p in (ROOT / "recipes").iterdir()
                     if p.is_dir() and (p / "run.py").exists())
    if args.only:
        wanted = {n.strip().zfill(2) for n in args.only.split(",")}
        recipes = [r for r in recipes if r.name[:2] in wanted]

    failed = []
    t0 = time.perf_counter()
    for r in recipes:
        print(f"\n{'=' * 70}\n  RUNNING {r.name}\n{'=' * 70}")
        code = subprocess.call([sys.executable, str(r / "run.py")])
        if code != 0:
            print(f"!! {r.name} exited with code {code} — continuing.")
            failed.append(r.name)

    print(f"\nTotal wall time: {(time.perf_counter() - t0) / 60:.1f} min")
    if failed:
        print(f"Failed: {', '.join(failed)}")

    print("\nRebuilding README results table...")
    subprocess.call([sys.executable, str(ROOT / "tools" / "build_table.py")])


if __name__ == "__main__":
    main()
