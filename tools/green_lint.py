"""
tools/green_lint.py — a heuristic static linter for "energy-heavy" PyTorch
patterns, each finding pointing at the recipe that addresses it.

This is pure `ast` pattern-matching on one file at a time: no type
inference, no cross-file tracing, no import resolution. It answers "does
this file contain pattern X and NOT pattern Y" -- nothing more. Expect
false positives (a file with a good reason to do things the "heavy" way)
and false negatives (patterns spread across files, or spelled differently
than this linter looks for). Findings are suggestions to go look, not verdicts.
Exit code is always 0 unless --strict is passed and at least one finding fired
-- this is meant to annotate PRs, not block them, by default.

    python tools/green_lint.py                        # scans recipes/ + common/
    python tools/green_lint.py path/to/file_or_dir ...
    python tools/green_lint.py --format github         # GitHub Step Summary markdown
    python tools/green_lint.py --strict                # exit 1 if anything fired
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = ["recipes", "common"]

# (rule id, message, pointer)
_RULES = {
    "hand-rolled-attention": (
        "softmax(...) combined with a `@`/matmul is present, and this file never "
        "calls scaled_dot_product_attention -- fused SDPA/FlashAttention computes "
        "the identical math with less memory and usually less time.",
        "recipes/07_flash_attention",
    ),
    "no-mixed-precision": (
        "a `.backward()` training step was found with no autocast/GradScaler in "
        "this file -- most training math doesn't need fp32; fp16/bf16 is a "
        "near-free speedup.",
        "recipes/01_mixed_precision",
    ),
    "no-8bit-optimizer": (
        "torch.optim.Adam/AdamW is used at full precision -- its optimizer state "
        "costs 8 bytes/param; bitsandbytes' 8-bit Adam is a one-line swap that "
        "cuts that ~4x if VRAM is tight.",
        "recipes/08_smaller_batch_tricks",
    ),
    "full-finetune-heavy": (
        "a pretrained model is loaded and trained (.backward() present) with no "
        "LoRA and no gradient checkpointing in this file -- if VRAM is the "
        "constraint, freezing the base model (LoRA) or trading compute for "
        "memory (checkpointing) are both cheap to try before reaching for a bigger GPU.",
        "recipes/02_lora and recipes/06_gradient_checkpointing",
    ),
}


class Finding:
    __slots__ = ("path", "line", "rule")

    def __init__(self, path: str, line: int, rule: str):
        self.path = path
        self.line = line
        self.rule = rule

    def render(self) -> str:
        msg, pointer = _RULES[self.rule]
        return f"{self.path}:{self.line}: [{self.rule}] {msg} (see {pointer})"


def _call_name(node: ast.Call):
    """Trailing identifier of a call: Name('foo') or Attribute(..., attr='foo')."""
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


class _Visitor(ast.NodeVisitor):
    def __init__(self):
        self.has_sdpa = False
        self.has_matmul = False
        self.has_backward = False
        self.has_autocast = False
        self.has_lora = False
        self.has_checkpoint = False
        self.has_8bit_optim = False
        self.softmax_lines: list[int] = []
        self.adam_lines: list[int] = []
        self.from_pretrained_lines: list[int] = []

    def visit_BinOp(self, node: ast.BinOp):
        if isinstance(node.op, ast.MatMult):
            self.has_matmul = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        name = _call_name(node)
        if name == "scaled_dot_product_attention":
            self.has_sdpa = True
        elif name in ("matmul", "bmm"):
            self.has_matmul = True
        elif name == "softmax":
            self.softmax_lines.append(node.lineno)
        elif name == "backward":
            self.has_backward = True
        elif name in ("autocast", "GradScaler"):
            self.has_autocast = True
        elif name in ("get_peft_model", "LoraConfig"):
            self.has_lora = True
        elif name in ("gradient_checkpointing_enable", "checkpoint"):
            self.has_checkpoint = True
        elif name in ("AdamW8bit", "Adam8bit"):
            self.has_8bit_optim = True
        elif name in ("AdamW", "Adam"):
            self.adam_lines.append(node.lineno)
        elif name == "from_pretrained":
            self.from_pretrained_lines.append(node.lineno)
        self.generic_visit(node)

    def findings(self, path: str) -> list[Finding]:
        out = []
        if self.softmax_lines and self.has_matmul and not self.has_sdpa:
            out.append(Finding(path, self.softmax_lines[0], "hand-rolled-attention"))
        if self.has_backward and not self.has_autocast:
            out.append(Finding(path, 1, "no-mixed-precision"))
        if self.adam_lines and not self.has_8bit_optim:
            out.append(Finding(path, self.adam_lines[0], "no-8bit-optimizer"))
        if (self.from_pretrained_lines and self.has_backward
                and not self.has_lora and not self.has_checkpoint):
            out.append(Finding(path, self.from_pretrained_lines[0], "full-finetune-heavy"))
        return out


def lint_file(path: Path) -> list[Finding]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []
    v = _Visitor()
    v.visit(tree)
    rel = str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
    return v.findings(rel)


def collect_files(paths: list[str]) -> list[Path]:
    files = []
    for p in paths:
        p = (ROOT / p).resolve() if not Path(p).is_absolute() else Path(p)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*.py") if "__pycache__" not in f.parts)
        elif p.suffix == ".py":
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=None,
                    help=f"files/dirs to lint (default: {', '.join(DEFAULT_PATHS)})")
    ap.add_argument("--format", choices=["text", "github"], default="text")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any finding fired (default: always exit 0, informational)")
    args = ap.parse_args()

    files = collect_files(args.paths or DEFAULT_PATHS)
    findings = [f for path in files for f in lint_file(path)]

    if args.format == "github":
        print("### green-lint (heuristic, non-blocking by default)\n")
        if not findings:
            print("No patterns flagged. This is a heuristic linter -- absence of "
                  "findings is not proof the code is optimal, just that these "
                  "specific patterns weren't detected.")
        else:
            print("| file:line | rule | suggests |")
            print("|---|---|---|")
            for f in findings:
                msg, pointer = _RULES[f.rule]
                print(f"| `{f.path}:{f.line}` | `{f.rule}` | {pointer} |")
        print(f"\n{len(findings)} finding(s) across {len(files)} file(s). "
              "Pure AST pattern-matching, no type/dataflow analysis -- expect "
              "false positives and false negatives; treat as prompts to look, not verdicts.")
    else:
        print(f"green-lint: scanned {len(files)} file(s)\n")
        for f in findings:
            print(f.render())
        print(f"\n{len(findings)} finding(s). Heuristic AST matching only -- "
              "read a flagged line before acting on it.")

    if args.strict and findings:
        sys.exit(1)


if __name__ == "__main__":
    main()
