"""Shared data helpers. Kept tiny on purpose — recipes should be readable alone."""
from __future__ import annotations

import random
import urllib.request
from pathlib import Path

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_tinyshakespeare() -> str:
    """~1 MB of public-domain Shakespeare. Downloaded once, cached in data/."""
    path = repo_root() / "data" / "tinyshakespeare.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading tiny-shakespeare -> {path}")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    return path.read_text(encoding="utf-8")


def set_seed(seed: int = 42):
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def tokenize_all(text: str, tokenizer, limit_tokens: int | None = None):
    """Tokenize in chunks (avoids the 'sequence too long' warning)."""
    ids, step = [], 20_000
    for i in range(0, len(text), step):
        ids += tokenizer(text[i : i + step], add_special_tokens=False)["input_ids"]
        if limit_tokens and len(ids) >= limit_tokens:
            break
    return ids


def lm_batches(text: str, tokenizer, seq_len: int, batch_size: int,
               n_batches: int, seed: int = 42):
    """Deterministic list of (batch_size, seq_len) LongTensors (kept on CPU)."""
    import torch
    need = seq_len * batch_size * n_batches * 3
    ids = tokenize_all(text, tokenizer, limit_tokens=need)
    rng = random.Random(seed)
    hi = len(ids) - seq_len - 1
    batches = []
    for _ in range(n_batches):
        rows = [ids[s : s + seq_len] for s in (rng.randrange(hi) for _ in range(batch_size))]
        batches.append(torch.tensor(rows, dtype=torch.long))
    return batches


def pick_device(arg: str = "auto") -> str:
    import torch
    if arg == "auto":
        arg = "cuda" if torch.cuda.is_available() else "cpu"
    if arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA GPU found")
    if arg == "cpu":
        print("NOTE: running on CPU — you get wall time only, no VRAM/energy numbers.")
    return arg
