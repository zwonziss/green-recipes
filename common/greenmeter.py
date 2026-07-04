"""
GreenMeter — the lab instrument of this repo.

Wrap any block of code and get a "receipt": wall time, peak VRAM,
GPU energy (integrated from NVML power samples), and estimated CO2.

    from common.greenmeter import GreenMeter, compare, save_result

    with GreenMeter("bf16") as m:
        train(...)
    m.add(final_loss=1.83, tokens_per_s=41_200)
    m.receipt()

    # a single run can't tell a real effect from GPU clock/thermal noise --
    # repeat the trial and aggregate (see any recipe's --repeats flag):
    trials = []
    for _ in range(n_repeats):
        with GreenMeter("bf16") as m:
            train(...)
        trials.append(m.result)
    r = aggregate_trials(trials)   # -> mean +/- std per metric, n_trials

Design notes / honesty:
  * Energy = whole-board GPU power integrated over the block (100 ms samples).
    It INCLUDES idle draw and EXCLUDES CPU/RAM/PSU losses. Treat it as a
    lower bound on true wall-socket energy, and as a fair *relative* metric.
  * Peak VRAM = torch's allocator peak (max_memory_allocated), reset on enter.
  * CO2 uses a grid intensity you can override:  GML_GCO2_PER_KWH=400 python run.py
    Default 450 gCO2/kWh (~global average). Türkiye grid is in the same ballpark.
  * Degrades gracefully: no GPU / no NVML -> you still get wall time.
  * Hardware: NVIDIA (NVML) is the only backend this repo's own runs are
    validated against. AMD (ROCm/pyrsmi), Apple Silicon (powermetrics) and
    Linux CPU (RAPL) backends are implemented against each tool's public
    docs but have NOT been run on that hardware by this repo -- treat their
    numbers as unverified until you (or a contributor) confirm them. See
    "power backend" in env_report() for what was actually detected.
  * Repeats: aggregate_trials() merges N GreenMeter.result dicts into one with
    mean/std per metric. compare()/print_receipt() show "mean +/- std (n=N)"
    and flag deltas with * (95% CIs don't overlap the baseline -- likely real)
    or ~ (CIs overlap -- could be noise). That's a normal-approx heuristic,
    not a real hypothesis test -- it exists to stop you from reading a story
    into single-run jitter, not to replace proper statistics.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

try:  # torch is optional so the meter itself can be unit-tested anywhere
    import torch
except Exception:  # pragma: no cover
    torch = None

GCO2_PER_KWH = float(os.environ.get("GML_GCO2_PER_KWH", 450))
PHONE_CHARGE_WH = 12.0   # ~ one smartphone full charge
LED_BULB_W = 10.0        # ~ one LED bulb


# --------------------------------------------------------------------------- #
# power backends -- anything with .start() / .read_watts() -> float|None / .stop()
# --------------------------------------------------------------------------- #
class _NVMLBackend:
    """NVIDIA GPUs via NVML (nvidia-ml-py). The only backend validated by
    real hardware runs in this repo."""
    name = "nvml"

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._pynvml = None
        self._handle = None

    def start(self):
        import pynvml
        pynvml.nvmlInit()
        self._pynvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

    def read_watts(self):
        return self._pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0

    def stop(self):
        pass  # nvmlShutdown() is process-global; leave it initialized


class _ROCmBackend:
    """AMD GPUs via ROCm SMI (pyrsmi). EXPERIMENTAL: written against pyrsmi's
    documented API (https://github.com/ROCm/pyrsmi), not yet run on real AMD
    hardware by this repo -- please validate and report back if you have a
    ROCm card. Requires `pip install pyrsmi` (not in requirements.txt since
    most users don't have AMD GPUs).
    """
    name = "rocm"

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._rocml = None

    def start(self):
        from pyrsmi import rocml
        rocml.smi_initialize()
        self._rocml = rocml

    def read_watts(self):
        # smi_get_device_average_power returns average socket power in Watts
        return float(self._rocml.smi_get_device_average_power(self.device_index))

    def stop(self):
        try:
            self._rocml.smi_shutdown()
        except Exception:
            pass


class _RAPLBackend:
    """CPU + DRAM package power via Linux's Intel/AMD RAPL counters
    (/sys/class/powercap/intel-rapl*), no extra dependency. EXPERIMENTAL:
    written against the documented sysfs interface, not yet run on real
    hardware by this repo. Many distros restrict reading these counters to
    root or a specific group after CVE-2020-8694 -- if `available()` is True
    but readings never arrive, try running with sudo.
    """
    name = "rapl"
    _ROOT = Path("/sys/class/powercap")

    def __init__(self):
        self._domains = []       # list of (energy_uj_path, max_range_uj|None)
        self._last_uj = []
        self._last_t = None

    def available(self) -> bool:
        return self._ROOT.exists() and any(self._ROOT.glob("intel-rapl:*"))

    def start(self):
        for p in sorted(self._ROOT.glob("intel-rapl:*")):
            # keep only top-level package domains ("intel-rapl:0"), skip
            # sub-domains ("intel-rapl:0:1" = core/uncore/dram) which the
            # package total already includes -- avoids double counting.
            if ":" in p.name[len("intel-rapl:"):]:
                continue
            energy_file = p / "energy_uj"
            if not energy_file.exists():
                continue
            max_file = p / "max_energy_range_uj"
            max_range = int(max_file.read_text()) if max_file.exists() else None
            self._domains.append((energy_file, max_range))
        if not self._domains:
            raise RuntimeError("no readable intel-rapl package domains")
        self._last_uj = [int(f.read_text()) for f, _ in self._domains]
        self._last_t = time.perf_counter()

    def read_watts(self):
        now = time.perf_counter()
        dt = now - self._last_t
        if dt <= 0:
            return None
        total_delta_uj = 0
        new_last = []
        for (f, max_range), prev in zip(self._domains, self._last_uj):
            cur = int(f.read_text())
            delta = cur - prev
            if delta < 0 and max_range:  # 32-bit counter wrapped around
                delta += max_range
            total_delta_uj += delta
            new_last.append(cur)
        self._last_uj = new_last
        self._last_t = now
        return (total_delta_uj / 1e6) / dt  # microjoules -> J, /s -> W

    def stop(self):
        pass


class _AppleBackend:
    """Apple Silicon combined CPU+GPU+ANE power via `powermetrics`.
    EXPERIMENTAL: written against powermetrics' documented plist output, not
    yet run on real Apple Silicon hardware by this repo -- please validate
    and report back if you have an M-series Mac. Requires sudo: macOS
    restricts powermetrics to root. Configure a passwordless sudoers rule
    for powermetrics, or run your script with `sudo python ...`, or this
    backend silently produces no readings (degrades to wall-time only).
    """
    name = "mps"

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self._proc = None
        self._reader_thread = None
        self._latest_w = None
        self._stop = threading.Event()

    def available(self) -> bool:
        import shutil
        return shutil.which("powermetrics") is not None

    def start(self):
        import subprocess
        self._proc = subprocess.Popen(
            ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power",
             "-i", str(self.interval_ms), "--format", "plist"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self):
        import plistlib
        buf = []
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            buf.append(line)
            if line.strip() == "</plist>":
                try:
                    doc = plistlib.loads("".join(buf).encode())
                    proc = doc.get("processor", {})
                    cpu_mw = proc.get("cpu_power", 0)
                    gpu_mw = proc.get("gpu_power", 0)
                    self._latest_w = (cpu_mw + gpu_mw) / 1000.0  # mW -> W
                except Exception:
                    pass
                buf = []

    def read_watts(self):
        return self._latest_w

    def stop(self):
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()


def _detect_power_backend(device_index: int = 0):
    """Best-effort autodetect, in order: NVIDIA (NVML) / AMD (ROCm) > Apple
    Silicon (powermetrics) > Linux CPU (RAPL). Returns None (wall-time-only
    receipts) if nothing usable is found or every candidate fails to start.
    """
    if torch is not None and torch.cuda.is_available():
        is_rocm = bool(getattr(getattr(torch, "version", None), "hip", None))
        backend = _ROCmBackend(device_index) if is_rocm else _NVMLBackend(device_index)
        try:
            backend.start()
            return backend
        except Exception:
            pass
    if (torch is not None and getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()):
        backend = _AppleBackend()
        if backend.available():
            try:
                backend.start()
                return backend
            except Exception:
                pass
    backend = _RAPLBackend()
    if backend.available():
        try:
            backend.start()
            return backend
        except Exception:
            pass
    return None


class _PowerSampler(threading.Thread):
    def __init__(self, backend, interval: float = 0.1):
        super().__init__(daemon=True)
        self.backend = backend
        self.interval = interval
        self.energy_j = 0.0
        self.peak_w = 0.0
        self.samples = 0
        self._stop = threading.Event()

    def run(self):
        last = time.perf_counter()
        while not self._stop.is_set():
            self._stop.wait(self.interval)
            now = time.perf_counter()
            try:
                watts = self.backend.read_watts()
            except Exception:
                continue
            if watts is None:
                continue
            self.energy_j += watts * (now - last)
            last = now
            self.peak_w = max(self.peak_w, watts)
            self.samples += 1

    def stop(self):
        self._stop.set()


# --------------------------------------------------------------------------- #
# the meter
# --------------------------------------------------------------------------- #
class GreenMeter:
    def __init__(self, label: str, device_index: int = 0):
        self.label = label
        self.device_index = device_index
        self.result: dict = {}
        self._sampler = None
        self._backend = None

    def __enter__(self):
        self.cuda = torch is not None and torch.cuda.is_available()
        self._mps = (not self.cuda and torch is not None
                    and getattr(torch.backends, "mps", None) is not None
                    and torch.backends.mps.is_available())
        if self.cuda:
            torch.cuda.synchronize(self.device_index)
            torch.cuda.reset_peak_memory_stats(self.device_index)

        self._backend = _detect_power_backend(self.device_index) if torch is not None else None
        if self._backend is not None:
            self._sampler = _PowerSampler(self._backend)
            self._sampler.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.cuda:
            torch.cuda.synchronize(self.device_index)
        wall = time.perf_counter() - self._t0

        r: dict = {"label": self.label, "wall_s": round(wall, 2)}
        if self.cuda:
            r["peak_vram_mb"] = round(
                torch.cuda.max_memory_allocated(self.device_index) / 2**20, 1
            )
            r["device"] = torch.cuda.get_device_name(self.device_index)
        elif self._mps:
            r["device"] = "Apple Silicon (MPS)"
            try:  # torch.mps memory accounting API varies by torch version
                r["peak_vram_mb"] = round(torch.mps.driver_allocated_memory() / 2**20, 1)
            except Exception:
                pass

        if self._sampler is not None:
            self._sampler.stop()
            self._sampler.join(timeout=2.0)
            try:
                self._backend.stop()
            except Exception:
                pass
            if self._sampler.samples > 0:  # a backend that started but never got
                ej = self._sampler.energy_j  # a real reading shouldn't report "0 Wh"
                r["gpu_energy_wh"] = round(ej / 3600.0, 4)
                r["gpu_avg_power_w"] = round(ej / wall, 1) if wall > 0 else 0.0
                r["gpu_peak_power_w"] = round(self._sampler.peak_w, 1)
                r["co2_g"] = round(ej / 3600.0 / 1000.0 * GCO2_PER_KWH, 3)
                r["power_source"] = self._backend.name
        self.result = r
        return False  # never swallow exceptions

    def add(self, **extra):
        """Attach recipe-specific numbers (final_loss, tokens_per_s, accuracy...)."""
        self.result.setdefault("extra", {}).update(
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in extra.items()}
        )
        return self

    def receipt(self):
        print_receipt(self.result)
        return self.result


# --------------------------------------------------------------------------- #
# repeated trials -> mean +/- std (a single run can't tell signal from GPU noise)
# --------------------------------------------------------------------------- #
_AGGREGATE_KEYS = ("wall_s", "peak_vram_mb", "gpu_energy_wh",
                   "gpu_avg_power_w", "gpu_peak_power_w", "co2_g")


def mean_std(xs: list[float]) -> tuple[float, float]:
    """Sample mean and (n-1)-denominator std. std is 0.0 for a single sample."""
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def aggregate_trials(trials: list[dict]) -> dict:
    """Collapse N repeated GreenMeter.result dicts (same label/variant) into one:
    each numeric field becomes its mean, with a f"{field}_std" sibling, plus
    n_trials and the raw per-trial dicts (kept for the saved JSON, not for
    display) so repeats show up as receipts you can audit, not a black box.
    """
    if len(trials) == 1:
        r = dict(trials[0])
        r["n_trials"] = 1
        r["trials"] = trials
        return r

    r: dict = {"label": trials[0].get("label"), "n_trials": len(trials)}
    for passthrough in ("device", "power_source"):
        if passthrough in trials[0]:
            r[passthrough] = trials[0][passthrough]

    for key in _AGGREGATE_KEYS:
        vals = [t[key] for t in trials if key in t]
        if vals:
            m, s = mean_std(vals)
            r[key] = round(m, 4)
            r[f"{key}_std"] = round(s, 4)

    extra_keys = {k for t in trials for k in t.get("extra", {})}
    extra: dict = {}
    for k in extra_keys:
        vals = [t["extra"][k] for t in trials if k in t.get("extra", {})]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            m, s = mean_std(vals)
            extra[k] = round(m, 4)
            if len(vals) > 1:
                extra[f"{k}_std"] = round(s, 4)
        else:
            extra[k] = vals[-1]  # non-numeric (e.g. optimizer name) -> last trial's
    r["extra"] = extra
    r["trials"] = trials
    return r


def ci95_overlap(mean_a, std_a, n_a, mean_b, std_b, n_b) -> bool | None:
    """Do the two ~95% CIs (normal approx, not a real t-test) overlap?
    None if either side has < 2 trials (nothing to compare). This is a
    cheap noise-vs-signal heuristic, not a hypothesis test -- treat it that way.
    """
    if n_a < 2 or n_b < 2:
        return None
    half_a = 1.96 * std_a / math.sqrt(n_a)
    half_b = 1.96 * std_b / math.sqrt(n_b)
    lo_a, hi_a = mean_a - half_a, mean_a + half_a
    lo_b, hi_b = mean_b - half_b, mean_b + half_b
    return not (hi_a < lo_b or hi_b < lo_a)


# --------------------------------------------------------------------------- #
# pretty printing
# --------------------------------------------------------------------------- #
def equivalents(wh: float) -> str:
    if not wh:
        return ""
    phones = wh / PHONE_CHARGE_WH
    led_min = wh / LED_BULB_W * 60
    return f"~= {phones:.2f} phone charges, {led_min:.0f} min of an LED bulb"


def _pm(r: dict, key: str, unit: str = "") -> str:
    """'value' or 'mean +/- std' (n>1) for a top-level metric, with a unit suffix."""
    v = r.get(key)
    std = r.get(f"{key}_std")
    if std:
        return f"{v}{unit} +/- {std}{unit}"
    return f"{v}{unit}"


def print_receipt(r: dict):
    line = "-" * 58
    n = r.get("n_trials", 1)
    title = r.get("label")
    if n > 1:
        title = f"{title}  (mean +/- std, n={n})"
    print(f"\n{line}\n  RECEIPT: {title}\n{line}")
    if "device" in r:
        print(f"  device        : {r['device']}")
    print(f"  wall time     : {_pm(r, 'wall_s', ' s')}")
    if "peak_vram_mb" in r:
        print(f"  peak VRAM     : {_pm(r, 'peak_vram_mb', ' MB')}")
    if "gpu_energy_wh" in r:
        print(f"  GPU energy    : {_pm(r, 'gpu_energy_wh', ' Wh')} "
              f"(avg {_pm(r, 'gpu_avg_power_w', ' W')}, "
              f"peak {_pm(r, 'gpu_peak_power_w', ' W')})")
        print(f"  est. CO2      : {_pm(r, 'co2_g', ' g')}  @ {GCO2_PER_KWH:.0f} gCO2/kWh")
        eq = equivalents(r["gpu_energy_wh"])
        if eq:
            print(f"  in real life  : {eq}")
        if r.get("power_source") and r["power_source"] != "nvml":
            print(f"  power source  : {r['power_source']} (experimental backend, see greenmeter.py)")
    for k, v in r.get("extra", {}).items():
        if k.endswith("_std"):
            continue
        std = r.get("extra", {}).get(f"{k}_std")
        print(f"  {k:<14}: {v}" + (f" +/- {std}" if std else ""))
    print(line)


_METRICS = [
    # key             header            lower-is-better (show delta %)
    ("wall_s",        "time (s)",       True),
    ("peak_vram_mb",  "VRAM (MB)",      True),
    ("gpu_energy_wh", "energy (Wh)",    True),
    ("gpu_avg_power_w", "avg W",        False),
    ("co2_g",         "CO2 (g)",        True),
]


def _fmt(v, std=None):
    if v is None:
        return "OOM"
    s = f"{v:g}" if isinstance(v, float) else str(v)
    if std:
        s += f" +/-{std:g}"
    return s


def _rows(results, extra_keys):
    base = results[0]
    active = [(k, h, d) for k, h, d in _METRICS if any(k in r for r in results)]
    headers = ["variant"] + [h for _, h, _ in active] + list(extra_keys)
    any_repeats = any(r.get("n_trials", 1) > 1 for r in results)
    rows = []
    for r in results:
        row = [str(r.get("label", "?"))]
        for k, _, delta in active:
            v = r.get(k)
            cell = _fmt(v, r.get(f"{k}_std"))
            if (delta and r is not base and isinstance(v, (int, float))
                    and isinstance(base.get(k), (int, float)) and base[k]):
                pct = (v - base[k]) / base[k] * 100
                marker = ""
                if any_repeats:
                    overlap = ci95_overlap(
                        v, r.get(f"{k}_std") or 0.0, r.get("n_trials", 1),
                        base.get(k), base.get(f"{k}_std") or 0.0, base.get("n_trials", 1))
                    marker = " ~" if overlap else (" *" if overlap is False else "")
                cell += f" ({pct:+.0f}%{marker})"
            row.append(cell)
        for k in extra_keys:
            extra = r.get("extra", {})
            row.append(_fmt(extra.get(k), extra.get(f"{k}_std")))
        rows.append(row)
    return headers, rows


def print_table(headers, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*r))


def to_markdown(results, extra_keys=()) -> str:
    headers, rows = _rows(results, extra_keys)
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def compare(results, extra_keys=(), markdown=True):
    """Print a comparison table; first result is the baseline for the % deltas."""
    headers, rows = _rows(results, extra_keys)
    print()
    print_table(headers, rows)
    if any(r.get("n_trials", 1) > 1 for r in results):
        print("\n(* = 95% CIs vs. baseline don't overlap, likely a real effect. "
              "~ = they overlap, could be GPU noise. Normal-approx CIs, not a "
              "real hypothesis test -- treat as a heuristic.)")
    if markdown:
        print("\nMarkdown (paste anywhere):\n")
        print(to_markdown(results, extra_keys))
    print()


# --------------------------------------------------------------------------- #
# persistence (feeds tools/build_table.py)
# --------------------------------------------------------------------------- #
def save_result(result: dict, recipe: str, results_dir, is_baseline: bool = False):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(result)
    payload["recipe"] = recipe
    payload["is_baseline"] = is_baseline
    payload["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["grid_gco2_per_kwh"] = GCO2_PER_KWH
    fname = results_dir / f"{result.get('label', 'run').replace(' ', '_')}.json"
    fname.write_text(json.dumps(payload, indent=2))
    print(f"saved -> {fname}")
    return fname


# --------------------------------------------------------------------------- #
# environment doctor (used by recipe 00)
# --------------------------------------------------------------------------- #
def _probe_power_backend(device_index: int = 0):
    """Detect + start a backend, wait long enough for one real sample, read
    it, then stop. Used only by env_report()'s one-off diagnostic print."""
    backend = _detect_power_backend(device_index)
    if backend is None:
        return None, None
    time.sleep(0.35)  # let RAPL see a nonzero dt, let powermetrics emit a block
    try:
        w = backend.read_watts()
    except Exception:
        w = None
    try:
        backend.stop()
    except Exception:
        pass
    return backend.name, w


def env_report():
    print("\n=== environment check ===")
    if torch is None:
        print("torch          : NOT INSTALLED")
        return
    print(f"torch          : {torch.__version__}")

    if torch.cuda.is_available():
        i = 0
        props = torch.cuda.get_device_properties(i)
        cap = (props.major, props.minor)
        is_rocm = bool(getattr(getattr(torch, "version", None), "hip", None))
        print(f"GPU            : {props.name}  ({props.total_memory / 2**30:.1f} GB)"
              + ("  [ROCm]" if is_rocm else ""))
        if not is_rocm:
            print(f"compute cap.   : sm_{props.major}{props.minor}")
            print(f"bf16 support   : {torch.cuda.is_bf16_supported()}")
            print(f"tf32 support   : {cap >= (8, 0)}  (Ampere+)")
            print(f"flash SDPA     : {'likely (Ampere+)' if cap >= (8, 0) else 'no -> falls back to mem-efficient SDPA'}")
        try:
            import bitsandbytes  # noqa: F401
            print("bitsandbytes   : OK")
        except Exception:
            print("bitsandbytes   : not installed (needed for recipes 03 & 08)")
    elif (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
        print("GPU            : Apple Silicon (MPS)")
        print("note           : bf16/tf32/flash-SDPA checks above are CUDA-specific and don't apply here")
    else:
        print("CUDA/MPS       : not available (recipes needing a GPU will exit; "
              "others run CPU/time-only unless a power backend is found below)")

    name, w = _probe_power_backend(0)
    if name is None:
        print("power backend  : none detected -> energy/CO2 columns will be empty")
    elif w is None:
        print(f"power backend  : {name} started but produced no reading "
              "(permissions? try sudo) -> energy/CO2 columns will be empty")
    else:
        experimental = "" if name == "nvml" else "  [EXPERIMENTAL -- see greenmeter.py]"
        print(f"power backend  : {name} OK (idle draw right now: {w:.0f} W){experimental}")

    print(f"grid intensity : {GCO2_PER_KWH:.0f} gCO2/kWh (override with GML_GCO2_PER_KWH)")
    print("=========================\n")
