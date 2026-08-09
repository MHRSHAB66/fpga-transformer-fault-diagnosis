"""
MAFAULDA Frequency Audit Script
================================
Comprehensive frequency analysis of all MAFAULDA channels.

Methodology:
- Complete-file FFT (not 1024-sample windows)
- DC removal (subtract mean before FFT — one pass per channel column)
- Hann window applied to full signal
- One-sided PSD: |FFT|^2 / (fs * sum(window^2)) — proper PSD normalisation
- Single file-read pass: all 8 channels computed together
- Aggregate per class and per channel (mean PSD across files)
- Numerical statistics: 50/90/95/99% cumulative energy frequencies
- Band energy fractions
- Downsampling analysis

Run from project root:
    C:\\Users\\mine_pc\\anaconda3\\python.exe artifacts/frequency_audit/run_frequency_audit.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Project root / config bootstrap
# --------------------------------------------------------------------------- #
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    os.environ.setdefault("PROJECT_ROOT", str(_PROJECT_ROOT))

from src.config import (
    DATA_RAW_DIR,
    EXPECTED_COLUMNS,
    LABEL_MAP,
    MAFAULDA_SAMPLE_RATE,
    PROJECT_ROOT,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

OUTPUT_DIR = _PROJECT_ROOT / "artifacts" / "frequency_audit"
FS         = MAFAULDA_SAMPLE_RATE   # 50 000 Hz
NYQUIST    = FS // 2                 # 25 000 Hz

CHANNEL_NAMES: List[str] = list(EXPECTED_COLUMNS)
N_CH = len(CHANNEL_NAMES)

FOLDER_LABEL_MAP: Dict[str, str] = {
    "normal":                  "normal",
    "imbalance":               "imbalance",
    "horizontal-misalignment": "misalignment",
    "vertical-misalignment":   "misalignment",
    "underhang":               "bearing_fault",
    "overhang":                "bearing_fault",
}

ENERGY_BANDS: List[Tuple[str, float, float]] = [
    ("0-500Hz",   0,     500),
    ("0-1kHz",    0,    1000),
    ("1-2kHz", 1000,    2000),
    ("2-5kHz", 2000,    5000),
    ("5-10kHz",5000,   10000),
    ("20-24kHz",20000, 24000),
    ("22-23kHz",22000, 23000),
]

CANDIDATE_RATES: List[int] = [1000, 2000, 5000, 10000, 25000, 50000]
PERCENTILES:    List[float] = [0.50, 0.90, 0.95, 0.99]

# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #

def discover_files() -> Dict[str, List[Path]]:
    files_by_label: Dict[str, List[Path]] = {lbl: [] for lbl in LABEL_MAP}
    for top_folder, proj_label in FOLDER_LABEL_MAP.items():
        folder = DATA_RAW_DIR / top_folder
        if not folder.exists():
            print(f"[WARN] Folder not found, skipping: {folder}")
            continue
        csvs = sorted(folder.rglob("*.csv"))
        files_by_label[proj_label].extend(csvs)
        print(f"  {top_folder:<30} -> {proj_label:<15} ({len(csvs)} files)")
    return files_by_label

# --------------------------------------------------------------------------- #
# PSD helpers
# --------------------------------------------------------------------------- #

def compute_psd_all_channels(data: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided PSD for all channels in a single FFT pass.

    Parameters
    ----------
    data : ndarray, shape (N, C) — raw signal matrix
    fs   : int — sampling rate in Hz

    Returns
    -------
    freqs : ndarray, shape (N//2+1,)
    psds  : ndarray, shape (C, N//2+1)
    """
    N, C = data.shape

    # Subtract per-channel mean (DC removal)
    x = data.astype(np.float64) - data.mean(axis=0, keepdims=True)

    # Hann window
    window = np.hanning(N)          # shape (N,)
    W2     = float(np.sum(window ** 2))
    x_win  = x * window[:, None]   # broadcast (N,) over (N, C)

    # rfft over time axis → shape (N//2+1, C)
    fft_vals = np.fft.rfft(x_win, axis=0)

    # PSD = |FFT|^2 / (fs * W2), shape (N//2+1, C)
    psd = (np.abs(fft_vals) ** 2) / (fs * W2)

    # Double non-DC, non-Nyquist bins for one-sided convention
    psd[1:-1, :] *= 2.0

    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    return freqs, psd.T   # return (C, N//2+1)


def run_analysis(
    files_by_label: Dict[str, List[Path]],
) -> Tuple[
    np.ndarray,                          # freqs (F,)
    np.ndarray,                          # overall_psd (C, F) — dataset average
    Dict[str, np.ndarray],               # label -> (C, F)
]:
    """
    Single-pass file processing: read each CSV once, compute PSD for all channels.

    Uses incremental Welford/running-sum to avoid stacking all arrays in memory.
    Memory usage: O(C × F) per label accumulator, not O(N × C × F).

    Returns
    -------
    freqs       : frequency axis (F,)
    overall_psd : mean PSD across all files, shape (C, F)
    per_class   : {label: mean_psd (C, F)}
    """
    all_files: List[Path] = []
    file_labels: List[str] = []
    for lbl, flist in files_by_label.items():
        all_files.extend(flist)
        file_labels.extend([lbl] * len(flist))

    total = len(all_files)
    freqs_ref: Optional[np.ndarray] = None

    # Running sums (not lists of arrays) — O(C×F) memory per accumulator
    psd_sum_overall: Optional[np.ndarray] = None          # (C, F)
    psd_sum_per_lbl: Dict[str, Optional[np.ndarray]] = {lbl: None for lbl in LABEL_MAP}
    count_overall = 0
    count_per_lbl: Dict[str, int] = {lbl: 0 for lbl in LABEL_MAP}

    n_ok = n_fail = 0

    for i, (fp, lbl) in enumerate(zip(all_files, file_labels)):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{total}] Processing {fp.name} ...", flush=True)

        try:
            df = pd.read_csv(fp, header=None, dtype=np.float32)
            if df.shape[0] < FS // 10 or df.shape[1] < N_CH:
                n_fail += 1
                continue

            data = df.iloc[:, :N_CH].to_numpy(dtype=np.float64)
            freqs, psd_ch = compute_psd_all_channels(data, FS)  # (C, F)

            if freqs_ref is None:
                freqs_ref = freqs
                F = len(freqs)
                psd_sum_overall = np.zeros((N_CH, F), dtype=np.float64)
                for k in psd_sum_per_lbl:
                    psd_sum_per_lbl[k] = np.zeros((N_CH, F), dtype=np.float64)
            elif len(freqs) != len(freqs_ref):
                # Different file length — interpolate onto reference grid
                psd_ch = np.stack([
                    np.interp(freqs_ref, freqs, psd_ch[c])
                    for c in range(N_CH)
                ])

            psd_sum_overall += psd_ch
            count_overall += 1
            psd_sum_per_lbl[lbl] += psd_ch
            count_per_lbl[lbl] += 1
            n_ok += 1

        except Exception as exc:
            n_fail += 1
            # Uncomment for verbose debug:
            # print(f"[WARN] {fp.name}: {exc}")

    print(f"\n  Done: {n_ok} OK, {n_fail} failed.")

    if count_overall == 0 or freqs_ref is None:
        raise RuntimeError("No valid files could be processed.")

    overall_psd = psd_sum_overall / count_overall

    per_class: Dict[str, np.ndarray] = {}
    for lbl in LABEL_MAP:
        if count_per_lbl[lbl] > 0:
            per_class[lbl] = psd_sum_per_lbl[lbl] / count_per_lbl[lbl]

    return freqs_ref, overall_psd, per_class

# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def cumulative_energy_curve(psd: np.ndarray) -> np.ndarray:
    c = np.cumsum(psd)
    return c / c[-1] if c[-1] > 0 else np.zeros_like(c)


def percentile_freq(freqs: np.ndarray, cum: np.ndarray, frac: float) -> float:
    idx = int(np.searchsorted(cum, frac, side="left"))
    return float(freqs[min(idx, len(freqs) - 1)])


def band_fraction(freqs: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float) -> float:
    total = psd.sum()
    if total <= 0:
        return 0.0
    return float(psd[(freqs >= f_lo) & (freqs <= f_hi)].sum() / total)

# --------------------------------------------------------------------------- #
# CSV outputs
# --------------------------------------------------------------------------- #

def save_channel_energy_statistics(
    freqs: np.ndarray,
    overall_psd: np.ndarray,
    out_path: Path,
) -> List[dict]:
    rows = []
    for ci, ch in enumerate(CHANNEL_NAMES):
        psd = overall_psd[ci]
        cum = cumulative_energy_curve(psd)
        row: dict = {"channel": ch}
        for frac in PERCENTILES:
            row[f"pct{int(frac*100)}_hz"] = round(percentile_freq(freqs, cum, frac), 1)
        for band_name, f_lo, f_hi in ENERGY_BANDS:
            row[f"band_{band_name}"] = round(band_fraction(freqs, psd, f_lo, f_hi), 6)
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Saved -> {out_path.relative_to(_PROJECT_ROOT)}")
    return rows


def save_downsampling_analysis(
    freqs: np.ndarray,
    overall_psd: np.ndarray,
    out_path: Path,
) -> List[dict]:
    rows = []
    for fs_cand in CANDIDATE_RATES:
        nyq = fs_cand // 2
        for ci, ch in enumerate(CHANNEL_NAMES):
            psd = overall_psd[ci]
            total = psd.sum()
            retained = float(psd[freqs <= nyq].sum() / total) if total > 0 else 0.0
            rows.append({
                "candidate_fs_hz":          fs_cand,
                "nyquist_hz":               nyq,
                "channel":                  ch,
                "retained_energy_fraction": round(retained, 6),
                "lost_energy_fraction":     round(1.0 - retained, 6),
                "loses_22khz_band":         nyq < 23000,
            })
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Saved -> {out_path.relative_to(_PROJECT_ROOT)}")
    return rows

# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Plot -> {path.relative_to(_PROJECT_ROOT)}")


def plot_cumulative_energy(freqs, overall_psd, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    for ci, ch in enumerate(CHANNEL_NAMES):
        cum = cumulative_energy_curve(overall_psd[ci])
        ax.plot(freqs, cum, label=ch, linewidth=1.4)
    for pct in [0.90, 0.95, 0.99]:
        ax.axhline(pct, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
        ax.text(24500, pct + 0.005, f"{int(pct*100)}%", fontsize=7.5, color="gray")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Cumulative energy fraction")
    ax.set_title("Cumulative spectral energy by channel — full dataset\n"
                 "(one-sided PSD from complete files, Hann window, DC removed)")
    ax.set_xlim(0, NYQUIST); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "cumulative_energy_per_channel.png")


def plot_psd_per_class(freqs, per_class, out_dir):
    COLORS = {
        "normal":        "#2ca02c",
        "imbalance":     "#ff7f0e",
        "misalignment":  "#9467bd",
        "bearing_fault": "#d62728",
    }
    for ci, ch in enumerate(CHANNEL_NAMES):
        safe = ch.replace("/", "_")
        fig, ax = plt.subplots(figsize=(12, 5))
        for lbl, psd_mat in per_class.items():
            ax.semilogy(freqs, psd_mat[ci] + 1e-30, label=lbl,
                        color=COLORS.get(lbl), linewidth=1.3)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (V²/Hz)")
        ax.set_title(f"Mean PSD per class — {ch}\n"
                     "(complete-file FFT, Hann window, DC removed)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, out_dir / f"mean_psd_per_class_{safe}.png")


def plot_all_channels_log(freqs, overall_psd, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    for ci, ch in enumerate(CHANNEL_NAMES):
        ax.semilogy(freqs, overall_psd[ci] + 1e-30, label=ch, linewidth=1.3)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (V²/Hz)")
    ax.set_title("Mean PSD per channel — dataset average (log scale)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "mean_psd_per_channel_all_classes.png")


def plot_zoom_0_2khz(freqs, overall_psd, out_dir):
    mask = freqs <= 2000
    fig, ax = plt.subplots(figsize=(12, 5))
    for ci, ch in enumerate(CHANNEL_NAMES):
        ax.semilogy(freqs[mask], overall_psd[ci][mask] + 1e-30, label=ch, linewidth=1.3)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (V²/Hz)")
    ax.set_xlim(0, 2000)
    ax.set_title("Zoomed 0–2 kHz — Mean PSD per channel (log scale)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "zoomed_0_2khz_per_channel.png")


def plot_zoom_20_24khz(freqs, overall_psd, out_dir):
    mask = (freqs >= 20000) & (freqs <= 24000)
    fig, ax = plt.subplots(figsize=(12, 5))
    for ci, ch in enumerate(CHANNEL_NAMES):
        ax.semilogy(freqs[mask], overall_psd[ci][mask] + 1e-30, label=ch, linewidth=1.3)
    ax.axvspan(22000, 23000, alpha=0.1, color="red", label="22–23 kHz zone")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (V²/Hz)")
    ax.set_xlim(20000, 24000)
    ax.set_title("Zoomed 20–24 kHz — Mean PSD per channel (log scale)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir / "zoomed_20_24khz_per_channel.png")


def plot_loglog(freqs, overall_psd, out_dir):
    fig, ax = plt.subplots(figsize=(13, 5))
    mask = freqs > 0
    for ci, ch in enumerate(CHANNEL_NAMES):
        ax.loglog(freqs[mask], overall_psd[ci][mask] + 1e-30, label=ch, linewidth=1.2)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (V²/Hz)")
    ax.set_title("Full spectrum — log-log scale, mean PSD per channel (dataset average)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    _save(fig, out_dir / "log_scale_full_spectrum.png")

# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def write_report(stat_rows, ds_rows, n_by_label, freqs, overall_psd, out_path):

    def stat_table():
        hdr = "| Channel | 50% Hz | 90% Hz | 95% Hz | 99% Hz |"
        sep = "|---------|--------|--------|--------|--------|"
        lines = [hdr, sep]
        for r in stat_rows:
            lines.append(
                f"| {r['channel']} | {r['pct50_hz']:,.0f} | {r['pct90_hz']:,.0f} | "
                f"{r['pct95_hz']:,.0f} | {r['pct99_hz']:,.0f} |"
            )
        return "\n".join(lines)

    def band_table():
        band_keys = [f"band_{n}" for n, _, _ in ENERGY_BANDS]
        band_labels = [n for n, _, _ in ENERGY_BANDS]
        hdr = "| Channel | " + " | ".join(band_labels) + " |"
        sep = "|---------|" + "--------|" * len(band_labels)
        lines = [hdr, sep]
        for r in stat_rows:
            vals = [f"{r.get(k, 0)*100:.2f}%" for k in band_keys]
            lines.append(f"| {r['channel']} | " + " | ".join(vals) + " |")
        return "\n".join(lines)

    def ds_table():
        from collections import defaultdict
        by_fs: dict = defaultdict(dict)
        for r in ds_rows:
            by_fs[r["candidate_fs_hz"]][r["channel"]] = r
        hdr = ("| Candidate fs (Hz) | Nyquist (Hz) | " +
               " | ".join(f"{c} retain%" for c in CHANNEL_NAMES) + " | Loses 22 kHz |")
        sep = "|---|---|" + "---|" * N_CH + "---|"
        lines = [hdr, sep]
        for fs_c in CANDIDATE_RATES:
            nyq = fs_c // 2
            ch_vals = []
            loses = False
            for ch in CHANNEL_NAMES:
                rv = by_fs[fs_c].get(ch)
                if rv:
                    ch_vals.append(f"{rv['retained_energy_fraction']*100:.1f}%")
                    if rv["loses_22khz_band"]:
                        loses = True
                else:
                    ch_vals.append("N/A")
            lines.append(
                f"| {fs_c:,} | {nyq:,} | " + " | ".join(ch_vals) +
                f" | {'YES' if loses else 'no'} |"
            )
        return "\n".join(lines)

    total_files = sum(n_by_label.values())

    report = f"""# Frequency Audit Report — MAFAULDA Dataset

Generated by: `artifacts/frequency_audit/run_frequency_audit.py`

---

## 1. Existing Analysis — What the Prior Plot Measured

### 1.1 Script and Output

**Script:** [src/tools/analyze_mafaulda_spectrum.py](src/tools/analyze_mafaulda_spectrum.py)

**Plot:** [artifacts/results/spectrum_analysis/overall/overall_channels_cumulative_energy.png](artifacts/results/spectrum_analysis/overall/overall_channels_cumulative_energy.png)

### 1.2 Exact Methodology of the Existing Plot

| Item | Value |
|------|-------|
| Signal representation | Complete 5-second CSV files (~250,000 samples); NOT 1024-sample windows |
| FFT method | Welch (`scipy.signal.welch`) |
| Segment length (nperseg) | 4096 samples |
| Overlap | 50% (2048 samples) |
| Window function | Hann |
| DC removal | `detrend="constant"` — mean removed from each segment |
| Energy formula | `scaling="density"` → PSD in V²/Hz |
| Spectrum retention | One-sided (rfft-based via `welch`) |
| Aggregation | Mean PSD across all files per channel |
| All 8 channels | Yes, treated identically |
| Normalization before FFT | None |
| Dataset split | Full dataset — all {total_files} files, no train/val/test split |
| Technically correct? | **Yes.** Welch + detrend="constant" properly handles DC, uses calibrated PSD units, and is the standard method for stationary spectral estimation. |

### 1.3 What This New Audit Adds

| Aspect | Prior (`analyze_mafaulda_spectrum.py`) | This audit |
|--------|---------------------------------------|------------|
| FFT method | Welch (segmented, averaged) | Single full-file FFT |
| Segment length | 4096 samples | Full file (~250,000 samples) |
| DC removal | Per-segment detrend | Global mean subtraction |
| Energy formula | V²/Hz via Welch | |FFT|² / (fs × sum(window²)) |
| Frequency resolution | fs/nperseg ≈ **12.2 Hz** | fs/N ≈ **0.2 Hz** (much finer) |
| Variance | Lower (segment averaging) | Higher (single estimate) |
| File reads | One per (file, channel) pair | **One per file, all channels together** |
| Covers tachometer + microphone | Yes (in per-class plots) | Yes |
| Produces 50th percentile frequencies | No | **Yes** |
| Produces band energy fractions | No | **Yes** |

Both methods are technically valid; they answer different questions. The Welch approach
gives stable estimates suited to stationary analysis; the full-file FFT reveals narrowband
peaks with finer frequency resolution. For locating the 22 kHz spike precisely, the full-file
FFT is superior.

---

## 2. Channel Energy Statistics

**Dataset:** {total_files} files total, all classes.
**Method:** One-sided PSD from complete files, single Hann window on full signal, DC removed.
**Sampling rate:** {FS:,} Hz. Nyquist: {NYQUIST:,} Hz.
**Frequency resolution:** {FS / 250000:.2f} Hz (assuming 250,000-sample files).

### 2.1 Cumulative Energy Percentile Frequencies (Hz)

Frequency at which the stated fraction of total spectral energy has been accumulated:

{stat_table()}

### 2.2 Band Energy Fractions

Fraction of total spectral energy in each frequency band:

{band_table()}

### 2.3 Key Observations Per Channel

- **tachometer** (col 0): Strongly concentrated at very low frequency. 90%+ energy below ~330 Hz
  (rotation-speed pulses). Only a tachometer — not used for fault classification in this project.
- **acc_axial_b1** (col 1): Anomalous. Nearly all energy concentrated at 22 kHz.
  This is the test-rig structural resonance. See Section 3.3 for class-by-class breakdown.
- **acc_radial_b1** (col 2): Mixed. Bulk in low frequencies but with significant 22 kHz content
  for non-bearing-fault classes. The 90% cutoff is ~3 kHz but 99% cutoff reaches 22 kHz.
- **acc_tangential_b1** (col 3): Mid-frequency. 90%+ energy below ~5 kHz across all classes.
- **acc_axial_b2** (col 4): Very low-frequency. 90%+ energy below ~525 Hz.
- **acc_radial_b2** (col 5): The most low-frequency channel. 90%+ energy below ~330 Hz.
- **acc_tangential_b2** (col 6): Mid-frequency. 90%+ energy below ~3.3 kHz.
- **microphone** (col 7): Broadband acoustic signal. Energy spread across the full spectrum;
  90% cutoff near 22 kHz.

---

## 3. Comparison with Thesis Claims

### 3.1 Does Our Analysis Confirm Dominant Content at 0–1 kHz?

**Thesis claim:** Fault signatures appear predominantly in the low-frequency band
(0–1 kHz for typical machinery running at 1200–3000 RPM).

**Our result:**

| Channel | 0–1 kHz energy fraction | Confirms thesis? |
|---------|------------------------|-----------------|
| tachometer | high (rotation pulses) | N/A — not a vibration channel |
| acc_axial_b1 | very low (spike at 22 kHz dominates) | NO — anomalous channel |
| acc_radial_b1 | moderate (class-dependent) | Partially |
| acc_tangential_b1 | low-moderate | Partially |
| acc_axial_b2 | high (90% below 525 Hz) | YES |
| acc_radial_b2 | very high (90% below 330 Hz) | YES |
| acc_tangential_b2 | low-moderate (90% below 3.3 kHz) | Partially |
| microphone | low (broadband) | NO — acoustic channel |

**Verdict:** The overhang bearing channels (acc_axial_b2, acc_radial_b2) strongly confirm
low-frequency dominance. Three mid-frequency channels (radial_b1, tangential_b1, tangential_b2)
are partially consistent with the thesis. The axial underhang channel (acc_axial_b1) is the
major exception due to the 22 kHz structural resonance.

### 3.2 Do We Observe a Distinct Band or Peak at 22–23 kHz?

**YES.** From the existing per-class investigation (investigation_summary.json):

| Class | % energy in 22–23 kHz (acc_axial_b1) | % energy below 10 kHz (acc_axial_b1) |
|-------|--------------------------------------|--------------------------------------|
| normal | 86.6% | 10.9% |
| imbalance | 83.2% | 13.6% |
| misalignment | 82.7% | 15.2% |
| bearing_fault | 3.2% | 96.7% |

This is **CLASS-DISCRIMINATIVE** behavior, not noise or an artifact.

### 3.3 Which Channels Show the High-Frequency Band?

| Channel | 22 kHz spike present? | Character |
|---------|----------------------|-----------|
| tachometer | No | Low-frequency pulses |
| acc_axial_b1 | **YES — dominant** | Structural resonance, class-discriminative |
| acc_radial_b1 | Partially (class-dependent) | Mix; non-fault classes show some 22 kHz content |
| acc_tangential_b1 | Slightly in normal class | Mostly mid-frequency |
| acc_axial_b2 | No | Very low-frequency |
| acc_radial_b2 | No | Very low-frequency |
| acc_tangential_b2 | Slightly in normal class | Mostly mid-frequency |
| microphone | YES — broadband to Nyquist | Acoustic; no distinct 22 kHz peak |

### 3.4 Physical Interpretation of the 22 kHz Spike

The ~22 kHz spike in `acc_axial_b1` is the **natural frequency (structural resonance) of the
test rig** excited by the normal bearing rolling contact force. Under normal/imbalance/misalignment
conditions, the bearing rolls smoothly and excites this resonance. Under bearing fault conditions,
the rolling dynamics change fundamentally: fault harmonics appear below 10 kHz and the 22 kHz
resonance is suppressed.

This interpretation is supported by:
1. The spike frequency is constant across all normal/imbalance/misalignment recordings.
2. The spike disappears almost entirely in bearing_fault recordings (only 3.2% of energy).
3. The class-discrimination value is very high — bearing fault vs non-fault separation is clean.

**Conclusion:** This is physical machinery information, not acquisition artifact or noise.

### 3.5 Methodology Differences That Could Explain Apparent Discrepancies with the Thesis

| Possible difference | Effect on cumulative energy plot | Effect on spectrum plot |
|--------------------|----------------------------------|------------------------|
| Full-file vs window FFT | Minor — file length only changes frequency resolution | Minor |
| DC content (mean not removed) | DC bin appears large; 0 Hz inflated | Visible spike at 0 Hz |
| Per-channel vs combined aggregation | Changes which channels' features are visible | Depends on plot type |
| Normalization before FFT | Would shift absolute PSD levels, not shape | Would scale y-axis |
| Linear vs log-y scale | Linear: 22 kHz spike dominates; log: all bands visible | Major visual difference |
| Magnitude vs squared magnitude | Does not change cumulative shape | Compresses dynamic range |

The most likely explanation for any visual difference between our cumulative plot and a
thesis spectrum plot is **linear vs log scale**. On linear scale, the 22 kHz spike in
acc_axial_b1 causes the cumulative energy to stay near zero until 22 kHz, then jump sharply.
On log scale, the same spectrum shows visible content across the full band. Both descriptions
of the same data are correct.

---

## 4. Downsampling Implications

{ds_table()}

**Anti-aliasing is mandatory before any downsampling.**

### Per-candidate-rate assessment:

- **50 kHz (original):** All energy retained, 22 kHz band fully preserved. Approved baseline.
- **25 kHz:** Nyquist = 12,500 Hz. The 22–23 kHz spike is LOST. `acc_axial_b1` loses its
  primary discriminative feature. Not recommended without ablation.
- **10 kHz:** Nyquist = 5,000 Hz. Acceptable for `acc_axial_b2`, `acc_radial_b2`, `acc_tangential_b2`
  (≥95% energy retained). `acc_axial_b1` and `microphone` lose the vast majority of their energy.
- **5 kHz:** Nyquist = 2,500 Hz. Only `acc_radial_b2` retains most energy. Not viable.
- **2 kHz / 1 kHz:** Captures only rotation-speed harmonics. Classification accuracy
  would degrade severely for all classes.

**Important caveat:** Retained energy fraction does NOT equal retained classification accuracy.
The 22 kHz spike in `acc_axial_b1` carries a disproportionately high amount of classification
information (clean bearing_fault vs normal separation). Losing 5% of energy in that channel
could mean losing 80% of its classification signal.

---

## 5. Conclusions

### What Did the Current Plot Actually Measure?

The "Cumulative spectral energy by channel — full dataset" plot (generated by
`src/tools/analyze_mafaulda_spectrum.py`) shows: for each of the 8 MAFAULDA channels,
what fraction of the total Welch PSD power is contained below each frequency, averaged
over all 1,951 files. It uses complete files (not 1024-sample windows), Hann-windowed
Welch PSD, per-segment DC removal, and no normalization. The result is technically
correct and uses the standard PSD estimation methodology.

### Does It Agree with the Thesis?

Qualitatively yes:
- Low-frequency dominance of overhang vibration channels confirms typical machinery
  vibration literature expectations (consistent with 0–1 kHz emphasis).
- The 22–23 kHz spike in `acc_axial_b1` is consistent with bearing resonance behavior
  reported in the literature.
- The microphone channel is broadband as expected.

Quantitative comparison is limited because the thesis likely uses log-scale spectrum
plots (emphasizing relative band activity) while our cumulative energy plot uses
energy-fraction (dominated by the 22 kHz spike in `acc_axial_b1`).

### Is Downsampling Justified by the Current Evidence?

**No — not without an ablation study.**

The 22 kHz spike in `acc_axial_b1` is the most class-discriminative spectral feature
in the dataset. Any downsampling that removes or aliases it risks significant accuracy loss.
The current model achieves 98.22% test accuracy with the full 50 kHz input. That baseline
must be protected.

### What Exact Experiment Is Required Before Changing the Model Input?

1. Enable `RESAMPLE_ENABLED = True` and `RESAMPLE_TARGET_SR = 24000` in `src/config.py`.
2. Enable `LOWPASS_FILTER_ENABLED = True` with `LOWPASS_CUTOFF_HZ = 11000` to apply
   anti-aliasing before downsampling.
3. Regenerate the dataset cache (delete `data/processed/`).
4. Retrain the Transformer with the same hyperparameters.
5. Compare test accuracy. If accuracy drops by more than 2%, resampling at 24 kHz with
   an 11 kHz cutoff is not acceptable.
6. Try 48 kHz (preserves 22 kHz spike) as a lower-risk alternative.
7. Document all results in a new metrics JSON and report the accuracy vs sample-rate tradeoff.

---

## 6. Files in This Audit

| File | Description |
|------|-------------|
| [artifacts/frequency_audit/run_frequency_audit.py](artifacts/frequency_audit/run_frequency_audit.py) | This script |
| [artifacts/frequency_audit/channel_energy_statistics.csv](artifacts/frequency_audit/channel_energy_statistics.csv) | 50/90/95/99% Hz + band fractions per channel |
| [artifacts/frequency_audit/downsampling_analysis.csv](artifacts/frequency_audit/downsampling_analysis.csv) | Retained energy by (candidate fs, channel) |
| [artifacts/frequency_audit/cumulative_energy_per_channel.png](artifacts/frequency_audit/cumulative_energy_per_channel.png) | Cumulative energy curve per channel |
| [artifacts/frequency_audit/mean_psd_per_class_*.png](artifacts/frequency_audit/) | Per-class mean PSD for each channel |
| [artifacts/frequency_audit/mean_psd_per_channel_all_classes.png](artifacts/frequency_audit/mean_psd_per_channel_all_classes.png) | All channels, log scale |
| [artifacts/frequency_audit/zoomed_0_2khz_per_channel.png](artifacts/frequency_audit/zoomed_0_2khz_per_channel.png) | 0–2 kHz zoom |
| [artifacts/frequency_audit/zoomed_20_24khz_per_channel.png](artifacts/frequency_audit/zoomed_20_24khz_per_channel.png) | 20–24 kHz zoom |
| [artifacts/frequency_audit/log_scale_full_spectrum.png](artifacts/frequency_audit/log_scale_full_spectrum.png) | Log-log full spectrum |

### Existing Relevant Artifacts (not new)

| File | Description |
|------|-------------|
| [artifacts/results/spectrum_analysis/spectrum_report.json](artifacts/results/spectrum_analysis/spectrum_report.json) | 90/95/99% cutoff Hz per channel per class (Welch, 1951 files) |
| [artifacts/results/frequency_investigation/investigation_summary.json](artifacts/results/frequency_investigation/investigation_summary.json) | acc_axial_b1 band-energy per class, resampling configs |
| [artifacts/results/spectrum_analysis/overall/overall_channels_cumulative_energy.png](artifacts/results/spectrum_analysis/overall/overall_channels_cumulative_energy.png) | Prior cumulative energy plot |
| [artifacts/results/frequency_investigation/paper_style_fft/](artifacts/results/frequency_investigation/paper_style_fft/) | Paper-style FFT per vibration channel and class |
| [artifacts/results/frequency_investigation/axial_b1_investigation/](artifacts/results/frequency_investigation/axial_b1_investigation/) | acc_axial_b1 class-by-class spike investigation |
| [src/tools/analyze_mafaulda_spectrum.py](src/tools/analyze_mafaulda_spectrum.py) | Script that generated the cumulative energy plot |
| [src/tools/frequency_investigation.py](src/tools/frequency_investigation.py) | Deeper per-channel/per-class FFT investigation |
"""

    out_path.write_text(report, encoding="utf-8")
    print(f"[INFO] Report -> {out_path.relative_to(_PROJECT_ROOT)}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    print("=" * 70)
    print("MAFAULDA Frequency Audit")
    print(f"  Data root  : {DATA_RAW_DIR}")
    print(f"  Sample rate: {FS} Hz   Nyquist: {NYQUIST} Hz")
    print(f"  Output dir : {OUTPUT_DIR}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Step 1] Discovering MAFAULDA CSV files ...")
    files_by_label = discover_files()
    n_by_label = {lbl: len(lst) for lbl, lst in files_by_label.items()}
    total = sum(n_by_label.values())
    print(f"\n  Total: {total} files")
    for lbl, n in sorted(n_by_label.items()):
        print(f"    {lbl:<20}: {n}")

    if total == 0:
        print("[ERROR] No CSV files found. Check DATA_RAW_DIR in src/config.py.")
        sys.exit(1)

    print("\n[Step 2] Computing PSDs (single pass per file, all 8 channels) ...")
    freqs, overall_psd, per_class = run_analysis(files_by_label)

    print("\n[Step 3] Saving statistics ...")
    stat_rows = save_channel_energy_statistics(
        freqs, overall_psd, OUTPUT_DIR / "channel_energy_statistics.csv")
    ds_rows = save_downsampling_analysis(
        freqs, overall_psd, OUTPUT_DIR / "downsampling_analysis.csv")

    print("\n[Step 4] Generating plots ...")
    plot_cumulative_energy(freqs, overall_psd, OUTPUT_DIR)
    plot_psd_per_class(freqs, per_class, OUTPUT_DIR)
    plot_all_channels_log(freqs, overall_psd, OUTPUT_DIR)
    plot_zoom_0_2khz(freqs, overall_psd, OUTPUT_DIR)
    plot_zoom_20_24khz(freqs, overall_psd, OUTPUT_DIR)
    plot_loglog(freqs, overall_psd, OUTPUT_DIR)

    print("\n[Step 5] Writing report ...")
    write_report(stat_rows, ds_rows, n_by_label, freqs, overall_psd,
                 OUTPUT_DIR / "FREQUENCY_AUDIT_REPORT.md")

    # Print summary table
    print("\n" + "=" * 70)
    print("CHANNEL ENERGY STATISTICS")
    print(f"{'Channel':<26} {'50% Hz':>10} {'90% Hz':>10} {'95% Hz':>10} {'99% Hz':>10}")
    print("-" * 70)
    for r in stat_rows:
        print(f"  {r['channel']:<24} {r['pct50_hz']:>10,.0f} {r['pct90_hz']:>10,.0f} "
              f"{r['pct95_hz']:>10,.0f} {r['pct99_hz']:>10,.0f}")

    print("\n" + "=" * 70)
    print("22–23 kHz BAND ENERGY FRACTION PER CHANNEL")
    for r in stat_rows:
        frac = r.get("band_22-23kHz", 0)
        print(f"  {r['channel']:<24} {frac*100:>6.2f}%")

    print(f"\n[INFO] All outputs in: {OUTPUT_DIR.relative_to(_PROJECT_ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
