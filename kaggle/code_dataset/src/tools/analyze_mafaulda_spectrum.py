"""
Spectrum analysis tool for MAFAULDA channels.

Analyzes frequency content across ALL labeled files and produces
publication-quality figures to justify anti-aliasing and resampling decisions.

Outputs (under --output-dir, default: artifacts/results/spectrum_analysis)
--------------------------------------------------------------------------
overall/
    overall_channels_average_psd.png       — average PSD for all channels
    overall_channels_cumulative_energy.png — cumulative energy for all channels

per_channel_per_label/
    <channel>_per_label_psd.png            — per-class PSD overlay per channel
    <channel>_per_label_cumulative_energy.png

representative/
    time_domain/<channel>_representative_time_domain.png
    psd/<channel>_representative_psd.png

report/
    report_psd_energy_cutoffs.png    — THESIS FIGURE: PSD + cumul energy + cutoff
                                       annotations for all vibration channels
    report_class_comparison_panel.png — per-class PSD overlaid on each channel
    report_lowfreq_zoom_<N>hz.png    — 0–<zoom_hz> detail, per-class

spectrum_report.json                 — cutoff frequencies, suggested rates,
                                       and final RESAMPLE / LOWPASS recommendation

Usage
-----
    # Analyze ALL files (recommended for final decision):
    python -m src.tools.analyze_mafaulda_spectrum

    # Quick preview on a subset:
    python -m src.tools.analyze_mafaulda_spectrum --max-files 80

    # Custom zoom and output:
    python -m src.tools.analyze_mafaulda_spectrum --zoom-hz 8000 --output-dir artifacts/results/spectrum_v2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.signal import welch
except Exception as exc:
    raise RuntimeError(
        "scipy is required. Install it with: pip install scipy"
    ) from exc

from src.config import (
    DATA_RAW_DIR,
    EXPECTED_COLUMNS,
    LABEL_MAP,
    MAFAULDA_SAMPLE_RATE,
    PROJECT_ROOT,
    RANDOM_SEED,
)
from src.data.load_data import build_file_index, load_one_csv

# Vibration channels only (exclude tachometer and microphone).
# Used for the report figures; spectral content of these channels drives the
# anti-aliasing cutoff and resampling rate decision.
VIBRATION_CHANNEL_NAMES: List[str] = [
    c for c in EXPECTED_COLUMNS if c not in ("tachometer", "microphone")
]

# Consistent class colors across all per-class figures.
_CLASS_COLORS: Dict[str, str] = {
    "normal":        "#2ca02c",
    "imbalance":     "#ff7f0e",
    "misalignment":  "#9467bd",
    "bearing_fault": "#d62728",
}

# Colors for 90 / 95 / 99 % energy cutoff lines.
_CUT_COLORS: Dict[str, str] = {
    "90%": "#e41a1c",
    "95%": "#ff7f0e",
    "99%": "#377eb8",
}
_CUT_FRACS: List[Tuple[str, float]] = [
    ("90%", 0.90),
    ("95%", 0.95),
    ("99%", 0.99),
]


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

def set_report_style() -> None:
    """Apply consistent publication-quality matplotlib settings."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "legend.framealpha": 0.85,
            "axes.grid": True,
            "grid.alpha": 0.30,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def sample_index_df(
    index_df: pd.DataFrame,
    max_files: Optional[int],
    per_class: Optional[int],
    random_seed: int,
) -> pd.DataFrame:
    """Return a (possibly sampled) subset of the file index."""
    if index_df.empty:
        return index_df.copy()

    rng = np.random.default_rng(random_seed)
    sampled = index_df.copy()

    if per_class is not None:
        chunks = []
        for _, group in sampled.groupby("label"):
            n = min(len(group), per_class)
            take_idx = rng.choice(group.index.to_numpy(), size=n, replace=False)
            chunks.append(group.loc[take_idx])
        sampled = pd.concat(chunks, axis=0).reset_index(drop=True)

    if max_files is not None and len(sampled) > max_files:
        take_idx = rng.choice(sampled.index.to_numpy(), size=max_files, replace=False)
        sampled = sampled.loc[take_idx].reset_index(drop=True)

    return sampled.reset_index(drop=True)


def load_signal_columns(
    file_path: Path,
    columns: List[str],
) -> Dict[str, Optional[np.ndarray]]:
    """
    Load raw signal columns from one CSV file by integer position.

    MAFAULDA CSV files have no header; columns are mapped by index order
    as defined in EXPECTED_COLUMNS.
    """
    try:
        df = load_one_csv(file_path)
    except Exception as exc:
        print(f"[WARN] Failed to read {file_path.name}: {exc}")
        return {col: None for col in columns}

    if df is None or df.empty:
        return {col: None for col in columns}

    result: Dict[str, Optional[np.ndarray]] = {}
    for idx, col in enumerate(columns):
        if idx >= df.shape[1]:
            result[col] = None
            continue

        series = pd.to_numeric(df.iloc[:, idx], errors="coerce").dropna()
        if series.empty or len(series) < 8:
            result[col] = None
            continue

        result[col] = series.to_numpy(dtype=np.float32, copy=True)

    return result


# ---------------------------------------------------------------------------
# PSD computation
# ---------------------------------------------------------------------------

def welch_psd(
    x: np.ndarray,
    sample_rate: int,
    nperseg: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD for one 1-D signal."""
    local_nperseg = min(nperseg, len(x))
    if local_nperseg < 8:
        raise ValueError("Signal too short for PSD estimation.")
    freqs, psd = welch(
        x,
        fs=sample_rate,
        window="hann",
        nperseg=local_nperseg,
        noverlap=local_nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    return freqs, psd.astype(np.float64)


def average_psd(
    signals: List[np.ndarray],
    sample_rate: int,
    nperseg: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute average PSD over a list of 1-D signals."""
    psds = []
    freqs_ref = None

    for x in signals:
        try:
            freqs, psd = welch_psd(x, sample_rate=sample_rate, nperseg=nperseg)
        except ValueError:
            continue

        if freqs_ref is None:
            freqs_ref = freqs
            psds.append(psd)
            continue

        if len(freqs) == len(freqs_ref) and np.allclose(freqs, freqs_ref):
            psds.append(psd)
        else:
            psds.append(np.interp(freqs_ref, freqs, psd))

    if not psds or freqs_ref is None:
        raise ValueError("No valid signals for PSD estimation.")

    return freqs_ref, np.mean(np.vstack(psds), axis=0)


def cumulative_energy_curve(psd: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(psd)
    if cumulative[-1] <= 0:
        return np.zeros_like(cumulative)
    return cumulative / cumulative[-1]


def energy_cutoff_frequency(
    freqs: np.ndarray,
    cumulative_energy: np.ndarray,
    fraction: float,
) -> float:
    idx = int(np.searchsorted(cumulative_energy, fraction, side="left"))
    idx = min(max(idx, 0), len(freqs) - 1)
    return float(freqs[idx])


def suggest_target_rates(cutoff_hz: float) -> Dict[str, int]:
    """Return strict Nyquist, mild (4×) margin, and practical common rate."""
    strict_nyquist = max(1, int(math.ceil(2.0 * cutoff_hz)))
    mild_margin = max(1, int(math.ceil(4.0 * cutoff_hz)))
    common_rates = np.array(
        [2000, 4000, 5000, 8000, 10000, 12000, 16000, 24000, 32000, 48000]
    )
    if np.any(common_rates >= mild_margin):
        practical = int(common_rates[common_rates >= mild_margin][0])
    else:
        practical = int(common_rates[-1])
    return {
        "strict_nyquist_hz": strict_nyquist,
        "mild_margin_hz": mild_margin,
        "practical_common_rate_hz": practical,
    }


def summarize_psd(freqs: np.ndarray, mean_psd: np.ndarray) -> Dict:
    cumulative = cumulative_energy_curve(mean_psd)
    cut90 = energy_cutoff_frequency(freqs, cumulative, 0.90)
    cut95 = energy_cutoff_frequency(freqs, cumulative, 0.95)
    cut99 = energy_cutoff_frequency(freqs, cumulative, 0.99)
    return {
        "energy_cutoff_hz": {
            "90_percent": cut90,
            "95_percent": cut95,
            "99_percent": cut99,
        },
        "suggested_rates_from_90_percent": suggest_target_rates(cut90),
        "suggested_rates_from_95_percent": suggest_target_rates(cut95),
        "suggested_rates_from_99_percent": suggest_target_rates(cut99),
    }


# ---------------------------------------------------------------------------
# Original overview plots (unchanged behavior)
# ---------------------------------------------------------------------------

def save_overall_plots(
    overall_results: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Save average PSD and cumulative energy plots across all channels."""
    output_dir.mkdir(parents=True, exist_ok=True)

    psd_path = output_dir / "overall_channels_average_psd.png"
    cum_path = output_dir / "overall_channels_cumulative_energy.png"

    fig, ax = plt.subplots(figsize=(12, 5))
    for col, result in overall_results.items():
        ax.semilogy(result["freqs"], result["mean_psd"] + 1e-20, label=col)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Average PSD")
    ax.set_title("Average PSD by channel — full dataset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(psd_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for col, result in overall_results.items():
        cumulative = cumulative_energy_curve(result["mean_psd"])
        ax.plot(result["freqs"], cumulative, label=col)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Cumulative energy fraction")
    ax.set_title("Cumulative spectral energy by channel — full dataset")
    ax.set_ylim(0.0, 1.01)
    ax.legend()
    fig.tight_layout()
    fig.savefig(cum_path, dpi=150)
    plt.close(fig)

    return psd_path, cum_path


def save_per_channel_per_label_plots(
    results_by_channel_and_label: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    output_dir: Path,
) -> List[str]:
    """Save per-channel plots with per-class PSD and cumulative energy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []

    for col, label_map in results_by_channel_and_label.items():
        if not label_map:
            continue

        safe_col = col.replace("/", "_")

        psd_path = output_dir / f"{safe_col}_per_label_psd.png"
        cum_path = output_dir / f"{safe_col}_per_label_cumulative_energy.png"

        fig, ax = plt.subplots(figsize=(12, 5))
        for label, result in label_map.items():
            ax.semilogy(
                result["freqs"], result["mean_psd"] + 1e-20,
                label=label, color=_CLASS_COLORS.get(label),
            )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Average PSD")
        ax.set_title(f"Average PSD by class — {col}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(psd_path, dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 5))
        for label, result in label_map.items():
            cumulative = cumulative_energy_curve(result["mean_psd"])
            ax.plot(
                result["freqs"], cumulative,
                label=label, color=_CLASS_COLORS.get(label),
            )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Cumulative energy fraction")
        ax.set_title(f"Cumulative spectral energy by class — {col}")
        ax.set_ylim(0.0, 1.01)
        ax.legend()
        fig.tight_layout()
        fig.savefig(cum_path, dpi=150)
        plt.close(fig)

        saved_paths.extend([str(psd_path), str(cum_path)])

    return saved_paths


def pick_representative_filepaths_by_label(
    sampled_df: pd.DataFrame,
) -> Dict[str, Path]:
    """Pick one representative file per label (first in index order)."""
    reps: Dict[str, Path] = {}
    for label in LABEL_MAP.keys():
        rows = sampled_df[sampled_df["label"] == label]
        if not rows.empty:
            reps[label] = Path(rows.iloc[0]["filepath"])
    return reps


def save_representative_time_domain_plots(
    representative_signals: Dict[str, Dict[str, np.ndarray]],
    sample_rate: int,
    output_dir: Path,
    max_seconds: float,
) -> List[str]:
    """Save representative time-domain plots per channel, overlaid by class."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []

    for col, label_signals in representative_signals.items():
        if not label_signals:
            continue

        safe_col = col.replace("/", "_")
        plot_path = output_dir / f"{safe_col}_representative_time_domain.png"

        fig, ax = plt.subplots(figsize=(12, 5))
        for label, signal in label_signals.items():
            max_points = min(len(signal), int(max_seconds * sample_rate))
            t = np.arange(max_points) / sample_rate
            ax.plot(t, signal[:max_points], label=label,
                    color=_CLASS_COLORS.get(label), linewidth=0.9)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (raw)")
        ax.set_title(f"Representative time-domain signals by class — {col}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        saved_paths.append(str(plot_path))

    return saved_paths


def save_representative_psd_plots(
    representative_signals: Dict[str, Dict[str, np.ndarray]],
    sample_rate: int,
    nperseg: int,
    output_dir: Path,
) -> List[str]:
    """Save representative PSD plots per channel, overlaid by class."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []

    for col, label_signals in representative_signals.items():
        if not label_signals:
            continue

        safe_col = col.replace("/", "_")
        plot_path = output_dir / f"{safe_col}_representative_psd.png"

        fig, ax = plt.subplots(figsize=(12, 5))
        for label, signal in label_signals.items():
            try:
                freqs, psd = welch_psd(signal, sample_rate=sample_rate, nperseg=nperseg)
            except ValueError:
                continue
            ax.semilogy(freqs, psd + 1e-20, label=label,
                        color=_CLASS_COLORS.get(label))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.set_title(f"Representative PSD by class — {col}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        saved_paths.append(str(plot_path))

    return saved_paths


# ---------------------------------------------------------------------------
# Report-quality figures (new)
# ---------------------------------------------------------------------------

def save_report_decision_figure(
    overall_results: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
) -> Optional[Path]:
    """
    Save the main thesis justification figure.

    Layout: 2 rows × 3 columns, one cell per vibration channel.
    Each cell: average PSD (left axis, semilogy) + cumulative energy (right axis).
    Vertical dashed lines annotate the 90 / 95 / 99 % energy cutoff frequencies.

    This figure directly answers: "at what frequency can we safely cut off
    and downsample without losing fault-relevant information?"
    """
    vib_channels = [c for c in VIBRATION_CHANNEL_NAMES if c in overall_results]
    if not vib_channels:
        return None

    n_cols = 3
    n_rows = math.ceil(len(vib_channels) / n_cols)

    fig, axes_2d = plt.subplots(
        n_rows, n_cols,
        figsize=(5.8 * n_cols, 4.8 * n_rows),
        squeeze=False,
    )
    axes_flat = axes_2d.flatten()

    for idx, col in enumerate(vib_channels):
        ax = axes_flat[idx]
        freqs = overall_results[col]["freqs"]
        psd = overall_results[col]["mean_psd"]
        cumulative = cumulative_energy_curve(psd)

        # Left axis: PSD (log scale)
        color_psd = "#1f77b4"
        ax.semilogy(freqs, psd + 1e-20, color=color_psd, linewidth=1.3,
                    label="Avg PSD")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Avg PSD (V²/Hz)", color=color_psd)
        ax.tick_params(axis="y", labelcolor=color_psd)

        # Right axis: cumulative energy (linear)
        ax2 = ax.twinx()
        color_cum = "#7b2d8b"
        ax2.plot(freqs, cumulative, color=color_cum, linewidth=1.3,
                 linestyle="--", label="Cumul. energy", alpha=0.85)
        ax2.set_ylabel("Cumul. energy fraction", color=color_cum)
        ax2.tick_params(axis="y", labelcolor=color_cum)
        ax2.set_ylim(-0.02, 1.10)
        ax2.axhline(1.0, color=color_cum, linewidth=0.5, linestyle=":", alpha=0.4)

        # Cutoff frequency annotations
        for cut_label, frac in _CUT_FRACS:
            cut_hz = energy_cutoff_frequency(freqs, cumulative, frac)
            c = _CUT_COLORS[cut_label]
            ax.axvline(cut_hz, color=c, linewidth=1.0, linestyle=":")
            # Place text at x=cut_hz (data), y=0.98 (axes fraction), rotated
            ax.text(
                cut_hz, 0.98,
                f"{cut_label}:{cut_hz:.0f}Hz",
                rotation=90, va="top", ha="right",
                fontsize=7, color=c,
                transform=ax.get_xaxis_transform(),
            )

        ax.set_title(col)

    for idx in range(len(vib_channels), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "MAFAULDA Vibration Channels — Average PSD with Spectral Energy Cutoffs\n"
        "Basis for anti-aliasing filter cutoff and resampling rate selection",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = output_dir / "report_psd_energy_cutoffs.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Report decision figure  -> {out.relative_to(PROJECT_ROOT)}")
    return out


def save_class_comparison_panel(
    results_by_channel_and_label: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    output_dir: Path,
) -> Optional[Path]:
    """
    Save a panel comparing per-fault-class average PSD across vibration channels.

    Layout: 2 rows × 3 columns, one cell per vibration channel.
    Each cell: 4 lines (one per fault class) on log-scale PSD.

    Shows which frequency bands discriminate between fault types —
    useful for justifying why we must preserve a certain bandwidth.
    """
    vib_channels = [
        c for c in VIBRATION_CHANNEL_NAMES
        if c in results_by_channel_and_label and results_by_channel_and_label[c]
    ]
    if not vib_channels:
        return None

    n_cols = 3
    n_rows = math.ceil(len(vib_channels) / n_cols)

    fig, axes_2d = plt.subplots(
        n_rows, n_cols,
        figsize=(5.8 * n_cols, 4.3 * n_rows),
        squeeze=False,
    )
    axes_flat = axes_2d.flatten()

    for idx, col in enumerate(vib_channels):
        ax = axes_flat[idx]
        label_results = results_by_channel_and_label[col]

        for label in LABEL_MAP.keys():
            if label not in label_results:
                continue
            freqs = label_results[label]["freqs"]
            psd = label_results[label]["mean_psd"]
            ax.semilogy(
                freqs, psd + 1e-20,
                label=label, color=_CLASS_COLORS.get(label),
                linewidth=1.3,
            )

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Avg PSD (V²/Hz)")
        ax.set_title(col)
        ax.legend(loc="upper right", fontsize=7.5)

    for idx in range(len(vib_channels), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Per-Class Average PSD — Fault Discrimination in Frequency Domain\n"
        "Each line is the mean Welch PSD averaged over all files of that class",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = output_dir / "report_class_comparison_panel.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Class comparison panel  -> {out.relative_to(PROJECT_ROOT)}")
    return out


def save_lowfreq_zoom_panel(
    results_by_channel_and_label: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    output_dir: Path,
    zoom_hz: float = 10000.0,
) -> Optional[Path]:
    """
    Save a zoomed per-class PSD panel focused on 0–zoom_hz Hz.

    Rotating machinery fault signatures (imbalance, misalignment, bearing)
    typically appear below 5–10 kHz.  This zoom makes the class-discriminating
    structure visible without the high-frequency tail dominating the view.
    """
    vib_channels = [
        c for c in VIBRATION_CHANNEL_NAMES
        if c in results_by_channel_and_label and results_by_channel_and_label[c]
    ]
    if not vib_channels:
        return None

    n_cols = 3
    n_rows = math.ceil(len(vib_channels) / n_cols)

    fig, axes_2d = plt.subplots(
        n_rows, n_cols,
        figsize=(5.8 * n_cols, 4.3 * n_rows),
        squeeze=False,
    )
    axes_flat = axes_2d.flatten()

    for idx, col in enumerate(vib_channels):
        ax = axes_flat[idx]
        label_results = results_by_channel_and_label[col]

        for label in LABEL_MAP.keys():
            if label not in label_results:
                continue
            freqs = label_results[label]["freqs"]
            psd = label_results[label]["mean_psd"]
            mask = freqs <= zoom_hz
            ax.semilogy(
                freqs[mask], psd[mask] + 1e-20,
                label=label, color=_CLASS_COLORS.get(label),
                linewidth=1.3,
            )

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Avg PSD (V²/Hz)")
        ax.set_xlim(0.0, zoom_hz)
        ax.set_title(col)
        ax.legend(loc="upper right", fontsize=7.5)

    for idx in range(len(vib_channels), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    zoom_label = f"{zoom_hz / 1000:.0f} kHz" if zoom_hz >= 1000 else f"{zoom_hz:.0f} Hz"
    fig.suptitle(
        f"Low-Frequency Detail (0 – {zoom_label}) — Fault-Relevant Spectral Bands\n"
        "Per-class average PSD, vibration channels only",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = output_dir / f"report_lowfreq_zoom_{int(zoom_hz)}hz.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Low-frequency zoom panel -> {out.relative_to(PROJECT_ROOT)}")
    return out


# ---------------------------------------------------------------------------
# Final recommendation
# ---------------------------------------------------------------------------

def compute_recommendation(
    overall_results: Dict[str, Dict[str, np.ndarray]],
) -> Dict:
    """
    Derive the resampling and anti-aliasing recommendation from per-channel PSDs.

    Strategy
    --------
    - Consider only the 6 vibration channels.
    - Take the MAXIMUM 95 % energy cutoff across all channels as the
      conservative reference (ensures no channel is under-sampled).
    - Anti-aliasing lowpass cutoff = that max 95 % value.
    - Resampling rate must be >= 2 × cutoff (Nyquist).
    - Report also shows 4 × cutoff (practical margin) and the nearest
      common integer rate above 4 × cutoff.
    """
    vib_channels = [c for c in VIBRATION_CHANNEL_NAMES if c in overall_results]
    if not vib_channels:
        return {}

    per_channel: Dict[str, Dict] = {}
    max_cut90 = max_cut95 = max_cut99 = 0.0

    for col in vib_channels:
        freqs = overall_results[col]["freqs"]
        psd = overall_results[col]["mean_psd"]
        cumulative = cumulative_energy_curve(psd)

        cut90 = energy_cutoff_frequency(freqs, cumulative, 0.90)
        cut95 = energy_cutoff_frequency(freqs, cumulative, 0.95)
        cut99 = energy_cutoff_frequency(freqs, cumulative, 0.99)

        per_channel[col] = {
            "cut90_hz": round(cut90, 1),
            "cut95_hz": round(cut95, 1),
            "cut99_hz": round(cut99, 1),
        }
        max_cut90 = max(max_cut90, cut90)
        max_cut95 = max(max_cut95, cut95)
        max_cut99 = max(max_cut99, cut99)

    rates_95 = suggest_target_rates(max_cut95)

    note = (
        f"The most demanding vibration channel reaches 95 % of its spectral energy "
        f"at {max_cut95:.0f} Hz.  "
        f"Set LOWPASS_CUTOFF_HZ = {max_cut95:.0f} as the anti-aliasing cutoff.  "
        f"Strict Nyquist requires RESAMPLE_TARGET_SR >= {rates_95['strict_nyquist_hz']} Hz; "
        f"the nearest practical common rate with a 4× safety margin is "
        f"{rates_95['practical_common_rate_hz']} Hz."
    )

    return {
        "per_channel_cutoffs_hz": per_channel,
        "conservative_reference": {
            "basis": "maximum 95% energy cutoff across all vibration channels",
            "max_90pct_hz": round(max_cut90, 1),
            "max_95pct_hz": round(max_cut95, 1),
            "max_99pct_hz": round(max_cut99, 1),
        },
        "recommendation": {
            "suggested_LOWPASS_CUTOFF_HZ": round(max_cut95, 1),
            "suggested_RESAMPLE_TARGET_SR_strict_nyquist_hz": rates_95["strict_nyquist_hz"],
            "suggested_RESAMPLE_TARGET_SR_practical_hz": rates_95["practical_common_rate_hz"],
            "note": note,
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spectrum analysis for MAFAULDA — all files by default."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_RAW_DIR,
        help="Path to MAFAULDA CSV root directory",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "artifacts" / "results" / "spectrum_analysis",
        help="Root output directory for all plots and the JSON report",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=MAFAULDA_SAMPLE_RATE,
        help="Original sample rate in Hz (default: MAFAULDA_SAMPLE_RATE from config)",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Optional cap on total files analyzed (default: all files)",
    )
    parser.add_argument(
        "--per-class", type=int, default=None,
        help="Optional per-class file cap (applied before --max-files)",
    )
    parser.add_argument(
        "--random-seed", type=int, default=RANDOM_SEED,
    )
    parser.add_argument(
        "--nperseg", type=int, default=4096,
        help="Welch segment length in samples (controls frequency resolution)",
    )
    parser.add_argument(
        "--zoom-hz", type=float, default=10000.0,
        help="Upper frequency limit for the low-frequency zoom panel (Hz)",
    )
    parser.add_argument(
        "--representative-seconds", type=float, default=0.1,
        help="Duration shown in representative time-domain plots",
    )
    args = parser.parse_args()

    set_report_style()

    # --- Build and filter file index ---
    index_df = build_file_index(args.data_dir)
    if index_df.empty:
        raise ValueError(f"No CSV files found under: {args.data_dir}")

    index_df = (
        index_df[index_df["label"].isin(LABEL_MAP.keys())]
        .copy()
        .reset_index(drop=True)
    )
    if index_df.empty:
        raise ValueError("No labeled files remained after filtering with LABEL_MAP.")

    sampled_df = sample_index_df(
        index_df=index_df,
        max_files=args.max_files,
        per_class=args.per_class,
        random_seed=args.random_seed,
    )

    print(f"[INFO] Total labeled files in dataset : {len(index_df)}")
    print(f"[INFO] Files selected for analysis    : {len(sampled_df)}")
    label_counts = sampled_df["label"].value_counts().to_dict()
    for lbl, cnt in sorted(label_counts.items()):
        print(f"        {lbl}: {cnt}")

    channels = list(EXPECTED_COLUMNS)

    signals_by_channel: Dict[str, List[np.ndarray]] = {col: [] for col in channels}
    signals_by_channel_and_label: Dict[str, Dict[str, List[np.ndarray]]] = {
        col: {label: [] for label in LABEL_MAP.keys()} for col in channels
    }

    # --- Load all files with progress reporting ---
    valid_files = 0
    n_total = len(sampled_df)
    print_every = max(1, n_total // 10)

    print(f"\n[INFO] Loading signals from {n_total} files ...")
    for i, (_, row) in enumerate(sampled_df.iterrows()):
        file_path = Path(row["filepath"])
        label = str(row["label"])

        channel_data = load_signal_columns(file_path, channels)
        any_valid = False

        for col, values in channel_data.items():
            if values is None:
                continue
            any_valid = True
            signals_by_channel[col].append(values)
            signals_by_channel_and_label[col][label].append(values)

        if any_valid:
            valid_files += 1

        if (i + 1) % print_every == 0 or (i + 1) == n_total:
            print(f"  {i + 1}/{n_total}  ({valid_files} valid so far)")

    if valid_files == 0:
        raise ValueError("No valid signals loaded for spectrum analysis.")

    print(f"\n[INFO] Valid files loaded: {valid_files}/{n_total}")

    # --- Compute average PSD per channel and per channel+label ---
    print("\n[INFO] Computing Welch PSD averages ...")
    overall_results: Dict[str, Dict[str, np.ndarray]] = {}
    results_by_channel_and_label: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    report: Dict = {
        "sample_rate_hz": int(args.sample_rate),
        "nperseg": int(args.nperseg),
        "files_requested": int(len(sampled_df)),
        "files_used": int(valid_files),
        "label_counts": label_counts,
        "channels": {},
    }

    for col in channels:
        ch_signals = signals_by_channel[col]
        if not ch_signals:
            print(f"[WARN] No valid signals for channel: {col}")
            continue

        freqs, mean_psd = average_psd(
            ch_signals, sample_rate=args.sample_rate, nperseg=args.nperseg
        )
        overall_results[col] = {"freqs": freqs, "mean_psd": mean_psd}

        per_label: Dict[str, Dict[str, np.ndarray]] = {}
        per_label_report: Dict = {}

        for label in LABEL_MAP.keys():
            lbl_signals = signals_by_channel_and_label[col][label]
            if not lbl_signals:
                continue
            lf, lp = average_psd(
                lbl_signals, sample_rate=args.sample_rate, nperseg=args.nperseg
            )
            per_label[label] = {"freqs": lf, "mean_psd": lp}
            per_label_report[label] = {
                "signal_count": len(lbl_signals),
                **summarize_psd(lf, lp),
            }

        results_by_channel_and_label[col] = per_label
        report["channels"][col] = {
            "overall": {
                "signal_count": len(ch_signals),
                **summarize_psd(freqs, mean_psd),
            },
            "per_label": per_label_report,
        }

    # --- Save plots ---
    overall_dir = args.output_dir / "overall"
    per_label_dir = args.output_dir / "per_channel_per_label"
    rep_dir = args.output_dir / "representative"
    report_dir = args.output_dir / "report"

    print("\n[INFO] Saving overview plots ...")
    overall_psd_path, overall_cum_path = save_overall_plots(overall_results, overall_dir)

    print("[INFO] Saving per-channel per-class plots ...")
    per_label_paths = save_per_channel_per_label_plots(
        results_by_channel_and_label, per_label_dir
    )

    print("[INFO] Saving representative file plots ...")
    rep_paths_by_label = pick_representative_filepaths_by_label(sampled_df)
    representative_signals: Dict[str, Dict[str, np.ndarray]] = {
        col: {} for col in channels
    }
    rep_file_report: Dict[str, str] = {}

    for label, rep_path in rep_paths_by_label.items():
        rep_file_report[label] = str(rep_path)
        for col, values in load_signal_columns(rep_path, channels).items():
            if values is not None:
                representative_signals[col][label] = values

    rep_time_paths = save_representative_time_domain_plots(
        representative_signals,
        sample_rate=args.sample_rate,
        output_dir=rep_dir / "time_domain",
        max_seconds=args.representative_seconds,
    )
    rep_psd_paths = save_representative_psd_plots(
        representative_signals,
        sample_rate=args.sample_rate,
        nperseg=args.nperseg,
        output_dir=rep_dir / "psd",
    )

    print("\n[INFO] Saving report-quality figures ...")
    save_report_decision_figure(overall_results, report_dir)
    save_class_comparison_panel(results_by_channel_and_label, report_dir)
    save_lowfreq_zoom_panel(results_by_channel_and_label, report_dir, zoom_hz=args.zoom_hz)

    # --- Compute and attach recommendation ---
    print("\n[INFO] Computing resampling recommendation ...")
    recommendation = compute_recommendation(overall_results)
    report["resampling_recommendation"] = recommendation

    report["representative_files"] = rep_file_report
    report["plots"] = {
        "overall_average_psd": str(overall_psd_path),
        "overall_cumulative_energy": str(overall_cum_path),
        "per_channel_per_label": per_label_paths,
        "representative_time_domain": rep_time_paths,
        "representative_psd": rep_psd_paths,
    }

    # --- Save JSON report ---
    report_path = args.output_dir / "spectrum_report.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # --- Console summary ---
    print("\n" + "=" * 65)
    print("Overall channel energy cutoffs (all files averaged)")
    print("=" * 65)
    for col in channels:
        if col not in report["channels"]:
            continue
        ec = report["channels"][col]["overall"]["energy_cutoff_hz"]
        print(
            f"  {col:<26}  "
            f"90%={ec['90_percent']:>7.1f} Hz  "
            f"95%={ec['95_percent']:>7.1f} Hz  "
            f"99%={ec['99_percent']:>7.1f} Hz"
        )

    if recommendation and "recommendation" in recommendation:
        rec = recommendation["recommendation"]
        print("\n" + "=" * 65)
        print("RECOMMENDATION")
        print("=" * 65)
        print(recommendation["recommendation"]["note"])
        print(f"\n  Suggested config values:")
        print(f"    LOWPASS_CUTOFF_HZ    = {rec['suggested_LOWPASS_CUTOFF_HZ']}")
        print(f"    RESAMPLE_TARGET_SR   = {rec['suggested_RESAMPLE_TARGET_SR_practical_hz']}  "
              f"(or >= {rec['suggested_RESAMPLE_TARGET_SR_strict_nyquist_hz']} minimum)")
        print("=" * 65)

    print(f"\n[INFO] JSON report saved: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
