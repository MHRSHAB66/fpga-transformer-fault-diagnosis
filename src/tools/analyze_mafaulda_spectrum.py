"""
Spectrum analysis tool for MAFAULDA channels.

This script uses the same file-indexing logic as the main project pipeline,
then analyzes frequency content for each expected signal channel with Welch PSD.

It produces:
1. Overall channel-wise average PSD and cumulative-energy plots
2. Per-channel per-label average PSD and cumulative-energy plots
3. Representative per-label time-domain plots (one sample per label)
4. Representative per-label PSD plots (one sample per label)
5. JSON report with cutoff frequencies and suggested sample rates

Examples
--------
python -m src.tools.analyze_mafaulda_spectrum --data-dir data/raw/mafaulda --max-files 120
python -m src.tools.analyze_mafaulda_spectrum --data-dir data/raw/mafaulda
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
        "scipy is required for this script. Install it with: pip install scipy"
    ) from exc

from src.config import (
    DATA_RAW_DIR,
    EXPECTED_COLUMNS,
    LABEL_MAP,
    RANDOM_SEED,
)
from src.data.load_data import build_file_index, load_one_csv


def sample_index_df(
    index_df: pd.DataFrame,
    max_files: Optional[int],
    per_class: Optional[int],
    random_seed: int,
) -> pd.DataFrame:
    """
    Sample a subset of files for spectrum analysis.
    """
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
    Load selected signal columns from one CSV file using fixed column positions.

    The MAFAULDA CSV files in this project do not have header names, so the
    expected channels are mapped by column index.
    """
    try:
        df = load_one_csv(file_path)
    except Exception as exc:
        print(f"[WARN] Failed to read {file_path}: {exc}")
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


def welch_psd(
    x: np.ndarray,
    sample_rate: int,
    nperseg: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Welch PSD for one 1D signal.
    """
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
    """
    Compute average PSD over many signals.

    If frequency grids differ, PSDs are interpolated to the first valid grid.
    """
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
            interp_psd = np.interp(freqs_ref, freqs, psd)
            psds.append(interp_psd)

    if not psds or freqs_ref is None:
        raise ValueError("No valid signals were available for PSD estimation.")

    mean_psd = np.mean(np.vstack(psds), axis=0)
    return freqs_ref, mean_psd


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


def summarize_psd(freqs: np.ndarray, mean_psd: np.ndarray) -> Dict[str, object]:
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


def save_overall_plots(
    overall_results: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """
    Save combined overall plots for all channels:
    - PSD
    - cumulative energy
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    psd_path = output_dir / "overall_channels_average_psd.png"
    cum_path = output_dir / "overall_channels_cumulative_energy.png"

    plt.figure(figsize=(12, 6))
    for col, result in overall_results.items():
        freqs = result["freqs"]
        psd = result["mean_psd"]
        plt.semilogy(freqs, psd + 1e-20, label=col)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Average PSD")
    plt.title("Average PSD by channel (overall dataset)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(psd_path, dpi=180)
    plt.close()

    plt.figure(figsize=(12, 6))
    for col, result in overall_results.items():
        freqs = result["freqs"]
        cumulative = cumulative_energy_curve(result["mean_psd"])
        plt.plot(freqs, cumulative, label=col)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Cumulative energy ratio")
    plt.title("Cumulative spectral energy by channel (overall dataset)")
    plt.ylim(0.0, 1.01)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cum_path, dpi=180)
    plt.close()

    return psd_path, cum_path


def save_per_channel_per_label_plots(
    results_by_channel_and_label: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    output_dir: Path,
) -> List[str]:
    """
    Save per-channel plots with label overlays:
    - PSD
    - cumulative energy
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []

    for col, label_map in results_by_channel_and_label.items():
        if not label_map:
            continue

        safe_col = col.replace("/", "_")

        psd_path = output_dir / f"{safe_col}_per_label_psd.png"
        cum_path = output_dir / f"{safe_col}_per_label_cumulative_energy.png"

        plt.figure(figsize=(12, 6))
        for label, result in label_map.items():
            freqs = result["freqs"]
            psd = result["mean_psd"]
            plt.semilogy(freqs, psd + 1e-20, label=label)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Average PSD")
        plt.title(f"Average PSD by label - {col}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(psd_path, dpi=180)
        plt.close()

        plt.figure(figsize=(12, 6))
        for label, result in label_map.items():
            freqs = result["freqs"]
            cumulative = cumulative_energy_curve(result["mean_psd"])
            plt.plot(freqs, cumulative, label=label)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Cumulative energy ratio")
        plt.title(f"Cumulative spectral energy by label - {col}")
        plt.ylim(0.0, 1.01)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(cum_path, dpi=180)
        plt.close()

        saved_paths.append(str(psd_path))
        saved_paths.append(str(cum_path))

    return saved_paths


def pick_representative_filepaths_by_label(sampled_df: pd.DataFrame) -> Dict[str, Path]:
    """
    Pick one representative file per label.
    Current rule: first file in sampled_df order for each label.
    """
    reps: Dict[str, Path] = {}
    for label in LABEL_MAP.keys():
        label_rows = sampled_df[sampled_df["label"] == label]
        if label_rows.empty:
            continue
        reps[label] = Path(label_rows.iloc[0]["filepath"])
    return reps


def save_representative_time_domain_plots(
    representative_signals: Dict[str, Dict[str, np.ndarray]],
    sample_rate: int,
    output_dir: Path,
    max_seconds: float,
) -> List[str]:
    """
    Save representative time-domain plots for each channel, overlaid by label.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []

    for col, label_signals in representative_signals.items():
        if not label_signals:
            continue

        safe_col = col.replace("/", "_")
        plot_path = output_dir / f"{safe_col}_representative_time_domain.png"

        plt.figure(figsize=(12, 6))
        for label, signal in label_signals.items():
            max_points = min(len(signal), int(max_seconds * sample_rate))
            x = signal[:max_points]
            t = np.arange(max_points) / sample_rate
            plt.plot(t, x, label=label)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(f"Representative time-domain signals by label - {col}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        plt.close()

        saved_paths.append(str(plot_path))

    return saved_paths


def save_representative_psd_plots(
    representative_signals: Dict[str, Dict[str, np.ndarray]],
    sample_rate: int,
    nperseg: int,
    output_dir: Path,
) -> List[str]:
    """
    Save representative PSD plots for each channel, overlaid by label.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []

    for col, label_signals in representative_signals.items():
        if not label_signals:
            continue

        safe_col = col.replace("/", "_")
        plot_path = output_dir / f"{safe_col}_representative_psd.png"

        plt.figure(figsize=(12, 6))
        for label, signal in label_signals.items():
            try:
                freqs, psd = welch_psd(signal, sample_rate=sample_rate, nperseg=nperseg)
            except ValueError:
                continue
            plt.semilogy(freqs, psd + 1e-20, label=label)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD")
        plt.title(f"Representative PSD by label - {col}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        plt.close()

        saved_paths.append(str(plot_path))

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_RAW_DIR,
        help="Path to MAFAULDA CSV root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/results/spectrum_analysis"),
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=50000,
        help="Original sample rate in Hz",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional total file cap",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=None,
        help="Optional per-class cap",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
    )
    parser.add_argument(
        "--nperseg",
        type=int,
        default=4096,
        help="Welch segment length",
    )
    parser.add_argument(
        "--representative-seconds",
        type=float,
        default=0.1,
        help="Duration shown in representative time-domain plots",
    )
    args = parser.parse_args()

    index_df = build_file_index(args.data_dir)
    if index_df.empty:
        raise ValueError(f"No CSV files found under: {args.data_dir}")

    index_df = index_df[index_df["label"].isin(LABEL_MAP.keys())].copy().reset_index(drop=True)
    if index_df.empty:
        raise ValueError("No labeled files remained after filtering with LABEL_MAP.")

    sampled_df = sample_index_df(
        index_df=index_df,
        max_files=args.max_files,
        per_class=args.per_class,
        random_seed=args.random_seed,
    )

    print(f"[INFO] Total labeled files found: {len(index_df)}")
    print(f"[INFO] Files selected for spectrum analysis: {len(sampled_df)}")

    label_counts = sampled_df["label"].value_counts().to_dict()
    print(f"[INFO] Label counts in analyzed subset: {label_counts}")

    channels = list(EXPECTED_COLUMNS)

    signals_by_channel: Dict[str, List[np.ndarray]] = {col: [] for col in channels}
    signals_by_channel_and_label: Dict[str, Dict[str, List[np.ndarray]]] = {
        col: {label: [] for label in LABEL_MAP.keys()} for col in channels
    }

    valid_files = 0

    for _, row in sampled_df.iterrows():
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

    if valid_files == 0:
        raise ValueError("No valid signals were loaded for spectrum analysis.")

    overall_results: Dict[str, Dict[str, np.ndarray]] = {}
    report: Dict[str, object] = {
        "sample_rate_hz": int(args.sample_rate),
        "files_used": int(len(sampled_df)),
        "valid_files_used": int(valid_files),
        "label_counts": label_counts,
        "channels": {},
    }

    for col in channels:
        channel_signals = signals_by_channel[col]
        if not channel_signals:
            print(f"[WARN] No valid signals found for channel: {col}")
            continue

        freqs, mean_psd = average_psd(
            signals=channel_signals,
            sample_rate=args.sample_rate,
            nperseg=args.nperseg,
        )

        overall_results[col] = {
            "freqs": freqs,
            "mean_psd": mean_psd,
        }

        report["channels"][col] = {
            "overall": {
                "signal_count": int(len(channel_signals)),
                **summarize_psd(freqs, mean_psd),
            },
            "per_label": {},
        }

    results_by_channel_and_label: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for col in channels:
        results_by_channel_and_label[col] = {}
        if col not in report["channels"]:
            continue

        for label in LABEL_MAP.keys():
            label_signals = signals_by_channel_and_label[col][label]
            if not label_signals:
                continue

            freqs, mean_psd = average_psd(
                signals=label_signals,
                sample_rate=args.sample_rate,
                nperseg=args.nperseg,
            )

            results_by_channel_and_label[col][label] = {
                "freqs": freqs,
                "mean_psd": mean_psd,
            }

            report["channels"][col]["per_label"][label] = {
                "signal_count": int(len(label_signals)),
                **summarize_psd(freqs, mean_psd),
            }

    representative_paths_by_label = pick_representative_filepaths_by_label(sampled_df)
    representative_signals: Dict[str, Dict[str, np.ndarray]] = {
        col: {} for col in channels
    }

    representative_file_report: Dict[str, str] = {}

    for label, rep_path in representative_paths_by_label.items():
        representative_file_report[label] = str(rep_path)
        channel_data = load_signal_columns(rep_path, channels)
        for col, values in channel_data.items():
            if values is not None:
                representative_signals[col][label] = values

    overall_dir = args.output_dir / "overall"
    per_label_dir = args.output_dir / "per_channel_per_label"
    representative_dir = args.output_dir / "representative"

    overall_psd_path, overall_cum_path = save_overall_plots(overall_results, overall_dir)
    per_label_plot_paths = save_per_channel_per_label_plots(
        results_by_channel_and_label,
        per_label_dir,
    )
    rep_time_paths = save_representative_time_domain_plots(
        representative_signals=representative_signals,
        sample_rate=args.sample_rate,
        output_dir=representative_dir / "time_domain",
        max_seconds=args.representative_seconds,
    )
    rep_psd_paths = save_representative_psd_plots(
        representative_signals=representative_signals,
        sample_rate=args.sample_rate,
        nperseg=args.nperseg,
        output_dir=representative_dir / "psd",
    )

    report["representative_files"] = representative_file_report
    report["plots"] = {
        "overall_average_psd": str(overall_psd_path),
        "overall_cumulative_energy": str(overall_cum_path),
        "per_channel_per_label_plots": per_label_plot_paths,
        "representative_time_domain_plots": rep_time_paths,
        "representative_psd_plots": rep_psd_paths,
    }

    report_path = args.output_dir / "spectrum_report.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== Overall Channel Summary ===")
    for col in channels:
        if col not in report["channels"]:
            continue
        overall_summary = report["channels"][col]["overall"]["energy_cutoff_hz"]
        print(
            f"{col}: "
            f"90%={overall_summary['90_percent']:.2f} Hz, "
            f"95%={overall_summary['95_percent']:.2f} Hz, "
            f"99%={overall_summary['99_percent']:.2f} Hz"
        )

    print("\nSaved files:")
    print(f"- {overall_psd_path}")
    print(f"- {overall_cum_path}")
    print(f"- {report_path}")
    print(f"- per-channel/per-label plots saved under: {per_label_dir}")
    print(f"- representative plots saved under: {representative_dir}")

if __name__ == "__main__":
    main()