"""
Educational figure: the three signal fingerprints used to identify MAFAULDA
channels (zero-crossing rate, skewness, positive/negative asymmetry).

Plots the tachometer (col 0), one accelerometer (col 2) and the microphone
(col 7) from a single real file, side by side, so the fingerprints are visible.

Usage
-----
    python -m src.tools.explain_fingerprints
Output
------
    artifacts/results/fingerprints_explained.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew as sp_skew

from src.config import DATA_RAW_DIR, PROJECT_ROOT, MAFAULDA_SAMPLE_RATE

SR = MAFAULDA_SAMPLE_RATE
OUT = PROJECT_ROOT / "artifacts" / "results" / "fingerprints_explained.png"

# One representative file; columns: 0=tacho, 2=accel(radial b1), 7=microphone
FILE = DATA_RAW_DIR / "normal" / "12.288.csv"
CHANNELS = [(0, "Tachometer (col 0)", "#d62728"),
            (2, "Accelerometer (col 2)", "#1f77b4"),
            (7, "Microphone (col 7)", "#2ca02c")]


def zero_crossings(sig: np.ndarray) -> int:
    mu = float(np.mean(sig))
    return int(np.sum(np.diff(np.sign(sig[:SR] - mu)) != 0))


def main() -> int:
    df = pd.read_csv(FILE, header=None, nrows=SR)

    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    n_show = SR  # show the full first second, matching the per-second ZC count

    for row, (col, title, color) in enumerate(CHANNELS):
        sig = df.iloc[:, col].to_numpy(dtype=np.float64)
        mu = float(np.mean(sig))
        zc = zero_crossings(sig)
        sk = float(sp_skew(sig))
        asymm = float(np.max(sig)) / (abs(float(np.min(sig))) + 1e-12)

        # --- Column 1: full-second waveform + zero crossings ---
        ax = axes[row, 0]
        seg = sig[:n_show]
        t = np.arange(n_show) / SR  # seconds
        ax.plot(t, seg, color=color, linewidth=0.4)
        ax.axhline(mu, color="black", linestyle="--", linewidth=1, label="mean")
        # Mark crossings only when few enough to see; otherwise they form a blob.
        crossings = np.where(np.diff(np.sign(seg - mu)) != 0)[0]
        if len(crossings) <= 200:
            ax.plot(t[crossings], np.full(len(crossings), mu), "k.", markersize=5,
                    label=f"{len(crossings)} crossings")
            ax.legend(fontsize=8, loc="upper right")
        else:
            ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"{title}\nFINGERPRINT 1: zero-crossings = {zc}/sec",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("time (s) — full 1 second")
        ax.set_ylabel("amplitude")

        # --- Column 2: histogram (skewness) ---
        ax = axes[row, 1]
        ax.hist(sig, bins=120, color=color, alpha=0.7)
        ax.axvline(mu, color="black", linestyle="--", linewidth=1)
        skew_word = ("right-skewed (positive)" if sk > 0.5 else
                     "left-skewed (negative)" if sk < -0.5 else "symmetric")
        ax.set_title(f"FINGERPRINT 2: skewness = {sk:+.2f}\n({skew_word})",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("value")
        ax.set_ylabel("count")

        # --- Column 3: asymmetry (max vs |min|) ---
        ax = axes[row, 2]
        pos_max, neg_min = float(np.max(sig)), float(np.min(sig))
        ax.bar(["max (+)", "|min| (-)"], [pos_max, abs(neg_min)],
               color=[color, "gray"], alpha=0.8)
        ax.set_title(f"FINGERPRINT 3: asymmetry = max/|min| = {asymm:.2f}",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("amplitude")

    fig.suptitle(
        "Three signal fingerprints on real MAFAULDA data (normal/12.288.csv)\n"
        "Tachometer: few crossings, strong positive skew, high asymmetry  |  "
        "Accelerometer: symmetric  |  Microphone: many crossings",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {OUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
