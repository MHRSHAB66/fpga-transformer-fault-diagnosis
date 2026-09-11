"""
Structural channel-order consistency check via inter-channel correlation.

`verify_channel_order.py` confirms the tachometer (col 0) is the min-zero-crossing
channel in nearly every file. That pins the tachometer end but cannot resolve the
microphone vs the accelerometers, nor the internal accelerometer order. This tool
adds independent evidence for the WHOLE 8-column order: if every file was recorded
with the same channel wiring, the 8x8 inter-channel correlation pattern should be
stable across files. A file whose columns were scrambled would have a permuted
correlation matrix and show up as a structural outlier.

Method
------
- For each file, compute the 8x8 Pearson correlation matrix of the channels
  (first second, for speed).
- Take the element-wise MEDIAN correlation matrix across all files as the robust
  reference structure.
- For each file, deviation = mean absolute difference of its correlation matrix
  from the reference.
- Flag files whose deviation exceeds median + k*IQR as structural outliers.

A low, tight deviation distribution with no outliers is evidence that the channel
order (including the 6 accelerometers) is consistent dataset-wide.

Usage
-----
    python -m src.tools.verify_channel_correlation
Output
------
    artifacts/results/channel_correlation_consistency.json
    artifacts/results/channel_correlation_reference.png
"""
from __future__ import annotations

import json
import sys
import time
from typing import List

import numpy as np
import pandas as pd

from src.config import (
    DATA_RAW_DIR, EXPECTED_COLUMNS, EXPECTED_NUM_COLUMNS,
    MAFAULDA_SAMPLE_RATE, PROJECT_ROOT,
)
from src.data.load_data import build_file_index, project_relative_or_absolute

SR = MAFAULDA_SAMPLE_RATE
OUT_JSON = PROJECT_ROOT / "artifacts" / "results" / "channel_correlation_consistency.json"
OUT_PNG = PROJECT_ROOT / "artifacts" / "results" / "channel_correlation_reference.png"


def file_corr(rel_path: str) -> np.ndarray | None:
    df = pd.read_csv(PROJECT_ROOT / rel_path, header=None, nrows=SR)
    if df.shape[1] != EXPECTED_NUM_COLUMNS:
        return None
    v = df.to_numpy(dtype=np.float64)
    # guard against constant channels (zero variance -> nan correlation)
    if np.any(v.std(axis=0) == 0):
        return None
    return np.corrcoef(v, rowvar=False)   # (8, 8)


def save_reference_heatmap(ref: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(ref, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(EXPECTED_NUM_COLUMNS))
    ax.set_yticks(range(EXPECTED_NUM_COLUMNS))
    ax.set_xticklabels(EXPECTED_COLUMNS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(EXPECTED_COLUMNS, fontsize=8)
    for i in range(EXPECTED_NUM_COLUMNS):
        for j in range(EXPECTED_NUM_COLUMNS):
            ax.text(j, i, f"{ref[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if abs(ref[i, j]) < 0.6 else "white")
    ax.set_title("Reference inter-channel correlation (median over all files)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    print("=" * 60)
    print("Structural channel-order consistency (inter-channel correlation)")
    print("=" * 60)

    index_df = build_file_index(DATA_RAW_DIR)
    total = len(index_df)
    if total == 0:
        print("[ERROR] no files")
        return 1

    t0 = time.perf_counter()
    mats: List[np.ndarray] = []
    paths: List[str] = []
    skipped = 0
    for i, rel in enumerate(index_df["filepath"].tolist(), 1):
        try:
            c = file_corr(rel)
        except Exception:
            c = None
        if c is None:
            skipped += 1
        else:
            mats.append(c)
            paths.append(rel)
        if i % 300 == 0:
            print(f"  ... {i}/{total}")

    stack = np.stack(mats)                       # (N, 8, 8)
    ref = np.median(stack, axis=0)               # robust reference structure
    # per-file deviation from the reference (mean abs diff over the 8x8 matrix)
    dev = np.abs(stack - ref).mean(axis=(1, 2))

    med, q1, q3 = np.median(dev), np.percentile(dev, 25), np.percentile(dev, 75)
    iqr = q3 - q1
    threshold = med + 3.0 * iqr
    outlier_idx = np.where(dev > threshold)[0]

    outliers = [{"filepath": paths[i], "deviation": float(dev[i])}
                for i in sorted(outlier_idx, key=lambda i: -dev[i])]

    save_reference_heatmap(ref)
    elapsed = time.perf_counter() - t0

    summary = {
        "method": ("per-file 8x8 Pearson correlation vs element-wise median "
                   "reference; deviation = mean|corr - ref|; outlier if "
                   "deviation > median + 3*IQR"),
        "expected_column_order": list(EXPECTED_COLUMNS),
        "files_used": len(mats),
        "files_skipped": skipped,
        "deviation_stats": {
            "median": float(med), "q1": float(q1), "q3": float(q3),
            "iqr": float(iqr), "max": float(dev.max()),
            "outlier_threshold": float(threshold),
        },
        "num_structural_outliers": int(len(outlier_idx)),
        "structural_outliers": outliers[:50],
        "reference_correlation_matrix": ref.round(4).tolist(),
        "elapsed_seconds": round(elapsed, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"files used: {len(mats)}  (skipped {skipped})")
    print(f"deviation  median={med:.4f}  IQR={iqr:.4f}  max={dev.max():.4f}")
    print(f"structural outliers (dev > {threshold:.4f}): {len(outlier_idx)}")
    if len(outlier_idx) == 0:
        print("=> inter-channel correlation structure is CONSISTENT across all files")
    else:
        for o in outliers[:10]:
            print(f"   {o['deviation']:.4f}  {o['filepath']}")
    print(f"-> {project_relative_or_absolute(OUT_JSON)}")
    print(f"-> {project_relative_or_absolute(OUT_PNG)}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
