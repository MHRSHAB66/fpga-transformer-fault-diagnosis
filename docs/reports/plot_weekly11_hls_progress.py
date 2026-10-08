"""Create the report-ready HLS progress figure from the verified JSON summary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "results" / "hls_z7020_current_summary.json"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "results" / "weekly11_hls_progress.png"


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    names = ["BRAM_18K", "LUT", "FF", "DSP"]
    current = summary["resources"]
    intermediate = summary["comparison_with_intermediate_z7020_version"][
        "intermediate_resources"
    ]
    old_percent = [100.0 * intermediate[name] / current[name]["available"] for name in names]
    new_percent = [current[name]["percent_exact"] for name in names]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    x = np.arange(len(names))
    width = 0.34
    axes[0].bar(x - width / 2, old_percent, width, label="Intermediate version", color="#9aa7b2")
    axes[0].bar(x + width / 2, new_percent, width, label="Current row-wise Q40", color="#0878bf")
    axes[0].axhline(100, color="#b3261e", linestyle="--", linewidth=1, label="Device capacity")
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("Zynq-7020 utilization (%)")
    axes[0].set_title("HLS core resource progression")
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    for positions, values in ((x - width / 2, old_percent), (x + width / 2, new_percent)):
        for xpos, value in zip(positions, values):
            axes[0].text(xpos, value + 1.4, f"{value:.1f}%", ha="center", fontsize=8)

    latency_ms = 1000.0 * summary["latency_seconds_at_target_clock"]
    acquisition_ms = 1000.0 * summary["window_acquisition_seconds"]
    bars = axes[1].bar(
        ["Current HLS latency", "Window acquisition"],
        [latency_ms, acquisition_ms],
        color=["#b3261e", "#2e7d32"],
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Time per 1024-sample window (ms, log scale)")
    axes[1].set_title(f"Throughput gap: {summary['latency_to_acquisition_ratio']:.2f}x")
    axes[1].grid(axis="y", which="both", alpha=0.25)
    for bar, value in zip(bars, (latency_ms, acquisition_ms)):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.12,
            f"{value:.2f} ms",
            ha="center",
            fontsize=9,
        )

    fig.suptitle("F_1layer int10 - verified Zynq-7020 C-synthesis results", fontweight="bold")
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
