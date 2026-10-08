"""Report-ready figure from MEASURED Vitis HLS results (baseline vs optimized v1).

Reads artifacts/results/hls_z7020_opt_ctx_measured_summary.json. All bars are
MEASURED (Vitis HLS 2020.2 csynth, xc7z020clg400-1, 10 ns). The optimized core is
2.0x faster but overflows BRAM (136%); the capacity line makes that explicit.

Run: python docs/reports/plot_measured_opt.py
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "results" / "hls_z7020_opt_ctx_measured_summary.json"
OUT = ROOT / "artifacts" / "results" / "hls_measured_opt_vs_baseline.png"


def main() -> None:
    d = json.loads(SRC.read_text())
    base = d["baseline_measured"]
    v1 = d["optimized_v1_aggressive_inline_MEASURED"]
    v3 = d["optimized_v3_bram_fixed_MEASURED"]
    cap = v1["device_capacity"]

    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.8))

    # panel 1: measured resource utilization %, with 100% capacity line
    names = ["BRAM_18K", "DSP", "FF", "LUT"]
    b = [100.0 * base["resources"][n] / cap[n] for n in names]
    o1 = [100.0 * v1["resources"][n] / cap[n] for n in names]
    o3 = [100.0 * v3["resources"][n] / cap[n] for n in names]
    x = np.arange(len(names)); w = 0.27
    ax[0].bar(x - w, b, w, label="baseline", color="#9aa7b2")
    ax[0].bar(x, o1, w, label="v1 inline (overflow)", color="#b3261e")
    ax[0].bar(x + w, o3, w, label="v3 BRAM-fixed (fits)", color="#0878bf")
    ax[0].axhline(100, color="#b3261e", ls="--", lw=1.2, label="Z7020 capacity")
    ax[0].set_xticks(x, names); ax[0].set_ylabel("utilization (% of Z7020)")
    ax[0].set_title("MEASURED resources: v1 overflows BRAM, v3 fits")
    ax[0].grid(axis="y", alpha=0.25); ax[0].legend(fontsize=7.5)
    for xs, vals in ((x - w, b), (x, o1), (x + w, o3)):
        for xp, v in zip(xs, vals):
            ax[0].text(xp, v + 2, f"{v:.0f}", ha="center", fontsize=7,
                       color=("#b3261e" if v > 100 else "black"),
                       fontweight=("bold" if v > 100 else "normal"))

    # panel 2: measured latency + real-time budget
    base_ms = base["latency_ms_at_target_clock"]
    v1_ms = v1["latency_ms_at_target_clock"]
    v3_ms = v3["latency_ms_at_target_clock"]
    acq_ms = d["realtime_assessment"]["window_acquisition_ms"]
    labels = ["baseline\n1.00x", f"v1 inline\n2.01x (no fit)", f"v3 BRAM-fixed\n1.65x (fits)"]
    bars = ax[1].bar(labels, [base_ms, v1_ms, v3_ms],
                     color=["#9aa7b2", "#b3261e", "#0878bf"])
    ax[1].axhline(acq_ms, color="#2e7d32", ls="--", lw=1.2,
                  label=f"real-time budget {acq_ms:.1f} ms")
    ax[1].set_ylabel("latency per window (ms)")
    ax[1].set_title("MEASURED latency per window")
    ax[1].grid(axis="y", alpha=0.25); ax[1].legend(fontsize=8)
    for bar, v in zip(bars, [base_ms, v1_ms, v3_ms]):
        ax[1].text(bar.get_x() + bar.get_width() / 2, v + 8, f"{v:.0f}", ha="center", fontsize=9)

    fig.suptitle("F_1layer int10 — MEASURED baseline vs v1 vs v3 (Vitis HLS 2020.2, xc7z020clg400-1, 10 ns)",
                 fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
