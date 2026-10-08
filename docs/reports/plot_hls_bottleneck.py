"""Report-ready figure for the HLS latency bottleneck + optimization projection.

Reads artifacts/results/hls_attention_bottleneck_analysis.json (analytical) and
artifacts/results/hls_z7020_current_summary.json (measured baseline). The
optimized latency/throughput bars are PROJECTIONS and are labelled as such; they
must be replaced with measured csynth numbers once the laptop synthesis runs.

Run: python docs/reports/plot_hls_bottleneck.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BOTT = ROOT / "artifacts" / "results" / "hls_attention_bottleneck_analysis.json"
SUMM = ROOT / "artifacts" / "results" / "hls_z7020_current_summary.json"
OUT = ROOT / "artifacts" / "results" / "hls_bottleneck_optimization.png"


def main() -> None:
    b = json.loads(BOTT.read_text())
    s = json.loads(SUMM.read_text())

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # --- panel 1: baseline cycle attribution -------------------------------
    shares = b["component_cycle_shares_baseline"]
    groups = {
        "attn context\n(baseline)": shares["attn_context_baseline"]["cycles"],
        "softmax\n(scores+max+exp+div)": (
            shares["attn_scores"]["cycles"] + shares["attn_rowmax"]["cycles"]
            + shares["attn_exp_denom"]["cycles"] + shares["attn_softmax_div"]["cycles"]
        ),
        "kv/in/pe/pool\nproj": (
            shares["kv_proj"]["cycles"] + shares["input_proj"]["cycles"]
            + shares["pe_add"]["cycles"] + shares["pooling"]["cycles"]
        ),
        "finish_token\n(res/LN/FFN)": shares["finish_token"]["cycles"],
        "residual\n(div-II/fill/hs)": b["analytical_cycle_model"]["reconciliation_residual_cycles"],
    }
    total = b["measured"]["total_latency_cycles"]
    labels = list(groups)
    vals = [100.0 * v / total for v in groups.values()]
    colors = ["#b3261e", "#ef8c00", "#0878bf", "#2e7d32", "#9aa7b2"]
    bars = ax[0].bar(range(len(labels)), vals, color=colors)
    ax[0].set_xticks(range(len(labels)), labels, fontsize=7.5)
    ax[0].set_ylabel("% of measured 74.25 M cycles")
    ax[0].set_title("Baseline cycle attribution (analytical)")
    ax[0].grid(axis="y", alpha=0.25)
    for bar, v in zip(bars, vals):
        ax[0].text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", fontsize=8)

    # --- panel 2: latency baseline vs projected optimized ------------------
    op = b["optimization_projection"]
    base_ms = 1000.0 * s["latency_seconds_at_target_clock"]
    cons_ms = op["projected_measured_latency_cycles_conservative"] * 10e-9 * 1000
    opt_ms = op["projected_measured_latency_cycles_optimistic"] * 10e-9 * 1000
    acq_ms = 1000.0 * s["window_acquisition_seconds"]
    names = ["baseline\n(measured)", "opt conservative\n(projected)", "opt optimistic\n(projected)"]
    lvals = [base_ms, cons_ms, opt_ms]
    bars = ax[1].bar(names, lvals, color=["#b3261e", "#ef8c00", "#f6c445"])
    ax[1].axhline(acq_ms, color="#2e7d32", linestyle="--", linewidth=1.2,
                  label=f"real-time budget {acq_ms:.1f} ms")
    ax[1].set_ylabel("Latency per window (ms)")
    ax[1].set_title("Latency: measured vs projected")
    ax[1].grid(axis="y", alpha=0.25)
    ax[1].legend(fontsize=8)
    for bar, v in zip(bars, lvals):
        ax[1].text(bar.get_x() + bar.get_width() / 2, v + 8, f"{v:.0f}", ha="center", fontsize=8)

    # --- panel 3: throughput vs real-time requirement (log) ----------------
    req = op["required_realtime_wps"]
    base_wps = s["estimated_throughput_windows_per_second"]
    cons_wps, opt_wps = op["projected_throughput_wps_range"]
    tnames = ["baseline\n(meas.)", "opt cons.\n(proj.)", "opt opt.\n(proj.)", "required\nreal-time"]
    tvals = [base_wps, cons_wps, opt_wps, req]
    bars = ax[2].bar(tnames, tvals, color=["#b3261e", "#ef8c00", "#f6c445", "#2e7d32"])
    ax[2].set_yscale("log")
    ax[2].set_ylabel("windows / second (log)")
    ax[2].set_title("Throughput gap to real-time")
    ax[2].grid(axis="y", which="both", alpha=0.25)
    for bar, v in zip(bars, tvals):
        ax[2].text(bar.get_x() + bar.get_width() / 2, v * 1.12, f"{v:.2f}", ha="center", fontsize=8)

    fig.suptitle("F_1layer int10 — HLS latency bottleneck & bit-exact context optimization "
                 "(optimized bars are analytical projections, not yet synthesized)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
