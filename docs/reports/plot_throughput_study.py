"""Figure: HLS throughput study on xc7z020clg400-1 (MEASURED) + analytical lower
bounds. Reads artifacts/results/hls_throughput_study_summary.json.
Run: python docs/reports/plot_throughput_study.py
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "results" / "hls_throughput_study_summary.json"
OUT = ROOT / "artifacts" / "results" / "hls_throughput_study.png"


def main() -> None:
    d = json.loads(SRC.read_text())
    tgt = d["real_time_requirement"]["required_interval_cycles"]
    mv = d["measured_variants"]
    names = ["baseline", "v1 fast", "v3", "T1\nK-bank", "T2\n+req-share"]
    interval = [m["interval"] for m in mv]
    lut = [m["LUT"] for m in mv]
    fit = [m["fit_7020"] for m in mv]

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.7))

    # panel 1: measured interval vs real-time target (log)
    colors = ["#9aa7b2", "#b3261e", "#9aa7b2", "#0878bf", "#0878bf"]
    bars = ax[0].bar(names, interval, color=colors)
    ax[0].axhline(tgt, color="#2e7d32", ls="--", lw=1.4, label=f"real-time {tgt:,} cyc")
    ax[0].set_yscale("log"); ax[0].set_ylabel("interval (cycles, log)")
    ax[0].set_title("MEASURED interval vs real-time target")
    ax[0].grid(axis="y", which="both", alpha=0.25); ax[0].legend(fontsize=8)
    for b, m in zip(bars, mv):
        ax[0].text(b.get_x() + b.get_width() / 2, m["interval"] * 1.1, f"{m['rt_gap']}x", ha="center", fontsize=8)

    # panel 2: LUT vs 7020 capacity
    cap = 53200
    colors2 = ["#b3261e" if l > cap or not f else "#0878bf" for l, f in zip(lut, fit)]
    bars = ax[1].bar(names, lut, color=colors2)
    ax[1].axhline(cap, color="#2e7d32", ls="--", lw=1.2, label=f"7020 LUT {cap}")
    ax[1].set_ylabel("LUT (measured)"); ax[1].set_title("LUT vs 7020 capacity")
    ax[1].grid(axis="y", alpha=0.25); ax[1].legend(fontsize=8)
    for b, l in zip(bars, lut):
        ax[1].text(b.get_x() + b.get_width() / 2, l + 800, f"{l}", ha="center", fontsize=7.5)

    # panel 3: analytical lower bounds vs target (log)
    lb = d["stage0_theoretical_lower_bounds_ANALYTICAL"]
    lbn = ["S1\nII=1", "S2\n+2h", "S3\n+4h", "S4\n+DF", "S5\n+split", "S6\n+ff"]
    lbv = [lb["S1_scoreII1_headsSeq"]["interval"], lb["S2_+2heads"]["interval"], lb["S3_+4heads"]["interval"],
           lb["S4_+queryDATAFLOW_4head_finishNotSplit"]["interval"], lb["S5_+finishSplit_ffRolled"]["interval"],
           lb["S6_+ffUnroll2"]["interval"]]
    bars = ax[2].bar(lbn, lbv, color="#ef8c00")
    ax[2].axhline(tgt, color="#2e7d32", ls="--", lw=1.4, label=f"real-time {tgt:,}")
    ax[2].set_yscale("log"); ax[2].set_ylabel("projected interval (cycles, log)")
    ax[2].set_title("ANALYTICAL lower bounds (projections)\nfloor ~3.9M > target: not real-time")
    ax[2].grid(axis="y", which="both", alpha=0.25); ax[2].legend(fontsize=8)

    fig.suptitle("F_1layer int10 — HLS throughput study, xc7z020clg400-1, 10 ns (measured T1/T2; S1–S6 analytical)",
                 fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
