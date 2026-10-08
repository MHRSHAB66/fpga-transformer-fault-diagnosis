"""Figure: Zynq-7010 LUT-fitting campaign (measured). Reads
artifacts/results/hls_z7010_fitting_campaign_summary.json. All bars MEASURED
(Vitis HLS 2020.2 csynth, xc7z010clg400-1, 10 ns). Variant E is the fitting one.
Run: python docs/reports/plot_z7010_campaign.py
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "results" / "hls_z7010_fitting_campaign_summary.json"
OUT = ROOT / "artifacts" / "results" / "hls_z7010_fitting_campaign.png"

# 7010-targeted variants in order
ORDER = ["v3 BRAM-fixed", "A shared ctx-requant", "B +shared layernorm",
         "C 4 context lanes", "D 2 context lanes", "E +shared requant/residual"]
LABELS = ["v3", "A\nreq-share", "B\n+LN-share", "C\n4 lanes", "D\n2 lanes", "E\n+req/res"]


def main() -> None:
    d = json.loads(SRC.read_text())
    by = {}
    for r in d["variants"]:
        if r["device"] == "xc7z010":
            by[r["variant"]] = r
    lut = [by[v]["lut"] for v in ORDER]
    lat = [by[v]["latency_ms_at_10ns"] for v in ORDER]
    cap = 17600
    colors = ["#b3261e" if x > cap else "#0878bf" for x in lut]

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.7))

    bars = ax[0].bar(LABELS, lut, color=colors)
    ax[0].axhline(cap, color="#2e7d32", ls="--", lw=1.4, label=f"7010 LUT capacity {cap}")
    ax[0].set_ylabel("LUT (measured)")
    ax[0].set_title("7010 LUT vs capacity — only E (blue) fits")
    ax[0].grid(axis="y", alpha=0.25); ax[0].legend(fontsize=8)
    for b, v in zip(bars, lut):
        pct = 100 * v / cap
        ax[0].text(b.get_x() + b.get_width() / 2, v + 300, f"{v}\n{pct:.0f}%", ha="center", fontsize=7.5,
                   color=("#b3261e" if v > cap else "#0878bf"), fontweight="bold")

    bars = ax[1].bar(LABELS, lat, color=colors)
    ax[1].set_ylabel("latency per window (ms @ 10 ns)")
    ax[1].set_title("7010 latency (lane cuts C/D are slower, don't help LUT)")
    ax[1].grid(axis="y", alpha=0.25)
    for b, v in zip(bars, lat):
        ax[1].text(b.get_x() + b.get_width() / 2, v + 6, f"{v:.0f}", ha="center", fontsize=8)

    fig.suptitle("F_1layer int10 — Zynq-7010 (xc7z010clg400-1) fitting campaign, MEASURED Vitis HLS 2020.2, 10 ns",
                 fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
