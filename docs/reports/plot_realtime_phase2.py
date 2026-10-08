"""Figure: real-time campaign — measured intervals vs the 2,048,000-cycle target,
and LUT/BRAM utilization. prefix_opt2 is the fastest fitting measured candidate;
the SEQ=256 token-reduction point is an ANALYTICAL estimate (Phase 3, not measured).
Run: python docs/reports/plot_realtime_phase2.py
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "results" / "hls_realtime_phase2.png"
TGT = 2_048_000

# measured candidates (interval cycles, LUT, BRAM) + one analytical Phase-3 point
rows = [
    ("attn2_ff4",      7_212_718, 50_197, 216, "meas"),
    ("prefix_opt2",    6_229_677, 49_889, 216, "meas"),
    ("kv_dataflow*",   6_229_681, 50_752, 296, "meas(no-fit)"),
    ("SEQ256 (est.)",  1_470_000, None,   None, "analytical"),
]
names = [r[0] for r in rows]
inter = [r[1] for r in rows]
colors = ["#9aa7b2", "#0878bf", "#b3261e", "#ef8c00"]

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

# panel 1: interval vs target (log)
bars = ax[0].bar(names, inter, color=colors)
ax[0].axhline(TGT, color="#2e7d32", ls="--", lw=1.5, label=f"real-time target {TGT:,}")
ax[0].set_yscale("log"); ax[0].set_ylabel("top-level interval (cycles, log)")
ax[0].set_title("Interval vs real-time target (10 ns)")
ax[0].grid(axis="y", which="both", alpha=0.25); ax[0].legend(fontsize=8)
for b, r in zip(bars, rows):
    ax[0].text(b.get_x()+b.get_width()/2, r[1]*1.08, f"{r[1]/TGT:.2f}x", ha="center", fontsize=8)
ax[0].text(3, 1.5e6, "analytical\n(Phase 3)", ha="center", fontsize=7, color="#ef8c00")

# panel 2: LUT and BRAM utilization (measured candidates only)
mnames = [r[0] for r in rows if r[2] is not None]
lut = [100*r[2]/53200 for r in rows if r[2] is not None]
bram = [100*r[3]/280 for r in rows if r[2] is not None]
x = np.arange(len(mnames)); w = 0.38
ax[1].bar(x-w/2, lut, w, label="LUT %", color="#0878bf")
ax[1].bar(x+w/2, bram, w, label="BRAM %", color="#8e44ad")
ax[1].axhline(100, color="#b3261e", ls="--", lw=1.3, label="device capacity")
ax[1].set_xticks(x, mnames); ax[1].set_ylabel("utilization (% of xc7z020)")
ax[1].set_title("Resource utilization (measured)")
ax[1].grid(axis="y", alpha=0.25); ax[1].legend(fontsize=8); ax[1].set_ylim(0, 140)
for i,(l,b) in enumerate(zip(lut,bram)):
    ax[1].text(i-w/2, l+2, f"{l:.0f}", ha="center", fontsize=7.5,
               color=("#b3261e" if l>100 else "black"))
    ax[1].text(i+w/2, b+2, f"{b:.0f}", ha="center", fontsize=7.5,
               color=("#b3261e" if b>100 else "black"))

fig.suptitle("F_1layer int10 real-time campaign — measured (Vitis HLS 2020.2, xc7z020clg400-1, 10 ns) + Phase-3 SEQ=256 analytical estimate",
             fontweight="bold", fontsize=10)
fig.tight_layout()
fig.savefig(OUT, dpi=170, bbox_inches="tight"); plt.close(fig)
print(OUT)
