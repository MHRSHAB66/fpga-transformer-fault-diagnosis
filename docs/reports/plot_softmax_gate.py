"""Figure: why hardware-only softmax pass-reduction cannot materially help.
Left: measured per-query dataflow stage balance (df_attn vs finish_token) and the
finish_token floor. Right: total interval of each softmax formulation vs the
2,048,000-cycle real-time target (all converge to the finish-limited floor).
df_attn/finish_token/per-query II are MEASURED; fused totals are ANALYTICAL.
Run: python docs/reports/plot_softmax_gate.py
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "results" / "hls_softmax_gate.png"
TGT = 2_048_000

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.7))

# Left: per-query stage balance
stages = ["df_attn\n(5-pass)", "df_attn\n(3-pass est.)", "finish_token\n(floor)"]
vals = [4773, 3730, 4452]
cols = ["#9aa7b2", "#0878bf", "#b3261e"]
b = ax[0].bar(stages, vals, color=cols)
ax[0].axhline(4452, color="#b3261e", ls="--", lw=1.3, label="finish_token floor 4,452")
ax[0].set_ylabel("cycles per query")
ax[0].set_title("Per-query stage balance (measured; 3-pass est.)")
ax[0].grid(axis="y", alpha=0.25); ax[0].legend(fontsize=8)
for bar, v in zip(b, vals):
    ax[0].text(bar.get_x()+bar.get_width()/2, v+60, f"{v}", ha="center", fontsize=8)
ax[0].text(1, 2400, "reducing softmax passes\ncannot go below the\nfinish_token floor",
           ha="center", fontsize=8, color="#b3261e")

# Right: total interval per formulation vs target (log)
names = ["prefix_opt2\n(5-pass)", "3-pass\n(bit-exact)", "2-pass\n(approx)", "online\n(approx)", "SEQ256\n(Phase3 est.)"]
tot = [6_229_677, 5_895_425, 5_895_425, 5_895_425, 1_470_000]
colc = ["#0878bf", "#ef8c00", "#ef8c00", "#ef8c00", "#2e7d32"]
bb = ax[1].bar(names, tot, color=colc)
ax[1].axhline(TGT, color="#2e7d32", ls="--", lw=1.5, label=f"real-time {TGT:,}")
ax[1].set_yscale("log"); ax[1].set_ylabel("total interval (cycles, log)")
ax[1].set_title("Total interval vs real-time target")
ax[1].grid(axis="y", which="both", alpha=0.25); ax[1].legend(fontsize=8)
for bar, v in zip(bb, tot):
    ax[1].text(bar.get_x()+bar.get_width()/2, v*1.07, f"{v/TGT:.2f}x", ha="center", fontsize=8)
ax[1].text(2.0, 8.2e6, "softmax fusion -> finish-limited\n~5.90M (1.06x, not material)",
           ha="center", fontsize=7.5, color="#ef8c00")

fig.suptitle("Hardware-only softmax pass-reduction gate — finish_token co-limits; SEQ256 is the analytical route (Phase 3, not started)",
             fontweight="bold", fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT, dpi=170, bbox_inches="tight"); plt.close(fig)
print(OUT)
