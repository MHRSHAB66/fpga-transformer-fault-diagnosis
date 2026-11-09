"""Board <-> PC clock mapping for the end-to-end criterion
(docs/reports/e2e_hard_rt_criterion_20261024.md, section 4). Pure functions, no I/O.

A sync round = (ts, tr) on the PC clock around one request, tb on the board clock at the
reply. The board time tb happened at some PC time in [ts, tr]; taking the midpoint gives an
error of at most RTT/2 = (tr - ts)/2, whatever the path asymmetry.

Mapping board -> PC: piecewise-linear between ANCHORS = the minimum-RTT round of each
`bucket_s` bucket (PC time). For a board time between anchors i and j:
    pc = pc_i + (tb - tb_i) * (pc_j - pc_i) / (tb_j - tb_i)
    error bound e = max(rtt_i, rtt_j)/2 + |drift_ij| * min(|tb - tb_i|, |tb - tb_j|)
where drift_ij is the segment's measured rate difference (dimensionless). Outside the anchor
range the nearest segment is extrapolated and the distance term uses the distance to the
outermost anchor.
"""
from __future__ import annotations

from bisect import bisect_right
from typing import Dict, List, Optional, Sequence, Tuple


def anchors(rounds: Sequence[Dict], bucket_s: float = 2.0) -> List[Tuple[float, float, float]]:
    """-> sorted [(pc_mid, tb, rtt)] : min-RTT round per bucket of PC time."""
    best: Dict[int, Tuple[float, float, float]] = {}
    for r in rounds:
        rtt = r["tr"] - r["ts"]
        if rtt < 0:
            continue
        b = int(r["ts"] // bucket_s)
        cand = ((r["ts"] + r["tr"]) / 2.0, r["tb"], rtt)
        if b not in best or rtt < best[b][2]:
            best[b] = cand
    out = sorted(best.values())
    # board time must increase with PC time; drop anchors that would break monotonicity
    mono = []
    for a in out:
        if not mono or a[1] > mono[-1][1]:
            mono.append(a)
    return mono


class ClockMap:
    def __init__(self, rounds: Sequence[Dict], bucket_s: float = 2.0):
        self.anc = anchors(rounds, bucket_s)
        if not self.anc:
            raise ValueError("no valid sync rounds")
        self._tb = [a[1] for a in self.anc]

    def _segment(self, tb: float) -> Tuple[int, int]:
        n = len(self.anc)
        if n == 1:
            return 0, 0
        j = bisect_right(self._tb, tb)
        j = min(max(j, 1), n - 1)
        return j - 1, j

    def board_to_pc(self, tb: float) -> Tuple[float, float]:
        """-> (pc_time, error_bound_s)."""
        i, j = self._segment(tb)
        pci, tbi, ri = self.anc[i]
        if i == j:
            return pci + (tb - tbi), ri / 2.0
        pcj, tbj, rj = self.anc[j]
        slope = (pcj - pci) / (tbj - tbi)
        drift = abs(slope - 1.0)
        pc = pci + (tb - tbi) * slope
        if tb < tbi:
            dist = tbi - tb
        elif tb > tbj:
            dist = tb - tbj
        else:
            dist = min(tb - tbi, tbj - tb)
        return pc, max(ri, rj) / 2.0 + drift * dist

    def summary(self) -> Dict:
        rt = [a[2] for a in self.anc]
        drifts = []
        for (p1, b1, _), (p2, b2, _) in zip(self.anc, self.anc[1:]):
            drifts.append(((p2 - p1) / (b2 - b1) - 1.0) * 1e6)
        return {"anchors": len(self.anc), "rtt_ms_min": min(rt) * 1e3, "rtt_ms_max": max(rt) * 1e3,
                "half_rtt_ms_max": max(rt) * 1e3 / 2.0,
                "drift_ppm_min": min(drifts) if drifts else None,
                "drift_ppm_max": max(drifts) if drifts else None,
                "span_s": self.anc[-1][0] - self.anc[0][0]}


def max_gap_s(rounds: Sequence[Dict]) -> Optional[float]:
    ts = sorted(r["ts"] for r in rounds)
    return max((b - a for a, b in zip(ts, ts[1:])), default=None)
