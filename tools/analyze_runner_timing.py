"""Analyze a board live-runner output captured with FPGA_RUNNER_TIMING=1.

Input: the runner's stdout (`R idx=... cls=... lat_ms=... logits=... [q= wq_ms= rd_ms=]
[resp_ms=] t_ms=` lines + SUMMARY). Checks every dataset index and class against the
recorded 2026-10-08 board predictions and computes:

  * output interval  = difference of consecutive t_ms (same meaning in both loops)
  * lat_ms           = the runner's own per-window latency field (its meaning
                       differs by loop; reported, not compared across loops)
  * resp_ms          = paced runs only: result time - simulated window arrival
  * queue depth at dequeue (prefetch runs)
Deadline = window period 1024 / 50 kHz = 20.48 ms. The first `--warmup` windows
are excluded from interval/latency statistics (reported separately).

Usage: python -m tools.analyze_runner_timing <workload.out> --out <summary.json> [--warmup 100]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FT = ROOT / "artifacts" / "results" / "fpga_fulltest_20261008_141335"
PERIOD_MS = 1024 / 50000 * 1000.0


def _kv(line):
    return dict(t.split("=", 1) for t in line.split()[1:] if "=" in t)


def _stats(a):
    if len(a) == 0:
        return None
    return {"n": int(len(a)), "mean": round(float(a.mean()), 3),
            "p50": round(float(np.percentile(a, 50)), 3), "p95": round(float(np.percentile(a, 95)), 3),
            "p99": round(float(np.percentile(a, 99)), 3), "p99.9": round(float(np.percentile(a, 99.9)), 3),
            "max": round(float(a.max()), 3),
            "over_20.48ms": int((a > PERIOD_MS).sum()), "over_50ms": int((a > 50).sum()),
            "over_100ms": int((a > 100).sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_file", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--indices", type=Path, default=None, help="expected index file (sampled runs)")
    ap.add_argument("--pc-rx", type=Path, default=None,
                    help="pc_rx.tsv from run_board_guarded --stream (PC receive time per line)")
    a = ap.parse_args()
    rows, summary = [], None
    for line in a.out_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("R idx="):
            rows.append(_kv(line))
        elif line.startswith("SUMMARY"):
            summary = line
    rec = np.loadtxt(FT / "fpga_pred.txt", dtype=np.int64)
    lab = np.loadtxt(FT / "labels.txt", dtype=np.int64)
    idx = np.array([int(r["idx"]) for r in rows], dtype=np.int64)
    cls = np.array([int(r["cls"]) for r in rows], dtype=np.int64)
    expect = (sorted(int(x) for x in a.indices.read_text().split()) if a.indices
              else list(range(len(rows))))
    w = a.warmup

    def col(k):
        return np.array([float(r[k]) for r in rows if k in r]) if rows and k in rows[0] else np.array([])
    t = col("t_ms")
    res = {
        "source": str(a.out_file), "windows": int(len(rows)), "warmup_excluded": w,
        "indices_as_expected": bool(len(rows) == len(expect) and (idx == np.array(expect)).all()),
        "match_recorded_fpga_pred": int((rec[idx] == cls).sum()) if len(rows) else 0,
        "correct_vs_labels": int((lab[idx] == cls).sum()) if len(rows) else 0,
        "deadline_ms": round(PERIOD_MS, 3),
        "summary_line": summary,
        "duration_s": round(float((t[-1] - t[0]) / 1000.0), 3) if len(t) > 1 else None,
        "rate_inf_s_after_warmup": round(float((len(t) - w - 1) / ((t[-1] - t[w]) / 1000.0)), 3)
        if len(t) > w + 1 else None,
        "output_interval_ms": _stats(np.diff(t)[w:]) if len(t) > 1 else None,
        "output_interval_ms_warmup": _stats(np.diff(t)[:w]) if len(t) > 1 else None,
        "lat_ms_field": _stats(col("lat_ms")[w:]),
        "resp_ms_paced": _stats(col("resp_ms")[w:]),
        "queue_depth": _stats(col("q")[w:]),
        "queue_empty_at_dequeue": int((col("q")[w:] == 0).sum()) if len(col("q")) else None,
        "queue_wait_ms": _stats(col("wq_ms")[w:]),
        "producer_read_ms": _stats(col("rd_ms")),
    }
    if a.pc_rx is not None:
        tab = chr(9)
        pc = np.array([float(l.split(tab, 1)[0]) for l in a.pc_rx.read_text(encoding="utf-8").splitlines()
                       if tab + "R idx=" in l])
        if len(pc) == len(t) and len(t) > w + 1:
            res["pc_receive_interval_ms"] = _stats(np.diff(pc)[w:])
            d = pc - t                        # PC receive time - board emit time (+ unknown offset)
            res["pc_extra_delay_ms_vs_min"] = _stats((d - d.min())[w:])
            res["pc_rate_inf_s_after_warmup"] = round(float((len(pc) - w - 1) / ((pc[-1] - pc[w]) / 1000.0)), 3)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("windows", "indices_as_expected", "match_recorded_fpga_pred",
                                          "rate_inf_s_after_warmup", "output_interval_ms",
                                          "resp_ms_paced", "queue_depth")}, indent=1))


if __name__ == "__main__":
    main()
