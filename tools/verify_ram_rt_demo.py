"""Independent check of ram_realtime_demo runs (CLI or GUI evidence dirs; no board).

From the RAW board record (remote/out/rt_raw.npz) it recomputes the board response
(done - release), deadline misses after the 200-window warm-up, classes vs the recorded
2026-10-08 FPGA predictions and all four logits vs the INT10 reference, both by the
REAL dataset index of each window. From the PC records (pc_results.tsv + clock sync in
pc_evidence.json) it recomputes PC-receipt and GUI-display latency -- reported as
measurements only, never as part of the hard-RT verdict.

Usage: python -m tools.verify_ram_rt_demo <evidence_dir> [<evidence_dir> ...]
Writes <evidence_dir>/verification.json for each.
"""
import json
import sys
from pathlib import Path

import numpy as np

from host.ram_rt_demo import clock_map, e2e_latencies, latency_stats

ROOT = Path(__file__).resolve().parents[1]
REC = np.loadtxt(ROOT / "artifacts/results/fpga_fulltest_20261008_141335/fpga_pred.txt", dtype=np.int64)
REF = np.loadtxt(ROOT / "artifacts/results/board_vs_reference_20261022/numpy_reference/numpy_export_logits.txt",
                 dtype=np.int64)[:, 1:]
P_MS, WARMUP = 20.48, 200


def _rc(events, key):
    for e in events:
        m = e.get("msg", "")
        if m.startswith(key + "="):
            return m.split("=", 1)[1]
    return None


def verify(d: Path) -> dict:
    ev = json.loads((d / "pc_evidence.json").read_text(encoding="utf-8"))
    events = [json.loads(l) for l in (d / "events_nonresult.jsonl").read_text(encoding="utf-8").splitlines() if l]
    rdir = d / "remote"
    z = np.load(rdir / "out" / "rt_raw.npz")
    board = json.loads((rdir / "out" / "summary.json").read_text())
    resp = (z["done"] - z["release"]) * 1e3
    acc = resp[WARMUP:]
    di = z["dataset_index"].astype(np.int64)
    cls, lg = z["cls"].astype(np.int64), z["logits"].astype(np.int64)
    n = int(len(resp))
    out = {"evidence_dir": str(d.relative_to(ROOT)).replace("\\", "/"), "mode": ev.get("mode"),
           "board_sched": board.get("sched"), "n_total_requested": board["n_total"], "executed": n,
           "aborted": board.get("aborted"),
           "board_response": {"acceptance_windows": int(len(acc)),
                              "deadline_misses": int((acc > P_MS).sum()),
                              "p50": round(float(np.percentile(acc, 50)), 3) if len(acc) else None,
                              "p99": round(float(np.percentile(acc, 99)), 3) if len(acc) else None,
                              "p99.9": round(float(np.percentile(acc, 99.9)), 3) if len(acc) else None,
                              "p99.99": round(float(np.percentile(acc, 99.99)), 3) if len(acc) else None,
                              "max": round(float(acc.max()), 3) if len(acc) else None,
                              "warmup_misses": int((resp[:WARMUP] > P_MS).sum())},
           "throughput_windows_per_s": round(float(n / (z["done"][-1] - z["release"][0])), 4) if n > 1 else None,
           "class_eq_recorded": int((cls == REC[di]).sum()),
           "logits_eq_reference_all4": int((lg == REF[di]).all(1).sum()),
           "true_classes_seen": sorted(set(int(x) for x in np.loadtxt(
               ROOT / "artifacts/results/fpga_fulltest_20261008_141335/labels.txt", dtype=np.int64)[di])),
           "display_sent": int(z["display_sent"].sum()), "display_dropped": int(n - z["display_sent"].sum()),
           "temp_c": board.get("temp_c"),
           "workload_rc": _rc(events, "WORKLOAD_RC"), "guard_rc": _rc(events, "GUARD_RC"),
           "post_run_board_processes": ev.get("post_run_board_processes")}
    # PC side, recomputed from the TSV and the raw sync rounds
    rows = [l.split("\t") for l in (d / "pc_results.tsv").read_text(encoding="utf-8").splitlines()[1:] if l]
    res = [{"k": int(r[0]), "rel_b": float(r[5]), "rx_pc": float(r[6]),
            "disp_pc": float(r[7]) if r[7] else None} for r in rows]
    cmap = clock_map(ev.get("sync_rounds") or [])
    out["pc_results_received"] = len(res)
    out["clock_map"] = cmap
    if cmap:
        post = [r for r in res if r["k"] >= WARMUP]
        out["pc_receipt_latency_ms"] = latency_stats(e2e_latencies(post, cmap, "rx_pc"))
        out["gui_display_latency_ms"] = latency_stats(e2e_latencies(post, cmap, "disp_pc"))
    full = (n == board["n_total"])
    out["board_hard_rt_pass"] = bool(
        full and out["board_response"]["deadline_misses"] == 0 and out["class_eq_recorded"] == n
        and out["logits_eq_reference_all4"] == n and out["guard_rc"] == "0" and out["workload_rc"] == "0"
        and out["board_response"]["acceptance_windows"] == board["n_total"] - WARMUP)
    out["note"] = ("board_hard_rt_pass covers ONLY board release -> FPGA class ready. "
                   "PC receipt / GUI display latencies are measurements, not hard-RT claims.")
    (d / "verification.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    for a in sys.argv[1:]:
        d = Path(a) if Path(a).is_absolute() else ROOT / a
        r = verify(d)
        print(d.name, json.dumps({k: r[k] for k in ("executed", "board_response", "class_eq_recorded",
                                                    "logits_eq_reference_all4", "display_dropped",
                                                    "guard_rc", "workload_rc", "board_hard_rt_pass")}))
        for k in ("pc_receipt_latency_ms", "gui_display_latency_ms", "clock_map"):
            if k in r:
                print("  ", k, json.dumps(r[k]))


if __name__ == "__main__":
    main()
