"""Independent check of end-to-end runs against the FROZEN criterion
docs/reports/e2e_hard_rt_criterion_20261024.md (sha256 in the .sha256 file next to it).

  python -m tools.verify_e2e_20261024 existing <harness_dir>   # ram_realtime_demo measured as-is
  python -m tools.verify_e2e_20261024 new <evidence_dir>        # ram_tcp_e2e_demo GUI evidence

Reads only raw files: board rt_raw.npz + summary.json, PC frame traces, raw sync rounds.
Writes <dir>/e2e_verification.json. A window is on time only if L + e <= 20.48 ms
(L = t_render - map(t_rel), e = clock-error bound at that window).
"""
import json
import sys
from pathlib import Path

import numpy as np

from host.e2e_clock import ClockMap, max_gap_s

ROOT = Path(__file__).resolve().parents[1]
REC = np.loadtxt(ROOT / "artifacts/results/fpga_fulltest_20261008_141335/fpga_pred.txt", dtype=np.int64)
REF = np.loadtxt(ROOT / "artifacts/results/board_vs_reference_20261022/numpy_reference/numpy_export_logits.txt",
                 dtype=np.int64)[:, 1:]
LAB = np.loadtxt(ROOT / "artifacts/results/fpga_fulltest_20261008_141335/labels.txt", dtype=np.int64)
D_MS, WARMUP, MAX_ERR_MS = 20.48, 200, 1.0


def stats(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return None
    q = {k: round(float(np.percentile(v, p)), 3) for k, p in
         (("p50", 50), ("p95", 95), ("p99", 99), ("p99.9", 99.9), ("p99.99", 99.99))}
    q.update(n=int(v.size), max=round(float(v.max()), 3), min=round(float(v.min()), 3))
    return q


def _rc(events, key):
    for e in events:
        m = e.get("msg", "")
        if m.startswith(key + "="):
            return m.split("=", 1)[1]
    return None


def board_part(remote: Path):
    z = np.load(remote / "out" / "rt_raw.npz")
    s = json.loads((remote / "out" / "summary.json").read_text())
    T0 = float(z["T0_board"][0])
    return z, s, T0


def finish(out, d: Path):
    (d / "e2e_verification.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    keys = ("mode", "accepted_expected", "accepted_rendered", "e2e_misses_with_bound", "max_clock_err_ms",
            "clock_causality_violations", "missing_count", "pc_sink_overflow", "board_overflow", "data_error",
            "missing", "duplicate", "out_of_order", "gui_not_rendered", "board_stage1_misses",
            "classes_ok", "logits_ok", "verdict")
    print(json.dumps({k: out.get(k) for k in keys}))
    for k in ("stage1_release_to_ready_ms", "stage2_ready_to_pub_ms", "stage3_pub_to_rx_ms",
              "stage23_ready_to_rx_ms", "stage4_rx_to_render_ms", "e2e_release_to_render_ms",
              "e2e_plus_bound_ms", "clock"):
        if out.get(k) is not None:
            print("  ", k, json.dumps(out[k]))


def verify_existing(d: Path):
    tr = json.loads((d / "existing_mode_gui_trace.json").read_text(encoding="utf-8"))
    ev_dir = Path(tr["evidence_dir"])
    pc = json.loads((ev_dir / "pc_evidence.json").read_text(encoding="utf-8"))
    events = [json.loads(l) for l in (ev_dir / "events_nonresult.jsonl").read_text(encoding="utf-8").splitlines() if l]
    z, s, T0 = board_part(ev_dir / "remote")
    rows = [l.split("\t") for l in (ev_dir / "pc_results.tsv").read_text(encoding="utf-8").splitlines()[1:] if l]
    renders = {int(k): v for k, v in tr["renders"].items()}
    cm = ClockMap(pc["sync_rounds"])
    n_total = s["n_total"]
    acc = [r for r in rows if int(r[0]) >= WARMUP]
    l_e2e, l_b, l23, l4, errs, notr = [], [], [], [], [], []
    cls_ok = log_ok = 0
    seqs = [int(r[0]) for r in rows]
    for r in acc:
        k, idx, c = int(r[0]), int(r[1]), int(r[2])
        lg = [int(x) for x in r[4].split(",")]
        rel_b = float(r[5])
        ready_b = float(z["done"][k]) + T0
        rx = float(r[6])
        cls_ok += int(c == REC[idx] == int(z["cls"][k]))
        log_ok += int(lg == list(REF[idx]) == list(z["logits"][k]))
        p_rel, e1 = cm.board_to_pc(rel_b)
        p_ready, e2 = cm.board_to_pc(ready_b)
        l_b.append((ready_b - rel_b) * 1e3)
        l23.append((rx - p_ready) * 1e3)
        if k in renders:
            l4.append((renders[k] - rx) * 1e3)
            l_e2e.append((renders[k] - p_rel) * 1e3)
            errs.append(e1 * 1e3)
        else:
            notr.append(k)
    lb = np.array(l_e2e) + np.array(errs)
    polls = np.diff(np.array(tr["poll_times"])) * 1e3
    out = {"criterion": "docs/reports/e2e_hard_rt_criterion_20261024.md", "mode": "ram_realtime_demo (existing, as-is)",
           "evidence_dir": str(ev_dir), "accepted_expected": n_total - WARMUP, "accepted_received": len(acc),
           "accepted_rendered": len(l_e2e),
           "missing": sorted(set(range(WARMUP, n_total)) - set(seqs)),
           "duplicate": len(seqs) - len(set(seqs)),
           "out_of_order": int(sum(1 for a, b in zip(seqs, seqs[1:]) if b < a)),
           "gui_not_rendered": len(notr), "display_dropped_board": pc.get("display_dropped"),
           "board_stage1_misses": s.get("deadline_misses"),
           "stage1_release_to_ready_ms": stats(l_b),
           "stage2_ready_to_pub_ms": None, "stage3_pub_to_rx_ms": None,
           "stage23_ready_to_rx_ms": stats(l23),
           "stage4_rx_to_render_ms": stats(l4), "e2e_release_to_render_ms": stats(l_e2e),
           "e2e_plus_bound_ms": stats(lb), "e2e_misses_with_bound": int((lb > D_MS).sum()),
           "e2e_misses_without_bound": int((np.array(l_e2e) > D_MS).sum()),
           "max_clock_err_ms": round(max(errs), 4) if errs else None, "clock": cm.summary(),
           "clock_rounds": len(pc["sync_rounds"]), "clock_max_gap_s": max_gap_s(pc["sync_rounds"]),
           "gui_poll_interval_ms": stats(polls),
           "classes_ok": "%d/%d" % (cls_ok, len(acc)), "logits_ok": "%d/%d" % (log_ok, len(acc)),
           "true_classes": sorted(set(int(LAB[int(r[1])]) for r in acc)),
           "guard_rc": _rc(events, "GUARD_RC"), "workload_rc": _rc(events, "WORKLOAD_RC"),
           "board_procs_after": tr.get("board_procs_after"),
           "limitations": ["stage 2 (class-ready -> publisher) and stage 3 are not separately timestamped "
                           "in the existing mode; reported combined as stage23",
                           "only start/end clock anchors exist in the existing mode (drift interpolated "
                           "over the whole run)"]}
    fails = []
    if out["e2e_misses_with_bound"]:
        fails.append("deadline")
    if out["max_clock_err_ms"] is None or out["max_clock_err_ms"] > MAX_ERR_MS:
        fails.append("clock_bound")
    if out["missing"] or out["duplicate"] or out["out_of_order"] or out["gui_not_rendered"]:
        fails.append("frames")
    if cls_ok != len(acc) or log_ok != len(acc):
        fails.append("class_logits")
    if out["guard_rc"] != "0" or out["workload_rc"] != "0" or s.get("deadline_misses"):
        fails.append("board")
    out["missing"] = out["missing"][:50]
    out["fail_reasons"] = fails
    out["verdict"] = "END-TO-END HARD REAL-TIME VERIFIED" if not fails else "END-TO-END HARD REAL-TIME NOT VERIFIED"
    finish(out, d)


def verify_new(d: Path):
    pc = json.loads((d / "pc_evidence.json").read_text(encoding="utf-8"))
    events = [json.loads(l) for l in (d / "events_control.jsonl").read_text(encoding="utf-8").splitlines() if l]
    z, s, T0 = board_part(d / "remote")
    pub = json.loads((d / "remote" / "out" / "publisher.json").read_text())
    rows = [l.split("\t") for l in (d / "frames.tsv").read_text(encoding="utf-8").splitlines()[1:] if l]
    cm = ClockMap(pc["sync_rounds"])
    n_total = s["n_total"]
    executed = s["executed"]
    bysq = {int(r[0]): r for r in rows}
    s1, s2, s3, s4, e2e, errs, notr = [], [], [], [], [], [], []
    causality_violations = []      # transport latency below -bound would disprove the clock bound
    cls_ok = log_ok = 0
    acc_seqs = range(WARMUP, executed)
    for k in acc_seqs:
        r = bysq.get(k)
        if r is None:
            continue
        idx, c = int(r[1]), int(r[2])
        lg = [int(x) for x in r[3].split(",")]
        t_rel, t_ready, t_pubrx, t_pub = (float(x) for x in r[4:8])
        t_rx = float(r[9])
        cls_ok += int(c == REC[idx] == int(z["cls"][k]) and idx == int(z["dataset_index"][k]))
        log_ok += int(lg == list(REF[idx]) == list(z["logits"][k]))
        # board record must match the raw npz timestamps exactly (same clock, same values)
        assert abs((t_rel - T0) - float(z["release"][k])) < 1e-6 and abs((t_ready - T0) - float(z["done"][k])) < 1e-6
        p_rel, e_rel = cm.board_to_pc(t_rel)
        p_pub, e_pub = cm.board_to_pc(t_pub)
        s1.append((t_ready - t_rel) * 1e3)
        s2.append((t_pub - t_ready) * 1e3)
        s3.append((t_rx - p_pub) * 1e3)
        if t_rx - p_pub < -e_pub:
            causality_violations.append(k)
        if r[10]:
            t_render = float(r[10])
            s4.append((t_render - t_rx) * 1e3)
            e2e.append((t_render - p_rel) * 1e3)
            errs.append(e_rel * 1e3)
        else:
            notr.append(k)
    lb = np.array(e2e) + np.array(errs)
    seq_all = sorted(bysq)
    missing = sorted(set(range(WARMUP, n_total)) - set(seq_all))
    ticks = np.diff(np.array(pc.get("gui_tick_times") or [0.0])) * 1e3
    out = {"criterion": "docs/reports/e2e_hard_rt_criterion_20261024.md", "mode": "ram_tcp_e2e_demo",
           "evidence_dir": str(d), "n_total": n_total, "executed": executed, "aborted": s.get("aborted"),
           "accepted_expected": n_total - WARMUP, "accepted_received": sum(1 for k in seq_all if k >= WARMUP),
           "accepted_rendered": len(e2e), "missing": missing[:50], "missing_count": len(missing),
           "duplicate": len(pc["seq_duplicate"]), "out_of_order": len(pc["seq_out_of_order"]),
           "gui_not_rendered": len(notr), "pc_sink_overflow": len(pc["pc_sink_overflow_seqs"]),
           "board_overflow": s.get("board_overflow"), "board_overflow_seqs": s.get("board_overflow_seqs", [])[:50],
           "publisher": pub, "data_error": pc.get("data_error"), "stall_log": pc.get("stall_log"),
           "board_stage1_misses": s.get("deadline_misses"),
           "stage1_release_to_ready_ms": stats(s1), "stage2_ready_to_pub_ms": stats(s2),
           "stage3_pub_to_rx_ms": stats(s3), "stage4_rx_to_render_ms": stats(s4),
           "e2e_release_to_render_ms": stats(e2e), "e2e_plus_bound_ms": stats(lb),
           "e2e_misses_with_bound": int((lb > D_MS).sum()),
           "e2e_misses_without_bound": int((np.array(e2e) > D_MS).sum()),
           "max_clock_err_ms": round(max(errs), 4) if errs else None, "clock": cm.summary(),
           "clock_rounds": len(pc["sync_rounds"]), "clock_max_gap_s": max_gap_s(pc["sync_rounds"]),
           "clock_causality_violations": len(causality_violations),
           "gui_tick_interval_ms": stats(ticks) if ticks.size > 1 else None,
           "classes_ok": "%d/%d" % (cls_ok, len(range(WARMUP, executed)) if executed > WARMUP else 0),
           "logits_ok": "%d/%d" % (log_ok, len(range(WARMUP, executed)) if executed > WARMUP else 0),
           "true_classes": sorted(set(int(LAB[int(bysq[k][1])]) for k in seq_all if k >= WARMUP)),
           "board_temp_c": s.get("temp_c"), "guard_rc": _rc(events, "GUARD_RC"),
           "workload_rc": _rc(events, "WORKLOAD_RC"),
           "post_run_board_processes": pc.get("post_run_board_processes")}
    fails = []
    if out["e2e_misses_with_bound"]:
        fails.append("deadline")
    if out["max_clock_err_ms"] is None or out["max_clock_err_ms"] > MAX_ERR_MS or causality_violations:
        fails.append("clock_bound")
    if missing or out["duplicate"] or out["out_of_order"] or notr or executed != n_total:
        fails.append("frames")
    nacc = max(0, executed - WARMUP)
    if cls_ok != nacc or log_ok != nacc:
        fails.append("class_logits")
    if out["guard_rc"] != "0" or out["workload_rc"] != "0" or s.get("deadline_misses"):
        fails.append("board")
    out["fail_reasons"] = fails
    out["verdict"] = "END-TO-END HARD REAL-TIME VERIFIED" if not fails else "END-TO-END HARD REAL-TIME NOT VERIFIED"
    finish(out, d)


def main():
    kind, d = sys.argv[1], Path(sys.argv[2])
    d = d if d.is_absolute() else ROOT / d
    (verify_existing if kind == "existing" else verify_new)(d)


if __name__ == "__main__":
    main()
