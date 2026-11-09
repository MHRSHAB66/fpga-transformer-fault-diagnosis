"""Timed defense-demo rehearsal on the real board (2026-10-23).

Follows docs/demo_runbook.md step by step and records, for every step, what it
includes, how long it took (wall clock on the PC) and what was observed:

  P1 ping                       network reachability (no SSH)
  P2 CLI preflight              `python -m host.live_preflight --n 40` (read-only, one SSH round trip)
  P3 GUI window up              build the real viewer window (both tabs, replay data loaded)
  P4 Replay fast-eval           recorded 95,404-window FPGA run re-scored vs its metrics file (no board)
  D1 Preflight button           Live tab preflight (read-only)
  D2 Start -> first result      push runner+guard, preflight, overlay load, gzip skip to the first window
  D3 Start -> 40 results        class-balanced run to completion (all four classes)
  D4 Start -> Stop              second start, Stop after >= 5 results, board clean
  D5 Close window               WM_DELETE_WINDOW handler, board clean
  H1 headless 120 (optional)    `python -m host.live_fpga --n 120 --sample-mode class_balanced`

GUI buttons are triggered with Tk Button.invoke() on the real window (OS mouse
input is not available in this session; see tools/gui_live_validation.py).
Usage: python -m tools.demo_rehearsal_20261023 --out artifacts/results/demo_rehearsal_<ts> [--headless]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import tools.gui_live_validation as g  # sets DPI awareness before Tk is created
from host.frame_protocol import LABELS

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--headless", action="store_true")
    a = ap.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    steps, t_all = [], time.perf_counter()

    def step(name, includes, t0, ok, **obs):
        steps.append({"step": name, "includes": includes, "seconds": round(time.perf_counter() - t0, 2),
                      "result": "PASS" if ok else "FAIL", **obs})
        print("[%s] %-28s %6.2f s" % ("PASS" if ok else "FAIL", name, steps[-1]["seconds"]))

    t0 = time.perf_counter()
    r = subprocess.run(["ping", "-n", "2", "-w", "1000", g.lf.BOARD_HOST], capture_output=True, text=True)
    step("P1 ping", "2 ICMP echoes to the board", t0, r.returncode == 0, out=r.stdout.strip().splitlines()[-1:])

    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", "host.live_preflight", "--n", "40"], capture_output=True,
                       text=True, cwd=ROOT)
    step("P2 CLI preflight", "read-only SSH probe: identity, input, DMA contract, temperature, no workload",
         t0, r.returncode == 0 and "PASS" in r.stdout, out=r.stdout.strip().splitlines())

    t0 = time.perf_counter()
    h = g.Harness(a.out)
    step("P3 GUI window up", "python start of the viewer + replay data (95,404 records) + both tabs", t0, True,
         window_build_s=h.t_window)

    t0 = time.perf_counter()
    res, diffs = h.replay.run_fast_eval_blocking()
    step("P4 Replay fast-eval", "re-score recorded FPGA predictions, compare with fpga_fulltest_metrics.json",
         t0, not diffs, accuracy=res["accuracy"], macro_f1=res["macro_f1"], discrepancies=diffs[:5],
         shot=None)

    t0 = time.perf_counter()
    h.set_run(40, "class_balanced")
    h.click("Preflight")
    ok = h.wait_for(lambda: "preflight RESULT" in h.log_text(), 60)
    step("D1 Preflight button", "Live tab read-only preflight", t0,
         ok and "PASS -- safe to Start" in h.log_text(), shot=h.shot("D1_preflight"))

    t0 = time.perf_counter()
    h.click("Start")
    first = h.wait_for(lambda: len(h.ev_since(t0, "result")) >= 1, 180)
    meta = [e for e in h.ev_since(t0, "meta")]
    step("D2 Start -> first result",
         "push runner+guard (new run dir), preflight, SSH launch, overlay load, gzip skip to first window",
         t0, first, class_source=(meta[0].get("class_source") if meta else None))
    h.wait_for(lambda: len(h.ev_since(t0, "result")) >= 20, 300)
    mid = h.shot("D3_mid_run")
    done = h.wait_worker(600)
    evs = h.ev_since(t0, "result")
    rdir = h.run_dir()
    procs = h.wait_board_clean()
    bad = h.check_results(evs)
    classes = sorted({e["cls"] for e in evs})
    step("D3 Start -> 40 results", "whole class-balanced run incl. gzip skipping between far indices",
         t0, done and len(evs) == 40 and classes == [0, 1, 2, 3] and not bad and not procs,
         results=len(evs), classes=[LABELS[c] for c in classes],
         correct_vs_labels=int(sum(g.LAB[e["idx"]] == e["cls"] for e in evs)),
         match_recorded=int(sum(g.REC[e["idx"]] == e["cls"] for e in evs)),
         latency_ms_mean=round(sum(e["lat_ms"] for e in evs) / max(len(evs), 1), 3),
         temp_c_max=max((e["temp_c"] for e in evs if e["temp_c"] is not None), default=None),
         guard_log_tail=h.guard_tail(rdir), run_dir=rdir, procs_after=procs, mismatches=bad[:5],
         shot_mid=mid, shot_end=h.shot("D3_end"))

    t0 = time.perf_counter()
    h.set_run(300, "class_balanced")
    h.click("Start")
    got = h.wait_for(lambda: len(h.ev_since(t0, "result")) >= 5, 180)
    rdir = h.run_dir()
    ts = time.perf_counter()
    h.click("Stop")
    stopped = h.wait_worker(40)
    stop_s = round(time.perf_counter() - ts, 2)
    procs = h.wait_board_clean()
    step("D4 Start -> Stop", "second start, Stop after >= 5 results, remote process check", t0,
         got and stopped and not procs, results=len(h.ev_since(t0, "result")), stop_to_end_s=stop_s,
         guard_log_tail=h.guard_tail(rdir), procs_after=procs, shot=h.shot("D4_after_stop"))

    t0 = time.perf_counter()
    h.root.on_close()
    h.closed = True
    procs = g.board_procs()
    step("D5 Close window", "close handler + remote process check", t0, not procs, procs_after=procs)

    if a.headless:
        t0 = time.perf_counter()
        hd = a.out / "headless_120"
        r = subprocess.run([sys.executable, "-m", "host.live_fpga", "--n", "120", "--sample-mode",
                            "class_balanced", "--out-dir", str(hd)], capture_output=True, text=True, cwd=ROOT)
        (a.out / "headless_120_console.txt").write_text(r.stdout + r.stderr, encoding="utf-8")
        ev = json.loads((hd / "live_evidence.json").read_text()) if (hd / "live_evidence.json").exists() else {}
        procs = g.board_procs()
        step("H1 headless 120", "CLI evidence run, 120 class-balanced windows (30/class)", t0,
             r.returncode == 0 and ev.get("executed_windows") == 120 and not procs,
             accuracy=ev.get("accuracy_vs_ground_truth"), match_recorded=ev.get("match_vs_recorded_fpga_pred"),
             class_source=ev.get("class_source_reported_by_board"), temp=ev.get("temperature_c"),
             latency_ms=ev.get("latency_ms"), errors=ev.get("errors"), procs_after=procs)

    out = {"pc_start": datetime.now().isoformat(timespec="seconds"), "deployment": g.lf.DEP_DIR,
           "total_seconds": round(time.perf_counter() - t_all, 1), "steps": steps,
           "gui_render_ms_per_event_mean": round(sum(h.apply_ms) / max(len(h.apply_ms), 1), 3),
           "all_pass": all(s["result"] == "PASS" for s in steps),
           "click_method": "Tk Button.invoke() on the real window"}
    (a.out / "rehearsal.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("ALL PASS" if out["all_pass"] else "SOME FAILED", a.out / "rehearsal.json")
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
