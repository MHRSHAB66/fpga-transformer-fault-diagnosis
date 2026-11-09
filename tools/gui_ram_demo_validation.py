"""Physical GUI validation of the Live tab in mode `ram_realtime_demo` (real window,
real board). Same harness as tools/gui_live_validation.py (Button.invoke() = the
command bound to the button; screenshots with PrintWindow; board process table over
SSH after every step).

Scenarios:
  default_mode      window opens in the legacy SD/gzip mode; banner wording visible
  select_ram_mode   switch the mode menu to ram_realtime_demo; banner changes
  preflight_ram     Preflight in RAM mode -> PASS (incl. RAM-set sha256 checks)
  run_complete      Start N=--n-complete windows to completion: FPGA classes ==
                    recorded, all 4 logits == reference, 4 true classes shown,
                    evidence saved (board raw npz + PC receipt/display times)
  stop_midrun       Start 30,200, Stop after >= 100 results: board clean, evidence saved
  close_midrun      Start 30,200, close the window (real close handler): board clean
  (legacy mode is re-selected at the end of select/preflight checks: default unchanged)

Usage: python -m tools.gui_ram_demo_validation --out artifacts/results/gui_ram_demo_validation_<ts>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tools import gui_live_validation as g

ROOT = g.ROOT
REF = np.loadtxt(ROOT / "artifacts/results/board_vs_reference_20261022/numpy_reference/numpy_export_logits.txt",
                 dtype=np.int64)[:, 1:]
g.PS_PATTERN = ("python3[.]6 .*[f]pga_(live_runner|ram_rt_demo|rt_acceptance)|[t]emp_guard[.]sh|"
                "[f]pga_stage_bench|[t]ail_diag")


def main() -> int:
    from host.fpga_result_viewer import LIVE_MODE_LEGACY, LIVE_MODE_RAM
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-complete", type=int, default=3200)
    a = ap.parse_args()
    out = a.out.resolve()
    if out.exists():
        raise SystemExit("refusing to reuse %s" % out)
    h = g.Harness(out)
    L = h.live

    def set_mode(m):
        L.mode_var.set(m)
        L._show_mode()
        h.root.update()

    # default_mode
    h.front()
    txt = L.mode_banner.cget("text")
    h.record("default_mode", L.mode_var.get() == LIVE_MODE_LEGACY
             and "AVERAGE-THROUGHPUT VERIFIED, HARD REAL-TIME NOT VERIFIED" in txt,
             banner=txt, screenshot=h.shot("01_default_legacy_mode"))

    # select_ram_mode
    set_mode(LIVE_MODE_RAM)
    txt = L.mode_banner.cget("text")
    h.record("select_ram_mode", "RAM REAL-TIME DEMO" in txt and "NOT hard real-time" in txt,
             banner=txt, screenshot=h.shot("02_ram_mode_selected"))

    # preflight_ram
    L.n_var.set("1200")
    t0 = time.perf_counter()
    h.click("Preflight")
    h.wait_for(lambda: "preflight RESULT" in h.log_text(), 60)
    lt = h.log_text()
    h.record("preflight_ram", "preflight RESULT: PASS" in lt and "ram_set_identity" in lt,
             note=[l for l in lt.splitlines() if "preflight" in l][:25],
             screenshot=h.shot("03_ram_preflight"))

    def run(n, label, stop_after=None, close_after=None):
        L.n_var.set(str(n))
        h.root.update()
        t0 = time.perf_counter()
        h.click("Start")
        if stop_after or close_after:
            need = stop_after or close_after
            h.wait_for(lambda: len(L.ram_results) >= need, 120)
            shot = h.shot(label + "_running")
            rdir = h.run_dir()
            if stop_after:
                h.click("Stop")
            else:
                h.root.on_close()
                h.closed = True
            return t0, shot, rdir
        h.wait_for(lambda: len(L.ram_results) >= min(n, 400), 120)
        shot = h.shot(label + "_running")
        h.wait_worker(n * 0.0205 + 120)
        return t0, shot, h.run_dir()

    def wait_evidence(timeout=120):
        h.wait_for(lambda: L.ram_evidence_dir is not None and
                   (Path(L.ram_evidence_dir) / "pc_evidence.json").exists(), timeout)
        return L.ram_evidence_dir

    # run_complete
    t0, shot, rdir = run(a.n_complete, "04_run_complete")
    ev_dir = wait_evidence()
    res = list(L.ram_results)
    bad = [r["idx"] for r in res if g.REC[r["idx"]] != r["cls"] or list(REF[r["idx"]]) != r["logits"]]
    shown = sorted(set(int(g.LAB[r["idx"]]) for r in res))
    procs = h.wait_board_clean()
    h.record("run_complete", len(res) == a.n_complete and not bad and shown == [0, 1, 2, 3] and not procs,
             results=len(res), mismatches=bad[:10], true_classes_shown=shown, evidence=ev_dir,
             summary=L.ram_summary, procs_after=procs, screenshot=shot,
             screenshot_end=h.shot("05_run_complete_end"))

    # stop_midrun
    L.ram_evidence_dir = None
    t0, shot, rdir = run(30200, "06_stop_midrun", stop_after=100)
    h.wait_worker(60)
    ev_dir = wait_evidence()
    procs = h.wait_board_clean()
    res = list(L.ram_results)
    h.record("stop_midrun", 100 <= len(res) < 30200 and not procs and ev_dir is not None,
             results=len(res), run_dir=rdir, guard_tail=h.guard_tail(rdir), evidence=ev_dir,
             procs_after=procs, screenshot=shot, screenshot_end=h.shot("07_stop_midrun_end"))

    # back to legacy, then RAM again for close test (mode switch is reversible)
    set_mode(LIVE_MODE_LEGACY)
    legacy_ok = "HARD REAL-TIME NOT VERIFIED" in L.mode_banner.cget("text")
    shot_leg = h.shot("08_back_to_legacy")
    set_mode(LIVE_MODE_RAM)

    # close_midrun (last: destroys the window)
    t0, shot, rdir = run(30200, "09_close_midrun", close_after=100)
    end = time.time() + 30
    procs = g.board_procs()
    while procs and time.time() < end:
        time.sleep(1.0)
        procs = g.board_procs()
    h.record("close_midrun", not procs, run_dir=rdir, guard_tail=g.ssh(
        "tail -2 %s/guard.log 2>/dev/null" % rdir).splitlines() if rdir else [],
        procs_after=procs, screenshot=shot, legacy_reselect_ok=legacy_ok, legacy_screenshot=shot_leg)

    rep = {"tool": "tools/gui_ram_demo_validation.py", "mode": LIVE_MODE_RAM,
           "method": "Tk Button.invoke() on the real window (same bound command as a click)",
           "scenarios": h.results,
           "all_pass": all(r["result"] == "PASS" for r in h.results)}
    (out / "gui_validation.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print("ALL_PASS", rep["all_pass"], out)
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
