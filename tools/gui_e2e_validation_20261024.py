"""Physical validation of the Live tab in mode `ram_tcp_e2e_demo` (real window, real board),
against docs/reports/e2e_hard_rt_criterion_20261024.md. Same harness as
tools/gui_live_validation.py (Button.invoke(), PrintWindow screenshots, board process table
after every scenario). PC CPU is sampled with psutil at 1 Hz in its own thread.

Scenarios (selected with --scenarios, run in the given order; `close` must be last):
  full1, full2  Start 30,200 windows to completion (criterion runs); smoke = 1,200
  stop          Start 30,200, Stop after >= 300 rendered frames
  disconnect    1,200 windows; PC closes the data connection after 8 s (transport loss)
  stall         1,200 windows; PC receiver stops reading for 10 s at 5 s (TCP backpressure)
  pause         3,200 windows; Pause Display 30 s (PC display-queue overflow), then Resume
  close         Start 30,200, close the window after >= 300 rendered frames
Usage: python -m tools.gui_e2e_validation_20261024 --out <dir> --scenarios full1 full2 stop disconnect stall pause close
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import psutil

from tools import gui_live_validation as g

g.PS_PATTERN = ("python3[.]6 .*[f]pga_(live_runner|ram_rt_demo|ram_e2e_demo|rt_acceptance)|[t]emp_guard[.]sh|"
                "[f]pga_stage_bench|[t]ail_diag")


def main() -> int:
    from host.fpga_result_viewer import LIVE_MODE_E2E
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenarios", nargs="+", required=True)
    a = ap.parse_args()
    out = a.out.resolve()
    if out.exists():
        raise SystemExit("refusing to reuse %s" % out)
    h = g.Harness(out)
    L = h.live
    L.mode_var.set(LIVE_MODE_E2E)
    L._show_mode()
    h.root.update()

    cpu, stop_cpu = [], threading.Event()

    def sampler():
        psutil.cpu_percent(percpu=True)
        while not stop_cpu.wait(1.0):
            cpu.append((time.perf_counter(), psutil.cpu_percent(percpu=True)))
    threading.Thread(target=sampler, daemon=True).start()

    def start(n, hooks=None):
        L.e2e_test_hooks = hooks or {}
        L.e2e_evidence_dir = None
        L.n_var.set(str(n))
        h.root.update()
        t0 = time.perf_counter()
        h.click("Start")
        return t0

    def run_loop(pred, timeout):
        """Drive the REAL Tk event loop (mainloop) until pred() or timeout, so the 1 ms
        render tick runs as in normal use (no harness sleep granularity)."""
        end = time.perf_counter() + timeout

        def chk():
            if pred() or time.perf_counter() > end:
                h.root.quit()
                return
            h.root.after(50, chk)
        h.root.after(50, chk)
        h.root.mainloop()
        return pred()

    def wait_rendered(k, timeout):
        return run_loop(lambda: L.e2e_rendered >= k, timeout)

    def wait_worker(timeout):
        return run_loop(lambda: (L.thread is None or not L.thread.is_alive()) and L.q.empty(), timeout)

    def wait_evidence(timeout=240):
        run_loop(lambda: L.e2e_evidence_dir is not None and
                 (Path(L.e2e_evidence_dir) / "pc_evidence.json").exists(), timeout)
        run_loop(lambda: False, 0.5)
        return L.e2e_evidence_dir

    def cpu_since(t0):
        return [c for c in cpu if c[0] >= t0]

    for sc in a.scenarios:
        if sc in ("full1", "full2", "smoke"):
            t0 = start(30200 if sc != "smoke" else 1200)
            wait_worker(L.n_target * 0.0205 + 300)  # no screenshot during a criterion run
            ev = wait_evidence()
            procs = h.wait_board_clean()
            h.record(sc, ev is not None and not procs, evidence=ev, rendered=L.e2e_rendered,
                     procs_after=procs, screenshot_end=h.shot("%s_end" % sc),
                     pc_cpu_percpu_1hz=cpu_since(t0))
        elif sc == "stop":
            t0 = start(30200)
            wait_rendered(300, 180)
            shot = h.shot("stop_running")
            rdir = h.run_dir()
            ts = time.perf_counter()
            h.click("Stop")
            wait_worker(60)
            ev = wait_evidence()
            procs = h.wait_board_clean()
            h.record("stop", ev is not None and not procs and L.e2e_rendered < 30200, evidence=ev,
                     rendered=L.e2e_rendered, stop_to_worker_end_s=round(time.perf_counter() - ts, 2),
                     guard_tail=h.guard_tail(rdir), procs_after=procs, screenshot=shot,
                     pc_cpu_percpu_1hz=cpu_since(t0))
        elif sc == "disconnect":
            t0 = start(1200, {"disconnect_at_s": 8.0})
            wait_worker(1200 * 0.0205 + 240)
            ev = wait_evidence()
            procs = h.wait_board_clean()
            h.record("disconnect", ev is not None and not procs, evidence=ev, rendered=L.e2e_rendered,
                     procs_after=procs, screenshot=h.shot("disconnect_end"), pc_cpu_percpu_1hz=cpu_since(t0))
        elif sc == "stall":
            t0 = start(1200, {"stall_at_s": 5.0, "stall_for_s": 10.0})
            wait_worker(1200 * 0.0205 + 240)
            ev = wait_evidence()
            procs = h.wait_board_clean()
            h.record("stall", ev is not None and not procs, evidence=ev, rendered=L.e2e_rendered,
                     procs_after=procs, screenshot=h.shot("stall_end"), pc_cpu_percpu_1hz=cpu_since(t0))
        elif sc == "pause":
            t0 = start(3200)
            wait_rendered(300, 180)
            h.click("Pause Display")
            shot = h.shot("pause_paused")
            run_loop(lambda: False, 30.0)
            h.click("Resume Display")
            wait_worker(3200 * 0.0205 + 240)
            ev = wait_evidence()
            procs = h.wait_board_clean()
            h.record("pause", ev is not None and not procs, evidence=ev, rendered=L.e2e_rendered,
                     pc_overflow=len(L.client.sink.overflow) if L.client else None, procs_after=procs,
                     screenshot=shot, screenshot_end=h.shot("pause_end"), pc_cpu_percpu_1hz=cpu_since(t0))
        elif sc == "close":
            start(30200)
            wait_rendered(300, 180)
            shot = h.shot("close_running")
            rdir = h.run_dir()
            ts = time.perf_counter()
            h.root.on_close()
            h.closed = True
            close_s = round(time.perf_counter() - ts, 2)
            end = time.time() + 30
            procs = g.board_procs()
            while procs and time.time() < end:
                time.sleep(1.0)
                procs = g.board_procs()
            h.record("close", not procs, close_handler_s=close_s, run_dir=rdir,
                     guard_tail=g.ssh("tail -2 %s/guard.log 2>/dev/null" % rdir).splitlines() if rdir else [],
                     procs_after=procs, screenshot=shot)
            break
        else:
            raise SystemExit("unknown scenario %s" % sc)
    stop_cpu.set()
    rep = {"tool": "tools/gui_e2e_validation_20261024.py", "mode": LIVE_MODE_E2E,
           "method": "Tk Button.invoke() on the real window (same bound command as a click)",
           "scenarios": h.results, "all_pass": all(r["result"] == "PASS" for r in h.results)}
    (out / "gui_validation.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    if not h.closed:
        h.root.on_close()
    print("ALL_PASS", rep["all_pass"], out)
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
