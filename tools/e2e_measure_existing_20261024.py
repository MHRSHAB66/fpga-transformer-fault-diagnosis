"""Measure the EXISTING `ram_realtime_demo` GUI path as-is against the frozen end-to-end
criterion (docs/reports/e2e_hard_rt_criterion_20261024.md). No product behaviour is changed:

- the real viewer window (host.fpga_result_viewer.build_app) runs the existing mode;
- render completion per result: a Tk after_idle callback queued right after the existing
  _apply() -- Tk runs idle callbacks FIFO, so it fires after that update's redraw handlers;
- GUI refresh evidence: every _poll() call time;
- PC CPU via psutil (1 Hz, own thread); board /proc/stat before/after (no polling during the run).

Usage: python -m tools.e2e_measure_existing_20261024 --out artifacts/results/e2e_existing_ram_mode_<ts> --n 30200
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import psutil

from tools import gui_live_validation as g


def board_stat():
    return g.ssh("cat /proc/stat | head -3; cat /proc/loadavg; date -u +%FT%TZ")


def main() -> int:
    from host.fpga_result_viewer import build_app, LIVE_MODE_RAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=30200)
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root, _, live = build_app(argparse.Namespace(mode="both", result_dir=None, auto=False, screenshot=None))
    nb = [c for c in root.winfo_children() if c.winfo_class() == "TNotebook"][0]
    nb.select(1)
    live.mode_var.set(LIVE_MODE_RAM)
    live._show_mode()
    live.n_var.set(str(a.n))
    root.update()

    renders = {}          # k -> render-complete perf_counter
    polls = []            # _poll call times
    orig_apply, orig_poll = live._apply, live._poll

    def apply(ev):
        orig_apply(ev)
        if ev.get("type") == "result" and "k" in ev:
            k = ev["k"]
            root.after_idle(lambda k=k: renders.setdefault(k, time.perf_counter()))

    def poll():
        polls.append(time.perf_counter())
        orig_poll()
    live._apply, live._poll = apply, poll
    # _poll was already scheduled with the original bound method; the next reschedule inside
    # orig_poll uses self._poll -> our wrapper from the second call on.

    cpu, stop = [], threading.Event()

    def cpu_sampler():
        psutil.cpu_percent(percpu=True)
        while not stop.wait(1.0):
            cpu.append((time.perf_counter(), psutil.cpu_percent(percpu=True)))
    threading.Thread(target=cpu_sampler, daemon=True).start()

    stat0 = board_stat()
    t0 = time.perf_counter()
    live.start()
    while True:
        root.update()
        th = live.thread
        if th is not None and not th.is_alive() and live.q.empty():
            break
        time.sleep(0.001)
    # evidence (write_evidence runs in a worker after 'ram_done')
    end = time.perf_counter() + 180
    while time.perf_counter() < end and not (live.ram_evidence_dir and
                                             (Path(live.ram_evidence_dir) / "pc_evidence.json").exists()):
        root.update()
        time.sleep(0.01)
    for _ in range(50):
        root.update()
        time.sleep(0.01)
    stop.set()
    stat1 = board_stat()
    rec = {"tool": "tools/e2e_measure_existing_20261024.py", "mode": LIVE_MODE_RAM,
           "n_total": a.n, "evidence_dir": live.ram_evidence_dir, "wall_s": time.perf_counter() - t0,
           "renders": {str(k): v for k, v in sorted(renders.items())},
           "poll_times": polls, "pc_cpu_percpu_1hz": cpu,
           "board_stat_before": stat0, "board_stat_after": stat1,
           "board_procs_after": g.board_procs()}
    (out / "existing_mode_gui_trace.json").write_text(json.dumps(rec), encoding="utf-8")
    live.shutdown()
    root.destroy()
    print("evidence:", live.ram_evidence_dir, "renders:", len(renders), "polls:", len(polls))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
