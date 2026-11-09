"""Physical GUI validation of the Live FPGA tab (real window, real board).

Builds the SAME application window as `python -m host.fpga_result_viewer`
(host.fpga_result_viewer.build_app), then triggers each real button with
Tk's Button.invoke() (same bound command as a mouse click; OS mouse input was
not available in this desktop session), waits for real board results,
checks the board's process table over SSH after every step, captures
screenshots, and writes <out>/gui_validation.json.

Scenarios (each recorded as PASS/FAIL with observations):
  startup          window + both tabs + CONF notes visible
  preflight        Preflight button -> PASS lines in the log
  run_balanced     Start, 40 class-balanced windows to completion: real FPGA
                   results, 4 classes, labels, CONF vs logits, matches recorded
  stop_midrun      Start 300 class-balanced, Stop after >=5 results; board clean
  stop_restart     second Start/Stop cycle on the same window; board clean
  stop_during_push Start then Stop immediately (before launch); nothing launched
  fault_preflight  harmless decoy `temp_guard.sh` (sleep) on the board -> Start is
                   refused by preflight; no runner launched; decoy removed
  fault_no_guard   guard path that does not exist -> no inference started
  fault_guard_exit real guard that exits after 10 readings (results already flowing)
                   -> workload terminated mid-run
  close_midrun     Start, then close the window (real close handler); board clean

Requires: desktop session, board reachable, no other workload. Usage:
  python -m tools.gui_live_validation --out artifacts/results/gui_live_validation_<ts>
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import json
import math
import os
import shlex
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from host import live_fpga as lf
from host.confidence import confidence_from_logits
from host.frame_protocol import LABELS

ROOT = Path(__file__).resolve().parents[1]
FT = ROOT / "artifacts" / "results" / "fpga_fulltest_20261008_141335"
REC = np.loadtxt(FT / "fpga_pred.txt", dtype=np.int64)
LAB = np.loadtxt(FT / "labels.txt", dtype=np.int64)
PS_PATTERN = "python3[.]6 .*[f]pga_live_runner|[t]emp_guard[.]sh|[f]pga_stage_bench|[t]ail_diag"


def ssh(cmd, timeout=30):
    r = subprocess.run(lf._ssh_base(lf.BOARD_HOST, lf.BOARD_USER, lf.SSH_KEY) + [cmd],
                       capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    return r.stdout.strip()


def board_procs():
    out = ssh("ps -eo pid=,args= | grep -E '%s' || true" % PS_PATTERN)
    return [l for l in out.splitlines() if l.strip()]


# --- real mouse clicks (Win32) ------------------------------------------------
_u32 = ctypes.windll.user32 if os.name == "nt" else None
if os.name == "nt":
    # Per-monitor DPI awareness BEFORE Tk exists, so Tk window coordinates,
    # SetCursorPos and the screenshot all use physical pixels (the desktop is
    # scaled; without this clicks and captures land on the wrong spot).
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        _u32.SetProcessDPIAware()


def window_capture(root):
    """Capture the Tk toplevel's own pixels with PrintWindow (works when other
    windows overlap it; does not take focus)."""
    from PIL import Image
    g32 = ctypes.windll.gdi32
    root.update()
    hwnd = int(root.frame(), 16)
    rect = ctypes.wintypes.RECT()
    _u32.GetWindowRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    hdc = _u32.GetWindowDC(hwnd)
    mdc = g32.CreateCompatibleDC(hdc)
    bmp = g32.CreateCompatibleBitmap(hdc, w, h)
    g32.SelectObject(mdc, bmp)
    _u32.PrintWindow(hwnd, mdc, 2)          # PW_RENDERFULLCONTENT

    class BIH(ctypes.Structure):
        _fields_ = [("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
                    ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
                    ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
                    ("biSizeImage", ctypes.c_uint32), ("a", ctypes.c_int32), ("b", ctypes.c_int32),
                    ("c", ctypes.c_uint32), ("d", ctypes.c_uint32)]
    bi = BIH(ctypes.sizeof(BIH), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
    buf = ctypes.create_string_buffer(w * h * 4)
    g32.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bi), 0)
    g32.DeleteObject(bmp); g32.DeleteDC(mdc); _u32.ReleaseDC(hwnd, hdc)
    return Image.frombuffer("RGBA", (w, h), buf, "raw", "BGRA", 0, 1).convert("RGB")


def real_click(widget):
    widget.update_idletasks()
    x = widget.winfo_rootx() + widget.winfo_width() // 2
    y = widget.winfo_rooty() + widget.winfo_height() // 2
    _u32.SetCursorPos(int(x), int(y))
    time.sleep(0.05)
    _u32.mouse_event(0x0002, 0, 0, 0, 0)   # left down
    time.sleep(0.03)
    _u32.mouse_event(0x0004, 0, 0, 0, 0)   # left up
    return (x, y)


class Harness:
    def __init__(self, out: Path):
        from host.fpga_result_viewer import build_app
        self.out = out
        self.shots = out / "screenshots"
        self.shots.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        args = argparse.Namespace(mode="both", result_dir=None, auto=False, screenshot=None)
        self.root, self.replay, self.live = build_app(args)
        self.root.update()
        self.t_window = round(time.perf_counter() - t0, 3)
        self.events = []          # (t, type, payload) recorded at render time
        self.apply_ms = []        # GUI render cost per event
        orig = self.live._apply

        def recorded_apply(ev):
            a = time.perf_counter()
            orig(ev)
            self.apply_ms.append((time.perf_counter() - a) * 1000.0)
            self.events.append((time.perf_counter(), ev))
        self.live._apply = recorded_apply
        self.results = []
        self.closed = False

    # -- helpers on the Tk thread --------------------------------------------
    def pump(self, seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end and not self.closed:
            self.root.update()
            time.sleep(0.02)

    def wait_for(self, pred, timeout):
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            self.root.update()
            if pred():
                return True
            time.sleep(0.02)
        return False

    def button(self, text):
        found = []

        def walk(w):
            for c in w.winfo_children():
                if c.winfo_class() == "Button" and c.cget("text") == text:
                    found.append(c)
                walk(c)
        walk(self.live.ui)
        assert found, text
        return found[0]

    def front(self):
        nb = [c for c in self.root.winfo_children() if c.winfo_class() == "TNotebook"][0]
        nb.select(1)                       # Live FPGA tab
        self.root.deiconify()
        self.root.update()
        self.pump(0.2)

    def click(self, text):
        """Trigger the real button. OS mouse input is not available in this
        desktop session (SetCursorPos has no effect while another application
        holds the foreground), so Tk's Button.invoke() is used: it runs exactly
        the command bound to the button, as a click would."""
        self.front()
        self.button(text).invoke()
        self.pump(0.3)

    def shot(self, name):
        self.front()
        self.pump(0.3)
        p = self.shots / (name + ".png")
        window_capture(self.root).save(p)
        return str(p.relative_to(ROOT))

    def log_text(self):
        return self.live.log.get("1.0", "end")

    def ev_since(self, t0, kind=None):
        return [e for (t, e) in self.events if t >= t0 and (kind is None or e["type"] == kind)]

    def worker_done(self):
        th = self.live.thread
        return th is None or not th.is_alive()

    def wait_worker(self, timeout):
        """Worker finished AND its last queued events rendered by the Tk loop."""
        ok = self.wait_for(self.worker_done, timeout)
        self.wait_for(lambda: self.live.q.empty(), 5)
        self.pump(0.5)
        return ok

    def wait_board_clean(self, timeout=20):
        end = time.time() + timeout
        procs = board_procs()
        while procs and time.time() < end:
            self.pump(1.0)
            procs = board_procs()
        return procs

    def record(self, name, ok, **obs):
        obs["procs_after"] = obs.get("procs_after", [])
        self.results.append({"scenario": name, "result": "PASS" if ok else "FAIL", **obs})
        print("[%s] %s %s" % ("PASS" if ok else "FAIL", name,
                             {k: v for k, v in obs.items() if k in ("note", "results", "classes")}))

    def set_run(self, n, mode):
        self.live.n_var.set(str(n))
        self.live.sample_var.set(mode)
        self.root.update()

    def run_dir(self):
        c = self.live.client
        return os.path.dirname(c.runner_remote) if c is not None else None

    def guard_tail(self, rdir):
        return ssh("tail -2 %s/guard.log 2>/dev/null" % shlex.quote(rdir)).splitlines() if rdir else []

    def check_results(self, evs):
        bad = []
        for e in evs:
            i, c = e["idx"], e["cls"]
            if LABELS[c] != e["label"]:
                bad.append(("label", i))
            if REC[i] != c:
                bad.append(("recorded", i))
            if e.get("logits") is None or e.get("conf") is None:
                bad.append(("no_logits_conf", i))
            elif abs(confidence_from_logits(e["logits"], c) - e["conf"]) > 1e-12:
                bad.append(("conf", i))
            if "CONF=" not in e["frame"]:
                bad.append(("frame", i))
        return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    meta = {"pc_start": datetime.now().isoformat(timespec="seconds"),
            "deployment": lf.DEP_DIR, "bit": lf.BIT, "input": lf.IN_GZ,
            "procs_before": board_procs()}
    if meta["procs_before"]:
        print("board busy:", meta["procs_before"])
        (a.out / "gui_validation.json").write_text(json.dumps(meta, indent=2))
        return 2
    h = Harness(a.out)
    meta["window_build_s"] = h.t_window
    want = lambda s: a.only is None or s in a.only

    # 1 startup ---------------------------------------------------------------
    if want("startup"):
        h.front()
        from host.fpga_result_viewer import LIVE_CONF_NOTE, REPLAY_CONF_NOTE
        texts = []

        def walk(w):
            for c in w.winfo_children():
                if c.winfo_class() == "Label":
                    try:
                        texts.append(c.cget("text"))
                    except Exception:
                        pass
                walk(c)
        walk(h.root)
        ok = LIVE_CONF_NOTE in texts and REPLAY_CONF_NOTE in texts
        h.record("startup", ok, window_build_s=h.t_window, shot=h.shot("01_startup_live_tab"),
                 note="live and replay CONF notes present" if ok else "CONF note missing")

    # 2 preflight -------------------------------------------------------------
    if want("preflight"):
        t0 = time.perf_counter()
        h.set_run(40, "class_balanced")
        h.click("Preflight")
        ok = h.wait_for(lambda: "preflight RESULT" in h.log_text(), 60)
        res_line = [l for l in h.log_text().splitlines() if "preflight" in l]
        h.record("preflight", ok and any("PASS" in l for l in res_line if "RESULT" in l),
                 seconds=round(time.perf_counter() - t0, 2), log=res_line[:20],
                 shot=h.shot("02_preflight_pass"))

    # 3 balanced run to completion ---------------------------------------------
    if want("run_balanced"):
        h.set_run(40, "class_balanced")
        t0 = time.perf_counter()
        h.click("Start")
        first = h.wait_for(lambda: len(h.ev_since(t0, "result")) >= 1, 120)
        t_first = round(time.perf_counter() - t0, 2)
        mid = None
        if first:
            h.wait_for(lambda: len(h.ev_since(t0, "result")) >= 20, 300)
            mid = h.shot("03_run_balanced_mid")
        h.wait_worker(600)
        h.pump(1.0)
        evs = h.ev_since(t0, "result")
        rdir = h.run_dir()
        procs = h.wait_board_clean()
        classes = sorted({e["cls"] for e in evs})
        bad = h.check_results(evs)
        errs = [e["msg"] for e in h.ev_since(t0, "error")]
        ok = len(evs) == 40 and classes == [0, 1, 2, 3] and not bad and not errs and not procs
        h.record("run_balanced", ok, results=len(evs), classes=[LABELS[c] for c in classes],
                 start_to_first_result_s=t_first, total_s=round(time.perf_counter() - t0, 2),
                 correct_vs_labels=int(sum(LAB[e["idx"]] == e["cls"] for e in evs)),
                 match_recorded=int(sum(REC[e["idx"]] == e["cls"] for e in evs)),
                 indices=[e["idx"] for e in evs], mismatches=bad[:10], errors=errs,
                 frames=[e["frame"] for e in evs[:6]], banner=h.live.big_var.get(),
                 guard_log_tail=h.guard_tail(rdir), run_dir=rdir, procs_after=procs,
                 shot_mid=mid, shot_end=h.shot("04_run_balanced_end"))

    # 4/5 stop mid-run, twice --------------------------------------------------
    for k, name in ((1, "stop_midrun"), (2, "stop_restart")):
        if not want(name):
            continue
        h.set_run(300, "class_balanced")
        t0 = time.perf_counter()
        h.click("Start")
        got = h.wait_for(lambda: len(h.ev_since(t0, "result")) >= 5, 180)
        rdir = h.run_dir()
        shot = h.shot("05_%s_before_stop" % name)
        ts = time.perf_counter()
        h.click("Stop")
        done = h.wait_worker(40)
        stop_s = round(time.perf_counter() - ts, 2)
        procs = h.wait_board_clean()
        n_res = len(h.ev_since(t0, "result"))
        gt = h.guard_tail(rdir)
        ok = got and done and not procs and 5 <= n_res < 300 and any("guard END" in l for l in gt)
        h.record(name, ok, results=n_res, stop_to_worker_end_s=stop_s, run_dir=rdir,
                 guard_log_tail=gt, procs_after=procs, status=h.live.conn_var.get(),
                 shot=shot, shot_after=h.shot("06_%s_after_stop" % name))

    # 6 stop during push ------------------------------------------------------
    if want("stop_during_push"):
        h.set_run(40, "class_balanced")
        t0 = time.perf_counter()
        h.click("Start")
        h.click("Stop")
        h.wait_worker(60)
        procs = h.wait_board_clean(5)
        evs = h.ev_since(t0)
        n_res = sum(e["type"] == "result" for e in evs)
        msgs = [e.get("msg") for e in evs if e["type"] in ("status", "error")]
        rdir = h.run_dir()
        launched = bool(rdir and ssh("test -s %s/guard.log && echo yes || echo no" % rdir) == "yes")
        ok = n_res == 0 and not procs and not launched
        h.record("stop_during_push", ok, results=n_res, messages=msgs, run_dir=rdir,
                 guard_log_written=launched, procs_after=procs)

    # 7 preflight fault: harmless decoy process named like a guard ------------
    if want("fault_preflight"):
        ssh("mkdir -p /tmp/gui_decoy_20261023 && printf 'sleep 120\\n' > /tmp/gui_decoy_20261023/temp_guard.sh"
            " ; setsid bash /tmp/gui_decoy_20261023/temp_guard.sh > /dev/null 2>&1 < /dev/null & echo ok")
        decoy = board_procs()
        h.set_run(40, "class_balanced")
        t0 = time.perf_counter()
        h.click("Start")
        h.wait_worker(90)
        evs = h.ev_since(t0)
        errs = [e["msg"] for e in evs if e["type"] == "error"]
        n_res = sum(e["type"] == "result" for e in evs)
        shot = h.shot("07_fault_preflight_refused")
        runners = [p for p in board_procs() if "fpga_live_runner" in p]
        ssh("pkill -f gui_decoy_20261023; rm -rf /tmp/gui_decoy_20261023; true")
        procs = h.wait_board_clean(10)
        ok = bool(decoy) and n_res == 0 and not runners and any("preflight failed" in m for m in errs) and not procs
        h.record("fault_preflight", ok, decoy_seen=decoy, errors=errs, results=n_res,
                 runners_during=runners, procs_after=procs, shot=shot)

    # 8 missing guard ---------------------------------------------------------
    if want("fault_no_guard"):
        real_push = lf.push_board_files

        def push_without_guard(*args):
            r, g = real_push(*args)
            return r, os.path.dirname(r) + "/MISSING_temp_guard.sh"
        lf.push_board_files = push_without_guard
        try:
            h.set_run(40, "class_balanced")
            t0 = time.perf_counter()
            h.click("Start")
            h.wait_worker(120)
        finally:
            lf.push_board_files = real_push
        evs = h.ev_since(t0)
        msgs = [e.get("msg") for e in evs if e["type"] in ("status", "error", "abort")]
        n_res = sum(e["type"] == "result" for e in evs)
        procs = h.wait_board_clean(5)
        ok = n_res == 0 and not procs and any("guard missing" in (m or "") for m in msgs)
        h.record("fault_no_guard", ok, results=n_res, messages=msgs, procs_after=procs,
                 shot=h.shot("08_fault_no_guard"))

    # 9 guard exits early -----------------------------------------------------
    if want("fault_guard_exit"):
        real_push = lf.push_board_files
        src = (ROOT / "tools" / "board" / "temp_guard.sh").read_text(encoding="utf-8")
        assert 'while kill -0 "$PID"' in src
        early = src.replace('while kill -0 "$PID" 2>/dev/null; do',
                            'N_EARLY=0\nwhile kill -0 "$PID" 2>/dev/null; do\n'
                            '  N_EARLY=$((N_EARLY+1)); if [ $N_EARLY -gt 10 ]; then '
                            'echo "FAULT_INJECTION guard exits early" >> "$LOG"; exit 0; fi', 1)
        tmp = a.out / "temp_guard_EARLY_EXIT_fault_injection.sh"
        tmp.write_bytes(early.encode())

        def push_early_guard(*args):
            r, g = real_push(*args)
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-i", lf.SSH_KEY, str(tmp),
                            "%s@%s:%s" % (lf.BOARD_USER, lf.BOARD_HOST, g)], check=True)
            return r, g
        lf.push_board_files = push_early_guard
        try:
            h.set_run(3000, "sequential")
            t0 = time.perf_counter()
            h.click("Start")
            h.wait_worker(120)
        finally:
            lf.push_board_files = real_push
        rdir = h.run_dir()
        evs = h.ev_since(t0)
        n_res = sum(e["type"] == "result" for e in evs)
        msgs = [e.get("msg") for e in evs if e["type"] in ("error", "abort")]
        procs = h.wait_board_clean()
        gt = h.guard_tail(rdir)
        ok = (0 < n_res < 3000 and not procs and any("FAULT_INJECTION" in l for l in gt)
              and any("guard exited" in (m or "") or "failed" in (m or "") for m in msgs))
        h.record("fault_guard_exit", ok, results=n_res, messages=msgs, guard_log_tail=gt,
                 run_dir=rdir, procs_after=procs, shot=h.shot("09_fault_guard_exit"))

    # 10 close mid-run (real close handler) ------------------------------------
    if want("close_midrun"):
        h.set_run(300, "class_balanced")
        t0 = time.perf_counter()
        h.click("Start")
        got = h.wait_for(lambda: len(h.ev_since(t0, "result")) >= 5, 180)
        rdir = h.run_dir()
        shot = h.shot("10_close_midrun_before_close")
        tc = time.perf_counter()
        h.root.on_close()                  # identical to the window's X button handler
        h.closed = True
        close_s = round(time.perf_counter() - tc, 2)
        procs = []
        end = time.time() + 20
        procs = board_procs()
        while procs and time.time() < end:
            time.sleep(1)
            procs = board_procs()
        gt = h.guard_tail(rdir)
        n_res = len(h.ev_since(t0, "result"))
        ok = got and not procs and any("guard END" in l for l in gt)
        h.record("close_midrun", ok, results=n_res, close_handler_s=close_s, run_dir=rdir,
                 guard_log_tail=gt, procs_after=procs, shot=shot,
                 note="window closed via its WM_DELETE_WINDOW handler (root.on_close)")

    if not h.closed:
        h.root.on_close()
    am = np.array(h.apply_ms) if h.apply_ms else np.array([0.0])
    meta.update({
        "pc_end": datetime.now().isoformat(timespec="seconds"),
        "click_method": "Tk Button.invoke() on the real buttons of the real window (OS mouse input unavailable: SetCursorPos had no effect while another application held the foreground; checked 2026-10-23)",
        "gui_render_ms_per_event": {"n": int(len(h.apply_ms)), "mean": round(float(am.mean()), 3),
                                    "p99": round(float(np.percentile(am, 99)), 3),
                                    "max": round(float(am.max()), 3)},
        "procs_final": board_procs(),
        "scenarios": h.results,
        "all_pass": all(r["result"] == "PASS" for r in h.results),
    })
    (a.out / "gui_validation.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print("ALL PASS" if meta["all_pass"] else "SOME FAILED", "->", a.out / "gui_validation.json")
    return 0 if meta["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
