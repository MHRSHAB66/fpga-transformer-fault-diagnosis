"""Stage 11 -- Windows/Tkinter viewer for the FPGA fault-classification results.

Display only. It NEVER re-classifies: every predicted class comes straight from
the FPGA run files and is paired against ground truth for scoring. The viewer
replays all 95404 test-set windows, or (optional) shows live FRAME=... lines
from the board, and reports the same metrics the board already recorded.

Replay source (primary, default):
    the fpga_fulltest result directory holding fpga_pred.txt / labels.txt /
    meta.json (+ fpga_fulltest_metrics.json used as the reference to check against)

Run
---
    python -m host.fpga_result_viewer
    python -m host.fpga_result_viewer --result-dir artifacts/results/fpga_fulltest_20261008_141335
    python -m host.fpga_result_viewer --source serial --port COM6   # optional live mode
    python -m host.fpga_result_viewer --screenshot shot.png --auto   # for smoke tests

Threading: the Tk widgets are only ever touched from the main thread. Replay
stepping uses root.after(); the Fast evaluation pass runs in a worker thread and
reports back through a queue polled with after(), so the window never freezes.
"""
from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from typing import List, Optional

from host.frame_protocol import LABELS
from host.replay_data import (
    ReplayDataset, MetricsAccumulator, DataError, load_meta, compare_to_reference,
)

# class -> background colour for the big "latest result" banner
COLORS = {"normal": "#2e7d32", "imbalance": "#f9a825",
          "misalignment": "#ef6c00", "bearing_fault": "#c62828"}

# replay speeds: label -> inter-frame delay (ms); "Max" renders as fast as Tk can
SPEEDS = [("0.5x (1 s)", 1000), ("1x (0.5 s)", 500), ("2x (0.25 s)", 250),
          ("5x (0.1 s)", 100), ("20x (0.025 s)", 25), ("Max", 1)]


# ---------------------------------------------------------------------------
# Default result dir: newest fpga_fulltest_* under artifacts/results
# ---------------------------------------------------------------------------
def default_result_dir() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, os.pardir, "artifacts", "results")
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return None
    cands = sorted(d for d in os.listdir(root)
                   if d.startswith("fpga_fulltest_")
                   and os.path.isdir(os.path.join(root, d)))
    return os.path.join(root, cands[-1]) if cands else None


class ViewerApp:
    """The replay viewer. Usable with mainloop (interactive) or driven directly
    (tests/screenshots) via step/jump/reset/run_fast_eval_blocking."""

    def __init__(self, root, dataset: ReplayDataset, reference: Optional[dict] = None,
                 title: str = "FPGA Fault-Classification Result Viewer",
                 container=None):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.root = root                   # toplevel: used for after()/title
        self.ui = container or root        # where widgets are packed (tab or root)
        self.ds = dataset
        self.reference = reference or {}

        # replay state: acc reflects the contiguous prefix [0 .. idx]
        self.idx = -1                      # -1 == nothing played yet
        self.acc = MetricsAccumulator()
        self.running = False
        self.delay = 500
        self._after_id = None
        self._eval_q: "queue.Queue[object]" = queue.Queue()
        self._eval_thread = None

        if container is None:
            root.title(title)
            root.geometry("920x620")

        # ---- latest-result banner ----------------------------------------
        self.big_var = tk.StringVar(value="--")
        self.result_var = tk.StringVar(value="load a result set to begin")
        self.big = tk.Label(self.ui, textvariable=self.big_var, fg="white", bg="#455a64",
                            font=("Segoe UI", 30, "bold"), pady=10)
        self.big.pack(fill="x")
        self.result_lbl = tk.Label(self.ui, textvariable=self.result_var,
                                   font=("Segoe UI", 13, "bold"))
        self.result_lbl.pack(pady=(2, 0))

        # ---- per-record detail line ---------------------------------------
        self.detail_var = tk.StringVar(value="")
        tk.Label(self.ui, textvariable=self.detail_var,
                 font=("Consolas", 10)).pack(pady=(0, 4))

        # ---- counters ------------------------------------------------------
        self.counter_var = tk.StringVar(value="")
        self.byclass_var = tk.StringVar(value="")
        cf = tk.Frame(self.ui)
        cf.pack(fill="x", padx=8)
        tk.Label(cf, textvariable=self.counter_var, font=("Consolas", 10),
                 fg="#222", justify="left", anchor="w").pack(fill="x")
        tk.Label(cf, textvariable=self.byclass_var, font=("Consolas", 10),
                 fg="#555", justify="left", anchor="w").pack(fill="x")

        # ---- controls ------------------------------------------------------
        ctrl = tk.Frame(self.ui)
        ctrl.pack(fill="x", padx=8, pady=6)
        self._btn(ctrl, "Start", self.start)
        self._btn(ctrl, "Pause", self.pause)
        self._btn(ctrl, "Resume", self.resume)
        self._btn(ctrl, "Stop", self.stop)
        self._btn(ctrl, "Reset", self.reset)
        self._btn(ctrl, "Prev", self.prev)
        self._btn(ctrl, "Next", self.next)
        self._btn(ctrl, "Fast eval", self.fast_eval)

        ctrl2 = tk.Frame(self.ui)
        ctrl2.pack(fill="x", padx=8)
        tk.Label(ctrl2, text="speed:").pack(side="left")
        self.speed_var = tk.StringVar(value=SPEEDS[1][0])
        om = ttk.OptionMenu(ctrl2, self.speed_var, SPEEDS[1][0],
                            *[s[0] for s in SPEEDS], command=self._on_speed)
        om.pack(side="left", padx=(2, 12))
        self._on_speed(self.speed_var.get())
        tk.Label(ctrl2, text="go to frame:").pack(side="left")
        self.goto_var = tk.StringVar()
        e = tk.Entry(ctrl2, textvariable=self.goto_var, width=10)
        e.pack(side="left", padx=2)
        e.bind("<Return>", lambda _ev: self.goto())
        self._btn(ctrl2, "Go", self.goto, side="left")

        # ---- recent table --------------------------------------------------
        tk.Label(self.ui, text="recent predictions (newest first)",
                 font=("Segoe UI", 9, "bold")).pack(pady=(6, 0))
        cols = ("frame", "class", "label", "true", "true_label", "result", "conf")
        self.tree = ttk.Treeview(self.ui, columns=cols, show="headings", height=11)
        widths = (70, 55, 120, 55, 120, 80, 60)
        heads = ("FRAME", "CLASS", "LABEL", "TRUE", "TRUE_LABEL", "RESULT", "CONF")
        for c, w, h in zip(cols, widths, heads):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="center")
        self.tree.tag_configure("correct", background="#e8f5e9")
        self.tree.tag_configure("wrong", background="#ffebee")
        self.tree.pack(fill="both", expand=True, padx=8, pady=6)

        self.status_var = tk.StringVar(value="ready")
        tk.Label(self.ui, textvariable=self.status_var, font=("Consolas", 9),
                 fg="#555", anchor="w").pack(fill="x", padx=8, pady=(0, 6))

        self._refresh_counters()
        self.root.after(80, self._poll_eval)

    # -- widget helpers -----------------------------------------------------
    def _btn(self, parent, text, cmd, side="left"):
        b = self.tk.Button(parent, text=text, command=cmd, width=8)
        b.pack(side=side, padx=2)
        return b

    def _on_speed(self, label):
        for lbl, ms in SPEEDS:
            if lbl == label:
                self.delay = ms
                return

    # -- replay engine ------------------------------------------------------
    def _set_index(self, target: int, incremental: bool):
        """Move current index to ``target`` and keep the prefix accumulator in
        sync. Incremental for the common forward-by-one path; otherwise rebuild
        the prefix [0..target] from scratch (used by Prev / Go to)."""
        target = max(-1, min(target, self.ds.n - 1))
        if incremental and target == self.idx + 1:
            self.acc.add(self.ds.labels[target], self.ds.preds[target])
            self.idx = target
        else:
            self.acc = MetricsAccumulator()
            if target >= 0:
                self.acc.add_all(self.ds.labels[:target + 1], self.ds.preds[:target + 1])
            self.idx = target
        if self.idx >= 0:
            self._show_record(self.ds.record(self.idx), rebuild_table=not incremental)
        self._refresh_counters()

    def _show_record(self, rec, rebuild_table: bool):
        label = rec["label"]
        ok = rec["correct"]
        self.big.configure(bg=COLORS.get(label, "#455a64"))
        self.big_var.set(label.replace("_", " ").upper())
        self.result_var.set("CORRECT" if ok else "WRONG")
        self.result_lbl.configure(fg="#2e7d32" if ok else "#c62828")
        self.detail_var.set(
            "FRAME={frame}  CLASS={class_id} {label}  "
            "TRUE_CLASS={true_class} {true_label}  {res}  CONF=N/A".format(
                res="CORRECT" if ok else "WRONG", **rec))
        if rebuild_table:
            self._rebuild_table_tail()
        else:
            self._table_insert(rec)

    def _table_insert(self, rec):
        tag = "correct" if rec["correct"] else "wrong"
        self.tree.insert("", 0, values=(
            rec["frame"], rec["class_id"], rec["label"],
            rec["true_class"], rec["true_label"],
            "CORRECT" if rec["correct"] else "WRONG", "N/A"), tags=(tag,))
        # keep the table light: only the most recent ~200 rows
        kids = self.tree.get_children()
        if len(kids) > 200:
            for item in kids[200:]:
                self.tree.delete(item)

    def _rebuild_table_tail(self):
        self.tree.delete(*self.tree.get_children())
        lo = max(0, self.idx - 199)
        for i in range(lo, self.idx + 1):
            self._table_insert(self.ds.record(i))  # inserts newest-first

    def _refresh_counters(self):
        a = self.acc
        cur = self.idx + 1
        self.counter_var.set(
            "frame {cur}/{tot}    correct={ok}   incorrect={bad}   "
            "running_accuracy={acc:.4f}".format(
                cur=cur, tot=self.ds.n, ok=a.correct, bad=a.incorrect,
                acc=a.running_accuracy))
        pc = a.predicted_counts()
        self.byclass_var.set("predicted:  " + "   ".join(
            "{}={}".format(LABELS[c], pc[c]) for c in range(len(LABELS))))

    def _tick(self):
        if not self.running:
            return
        if self.idx + 1 >= self.ds.n:
            self.running = False
            self.status_var.set("replay finished at last frame")
            return
        self._set_index(self.idx + 1, incremental=True)
        self._after_id = self.root.after(self.delay, self._tick)

    # -- controls -----------------------------------------------------------
    def start(self):
        self.reset()
        self.resume()

    def resume(self):
        if self.running:
            return
        self.running = True
        self.status_var.set("playing")
        self._tick()

    def pause(self):
        self.running = False
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self.status_var.set("paused at frame {}".format(self.idx + 1))

    def stop(self):
        self.pause()
        self.status_var.set("stopped at frame {}".format(self.idx + 1))

    def reset(self):
        self.pause()
        self.idx = -1
        self.acc = MetricsAccumulator()
        self.big.configure(bg="#455a64")
        self.big_var.set("--")
        self.result_var.set("reset")
        self.result_lbl.configure(fg="black")
        self.detail_var.set("")
        self.tree.delete(*self.tree.get_children())
        self._refresh_counters()
        self.status_var.set("reset")

    def next(self):
        self.pause()
        if self.idx + 1 < self.ds.n:
            self._set_index(self.idx + 1, incremental=True)
            self.status_var.set("step -> frame {}".format(self.idx + 1))

    def prev(self):
        self.pause()
        if self.idx >= 0:
            self._set_index(self.idx - 1, incremental=False)
            self.status_var.set("step <- frame {}".format(self.idx + 1))

    def goto(self):
        self.pause()
        raw = self.goto_var.get().strip()
        if not raw:
            return
        try:
            target = int(raw)
        except ValueError:
            self.status_var.set("go to frame: {!r} is not an integer".format(raw))
            return
        if not (0 <= target < self.ds.n):
            self.status_var.set("go to frame: {} out of range 0..{}".format(
                target, self.ds.n - 1))
            return
        self._set_index(target, incremental=False)
        self.status_var.set("jumped to frame {}".format(target))

    # -- Fast / Full evaluation --------------------------------------------
    def fast_eval(self):
        if self._eval_thread and self._eval_thread.is_alive():
            return
        self.pause()
        self.status_var.set("fast evaluation running over all {} windows...".format(self.ds.n))

        def work():
            acc = MetricsAccumulator()
            acc.add_all(self.ds.labels, self.ds.preds)
            res = acc.finalize()
            diffs = compare_to_reference(res, self.reference) if self.reference else []
            self._eval_q.put(("eval", res, diffs))

        self._eval_thread = threading.Thread(target=work, daemon=True)
        self._eval_thread.start()

    def run_fast_eval_blocking(self):
        """Synchronous variant for tests/screenshots (no Tk mainloop needed)."""
        acc = MetricsAccumulator()
        acc.add_all(self.ds.labels, self.ds.preds)
        res = acc.finalize()
        diffs = compare_to_reference(res, self.reference) if self.reference else []
        return res, diffs

    def _poll_eval(self):
        try:
            while True:
                item = self._eval_q.get_nowait()
                if item[0] == "eval":
                    self._show_eval(item[1], item[2])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_eval)

    def _show_eval(self, res: dict, diffs: List[str]):
        self.status_var.set(
            "fast eval done: acc={:.6f} macro_f1={:.6f}  {}".format(
                res["accuracy"], res["macro_f1"],
                "MATCHES reference" if (self.reference and not diffs)
                else ("{} discrepancy(ies) vs reference".format(len(diffs)) if self.reference
                      else "no reference file")))
        self._eval_popup(res, diffs)

    def _eval_popup(self, res: dict, diffs: List[str]):
        tk = self.tk
        top = tk.Toplevel(self.root)
        top.title("Fast evaluation -- full test set")
        top.geometry("640x560")
        head = ("windows={windows}  correct={correct}\n"
                "accuracy        = {acc:.6f}\n"
                "macro precision = {mp:.6f}\n"
                "macro recall    = {mr:.6f}\n"
                "macro F1        = {mf:.6f}\n"
                "weighted F1     = {wf:.6f}").format(
            windows=res["test_windows"], correct=res["correct"],
            acc=res["accuracy"], mp=res["macro_precision"],
            mr=res["macro_recall"], mf=res["macro_f1"], wf=res["weighted_f1"])
        tk.Label(top, text=head, font=("Consolas", 11), justify="left",
                 anchor="w").pack(fill="x", padx=10, pady=8)

        # confusion matrix (rows=true, cols=pred)
        tk.Label(top, text="confusion matrix  (rows = TRUE, cols = PRED)",
                 font=("Segoe UI", 9, "bold")).pack()
        grid = tk.Frame(top)
        grid.pack(padx=10, pady=4)
        short = ["0 norm", "1 imb", "2 mis", "3 bear"]
        tk.Label(grid, text="T\\P", font=("Consolas", 9, "bold"),
                 width=8, borderwidth=1, relief="solid").grid(row=0, column=0)
        for j, s in enumerate(short):
            tk.Label(grid, text=s, font=("Consolas", 9, "bold"), width=9,
                     borderwidth=1, relief="solid").grid(row=0, column=j + 1)
        cm = res["confusion_matrix"]
        for i, s in enumerate(short):
            tk.Label(grid, text=s, font=("Consolas", 9, "bold"), width=8,
                     borderwidth=1, relief="solid").grid(row=i + 1, column=0)
            for j in range(len(short)):
                bg = "#e8f5e9" if i == j else "white"
                tk.Label(grid, text=str(cm[i][j]), font=("Consolas", 9), width=9,
                         bg=bg, borderwidth=1, relief="solid").grid(row=i + 1, column=j + 1)

        # per-class P/R/F1
        pc = "per-class  precision / recall / f1\n" + "\n".join(
            "  {:<14} {:.4f} / {:.4f} / {:.4f}  (support {})".format(
                LABELS[c], res["per_class_precision"][c], res["per_class_recall"][c],
                res["per_class_f1"][c], res["support"][c])
            for c in range(len(LABELS)))
        tk.Label(top, text=pc, font=("Consolas", 10), justify="left",
                 anchor="w").pack(fill="x", padx=10, pady=8)

        if self.reference:
            if diffs:
                msg = "DISCREPANCIES vs fpga_fulltest_metrics.json:\n" + "\n".join(diffs)
                fg = "#c62828"
            else:
                msg = "Matches fpga_fulltest_metrics.json within 1e-6."
                fg = "#2e7d32"
        else:
            msg = "No fpga_fulltest_metrics.json reference loaded."
            fg = "#555"
        tk.Label(top, text=msg, font=("Consolas", 10), fg=fg, justify="left",
                 anchor="w", wraplength=600).pack(fill="x", padx=10, pady=4)
        return top


# ---------------------------------------------------------------------------
# LIVE FPGA MODE -- real on-board AXI-DMA inference streamed over SSH/Ethernet.
# The PC performs NO classification; it displays what the FPGA returned. Windows
# are fed on the board (see host.live_fpga / tools/board/fpga_live_runner.py).
# ---------------------------------------------------------------------------
class LiveFrame:
    """Embeddable Live-FPGA tab: drives host.live_fpga.LiveFpgaClient in a worker
    thread and shows the latest class, per-inference latency, throughput, board
    temperature, connection status and error count, with Start/Pause/Resume/Stop.
    """

    def __init__(self, container, tkroot):
        import tkinter as tk
        self.tk = tk
        self.ui = container
        self.root = tkroot
        # Bounded queue so a long Pause (display frozen, board still streaming)
        # cannot grow memory without limit; overflow drops the OLDEST buffered
        # event and is surfaced honestly as a warning + counter.
        self.q = queue.Queue(maxsize=5000)
        self.dropped = 0
        self.client = None
        self.thread = None
        self.paused = False
        self.n_target = 150
        self._alive = True
        self._poll_after = None
        # counters
        self.executed = 0
        self.errors = 0
        self.by_class = {c: 0 for c in LABELS}
        self.lat_sum = 0.0
        self.lat_last = None
        self.temp_last = None
        self.temp_max = None
        self._t0 = None

        self.big_var = tk.StringVar(value="--")
        self.sub_var = tk.StringVar(value="idle -- press Start to run real FPGA inference")
        self.conn_var = tk.StringVar(value="connection: not started")
        self.metric_var = tk.StringVar(value="")
        self.byclass_var = tk.StringVar(value="")

        self.big = tk.Label(self.ui, textvariable=self.big_var, fg="white", bg="#455a64",
                            font=("Segoe UI", 30, "bold"), pady=10)
        self.big.pack(fill="x")
        tk.Label(self.ui, textvariable=self.sub_var, font=("Consolas", 11)).pack(pady=(2, 0))
        tk.Label(self.ui, textvariable=self.conn_var, font=("Consolas", 10),
                 fg="#1565c0").pack(pady=(2, 0))
        tk.Label(self.ui, textvariable=self.metric_var, font=("Consolas", 11),
                 anchor="w", justify="left").pack(fill="x", padx=10, pady=4)
        tk.Label(self.ui, textvariable=self.byclass_var, font=("Consolas", 10),
                 fg="#555", anchor="w").pack(fill="x", padx=10)

        ctrl = tk.Frame(self.ui)
        ctrl.pack(fill="x", padx=8, pady=6)
        tk.Label(ctrl, text="windows:").pack(side="left")
        self.n_var = tk.StringVar(value="150")
        tk.Entry(ctrl, textvariable=self.n_var, width=8).pack(side="left", padx=(2, 12))
        # Note: "Pause Display" only pauses the PC-side view; the FPGA keeps
        # running on the board. "Stop" is the only control that ends on-board
        # inference. Labelled honestly so no one reads it as pausing the FPGA.
        for text, cmd, w in (("Preflight", self.preflight, 9), ("Start", self.start, 8),
                             ("Pause Display", self.pause, 12),
                             ("Resume Display", self.resume, 13), ("Stop", self.stop, 8)):
            tk.Button(ctrl, text=text, width=w, command=cmd).pack(side="left", padx=2)

        tk.Label(self.ui, text="live FPGA frames (newest first)",
                 font=("Segoe UI", 9, "bold")).pack(pady=(6, 0))
        self.log = tk.Text(self.ui, height=12, width=70, font=("Consolas", 9), state="disabled")
        self.log.pack(fill="both", expand=True, padx=8, pady=6)

        self._refresh()
        self._poll_after = self.root.after(80, self._poll)

    # -- controls -----------------------------------------------------------
    def preflight(self):
        """Run the non-destructive preflight (read-only; no overlay/inference)
        in a worker thread and report each check in the log."""
        if self.thread and self.thread.is_alive():
            self.conn_var.set("preflight skipped: a run is active (Stop first)")
            return
        try:
            self.n_target = max(1, int(self.n_var.get()))
        except ValueError:
            self.n_target = 150
        self.conn_var.set("preflight: probing board (read-only)...")

        def work():
            try:
                from host.live_preflight import preflight as run_pf, all_ok
                checks = run_pf(n=self.n_target)
            except Exception as e:
                self._put({"type": "status", "msg": "preflight error: %s" % e})
                return
            for c in checks:
                self._put({"type": "status",
                           "msg": "preflight [%s] %s -- %s"
                                  % ("OK" if c["ok"] else "XX", c["check"], c["detail"])})
            self._put({"type": "status",
                       "msg": "preflight RESULT: %s"
                              % ("PASS -- safe to Start" if all_ok(checks)
                                 else "FAIL -- fix XX items before Start")})

        threading.Thread(target=work, daemon=True).start()

    def start(self):
        if self.thread and self.thread.is_alive():
            self.conn_var.set("already running (Start ignored); use Stop first")
            return
        try:
            self.n_target = max(1, int(self.n_var.get()))
        except ValueError:
            self.n_target = 150
        # reset counters
        self.executed = 0; self.errors = 0
        self.by_class = {c: 0 for c in LABELS}
        self.lat_sum = 0.0; self.lat_last = None
        self.temp_last = None; self.temp_max = None; self._t0 = None
        self.paused = False
        self._logclear()
        try:
            from host import live_fpga as lf
        except Exception as e:
            self.conn_var.set("live client import failed: {}".format(e))
            return
        self.conn_var.set("connection: pushing runner + connecting...")

        def work():
            try:
                lf.push_board_files(lf.BOARD_HOST, lf.BOARD_USER, lf.SSH_KEY, 22)
            except Exception as e:
                self._put({"type": "status", "msg": "push warning: %s" % e})
            self.client = lf.LiveFpgaClient(n=self.n_target)
            for ev in self.client.stream():
                self._put(ev)
            self._put({"type": "status", "msg": "stream ended"})

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def _put(self, ev):
        """Bounded enqueue: if the display is paused long enough to fill the
        buffer, drop the OLDEST event (the FPGA still ran it; it is just not
        shown) and count it, instead of growing memory without limit."""
        try:
            self.q.put_nowait(ev)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(ev)
            except queue.Full:
                pass

    def pause(self):
        """Pause only the PC-side DISPLAY. The FPGA keeps running on the board;
        incoming results buffer and catch up on Resume. Use Stop to actually end
        on-board inference."""
        self.paused = True
        self.conn_var.set("DISPLAY PAUSED -- FPGA still running on board (press Stop to end it)")

    def resume(self):
        self.paused = False
        self.conn_var.set("connection: display resumed")

    def stop(self):
        """Actually end on-board inference: signal the client, which pkills the
        remote runner and terminates the SSH process."""
        if self.client:
            self.client.stop()
        self.conn_var.set("connection: stopped by user (on-board runner terminated)")

    def shutdown(self, join_timeout=5.0):
        """Tear down cleanly on window close: stop the board runner and wait
        briefly for the SSH worker thread to finish. Returns True if the worker
        is gone. Safe to call repeatedly and when nothing is running."""
        self._alive = False
        if self._poll_after is not None:
            try:
                self.root.after_cancel(self._poll_after)
            except Exception:
                pass
            self._poll_after = None
        try:
            if self.client:
                self.client.stop()
        except Exception:
            pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=join_timeout)
        return not (self.thread and self.thread.is_alive())

    # -- event pump ---------------------------------------------------------
    def _poll(self):
        if not self._alive:
            return  # window is being torn down; stop rescheduling after() callbacks
        if not self.paused:
            drained = 0
            while drained < 200:
                try:
                    ev = self.q.get_nowait()
                except queue.Empty:
                    break
                self._apply(ev)
                drained += 1
        self._poll_after = self.root.after(80, self._poll)

    def _apply(self, ev):
        t = ev["type"]
        if t == "status":
            self.conn_var.set("connection: " + ev["msg"])
            self._logline("[status] " + ev["msg"])
        elif t == "error":
            self.errors += 1
            self.conn_var.set("connection ERROR: " + ev["msg"])
            self._logline("[ERROR] " + ev["msg"])
        elif t == "temp_start":
            self._logline("[temp_start] %s" % ev["temp_c"])
        elif t == "meta":
            self.conn_var.set("connection: running on board, N=%d" % ev["n"])
        elif t == "result":
            if self._t0 is None:
                self._t0 = time.time()
            self.executed += 1
            self.by_class[ev["cls"]] = self.by_class.get(ev["cls"], 0) + 1
            self.lat_last = ev["lat_ms"]
            self.lat_sum += ev["lat_ms"]
            if ev["temp_c"] is not None:
                self.temp_last = ev["temp_c"]
                self.temp_max = ev["temp_c"] if self.temp_max is None else max(self.temp_max, ev["temp_c"])
            self.big.configure(bg=COLORS.get(ev["label"], "#455a64"))
            self.big_var.set(ev["label"].replace("_", " ").upper())
            self.sub_var.set(ev["frame"])
            self._logline(ev["frame"] + "  lat=%.1fms temp=%s" % (ev["lat_ms"], ev["temp_c"]))
        elif t == "abort":
            self.conn_var.set("connection: ABORTED -- " + ev["msg"])
            self._logline("[ABORT] " + ev["msg"])
        elif t == "summary":
            self._logline("[summary] executed=%d aborted=%s throughput=%.2f temp_max=%s"
                          % (ev["executed"], ev["aborted"], ev["throughput"], ev["temp_max"]))
        self._refresh()

    def _refresh(self):
        mean_lat = (self.lat_sum / self.executed) if self.executed else 0.0
        elapsed = (time.time() - self._t0) if self._t0 else 0.0
        thr = (self.executed / elapsed) if elapsed > 0 else 0.0
        self.metric_var.set(
            "executed={ex}/{tgt}   errors={err}{drop}   latency last={ll}  mean={lm:.1f} ms\n"
            "throughput={thr:.1f} inf/s   board temp last={tl}  max={tm}  (stop>=75 C)".format(
                ex=self.executed, tgt=self.n_target, err=self.errors,
                drop=("   dropped(display backlog)=%d" % self.dropped) if self.dropped else "",
                ll=("%.1f" % self.lat_last) if self.lat_last is not None else "--",
                lm=mean_lat, thr=thr,
                tl=("%.2f" % self.temp_last) if self.temp_last is not None else "--",
                tm=("%.2f" % self.temp_max) if self.temp_max is not None else "--"))
        self.byclass_var.set("predicted:  " + "   ".join(
            "{}={}".format(LABELS[c], self.by_class.get(c, 0)) for c in range(len(LABELS))))

    def _logline(self, text):
        self.log.configure(state="normal")
        self.log.insert("1.0", text + "\n")
        self.log.configure(state="disabled")

    def _logclear(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def build_dataset(result_dir: str):
    ds = ReplayDataset.from_dir(result_dir)
    ref_path = os.path.join(result_dir, "fpga_fulltest_metrics.json")
    reference = load_meta(ref_path) if os.path.isfile(ref_path) else {}
    return ds, reference


def main(argv=None):
    ap = argparse.ArgumentParser(description="FPGA fault-classification result viewer.")
    ap.add_argument("--mode", choices=["both", "replay", "live"], default="both",
                    help="which tab(s) to open (default both)")
    ap.add_argument("--result-dir", help="fpga_fulltest_* dir (replay source)")
    ap.add_argument("--auto", action="store_true", help="start replay automatically")
    ap.add_argument("--screenshot", help="render replay tab, save a PNG here, exit (smoke test)")
    args = ap.parse_args(argv)

    import tkinter as tk
    from tkinter import ttk

    rdir = args.result_dir or default_result_dir()
    ds = reference = None
    if args.mode in ("both", "replay"):
        if not rdir or not os.path.isdir(rdir):
            raise SystemExit("no replay result dir found; pass --result-dir <fpga_fulltest_*>")
        try:
            ds, reference = build_dataset(rdir)
        except DataError as e:
            raise SystemExit("failed to load replay data: {}".format(e))

    root = tk.Tk()
    root.title("FPGA Fault-Classification -- Replay + Live FPGA")
    root.geometry("940x680")
    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)

    app = None
    if args.mode in ("both", "replay"):
        tab = ttk.Frame(nb)
        nb.add(tab, text="Replay Mode (recorded full test)")
        app = ViewerApp(root, ds, reference, container=tab)
        app.status_var.set("loaded {} windows from {}".format(ds.n, rdir))
    live = None
    if args.mode in ("both", "live"):
        ltab = ttk.Frame(nb)
        nb.add(ltab, text="Live FPGA Mode (real on-board inference)")
        live = LiveFrame(ltab, root)

    def on_close():
        # Never leave the board running: stop the on-board runner and drain the
        # SSH worker before destroying the window.
        if live is not None:
            try:
                live.conn_var.set("closing: stopping on-board runner...")
                root.update_idletasks()
            except Exception:
                pass
            live.shutdown(join_timeout=5.0)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    if args.screenshot and app is not None:
        _screenshot_and_exit(root, app, args.screenshot)
        return
    if args.auto and app is not None:
        root.after(300, app.start)
    root.mainloop()


def _screenshot_and_exit(root, app, path):
    """Render a representative state, grab the window to PNG, then quit."""
    # populate some records + a blocking fast-eval, summarised into the status bar
    app._set_index(min(42, app.ds.n - 1), incremental=False)
    res, diffs = app.run_fast_eval_blocking()
    app.status_var.set(
        "FAST EVAL: acc={:.6f} macro_f1={:.6f}  {}".format(
            res["accuracy"], res["macro_f1"],
            "MATCHES fpga_fulltest_metrics.json" if (app.reference and not diffs)
            else ("{} discrepancies".format(len(diffs)) if app.reference else "no reference")))
    root.update_idletasks()
    root.update()

    def grab():
        from PIL import ImageGrab
        root.update_idletasks(); root.update()
        x, y = root.winfo_rootx(), root.winfo_rooty()
        w, h = root.winfo_width(), root.winfo_height()
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        img.save(path)
        print("saved screenshot:", path)
        root.destroy()

    root.after(600, grab)
    root.mainloop()


if __name__ == "__main__":
    main()
