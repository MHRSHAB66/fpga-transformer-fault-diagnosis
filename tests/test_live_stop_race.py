"""Stop/Close at every point of the Live lifecycle must never leave (or start)
an unprotected or orphaned workload. No board: subprocess is mocked.

These validate client/GUI control logic only, not the physical board.
"""
import threading
import time
import unittest
from unittest import mock

from host import live_fpga

RUNNER = "/home/xilinx/live_runs/abc123/fpga_live_runner.py"
GUARD = "/home/xilinx/live_runs/abc123/temp_guard.sh"
OK_CHECKS = [{"check": "ok", "ok": True, "detail": "ok"}]


class FakeProc:
    """Popen stand-in: emits lines until killed; records wait/terminate."""

    def __init__(self, lines, block=True):
        self._lines = list(lines)
        self._killed = threading.Event()
        self._block = block
        self.terminated = False
        self.waited = False
        self.rc = None
        self.stdout = self._iter()

    def _iter(self):
        for ln in self._lines:
            yield ln + "\n"
        if self._block:
            self._killed.wait(5)
        self.rc = 0

    def kill_remote(self):
        self._killed.set()

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        self.waited = True
        self._killed.wait(timeout)
        if self._killed.is_set():
            self.rc = 0
        return self.rc

    def terminate(self):
        self.terminated = True
        self._killed.set()
        self.rc = -15


def _client():
    return live_fpga.LiveFpgaClient(n=10, runner_remote=RUNNER, monitor_remote=GUARD,
                                    line_timeout=2.0)


@mock.patch("host.live_preflight.evaluate", return_value=OK_CHECKS)
@mock.patch("host.live_preflight.run_probe", return_value={})
@mock.patch.object(live_fpga.LiveFpgaClient, "check_connection", return_value=True)
class ClientStopTests(unittest.TestCase):
    def test_stop_before_stream_prevents_launch(self, conn, probe, evaluate):
        c = _client()
        c.stop()
        with mock.patch("host.live_fpga.subprocess.Popen") as popen:
            events = list(c.stream())
        popen.assert_not_called()
        probe.assert_not_called()
        self.assertIn("nothing started", events[-1]["msg"])

    def test_stop_during_preflight_prevents_launch(self, conn, probe, evaluate):
        c = _client()

        def stop_then_ok(*a, **k):
            c.stop()          # Stop pressed while preflight is running
            return OK_CHECKS
        evaluate.side_effect = stop_then_ok
        with mock.patch("host.live_fpga.subprocess.Popen") as popen:
            events = list(c.stream())
        popen.assert_not_called()
        self.assertIn("nothing started", events[-1]["msg"])

    def test_stop_during_run_kills_this_run_and_reaps(self, conn, probe, evaluate):
        c = _client()
        proc = FakeProc(["TEMP_START=48.0", "META n=10 seq=1024 ch=8",
                         "R idx=0 cls=3 lat_ms=16.0 temp_c=48.0 logits=-116,-129,-36,112"])
        kills = []

        def fake_run(argv, **kw):
            kills.append(argv[-1])
            proc.kill_remote()
            return mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch("host.live_fpga.subprocess.Popen", return_value=proc), \
                mock.patch("host.live_fpga.subprocess.run", side_effect=fake_run):
            got = []
            for ev in c.stream():
                got.append(ev)
                if ev["type"] == "result":
                    c.stop()
        self.assertEqual(len(kills), 1, kills)
        self.assertIn("^python3[.]6 -u /home/xilinx/live_runs/abc123/[f]pga_live_runner[.]py", kills[0])
        self.assertTrue(proc.waited)
        self.assertFalse(proc.terminated)          # wrapper exited by itself
        self.assertEqual(sum(e["type"] == "result" for e in got), 1)

    def test_line_timeout_ends_remote_workload(self, conn, probe, evaluate):
        c = _client()
        c.line_timeout = 0.3
        proc = FakeProc(["TEMP_START=48.0"])
        kills = []

        def fake_run(argv, **kw):
            kills.append(argv[-1])
            proc.kill_remote()
            return mock.Mock(returncode=0)
        with mock.patch("host.live_fpga.subprocess.Popen", return_value=proc), \
                mock.patch("host.live_fpga.subprocess.run", side_effect=fake_run):
            events = list(c.stream())
        self.assertTrue(any(e["type"] == "error" and "timeout" in e["msg"] for e in events))
        self.assertEqual(len(kills), 1)

    def test_unkillable_wrapper_is_terminated(self, conn, probe, evaluate):
        c = _client()
        proc = FakeProc(["META n=10 seq=1024 ch=8"])
        with mock.patch("host.live_fpga.subprocess.Popen", return_value=proc), \
                mock.patch("host.live_fpga.subprocess.run", return_value=mock.Mock(returncode=0)):
            it = c.stream()
            next(it)                         # status "connected ..." (before launch)
            self.assertEqual(next(it)["type"], "meta")   # launched
            c._end_remote(grace_s=0.1)       # remote kill had no effect
        self.assertTrue(proc.terminated)

    def test_kill_pattern_is_run_scoped(self, conn, probe, evaluate):
        import re
        pat = _client()._kill_pattern()
        self.assertEqual(pat, "^python3[.]6 -u /home/xilinx/live_runs/abc123/[f]pga_live_runner[.]py")
        runner = "python3.6 -u /home/xilinx/live_runs/abc123/fpga_live_runner.py bit in 1024 8 40"
        wrapper = "bash -c cd x || exit 1; PYTHONPATH=y python3.6 -u /home/xilinx/live_runs/abc123/fpga_live_runner.py bit"
        other = "python3.6 -u /home/xilinx/live_runs/zzz999/fpga_live_runner.py bit"
        self.assertIsNotNone(re.search(pat, runner))
        self.assertIsNone(re.search(pat, wrapper))
        self.assertIsNone(re.search(pat, other))


try:
    import tkinter as tk
    from tkinter import ttk
    _r = tk.Tk(); _r.withdraw(); _r.destroy()
    _TK = True
except Exception:
    _TK = False


@unittest.skipUnless(_TK, "Tkinter/display not available")
class GuiStopRaceTests(unittest.TestCase):
    def setUp(self):
        from host.fpga_result_viewer import LiveFrame, LIVE_CONF_NOTE
        self.note = LIVE_CONF_NOTE
        self.root = tk.Tk(); self.root.withdraw()
        self.frame = LiveFrame(ttk.Frame(self.root), self.root)
        self.frame.n_var.set("4")

    def tearDown(self):
        self.frame.shutdown(join_timeout=2)
        self.root.destroy()

    def test_stop_while_pushing_never_launches(self):
        gate = threading.Event()

        def slow_push(*a):
            gate.wait(2)
            return RUNNER, GUARD
        with mock.patch("host.live_fpga.push_board_files", side_effect=slow_push), \
                mock.patch("host.live_fpga.subprocess.Popen") as popen, \
                mock.patch.object(live_fpga.LiveFpgaClient, "check_connection",
                                  return_value=True) as conn:
            self.frame.start()
            time.sleep(0.1)
            self.frame.stop()                 # Stop while the push is in flight
            gate.set()
            self.frame.thread.join(3)
        popen.assert_not_called()
        conn.assert_not_called()

    def test_close_while_pushing_never_launches(self):
        gate = threading.Event()
        with mock.patch("host.live_fpga.push_board_files",
                        side_effect=lambda *a: (gate.wait(2), (RUNNER, GUARD))[1]), \
                mock.patch("host.live_fpga.subprocess.Popen") as popen:
            self.frame.start()
            time.sleep(0.1)
            threading.Timer(0.2, gate.set).start()
            self.assertTrue(self.frame.shutdown(join_timeout=3))
        popen.assert_not_called()

    def test_confidence_note_is_explicit(self):
        self.assertIn("CONF=NA", self.note)
        self.assertIn("decided on the FPGA", self.note)
        # either explicitly uncalibrated, or explicitly calibrated with T and the caveat
        self.assertTrue(("UNCALIBRATED" in self.note and "NOT a guaranteed probability" in self.note)
                        or ("CALIBRATED" in self.note and "T=" in self.note
                            and "not a guarantee" in self.note), self.note)


if __name__ == "__main__":
    unittest.main()
