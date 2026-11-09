"""tools/board/fpga_rt_acceptance.py control logic with a fake driver (no board).

Checks: every window processed exactly once in release order (no skipping),
frame integrity (the fake core derives logits from the whole frame), late
windows counted as misses (never dropped), DMA timeout and SIGTERM abort with
partial evidence, and the SCHED_FIFO/affinity path where the OS supports it
(Linux as root, e.g. on the board). Validates logic only, not physical timing.
"""
import gzip
import importlib.util
import io
import json
import os
import signal
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout

import numpy as np

SCRIPT = os.path.join(os.path.dirname(__file__), os.pardir, "tools", "board", "fpga_rt_acceptance.py")
REC = 1024 * 8 * 4


def _fake_driver(delay_s=0.0, hang_at=None):
    m = types.ModuleType("fault_classifier")
    import time as _t

    class Chan:
        def __init__(self, be, name):
            self.be, self.name = be, name

        def transfer(self, buf):
            if self.name == "send":
                self.be.frames.append(np.array(buf, dtype=np.int64))
            self.buf = buf

        def wait(self):
            if self.name == "recv":
                if hang_at is not None and len(self.be.frames) - 1 == hang_at:
                    raise self.be.hang_exc
                if delay_s:
                    _t.sleep(delay_s)
                x = self.be.frames[-1]
                self.buf[:] = [x[0] % 5, x[1] % 5, x[2] % 5, int(x.sum()) % 5]

    class HardwareBackend:
        def __init__(self, bit, wait_timeout=None):
            self.frames = []
            self.dma = types.SimpleNamespace(sendchannel=Chan(self, "send"), recvchannel=Chan(self, "recv"))
            self._in = np.zeros(8192, np.int32)
            self._out = np.zeros(4, np.int32)
            self.hang_exc = RuntimeError("hang")

        def _wait(self, ch):
            ch.wait()

    class FaultClassifier:
        class_source = "fpga_register"

        def __init__(self, backend):
            self.backend = backend

        def _decide(self, lg):
            return int(np.argmax(lg))

    m.HardwareBackend, m.FaultClassifier = HardwareBackend, FaultClassifier
    return m


def _load(driver):
    sys.modules["fault_classifier"] = driver
    try:
        import resource  # noqa: F401  (Unix; present on the board)
    except ImportError:   # Windows test host: counters are not under test here
        r = types.ModuleType("resource")
        r.RUSAGE_SELF = 0
        r.getrusage = lambda who: types.SimpleNamespace(ru_maxrss=0, ru_nivcsw=0, ru_nvcsw=0, ru_majflt=0)
        sys.modules["resource"] = r
    if not hasattr(signal, "SIGALRM"):
        signal.SIGALRM = signal.SIGINT
        signal.alarm = lambda s: 0
    spec = importlib.util.spec_from_file_location("rt_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RtAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.data = np.random.RandomState(3).randint(-511, 512, size=(12, 8192)).astype("<i4")
        cls.gz = os.path.join(cls.tmp, "in.i32.gz")
        with gzip.open(cls.gz, "wb") as f:
            f.write(cls.data.tobytes())
        for n, v in (("raw", "2600"), ("offset", "-2219"), ("scale", "123.04")):
            open(os.path.join(cls.tmp, "in_temp0_" + n), "w").write(v)

    def run_main(self, mod, args, tag):
        mod.XADC = self.tmp
        out = os.path.join(self.tmp, tag)
        old = sys.argv
        sys.argv = ["x", "bit", self.gz] + [str(a) for a in args] + [out]
        buf = io.StringIO()
        code = 0
        try:
            with redirect_stdout(buf):
                mod.main()
        except SystemExit as e:
            code = e.code
        finally:
            sys.argv = old
        summ = json.load(open(os.path.join(out, "summary.json")))
        raw = np.load(os.path.join(out, "rt_raw.npz"))
        return code, summ, raw, buf.getvalue()

    def test_every_window_once_in_order_with_frame_integrity(self):
        mod = _load(_fake_driver())
        mod.P = 0.004                                    # fast clock for the test
        code, s, raw, _ = self.run_main(mod, [5, 23, 3, 0, 1, 50], "ok")
        self.assertEqual((code, s["executed"], s["aborted"]), (0, 23, None))
        self.assertEqual(list(raw["window_index"]), [k % 5 for k in range(23)])
        for k in range(23):
            x = self.data[k % 5].astype(np.int64)
            self.assertEqual(list(raw["logits"][k]), [x[0] % 5, x[1] % 5, x[2] % 5, int(x.sum()) % 5])
        rel = raw["release"]
        self.assertTrue(np.all(np.diff(rel) > 0))
        self.assertTrue(np.all(raw["start"] >= rel - 1e-9))             # never before its release
        self.assertEqual(s["acceptance_windows"], 20)
        self.assertEqual(s["class_source"], "fpga_register")

    def test_late_windows_are_counted_not_skipped(self):
        mod = _load(_fake_driver(delay_s=0.006))         # core slower than the period
        mod.P = 0.004
        code, s, raw, _ = self.run_main(mod, [5, 15, 2, 0, 1, 50], "late")
        self.assertEqual(s["executed"], 15)
        self.assertEqual(s["deadline_misses"], 13)
        resp = raw["done"] - raw["release"]
        self.assertTrue(resp[-1] > resp[2])              # backlog grows, visibly

    def test_dma_timeout_aborts_with_partial_evidence(self):
        drv = _fake_driver(hang_at=4)
        mod = _load(drv)
        mod.P = 0.004
        orig = drv.HardwareBackend.__init__

        def init(self, bit, wait_timeout=None):
            orig(self, bit, wait_timeout)
            self.hang_exc = mod.Abort("DMA timeout")
        drv.HardwareBackend.__init__ = init
        code, s, raw, _ = self.run_main(mod, [5, 20, 2, 0, 1, 50], "timeout")
        self.assertEqual(code, 7)
        self.assertEqual(s["executed"], 4)
        self.assertIn("DMA timeout", s["aborted"])

    def test_sigterm_aborts_cleanly(self):
        drv = _fake_driver(hang_at=6)
        mod = _load(drv)
        mod.P = 0.004
        orig = drv.HardwareBackend.__init__

        def init(self, bit, wait_timeout=None):
            orig(self, bit, wait_timeout)
            self.hang_exc = mod.Abort("SIGTERM (guard or operator)")
        drv.HardwareBackend.__init__ = init
        code, s, _, _ = self.run_main(mod, [5, 20, 2, 0, 1, 50], "term")
        self.assertEqual((code, s["executed"]), (7, 6))
        self.assertIn("SIGTERM", s["aborted"])

    @unittest.skipUnless(hasattr(os, "sched_setscheduler") and hasattr(os, "geteuid") and os.geteuid() == 0,
                         "SCHED_FIFO needs Linux and root (runs on the board)")
    def test_rt_path_sets_fifo_and_affinity(self):
        mod = _load(_fake_driver())
        mod.P = 0.004
        code, s, _, _ = self.run_main(mod, [5, 10, 2, 1, 0, 10], "rt")
        self.assertEqual(code, 0)
        self.assertIn("SCHED_FIFO prio 10", s["sched"])
        self.assertIn("[0]", s["sched"])


if __name__ == "__main__":
    unittest.main()
