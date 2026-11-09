"""Board runner input buffer (RecordSource / Prefetcher / main loop) without a board.

A fake `fault_classifier` and fake XADC are injected; the DMA "core" returns
logits derived from the frame, so ordering and frame integrity are checked.
Validates control logic only -- not physical DMA or timing.
"""
import gzip
import importlib.util
import io
import os
import signal
import sys
import tempfile
import threading
import time
import types
import unittest
from contextlib import redirect_stdout

import numpy as np

RUNNER = os.path.join(os.path.dirname(__file__), os.pardir, "tools", "board", "fpga_live_runner.py")
REC = 1024 * 8 * 4


def _fake_driver():
    m = types.ModuleType("fault_classifier")

    class Chan:
        def __init__(self, be, name):
            self.be, self.name = be, name

        def transfer(self, buf):
            if self.name == "send":
                # the whole 8192-word frame must be present at submit time
                self.be.frames.append(np.array(buf, dtype=np.int64))
            self.buf = buf

        def wait(self):
            if self.name == "recv":
                x = self.be.frames[-1]
                self.buf[:] = [x[0] % 7, x[1] % 7, x[2] % 7, int(x.sum()) % 7]

    class HardwareBackend:
        def __init__(self, bit, wait_timeout=None):
            self.frames = []
            self.dma = types.SimpleNamespace(sendchannel=Chan(self, "send"),
                                             recvchannel=Chan(self, "recv"))
            self._in = np.zeros(8192, np.int32)
            self._out = np.zeros(4, np.int32)

        def _wait(self, ch):
            ch.wait()

    class FaultClassifier:
        def __init__(self, backend):
            self.backend = backend

    m.HardwareBackend, m.FaultClassifier = HardwareBackend, FaultClassifier
    return m


HAS_FORK = "fork" in __import__("multiprocessing").get_all_start_methods()


def _load(env):
    for k in ("FPGA_RUNNER_PREFETCH", "FPGA_RUNNER_PACE_HZ", "FPGA_RUNNER_TIMING",
              "FPGA_RUNNER_PREFETCH_KIND", "FPGA_RUNNER_READAHEAD_MB"):
        os.environ.pop(k, None)
    env = dict(env)
    # the process reader needs fork (Linux / the board); elsewhere test the
    # same queue logic with the thread reader
    env.setdefault("FPGA_RUNNER_PREFETCH_KIND", "process" if HAS_FORK else "thread")
    os.environ.update(env)
    sys.modules["fault_classifier"] = _fake_driver()
    if not hasattr(signal, "SIGALRM"):
        signal.SIGALRM = signal.SIGINT
        signal.alarm = lambda s: 0
    spec = importlib.util.spec_from_file_location("runner_under_test", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        rng = np.random.RandomState(1)       # numpy 1.13 on the board
        cls.data = rng.randint(-511, 512, size=(40, 8192)).astype("<i4")
        cls.gz = os.path.join(cls.tmp, "in.i32.gz")
        with gzip.open(cls.gz, "wb") as f:
            f.write(cls.data.tobytes())
        for name, val in (("raw", "2600"), ("offset", "-2219"), ("scale", "123.04")):
            open(os.path.join(cls.tmp, "in_temp0_" + name), "w").write(val)

    def run_main(self, args, env=None):
        mod = _load(env or {})
        mod.XADC = self.tmp
        out = io.StringIO()
        old = sys.argv
        sys.argv = ["runner"] + args
        code = 0
        try:
            with redirect_stdout(out):
                mod.main()
        except SystemExit as e:
            code = e.code
        finally:
            sys.argv = old
        lines = out.getvalue().splitlines()
        res = [l for l in lines if l.startswith("R ")]
        summ = [l for l in lines if l.startswith("SUMMARY")]
        return res, summ[0] if summ else "", code, mod


def _strip(lines):
    import re
    return [re.sub(r"lat_ms=[0-9.]+ ", "", l) for l in lines]


class RunnerPrefetchTests(Base):
    def test_default_is_original_loop(self):
        a, sa, ca, _ = self.run_main(["bit", self.gz, "1024", "8", "5"])
        self.assertIn("prefetch=0 prefetch_kind=none", sa)
        self.assertIn("readahead_mb=0", sa)

    def test_prefetch_equals_legacy_sequential(self):
        a, sa, ca, _ = self.run_main(["bit", self.gz, "1024", "8", "40"], {"FPGA_RUNNER_PREFETCH": "128"})
        b, sb, cb, _ = self.run_main(["bit", self.gz, "1024", "8", "40"], {"FPGA_RUNNER_PREFETCH": "0"})
        self.assertEqual(len(a), 40)
        self.assertEqual(_strip(a), _strip(b))
        self.assertEqual([int(l.split()[1][4:]) for l in a], list(range(40)))
        self.assertIn("prefetch=128", sa)
        self.assertIn("prefetch=0", sb)
        self.assertEqual((ca, cb), (0, 0))

    def test_small_queue_preserves_order_and_frames(self):
        res, summ, code, mod = self.run_main(["bit", self.gz, "1024", "8", "40"],
                                             {"FPGA_RUNNER_PREFETCH": "2"})
        self.assertEqual(code, 0)
        self.assertEqual([int(l.split()[1][4:]) for l in res], list(range(40)))
        # every logit line reflects the exact frame that was sent, in order
        for i, l in enumerate(res):
            lg = [int(v) for v in l.split("logits=")[1].split()[0].split(",")]
            x = self.data[i].astype(np.int64)
            self.assertEqual(lg, [x[0] % 7, x[1] % 7, x[2] % 7, int(x.sum()) % 7])

    def test_indices_mode_equal_and_ordered(self):
        idx = os.path.join(self.tmp, "idx.txt")
        open(idx, "w").write("3\n7\n8\n20\n39\n")
        a, _, _, _ = self.run_main(["bit", self.gz, "1024", "8", "0", "-", idx], {"FPGA_RUNNER_PREFETCH": "128"})
        b, _, _, _ = self.run_main(["bit", self.gz, "1024", "8", "0", "-", idx], {"FPGA_RUNNER_PREFETCH": "0"})
        self.assertEqual(_strip(a), _strip(b))
        self.assertEqual([int(l.split()[1][4:]) for l in a], [3, 7, 8, 20, 39])

    def test_input_exhausted_abort_message_unchanged(self):
        a, sa, ca, _ = self.run_main(["bit", self.gz, "1024", "8", "45"], {"FPGA_RUNNER_PREFETCH": "128"})
        b, sb, cb, _ = self.run_main(["bit", self.gz, "1024", "8", "45"], {"FPGA_RUNNER_PREFETCH": "0"})
        self.assertEqual(len(a), 40)
        self.assertIn("aborted=input exhausted at window 40 (requested 45)", sa)
        self.assertIn("aborted=input exhausted at window 40 (requested 45)", sb)
        self.assertEqual((ca, cb), (7, 7))

    def test_timing_fields_only_when_enabled(self):
        a, _, _, _ = self.run_main(["bit", self.gz, "1024", "8", "3"])
        self.assertNotIn("q=", a[0])
        b, s, _, _ = self.run_main(["bit", self.gz, "1024", "8", "3"],
                                   {"FPGA_RUNNER_TIMING": "1", "FPGA_RUNNER_PACE_HZ": "200",
                                    "FPGA_RUNNER_PREFETCH": "8"})
        self.assertIn(" q=", b[0])
        self.assertIn(" resp_ms=", b[0])
        self.assertIn("pace_hz=200", s)


@unittest.skipUnless(HAS_FORK, "process reader needs fork (runs on the board)")
class ProcessReaderTests(Base):
    def test_process_reader_equals_legacy_and_reports_kind(self):
        a, sa, ca, _ = self.run_main(["bit", self.gz, "1024", "8", "40"],
                                     {"FPGA_RUNNER_PREFETCH_KIND": "process", "FPGA_RUNNER_PREFETCH": "4"})
        b, _, _, _ = self.run_main(["bit", self.gz, "1024", "8", "40"], {"FPGA_RUNNER_PREFETCH": "0"})
        self.assertEqual(_strip(a), _strip(b))
        self.assertIn("prefetch_kind=process", sa)
        self.assertEqual(ca, 0)

    def test_process_reader_abort_and_indices(self):
        a, sa, ca, _ = self.run_main(["bit", self.gz, "1024", "8", "45"],
                                     {"FPGA_RUNNER_PREFETCH_KIND": "process", "FPGA_RUNNER_PREFETCH": "8"})
        self.assertEqual(len(a), 40)
        self.assertIn("aborted=input exhausted at window 40 (requested 45)", sa)
        idx = os.path.join(self.tmp, "idx_p.txt")
        open(idx, "w").write("3 7 39")
        b, _, _, _ = self.run_main(["bit", self.gz, "1024", "8", "0", "-", idx],
                                   {"FPGA_RUNNER_PREFETCH_KIND": "process", "FPGA_RUNNER_PREFETCH": "8"})
        self.assertEqual([int(l.split()[1][4:]) for l in b], [3, 7, 39])

    def test_reader_exits_when_parent_is_gone(self):
        import multiprocessing as mp
        mod = _load({})
        q, stop = mp.Queue(maxsize=1), mp.Event()
        p = mp.Process(target=mod._reader_process,
                       args=(self.gz, REC, 40, None, 39, q, stop, -12345))  # wrong parent pid
        p.start()
        p.join(5)
        self.assertFalse(p.is_alive())


class PrefetcherUnitTests(Base):
    def setUp(self):
        self.mod = _load({})

    def _source(self, n, want=None):
        f = gzip.open(self.gz, "rb")
        self.addCleanup(f.close)
        return self.mod.RecordSource(f, REC, n, want, max(want) if want else n - 1)

    def test_bounded_queue_blocks_and_stop_exits(self):
        pf = self.mod.Prefetcher(self._source(40), 40, 3)
        pf.start()
        time.sleep(1.2)                   # consumer absent: producer must block, not grow
        self.assertEqual(pf.q.qsize(), 3)
        self.assertGreater(pf.full_blocks, 0)
        pf.stop.set()
        pf.join(2.0)
        self.assertFalse(pf.is_alive())

    def test_producer_error_reaches_consumer(self):
        class Boom:
            aborted = None

            def next(self, pos):
                raise IOError("disk gone")
        pf = self.mod.Prefetcher(Boom(), 5, 2)
        pf.start()
        item = pf.q.get(timeout=2)
        self.assertEqual(item[0], "END")
        self.assertIn("disk gone", item[1])

    def test_past_last_index_ends_with_reason(self):
        pf = self.mod.Prefetcher(self._source(2, {3, 50}), 2, 4)
        pf.start()
        first = pf.q.get(timeout=2)
        self.assertEqual(first[0], 3)
        self.assertEqual(len(first[1]), 8192)       # one whole frame per item
        end = pf.q.get(timeout=2)
        self.assertEqual(end[0], "END")
        self.assertIn("input exhausted at window 1", end[1])

    def test_consumer_times_out_on_stalled_producer(self):
        mod = _load({"FPGA_RUNNER_PREFETCH": "8"})
        mod.QUEUE_TIMEOUT_S = 0.3
        mod.XADC = self.tmp

        class Slow:
            aborted = None

            def __init__(self, *a, **k):
                pass

            def next(self, pos):
                time.sleep(2)
                return None
        mod.RecordSource = Slow
        old = sys.argv
        sys.argv = ["runner", "bit", self.gz, "1024", "8", "3"]
        out = io.StringIO()
        try:
            with redirect_stdout(out), self.assertRaises(SystemExit) as cm:
                mod.main()
        finally:
            sys.argv = old
        self.assertEqual(cm.exception.code, 7)
        self.assertIn("input producer stalled", out.getvalue())


if __name__ == "__main__":
    unittest.main()


class ReadaheadTests(Base):
    def test_readahead_does_not_change_results(self):
        a, sa, ca, _ = self.run_main(["bit", self.gz, "1024", "8", "40"],
                                     {"FPGA_RUNNER_PREFETCH": "0", "FPGA_RUNNER_READAHEAD_MB": "4"})
        b, _, _, _ = self.run_main(["bit", self.gz, "1024", "8", "40"], {"FPGA_RUNNER_PREFETCH": "0"})
        self.assertEqual(_strip(a), _strip(b))
        self.assertIn("readahead_mb=4", sa)
        self.assertEqual(ca, 0)

    def test_readahead_requests_ranges_ahead_of_reader(self):
        mod = _load({})
        calls = []
        real = getattr(os, "posix_fadvise", None)
        os.posix_fadvise = lambda fd, off, ln, adv: calls.append((off, ln))
        if not hasattr(os, "POSIX_FADV_WILLNEED"):
            os.POSIX_FADV_WILLNEED = 3
        try:
            with gzip.open(self.gz, "rb") as f:
                stop = mod.start_readahead(self.gz, f, 1, period_s=0.01)
                time.sleep(0.1)
                f.read(REC * 30)           # reader advances in the compressed file
                time.sleep(0.1)
                stop.set()
                time.sleep(0.05)
        finally:
            if real is None:
                del os.posix_fadvise
            else:
                os.posix_fadvise = real
        self.assertGreaterEqual(len(calls), 1)
        self.assertEqual(calls[0], (0, 1 << 20))
        ends = [o + l for o, l in calls]
        self.assertEqual(ends, sorted(ends))            # always moving forward
        for (o1, l1), (o2, l2) in zip(calls, calls[1:]):
            self.assertEqual(o2, o1 + l1)               # contiguous, no gaps/overlap
