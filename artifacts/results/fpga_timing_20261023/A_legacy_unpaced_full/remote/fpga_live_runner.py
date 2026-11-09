#!/usr/bin/env python3
"""Live FPGA inference runner (runs ON the MYIR Z-Turn board, Python 3.6).

Streams the FIRST N test-set windows through the REAL transformer_top core via
AXI-DMA and prints one result line per window as it completes, so a PC-side UI
can display live FPGA output over SSH/Ethernet. Classification happens 100% on
the FPGA; this script only feeds windows and reports what the DMA returned.

Each result line (ASCII, flushed immediately):

    R idx=<i> cls=<c> lat_ms=<f> temp_c=<t|NA> logits=<l0>,<l1>,<l2>,<l3>

Framing/contract note: the canonical Stage-10 wire frame
``FRAME=<id>,CLASS=<c>,LABEL=<label>,CONF=NA`` is built on the PC from these
class ids via host.frame_protocol.format_frame (single source of truth); the
board emits the compact form plus per-window latency and board temperature.

Temperature safety -- the watchdog runs in its OWN thread (TempMonitor), fully
independent of the blocking DMA thread. The DMA loop only reads the cached value
that thread publishes, so a stuck DMA can never stall temperature sampling:
  * temp >= 75 C                                   -> immediate stop
  * a single invalid/None reading                  -> keep going
  * no valid temperature for >= 30 s continuously  -> stop
  * per-window DMA watchdog via SIGALRM            -> stop on a stuck transfer
Start / maximum / end temperature are reported in the final summary.

Window selection: by default the FIRST N windows are run (sequential, the
original behaviour). Passing an indices file (one 0-based dataset index per line)
runs exactly those windows instead, in that order, preserving the REAL dataset
frame id in each result line (`idx=<dataset index>`). The gzip input is read
sequentially and only the requested indices are fed to the DMA -- no seeking,
and classification still happens entirely on the FPGA.

Usage (mirrors the proven full-test launch):
    cd <DEP>/driver_py36compat
    PYTHONPATH=<DEP>/driver_py36compat python3.6 -u fpga_live_runner.py \
        <bit> <in.i32.gz> <seq> <ch> <N> [out_pred.txt] [indices_file]
"""
import os
import sys
import time
import signal
import gzip
import threading
try:
    import queue
except ImportError:  # pragma: no cover
    import Queue as queue

import numpy as np
from fault_classifier import HardwareBackend, FaultClassifier

XADC = "/sys/bus/iio/devices/iio:device0"
STOP_C = 75.0
TEMP_STALE_S = 30.0
DMA_WATCHDOG_S = 20
# Bounded input prefetch (producer thread -> queue -> DMA loop). Measured
# 2026-10-23: reads of the gzip input stall in blocking I/O for up to ~0.8 s
# while decompression itself stays < 4 ms; the queue absorbs such stalls.
# 0 = original single-thread loop. Each slot holds one 32,768-byte window.
PREFETCH_DEPTH = int(os.environ.get("FPGA_RUNNER_PREFETCH", "128"))
QUEUE_TIMEOUT_S = 30.0          # consumer gives up if the producer stalls this long
# Measurement-only options (default off): release windows on a fixed clock to
# emulate a sensor (Hz), and append per-window timing fields to result lines.
PACE_HZ = float(os.environ.get("FPGA_RUNNER_PACE_HZ", "0") or 0)
TIMING = os.environ.get("FPGA_RUNNER_TIMING", "0") == "1"


class RecordSource(object):
    """Sequential reader of the int32 window file with optional index selection.

    next(pos) -> (dataset_index, bytes) or None (reason in .aborted).
    `pos` = number of windows completed when the record will be used (keeps the
    original abort messages)."""

    def __init__(self, fin, rec_bytes, n, want_set=None, max_idx=None):
        self.fin, self.rec_bytes, self.n = fin, rec_bytes, n
        self.want_set, self.max_idx = want_set, max_idx
        self.src_idx = -1
        self.aborted = None

    def next(self, pos):
        while True:
            buf = self.fin.read(self.rec_bytes)
            if not buf:
                self.aborted = "input exhausted at window %d (requested %d)" % (pos, self.n)
                return None
            if len(buf) != self.rec_bytes:
                self.aborted = "short read %d at window %d" % (len(buf), pos)
                return None
            self.src_idx += 1
            # Skip records not in the requested index set (gz read sequentially).
            if self.want_set is not None and self.src_idx not in self.want_set:
                if self.src_idx > self.max_idx:
                    self.aborted = "input passed last requested index at %d" % self.src_idx
                    return None
                continue
            return self.src_idx, buf


class Prefetcher(threading.Thread):
    """Producer: reads up to n selected windows in order into a bounded queue.

    Items: (dataset_index, int32 array of one whole window, read_ms), or
    ("END", reason) when the input ends early / fails. Blocks when the queue
    is full (bounded memory); exits promptly when `stop` is set."""

    def __init__(self, source, n, depth):
        threading.Thread.__init__(self)
        self.daemon = True
        self.source, self.n = source, n
        self.q = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.full_blocks = 0

    def _put(self, item):
        while not self.stop.is_set():
            try:
                self.q.put(item, timeout=0.5)
                return True
            except queue.Full:
                self.full_blocks += 1
        return False

    def run(self):
        try:
            for pos in range(self.n):
                if self.stop.is_set():
                    return
                t = time.perf_counter()
                rec = self.source.next(pos)
                rd_ms = (time.perf_counter() - t) * 1000.0
                if rec is None:
                    self._put(("END", self.source.aborted))
                    return
                idx, buf = rec
                if not self._put((idx, np.frombuffer(buf, dtype=np.int32), rd_ms)):
                    return
        except Exception as exc:  # never die silently: the consumer must see it
            self._put(("END", "input producer error: %r" % (exc,)))


class TempMonitor(threading.Thread):
    """Independent die-temperature sampler (NOT in the DMA thread).

    Reads the Zynq XADC sysfs every ``interval`` seconds and publishes the most
    recent valid reading plus the time it was taken. A single bad read does not
    clear the last value; callers decide staleness from ``last_valid_monotonic``.
    """

    def __init__(self, interval=2.0):
        threading.Thread.__init__(self)
        self.daemon = True
        self.interval = interval
        self._lock = threading.Lock()
        self._temp = None
        self._last_valid = None
        self._stop = False
        try:
            self._offset = float(open(XADC + "/in_temp0_offset").read().strip())
            self._scale = float(open(XADC + "/in_temp0_scale").read().strip())
        except Exception:
            self._offset, self._scale = None, None

    def _read_once(self):
        if self._offset is None:
            return None
        try:
            raw = float(open(XADC + "/in_temp0_raw").read().strip())
        except Exception:
            return None
        return (raw + self._offset) * self._scale / 1000.0

    def run(self):
        while not self._stop:
            t = self._read_once()
            if t is not None:
                with self._lock:
                    self._temp = t
                    self._last_valid = time.monotonic()
            time.sleep(self.interval)

    def snapshot(self):
        """(temp_or_None, seconds_since_last_valid_or_None)."""
        with self._lock:
            if self._last_valid is None:
                return None, None
            return self._temp, time.monotonic() - self._last_valid

    def stop(self):
        self._stop = True


class DmaTimeout(Exception):
    pass


signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(DmaTimeout()))


def main():
    bit = sys.argv[1]
    in_path = sys.argv[2]
    seq = int(sys.argv[3])
    ch = int(sys.argv[4])
    n = int(sys.argv[5])
    out_path = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] not in ("-", "") else None
    idx_path = sys.argv[7] if len(sys.argv) > 7 else None
    rec_bytes = seq * ch * 4  # int32 words

    # Optional explicit window selection (real dataset indices, ascending-read).
    want = None            # None -> first n sequential
    if idx_path:
        with open(idx_path) as fh:
            want = [int(x) for x in fh.read().split() if x.strip() != ""]
        n = len(want)
    want_set = set(want) if want is not None else None
    max_idx = max(want) if want else (n - 1)

    mon = TempMonitor(interval=2.0)
    mon.start()

    # --- wait for a first valid temperature, then gate before any DMA ----
    start_wait = time.monotonic()
    t0, _ = mon.snapshot()
    while t0 is None and time.monotonic() - start_wait < TEMP_STALE_S:
        time.sleep(0.5)
        t0, _ = mon.snapshot()
    print("TEMP_START=%s" % (("%.3f" % t0) if t0 is not None else "None"), flush=True)
    if t0 is None:
        print("ABORT no valid temperature for %.0f s before start" % TEMP_STALE_S, flush=True)
        sys.exit(5)
    if t0 >= STOP_C:
        print("ABORT temperature %.3f C >= %.1f C" % (t0, STOP_C), flush=True)
        sys.exit(5)

    clf = FaultClassifier(backend=HardwareBackend(bit, wait_timeout=10.0))
    print("META n=%d seq=%d ch=%d" % (n, seq, ch), flush=True)

    maxt = t0
    aborted = None
    done = 0
    t_start = time.time()
    fout = open(out_path, "w") if out_path else None

    # Pipelining (host-side only): after the single 32,768-byte MM2S frame has
    # been accepted, the core holds the whole window internally, so the next
    # window is read/decompressed from gzip WHILE the core computes, instead of
    # after it. Same bitstream, same frame, same order, same class decision.
    # Uses the deployed driver's transport (dma/_in/_out/_wait); if a driver
    # without them is imported, the original blocking path is used.
    # Overlap is used only for contiguous runs: with an index file the next
    # read may gzip-skip thousands of records, which would then be counted in
    # lat_ms; there the read happens after the result, as before.
    be = clf.backend
    pipelined = all(hasattr(be, a) for a in ("dma", "_in", "_out", "_wait"))
    overlap = pipelined and want_set is None
    use_prefetch = pipelined and PREFETCH_DEPTH > 0
    decide = getattr(clf, "_decide", None) or (lambda lg: int(np.argmax(lg)))
    stats = {"underruns": 0, "qmin": None, "late_release": 0}

    t_perf0 = time.perf_counter()

    def emit(src_idx, cls, logits, lat_ms, tc, extra=""):
        if TIMING:   # emission time since start: same meaning in both loops
            extra += " t_ms=%.3f" % ((time.perf_counter() - t_perf0) * 1000.0)
        # logits= : the 4 raw INT10 logits returned by the core (extra
        # trailing field; older PC parsers ignore it). The PC derives an
        # optional confidence from them; the class is the FPGA-path result.
        print("R idx=%d cls=%d lat_ms=%.3f temp_c=%s logits=%s%s"
              % (src_idx, int(cls), lat_ms, ("%.3f" % tc) if tc is not None else "NA",
                 ",".join(str(int(v)) for v in logits), extra),
              flush=True)
        if fout is not None:
            fout.write("%d\n" % int(cls))
            fout.flush()

    def dma_infer():
        """One intact 32,768-byte MM2S frame (already in be._in) -> class."""
        signal.alarm(DMA_WATCHDOG_S)
        be.dma.recvchannel.transfer(be._out)    # arm S2MM first
        be.dma.sendchannel.transfer(be._in)     # one 32,768-byte MM2S frame
        be._wait(be.dma.sendchannel)
        be._wait(be.dma.recvchannel)
        signal.alarm(0)
        logits = np.array(be._out, dtype=np.int64)
        return decide(logits), logits

    with gzip.open(in_path, "rb") as fin:
        src = RecordSource(fin, rec_bytes, n, want_set, max_idx)
        if use_prefetch:
            # Producer/consumer: the reader thread keeps up to PREFETCH_DEPTH
            # windows ready, in file order; this loop only does DMA + decision.
            # lat_ms = submit -> result collected (DMA + core + decision only).
            pf = Prefetcher(src, n, PREFETCH_DEPTH)
            pf.start()
            t_run0 = time.perf_counter()
            try:
                while done < n:
                    t_get = time.perf_counter()
                    depth = pf.q.qsize()
                    if depth == 0:
                        stats["underruns"] += 1
                    try:
                        item = pf.q.get(timeout=QUEUE_TIMEOUT_S)
                    except queue.Empty:
                        aborted = "input producer stalled > %.0f s at window %d" % (QUEUE_TIMEOUT_S, done)
                        break
                    wq_ms = (time.perf_counter() - t_get) * 1000.0
                    if item[0] == "END":
                        aborted = item[1]
                        break
                    src_idx, arr, rd_ms = item
                    stats["qmin"] = depth if stats["qmin"] is None else min(stats["qmin"], depth)
                    release = None
                    if PACE_HZ > 0:
                        release = t_run0 + done / PACE_HZ
                        now = time.perf_counter()
                        if now < release:
                            time.sleep(release - now)
                        else:
                            stats["late_release"] += 1
                    be._in[:] = arr
                    w0 = time.perf_counter()
                    try:
                        cls, logits = dma_infer()
                    except DmaTimeout:
                        aborted = "DMA timeout at window %d" % done
                        break
                    finally:
                        signal.alarm(0)
                    t_done = time.perf_counter()
                    lat_ms = (t_done - w0) * 1000.0
                    tc, stale = mon.snapshot()
                    if tc is not None and tc > maxt:
                        maxt = tc
                    extra = ""
                    if TIMING:
                        extra = " q=%d wq_ms=%.3f rd_ms=%.3f" % (depth, wq_ms, rd_ms)
                        if release is not None:
                            extra += " resp_ms=%.3f" % ((t_done - release) * 1000.0)
                    emit(src_idx, cls, logits, lat_ms, tc, extra)
                    done += 1
                    if tc is not None and tc >= STOP_C:
                        aborted = "temperature %.3f C >= %.1f C at window %d" % (tc, STOP_C, done)
                        break
                    if stale is not None and stale >= TEMP_STALE_S:
                        aborted = "no valid temperature for %.1f s at window %d" % (stale, done)
                        break
            finally:
                pf.stop.set()
                pf.join(timeout=2.0)
        else:
            # Original loop (no prefetch thread). With the overlap, the next
            # window is read while the core computes (contiguous runs only).
            nxt = src.next(0)
            while done < n and nxt is not None:
                src_idx, buf = nxt
                nxt = None
                w0 = time.time()
                try:
                    if pipelined:
                        be._in[:] = np.frombuffer(buf, dtype=np.int32)
                        signal.alarm(DMA_WATCHDOG_S)
                        be.dma.recvchannel.transfer(be._out)    # arm S2MM first
                        be.dma.sendchannel.transfer(be._in)     # one 32,768-byte MM2S frame
                        be._wait(be.dma.sendchannel)
                        signal.alarm(0)
                        if overlap and done + 1 < n:
                            nxt = src.next(done + 1)             # overlaps core compute
                        signal.alarm(DMA_WATCHDOG_S)
                        be._wait(be.dma.recvchannel)
                        signal.alarm(0)
                        logits = np.array(be._out, dtype=np.int64)
                        cls = decide(logits)
                    else:
                        w = np.frombuffer(buf, dtype=np.int32).astype(np.int64).reshape(seq, ch)
                        signal.alarm(DMA_WATCHDOG_S)
                        cls, logits = clf.infer_from_codes(w)
                        signal.alarm(0)
                except DmaTimeout:
                    signal.alarm(0)
                    aborted = "DMA timeout at window %d" % done
                    break
                finally:
                    signal.alarm(0)
                # overlap: submit -> result collected (includes the overlapped read
                # of the next window only if that took longer than the core)
                lat_ms = (time.time() - w0) * 1000.0
                if not overlap and done + 1 < n:
                    nxt = src.next(done + 1)

                tc, stale = mon.snapshot()
                if tc is not None and tc > maxt:
                    maxt = tc
                emit(src_idx, cls, logits, lat_ms, tc)
                done += 1

                # safety (reads only the cached value; sampling is in TempMonitor)
                if tc is not None and tc >= STOP_C:
                    aborted = "temperature %.3f C >= %.1f C at window %d" % (tc, STOP_C, done)
                    break
                if stale is not None and stale >= TEMP_STALE_S:
                    aborted = "no valid temperature for %.1f s at window %d" % (stale, done)
                    break
            if aborted is None and done < n:
                aborted = src.aborted

    if fout is not None:
        fout.flush()
        fout.close()
    dur = time.time() - t_start
    tend, _ = mon.snapshot()
    if tend is not None and tend > maxt:
        maxt = tend
    mon.stop()
    print("SUMMARY executed=%d aborted=%s wall_s=%.3f throughput=%.3f "
          "temp_start=%.3f temp_max=%.3f temp_end=%s prefetch=%d underruns=%d qmin=%s "
          "pace_hz=%g late_release=%d"
          % (done, aborted, dur, (done / dur) if dur else 0.0,
             t0, maxt, ("%.3f" % tend) if tend is not None else "None",
             PREFETCH_DEPTH if use_prefetch else 0, stats["underruns"], stats["qmin"],
             PACE_HZ, stats["late_release"]), flush=True)
    if aborted:
        sys.exit(7)


if __name__ == "__main__":
    main()
