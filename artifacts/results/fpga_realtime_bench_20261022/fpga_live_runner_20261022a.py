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
import sys
import time
import signal
import gzip
import threading

import numpy as np
from fault_classifier import HardwareBackend, FaultClassifier

XADC = "/sys/bus/iio/devices/iio:device0"
STOP_C = 75.0
TEMP_STALE_S = 30.0
DMA_WATCHDOG_S = 20


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
    be = clf.backend
    pipelined = all(hasattr(be, a) for a in ("dma", "_in", "_out", "_wait"))
    decide = getattr(clf, "_decide", None) or (lambda lg: int(np.argmax(lg)))
    state = {"src_idx": -1, "aborted": None}

    def fetch(fin, pos):
        """Next selected record as (dataset_index, bytes), or None (+abort reason).

        `pos` = number of windows completed when this record will be used
        (keeps the original abort messages)."""
        while True:
            buf = fin.read(rec_bytes)
            if not buf:
                state["aborted"] = "input exhausted at window %d (requested %d)" % (pos, n)
                return None
            if len(buf) != rec_bytes:
                state["aborted"] = "short read %d at window %d" % (len(buf), pos)
                return None
            state["src_idx"] += 1
            # Skip records not in the requested index set (gz read sequentially).
            if want_set is not None and state["src_idx"] not in want_set:
                if state["src_idx"] > max_idx:
                    state["aborted"] = "input passed last requested index at %d" % state["src_idx"]
                    return None
                continue
            return state["src_idx"], buf

    with gzip.open(in_path, "rb") as fin:
        nxt = fetch(fin, 0)
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
                    if done + 1 < n:
                        nxt = fetch(fin, done + 1)           # overlaps core compute
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
            # pipelined: submit -> result collected (includes the overlapped read
            # of the next window if that took longer than the core)
            lat_ms = (time.time() - w0) * 1000.0
            if not pipelined and done + 1 < n:
                nxt = fetch(fin, done + 1)

            tc, stale = mon.snapshot()
            if tc is not None and tc > maxt:
                maxt = tc

            # logits= : the 4 raw INT10 logits returned by the core (extra
            # trailing field; older PC parsers ignore it). The PC derives an
            # optional confidence from them; the class is the FPGA-path result.
            print("R idx=%d cls=%d lat_ms=%.3f temp_c=%s logits=%s"
                  % (src_idx, int(cls), lat_ms, ("%.3f" % tc) if tc is not None else "NA",
                     ",".join(str(int(v)) for v in logits)),
                  flush=True)
            if fout is not None:
                fout.write("%d\n" % int(cls))
                fout.flush()
            done += 1

            # safety (reads only the cached value; sampling is in TempMonitor)
            if tc is not None and tc >= STOP_C:
                aborted = "temperature %.3f C >= %.1f C at window %d" % (tc, STOP_C, done)
                break
            if stale is not None and stale >= TEMP_STALE_S:
                aborted = "no valid temperature for %.1f s at window %d" % (stale, done)
                break
        if aborted is None and done < n:
            aborted = state["aborted"]

    if fout is not None:
        fout.flush()
        fout.close()
    dur = time.time() - t_start
    tend, _ = mon.snapshot()
    if tend is not None and tend > maxt:
        maxt = tend
    mon.stop()
    print("SUMMARY executed=%d aborted=%s wall_s=%.3f throughput=%.3f "
          "temp_start=%.3f temp_max=%.3f temp_end=%s"
          % (done, aborted, dur, (done / dur) if dur else 0.0,
             t0, maxt, ("%.3f" % tend) if tend is not None else "None"), flush=True)
    if aborted:
        sys.exit(7)


if __name__ == "__main__":
    main()
