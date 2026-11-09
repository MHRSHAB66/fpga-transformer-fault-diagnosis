#!/usr/bin/env python3
"""Stage-timed on-board throughput benchmark (runs ON the Z-Turn, Python 3.6).

Measures the board-side inference loop over the prepared test-set file with
per-stage timing, in one of two loop modes that use the SAME bitstream, the SAME
driver, the SAME int32 input file and the SAME single 32,768-byte MM2S frame:

  baseline   exact loop of fpga_fulltest_bin_runner.py (the 2026-10-08 46.4 inf/s
             run): gzip read -> frombuffer.astype(int64) -> infer_from_codes
             (blocking send+recv wait) -> write.
  pipelined  arm S2MM, send the frame, wait MM2S only (the core buffers the whole
             window before computing), THEN read+parse the next window from gzip
             while the core computes, then wait S2MM. Same frame, same order,
             same core; only host-side work is overlapped with core compute.

Classification stays on the FPGA path exactly as in the deployed system
(argmax of the 4 hardware logits, np.argmax = lowest index on ties).

Safety: an in-process XADC thread stops the run at >= 75 C or after 30 s
without a valid reading; the companion temp_guard.sh is an independent process
that kills this PID at >= 75 C. Nothing restarts automatically after a stop.

Usage (from the deployment's driver dir, so the deployed driver is imported):
    PYTHONPATH=<DEP>/driver_py36compat python3.6 -u fpga_stage_bench.py \
        <bit> <in.i32.gz> <n> <baseline|pipelined> <out_dir> [lines]
`lines` additionally prints the live protocol line per window
(`R idx=<i> cls=<c> lat_ms=<f> temp_c=<t>`), for PC-side end-to-end timing.
"""
import gzip
import json
import os
import signal
import sys
import threading
import time

import numpy as np
from fault_classifier import HardwareBackend

XADC = "/sys/bus/iio/devices/iio:device0"
STOP_C = 75.0
TEMP_STALE_S = 30.0
SEQ, CH = 1024, 8
REC = SEQ * CH * 4


class TempMonitor(threading.Thread):
    def __init__(self, interval=1.0):
        threading.Thread.__init__(self)
        self.daemon = True
        self.interval = interval
        self._lock = threading.Lock()
        self._temp = None
        self._last = None
        self._stop = False
        self.off = float(open(XADC + "/in_temp0_offset").read())
        self.scl = float(open(XADC + "/in_temp0_scale").read())

    def run(self):
        while not self._stop:
            try:
                raw = float(open(XADC + "/in_temp0_raw").read())
                t = (raw + self.off) * self.scl / 1000.0
                with self._lock:
                    self._temp, self._last = t, time.monotonic()
            except Exception:
                pass
            time.sleep(self.interval)

    def snapshot(self):
        with self._lock:
            if self._last is None:
                return None, None
            return self._temp, time.monotonic() - self._last


class Stop(Exception):
    pass


def _term(*_):
    raise Stop("SIGTERM (external temperature guard or operator)")


signal.signal(signal.SIGTERM, _term)


def main():
    bit, in_path, n, mode, out_dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
    emit = len(sys.argv) > 6 and sys.argv[6] == "lines"
    if mode not in ("baseline", "pipelined"):
        raise SystemExit("mode must be baseline|pipelined")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    mon = TempMonitor()
    mon.start()
    w0 = time.monotonic()
    t0, _ = mon.snapshot()
    while t0 is None and time.monotonic() - w0 < TEMP_STALE_S:
        time.sleep(0.2)
        t0, _ = mon.snapshot()
    print("TEMP_START=%s" % (("%.3f" % t0) if t0 is not None else "None"), flush=True)
    if t0 is None or t0 >= STOP_C:
        print("ABORT temperature gate", flush=True)
        sys.exit(5)

    hw = HardwareBackend(bit, wait_timeout=10.0)
    dma = hw.dma
    print("META n=%d seq=%d ch=%d mode=%s" % (n, SEQ, CH, mode), flush=True)

    preds = np.full(n, -1, dtype=np.int8)
    logits = np.zeros((n, 4), dtype=np.int32)
    t_read = np.zeros(n)     # gzip read + parse (+ copy into DMA buffer)
    t_core = np.zeros(n)     # time blocked on DMA/core (send+recv wait)
    t_iter = np.zeros(n)     # full loop iteration
    maxt = t0
    aborted = None
    done = 0
    fout = open(os.path.join(out_dir, "pred.txt"), "w")
    t_start = time.time()
    try:
        with gzip.open(in_path, "rb") as fin:
            if mode == "baseline":
                while done < n:
                    a = time.perf_counter()
                    buf = fin.read(REC)
                    if len(buf) != REC:
                        aborted = "short read at %d" % done
                        break
                    w = np.frombuffer(buf, dtype=np.int32).astype(np.int64).reshape(SEQ, CH)
                    b = time.perf_counter()
                    lg = hw.infer_codes(w)          # same call chain as infer_from_codes
                    cls = int(np.argmax(np.asarray(lg, dtype=np.int64)))
                    c = time.perf_counter()
                    fout.write("%d\n" % cls)
                    preds[done], logits[done] = cls, lg
                    t_read[done], t_core[done] = b - a, c - b
                    done += 1
                    if emit:
                        tc, _ = mon.snapshot()
                        print("R idx=%d cls=%d lat_ms=%.3f temp_c=%s" % (
                            done - 1, cls, (c - b) * 1e3, ("%.3f" % tc) if tc is not None else "NA"),
                            flush=True)
                    t_iter[done - 1] = time.perf_counter() - a
                    if done % 500 == 0:
                        aborted = _check(mon, done, n, t_start, emit)
                        maxt = max(maxt, mon.snapshot()[0] or maxt)
                        if aborted:
                            break
            else:
                # prime: first window into the DMA buffer
                a = time.perf_counter()
                buf = fin.read(REC)
                hw._in[:] = np.frombuffer(buf, dtype=np.int32)
                pending_read = time.perf_counter() - a
                while done < n:
                    a = time.perf_counter()
                    dma.recvchannel.transfer(hw._out)       # arm S2MM first (as driver)
                    dma.sendchannel.transfer(hw._in)        # one 32,768-byte MM2S frame
                    hw._wait(dma.sendchannel)
                    b = time.perf_counter()
                    nxt = None
                    if done + 1 < n:                        # overlap: read next window
                        nxt = fin.read(REC)
                        if len(nxt) != REC:
                            aborted = "short read at %d" % (done + 1)
                    r = time.perf_counter()
                    hw._wait(dma.recvchannel)
                    c = time.perf_counter()
                    lg = np.array(hw._out, dtype=np.int64)
                    cls = int(np.argmax(lg))
                    fout.write("%d\n" % cls)
                    preds[done], logits[done] = cls, lg
                    t_read[done] = pending_read
                    t_core[done] = (b - a) + (c - r)
                    if nxt is not None and not aborted:
                        hw._in[:] = np.frombuffer(nxt, dtype=np.int32)   # buffer free after MM2S done
                    pending_read = (r - b) + (time.perf_counter() - c)
                    done += 1
                    if emit:
                        tc, _ = mon.snapshot()
                        print("R idx=%d cls=%d lat_ms=%.3f temp_c=%s" % (
                            done - 1, cls, (c - a) * 1e3, ("%.3f" % tc) if tc is not None else "NA"),
                            flush=True)
                    t_iter[done - 1] = time.perf_counter() - a
                    if aborted:
                        break
                    if done % 500 == 0:
                        aborted = _check(mon, done, n, t_start, emit)
                        maxt = max(maxt, mon.snapshot()[0] or maxt)
                        if aborted:
                            break
    except Stop as exc:
        aborted = str(exc)
    finally:
        fout.close()
    dur = time.time() - t_start
    tend, _ = mon.snapshot()
    maxt = max(maxt, tend or maxt)
    mon._stop = True
    logits[:done].tofile(os.path.join(out_dir, "logits_i32.bin"))
    np.savez(os.path.join(out_dir, "timing.npz"), t_read=t_read[:done],
             t_core=t_core[:done], t_iter=t_iter[:done])
    summ = {"mode": mode, "requested": n, "executed": done, "aborted": aborted,
            "wall_s": round(dur, 3), "throughput_inf_s": round(done / dur, 3) if dur else 0.0,
            "temp_c": {"start": t0, "max": maxt, "end": tend, "stop_threshold": STOP_C}}
    for name, arr in (("t_read_ms", t_read[:done]), ("t_core_ms", t_core[:done]),
                      ("t_iter_ms", t_iter[:done])):
        if done:
            summ[name] = {"mean": round(float(arr.mean()) * 1e3, 3),
                          "p50": round(float(np.percentile(arr, 50)) * 1e3, 3),
                          "p95": round(float(np.percentile(arr, 95)) * 1e3, 3),
                          "max": round(float(arr.max()) * 1e3, 3)}
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    print("SUMMARY " + json.dumps(summ), flush=True)
    sys.exit(7 if aborted else 0)


def _check(mon, done, n, t_start, emit):
    tc, stale = mon.snapshot()
    el = time.time() - t_start
    if not emit:
        print("progress %d/%d temp=%s %.1f inf/s elapsed=%.0fs" % (
            done, n, ("%.3f" % tc) if tc is not None else "NA", done / el if el else 0, el), flush=True)
    if tc is not None and tc >= STOP_C:
        return "temperature %.3f C >= %.1f C at window %d" % (tc, STOP_C, done)
    if stale is not None and stale >= TEMP_STALE_S:
        return "no valid temperature for %.1f s at window %d" % (stale, done)
    return None


if __name__ == "__main__":
    main()
