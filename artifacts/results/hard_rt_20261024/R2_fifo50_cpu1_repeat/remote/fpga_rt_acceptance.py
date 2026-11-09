#!/usr/bin/env python3
"""Hard real-time acceptance test on the board (Python 3.6, deployed driver).

Criterion: docs/reports/hard_rt_acceptance_criterion_20261024.md (fixed before
the runs). Windows are PRELOADED into RAM (sensor-DMA stand-in), released on a
fixed clock every P = 20.48 ms, each processed once (late windows are processed
late and counted, never skipped). Per window it records, on time.perf_counter:

  release   scheduled arrival  t0 + k*P
  start     copy into the DMA buffer begins
  send_done MM2S complete (frame accepted by the core)
  done      S2MM complete + class read (FPGA pred_class register via the driver)

response = done - release; miss = response > P.

Usage (run inside <DEP>/driver with PYTHONPATH there, under temp_guard.sh):
  python3.6 -u fpga_rt_acceptance.py <bit> <in.i32.gz> <n_unique> <n_total> <warmup> \
      <rt:0|1> <cpu> <prio> <out_dir>
rt=1: this process only gets SCHED_FIFO <prio> and CPU affinity {cpu} (no global
setting is changed; it ends with the process). The temperature thread is started
BEFORE that, so it keeps normal priority. Kernel RT throttling (95 %) is left as is.
Exit: 0 ok, 5 temperature gate, 7 aborted (timeout / temperature / SIGTERM).
"""
import gzip
import json
import os
import resource
import signal
import sys
import threading
import time

import numpy as np
from fault_classifier import FaultClassifier, HardwareBackend

XADC = "/sys/bus/iio/devices/iio:device0"
STOP_C = 75.0
TEMP_STALE_S = 30.0
DMA_WATCHDOG_S = 5
P = 1024.0 / 50000.0
REC = 1024 * 8 * 4


class TempMonitor(threading.Thread):
    def __init__(self, interval=1.0):
        threading.Thread.__init__(self)
        self.daemon = True
        self.interval = interval
        self.lock = threading.Lock()
        self.temp, self.last, self.stop_flag = None, None, False
        self.off = float(open(XADC + "/in_temp0_offset").read())
        self.scl = float(open(XADC + "/in_temp0_scale").read())
        self.series = []

    def run(self):
        while not self.stop_flag:
            try:
                with open(XADC + "/in_temp0_raw") as f:
                    t = (float(f.read()) + self.off) * self.scl / 1000.0
                with self.lock:
                    self.temp, self.last = t, time.monotonic()
                    self.series.append(round(t, 3))
            except Exception:
                pass
            time.sleep(self.interval)

    def snapshot(self):
        with self.lock:
            return self.temp, (None if self.last is None else time.monotonic() - self.last)


class Abort(Exception):
    pass


def _raise(msg):
    def h(*_):
        raise Abort(msg)
    return h


def main():
    bit, in_path = sys.argv[1], sys.argv[2]
    n_unique, n_total, warmup = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    rt, cpu, prio, out = sys.argv[6] == "1", int(sys.argv[7]), int(sys.argv[8]), sys.argv[9]
    os.makedirs(out)
    signal.signal(signal.SIGTERM, _raise("SIGTERM (guard or operator)"))
    signal.signal(signal.SIGALRM, _raise("DMA timeout"))

    mon = TempMonitor()
    mon.start()
    w0 = time.monotonic()
    t0c, _ = mon.snapshot()
    while t0c is None and time.monotonic() - w0 < TEMP_STALE_S:
        time.sleep(0.2)
        t0c, _ = mon.snapshot()
    print("TEMP_START=%s" % t0c, flush=True)
    if t0c is None or t0c >= STOP_C:
        print("ABORT temperature gate", flush=True)
        sys.exit(5)

    # --- preload (outside the timed interval; footprint measured) ----------
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    a = time.perf_counter()
    buf = np.empty((n_unique, 8192), dtype=np.int32)
    with gzip.open(in_path, "rb") as f:
        for i in range(n_unique):
            b = f.read(REC)
            if len(b) != REC:
                print("ABORT short input at %d" % i, flush=True)
                sys.exit(7)
            buf[i] = np.frombuffer(b, dtype="<i4")
    load_s = time.perf_counter() - a
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    clf = FaultClassifier(backend=HardwareBackend(bit, wait_timeout=10.0))
    be = clf.backend
    decide = clf._decide
    sched = "SCHED_OTHER"
    if rt:
        os.sched_setaffinity(0, {cpu})
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(prio))
        sched = "SCHED_FIFO prio %d, cpu %s" % (prio, sorted(os.sched_getaffinity(0)))
    print("META n_unique=%d n_total=%d warmup=%d sched=%s class_source=%s preload_s=%.2f "
          "preload_maxrss_kb=%d->%d" % (n_unique, n_total, warmup, sched.replace(" ", "_"),
                                       clf.class_source, load_s, rss0, rss1), flush=True)

    rel = np.zeros(n_total)
    st = np.zeros(n_total)
    sd = np.zeros(n_total)
    dn = np.zeros(n_total)
    cls = np.full(n_total, -1, dtype=np.int8)
    lg = np.zeros((n_total, 4), dtype=np.int16)
    aborted, done, maxt = None, 0, t0c
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    T0 = time.perf_counter() + 0.5
    try:
        for k in range(n_total):
            release = T0 + k * P
            now = time.perf_counter()
            # sleep until 0.5 ms before the release, then spin. Keeps the busy
            # share (spin + polling DMA wait) ~80 % of each period, below the
            # kernel's 95 % RT throttling limit, which would itself cause stalls.
            if release - now > 0.0005:
                time.sleep(release - now - 0.0005)
            while time.perf_counter() < release:
                pass
            s0 = time.perf_counter()
            be._in[:] = buf[k % n_unique]
            signal.alarm(DMA_WATCHDOG_S)
            be.dma.recvchannel.transfer(be._out)       # arm S2MM first (as the driver)
            be.dma.sendchannel.transfer(be._in)        # one intact 32,768-byte frame
            be._wait(be.dma.sendchannel)
            s1 = time.perf_counter()
            be._wait(be.dma.recvchannel)
            logits = np.array(be._out, dtype=np.int64)
            c = decide(logits)                          # FPGA register, checked vs logits
            s2 = time.perf_counter()
            signal.alarm(0)
            rel[k], st[k], sd[k], dn[k] = release, s0, s1, s2
            cls[k], lg[k] = c, logits
            done = k + 1
            if k % 250 == 0:
                tc, stale = mon.snapshot()
                if tc is not None:
                    maxt = max(maxt, tc)
                if tc is not None and tc >= STOP_C:
                    aborted = "temperature %.3f C at window %d" % (tc, k)
                    break
                if stale is not None and stale >= TEMP_STALE_S:
                    aborted = "no valid temperature for %.0f s at window %d" % (stale, k)
                    break
    except Abort as e:
        signal.alarm(0)
        aborted = "%s at window %d" % (e, done)
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    mon.stop_flag = True
    n = done
    resp = (dn[:n] - rel[:n]) * 1e3
    np.savez(os.path.join(out, "rt_raw.npz"), release=rel[:n] - T0, start=st[:n] - T0,
             send_done=sd[:n] - T0, done=dn[:n] - T0, cls=cls[:n], logits=lg[:n],
             window_index=np.arange(n) % n_unique)
    acc = resp[warmup:]
    summ = {"n_unique": n_unique, "n_total": n_total, "executed": n, "warmup": warmup,
            "aborted": aborted, "sched": sched, "class_source": clf.class_source,
            "preload_s": round(load_s, 3), "preload_maxrss_kb": [rss0, rss1],
            "period_ms": P * 1e3,
            "acceptance_windows": int(len(acc)),
            "deadline_misses": int((acc > P * 1e3).sum()),
            "response_ms": {q: round(float(np.percentile(acc, p)), 3) for q, p in
                            (("p50", 50), ("p95", 95), ("p99", 99), ("p99.9", 99.9))} if len(acc) else None,
            "response_ms_max": round(float(acc.max()), 3) if len(acc) else None,
            "warmup_misses": int((resp[:warmup] > P * 1e3).sum()),
            "involuntary_ctx_switches": ru1.ru_nivcsw - ru0.ru_nivcsw,
            "voluntary_ctx_switches": ru1.ru_nvcsw - ru0.ru_nvcsw,
            "major_faults": ru1.ru_majflt - ru0.ru_majflt,
            "temp_c": {"start": t0c, "max": maxt, "end": mon.snapshot()[0]}}
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print("SUMMARY " + json.dumps(summ), flush=True)
    sys.exit(7 if aborted else 0)


if __name__ == "__main__":
    main()
