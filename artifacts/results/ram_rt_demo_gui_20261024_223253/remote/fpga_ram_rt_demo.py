#!/usr/bin/env python3
"""RAM real-time demo runner (mode `ram_realtime_demo`; Python 3.6, deployed driver).

Same timing core as fpga_rt_acceptance.py (validated hard-RT workload): a documented
finite input set is PRELOADED into RAM, windows are released on a fixed clock every
P = 20.48 ms, each is processed once (late = counted, never skipped), and only this
main thread runs with SCHED_FIFO <prio> on CPU {cpu}. Board response = done - release.

Differences, all outside the board-response boundary:
  * the input set is a raw file of selected test-set windows (tools/make_ram_demo_set.py)
    with their dataset indices, cycled n_total times;
  * each result line is handed to a separate PUBLISHER process (forked first, normal
    scheduling, no affinity) through a NON-BLOCKING pipe; the publisher forwards it to
    stdout (SSH -> PC). If the PC/SSH falls behind and the pipe is full, the line is
    dropped for display and counted (`display_dropped`); timing and the raw record are
    unaffected. The timed thread never blocks on SSH;
  * optional clock-sync rounds with the PC over stdin/stdout BEFORE and AFTER the timed
    loop (never inside it), so the PC can map board release times to its own clock.

Usage (inside <DEP>/driver with PYTHONPATH there, under temp_guard.sh):
  python3.6 -u fpga_ram_rt_demo.py <bit> <set.i32> <indices.txt> <n_total> <warmup> \
      <cpu> <prio> <out_dir> <sync_rounds>
Output lines (stdout): TEMP_START=, SYNC_REQ/SYNC, META, R idx=.. cls=.. lat_ms=..
temp_c=.. logits=a,b,c,d k=.. rel_b=.., SUMMARY ... ; files in <out_dir>: rt_raw.npz,
summary.json. Exit: 0 ok, 5 temperature gate, 7 aborted (timeout/temperature/SIGTERM).
"""
import json
import os
import resource
import select
import signal
import sys
import threading
import time

import numpy as np

XADC = "/sys/bus/iio/devices/iio:device0"
STOP_C = 75.0
TEMP_STALE_S = 30.0
DMA_WATCHDOG_S = 5
SYNC_TIMEOUT_S = 5.0
P = 1024.0 / 50000.0
REC = 1024 * 8 * 4


def start_publisher():
    """Fork the publisher before any thread exists. Returns the write end (parent)."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:                                   # publisher: relay pipe -> stdout
        os.close(w)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)   # ends on EOF (parent exit)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        broken = False
        while True:
            b = os.read(r, 65536)
            if not b:
                break
            while b and not broken:
                try:
                    n = os.write(1, b)
                    b = b[n:]
                except OSError:
                    broken = True                  # SSH gone: keep draining, never block parent
        os._exit(0)
    os.close(r)
    return w, pid


class Out:
    def __init__(self, fd):
        self.fd = fd
        self.dropped = 0

    def block(self, line):
        os.set_blocking(self.fd, True)
        try:
            b = (line + "\n").encode()
            while b:
                b = b[os.write(self.fd, b):]
        except OSError:
            pass

    def nowait(self, b):
        """Non-blocking; lines < PIPE_BUF are written whole or not at all."""
        try:
            os.write(self.fd, b)
            return True
        except (BlockingIOError, InterruptedError):
            self.dropped += 1
            return False
        except OSError:
            self.dropped += 1
            return False


def clock_sync(out, phase, rounds):
    """PC sends 'S <i>' on stdin, board answers 'SYNC <i> tb=<perf_counter>'."""
    if rounds <= 0:
        return
    out.block("SYNC_REQ phase=%s rounds=%d" % (phase, rounds))
    for i in range(rounds):
        ready, _, _ = select.select([sys.stdin], [], [], SYNC_TIMEOUT_S)
        line = sys.stdin.readline() if ready else ""
        if not line.startswith("S "):
            out.block("SYNC_SKIPPED phase=%s at=%d" % (phase, i))
            return
        out.block("SYNC %s tb=%.7f" % (line.split()[1], time.perf_counter()))


class TempMonitor(threading.Thread):
    def __init__(self, interval=1.0):
        threading.Thread.__init__(self)
        self.daemon = True
        self.interval = interval
        self.lock = threading.Lock()
        self.temp, self.last, self.stop_flag = None, None, False
        self.off = float(open(XADC + "/in_temp0_offset").read())
        self.scl = float(open(XADC + "/in_temp0_scale").read())

    def run(self):
        while not self.stop_flag:
            try:
                with open(XADC + "/in_temp0_raw") as f:
                    t = (float(f.read()) + self.off) * self.scl / 1000.0
                with self.lock:
                    self.temp, self.last = t, time.monotonic()
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
    bit, set_path, idx_path = sys.argv[1], sys.argv[2], sys.argv[3]
    n_total, warmup = int(sys.argv[4]), int(sys.argv[5])
    cpu, prio, out_dir, sync_rounds = int(sys.argv[6]), int(sys.argv[7]), sys.argv[8], int(sys.argv[9])
    os.makedirs(out_dir)
    wfd, pub_pid = start_publisher()
    out = Out(wfd)
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _raise("SIGTERM_(guard_or_operator)"))
    signal.signal(signal.SIGALRM, _raise("DMA_timeout"))

    mon = TempMonitor()
    mon.start()
    w0 = time.monotonic()
    t0c, _ = mon.snapshot()
    while t0c is None and time.monotonic() - w0 < TEMP_STALE_S:
        time.sleep(0.2)
        t0c, _ = mon.snapshot()
    out.block("TEMP_START=%s" % t0c)
    if t0c is None or t0c >= STOP_C:
        out.block("ABORT temperature gate")
        sys.exit(5)

    # --- preload the documented set (outside the timed interval) -------------
    a = time.perf_counter()
    raw = np.fromfile(set_path, dtype="<i4")
    if raw.size % 8192:
        out.block("ABORT set size not a multiple of one window")
        sys.exit(7)
    buf = raw.reshape(-1, 8192)
    ds_idx = np.array([int(x) for x in open(idx_path).read().split()], dtype=np.int64)
    if len(ds_idx) != len(buf):
        out.block("ABORT indices/set length mismatch")
        sys.exit(7)
    n_unique = len(buf)
    load_s = time.perf_counter() - a

    from fault_classifier import FaultClassifier, HardwareBackend
    clf = FaultClassifier(backend=HardwareBackend(bit, wait_timeout=10.0))
    be = clf.backend
    decide = clf._decide

    clock_sync(out, "start", sync_rounds)
    # Only this (main) thread: the temperature thread and the publisher process
    # keep normal scheduling. Nothing global is changed; it ends with the process.
    os.sched_setaffinity(0, {cpu})
    os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(prio))
    sched = "SCHED_FIFO prio %d, cpu %s" % (prio, sorted(os.sched_getaffinity(0)))
    out.block("META n=%d seq=1024 ch=8 class_source=%s mode=ram_realtime_demo n_unique=%d "
              "n_total=%d warmup=%d sched=%s preload_s=%.2f publisher_pid=%d"
              % (n_total, clf.class_source, n_unique, n_total, warmup, sched.replace(" ", "_"),
                 load_s, pub_pid))
    os.set_blocking(wfd, False)

    rel = np.zeros(n_total)
    st = np.zeros(n_total)
    sd = np.zeros(n_total)
    dn = np.zeros(n_total)
    cls = np.full(n_total, -1, dtype=np.int8)
    lg = np.zeros((n_total, 4), dtype=np.int16)
    sent = np.zeros(n_total, dtype=np.int8)
    aborted, done, maxt, tlast = None, 0, t0c, t0c
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    T0 = time.perf_counter() + 0.5
    try:
        for k in range(n_total):
            release = T0 + k * P
            now = time.perf_counter()
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
            # board response ends at s2; the display hand-off below is outside it
            sent[k] = out.nowait(("R idx=%d cls=%d lat_ms=%.3f temp_c=%s logits=%d,%d,%d,%d k=%d rel_b=%.7f\n"
                                  % (ds_idx[k % n_unique], c, (s2 - release) * 1e3,
                                     "NA" if tlast is None else "%.2f" % tlast,
                                     logits[0], logits[1], logits[2], logits[3], k, release)).encode())
            if k % 250 == 0:
                tc, stale = mon.snapshot()
                if tc is not None:
                    maxt, tlast = max(maxt, tc), tc
                if tc is not None and tc >= STOP_C:
                    aborted = "temperature_%.3f_C_at_window_%d" % (tc, k)
                    break
                if stale is not None and stale >= TEMP_STALE_S:
                    aborted = "no_valid_temperature_for_%.0f_s_at_window_%d" % (stale, k)
                    break
    except Abort as e:
        signal.alarm(0)
        aborted = "%s_at_window_%d" % (e, done)
    wall = time.perf_counter() - T0
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    # leave the RT class before any blocking I/O
    os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
    mon.stop_flag = True
    n = done
    resp = (dn[:n] - rel[:n]) * 1e3
    np.savez(os.path.join(out_dir, "rt_raw.npz"), release=rel[:n] - T0, start=st[:n] - T0,
             send_done=sd[:n] - T0, done=dn[:n] - T0, cls=cls[:n], logits=lg[:n],
             window_index=np.arange(n) % n_unique, dataset_index=ds_idx[np.arange(n) % n_unique],
             display_sent=sent[:n], T0_board=np.array([T0]))
    acc = resp[warmup:]
    summ = {"mode": "ram_realtime_demo", "n_unique": n_unique, "n_total": n_total, "executed": n,
            "warmup": warmup, "aborted": aborted, "sched": sched, "class_source": clf.class_source,
            "preload_s": round(load_s, 3), "period_ms": P * 1e3,
            "acceptance_windows": int(len(acc)),
            "deadline_misses": int((acc > P * 1e3).sum()),
            "response_ms": {q: round(float(np.percentile(acc, p)), 3) for q, p in
                            (("p50", 50), ("p99", 99), ("p99.9", 99.9), ("p99.99", 99.99))} if len(acc) else None,
            "response_ms_max": round(float(acc.max()), 3) if len(acc) else None,
            "display_dropped": int(n - int(sent[:n].sum())),
            "involuntary_ctx_switches": ru1.ru_nivcsw - ru0.ru_nivcsw,
            "temp_c": {"start": t0c, "max": maxt, "end": mon.snapshot()[0]}}
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    clock_sync(out, "end", sync_rounds if not aborted else 0)
    out.block("SUMMARY executed=%d aborted=%s wall_s=%.3f throughput=%.4f temp_start=%s temp_max=%s "
              "temp_end=%s deadline_misses=%d acceptance_windows=%d resp_max_ms=%s display_dropped=%d"
              % (n, aborted, wall, n / wall if wall > 0 else 0.0, t0c, maxt, summ["temp_c"]["end"],
                 summ["deadline_misses"], summ["acceptance_windows"], summ["response_ms_max"],
                 summ["display_dropped"]))
    sys.exit(7 if aborted else 0)


if __name__ == "__main__":
    main()
