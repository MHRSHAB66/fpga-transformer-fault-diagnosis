#!/usr/bin/env python3
"""End-to-end demo runner, mode `ram_tcp_e2e_demo` (Python 3.6, deployed driver).

Timing core = the validated RAM-backed loop of fpga_rt_acceptance.py / fpga_ram_rt_demo.py
(release every P = 20.48 ms, late windows counted never skipped, SCHED_FIFO <prio> on CPU
{cpu} for the timed main thread only). Data plane WITHOUT SSH:

  timed thread --(non-blocking, bounded pipe, fixed 40-byte records)--> PUBLISHER process
  publisher --(persistent TCP, TCP_NODELAY, small bounded buffers)--> PC

* The timed thread never blocks: a full pipe => the record is not queued, its sequence
  number is recorded as a board overflow (npz `queued`=0, summary list) and the cumulative
  overflow count travels in every later record. Nothing is discarded silently.
* The publisher (forked first; own scheduling, default SCHED_FIFO 40 on CPU 0) listens on
  <port>, accepts ONE client, answers clock-sync requests, timestamps each record when read
  from the pipe (t_pubrx) and right before send (t_pub), and keeps draining (and counting
  `undelivered`) if the client disconnects. stdout/SSH carry only control lines.

Wire format (little endian). Board->PC:
  b'R' + REC(40) + d t_pubrx + d t_pub                         (57 bytes)  result
  b'Y' + Q sync_id + d t_board                                 (17 bytes)  sync reply
  b'E' + I records_read + I undelivered + I overflow_cum       (13 bytes)  end of stream
PC->board: b'S' + Q sync_id (9) | b'G' (start timed run) | b'Q' (quit after final sync)
REC = <IIb4hddI3x: seq, dataset_idx, cls, logits[4], t_rel, t_ready, overflow_cum.

Usage (inside <DEP>/driver, PYTHONPATH there, under temp_guard.sh):
  python3.6 -u fpga_ram_e2e_demo.py <bit> <set.i32> <indices.txt> <n_total> <warmup> <cpu> <prio>
      <out_dir> <port> <pub_cpu> <pub_prio>
Exit: 0 ok, 5 temperature gate, 6 no client / not started, 7 aborted.
"""
import fcntl
import json
import os
import resource
import select
import signal
import socket
import struct
import sys
import threading
import time

import numpy as np

XADC = "/sys/bus/iio/devices/iio:device0"
STOP_C = 75.0
TEMP_STALE_S = 30.0
DMA_WATCHDOG_S = 5
P = 1024.0 / 50000.0
REC_FMT = "<IIb4hddI3x"
REC_SIZE = struct.calcsize(REC_FMT)          # 40
END_SEQ = 0xFFFFFFFF
PIPE_BYTES = 4096                              # bounded queue: 102 records ~ 2.1 s
SOCK_BUF = 8192
ACCEPT_TIMEOUT_S = 60.0
F_SETPIPE_SZ, F_GETPIPE_SZ = 1031, 1032


def _sched(cpu, prio):
    if cpu >= 0:
        os.sched_setaffinity(0, {cpu})
    if prio > 0:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(prio))


def publisher(data_r, ready_w, port, out_dir, pub_cpu, pub_prio):
    """Child process. Never returns."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)      # ends on pipe EOF (runner exit)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    stats = {"port": port, "pub_cpu": pub_cpu, "pub_prio": pub_prio, "client": None,
             "records_read": 0, "sent": 0, "undelivered": 0, "undelivered_first_seq": None,
             "disconnect_t": None, "max_send_s": 0.0, "sync_replies": 0, "error": None}
    try:
        _sched(pub_cpu, pub_prio)
        stats["sched"] = "%s cpu %s" % ("SCHED_FIFO %d" % pub_prio if pub_prio > 0 else "SCHED_OTHER",
                                       sorted(os.sched_getaffinity(0)))
        ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ls.bind(("0.0.0.0", port))
        ls.listen(1)
        ls.settimeout(ACCEPT_TIMEOUT_S)
        try:
            c, addr = ls.accept()
        except socket.timeout:
            stats["error"] = "no client within %.0f s" % ACCEPT_TIMEOUT_S
            os.write(ready_w, b"X")
            raise SystemExit
        ls.close()
        stats["client"] = "%s:%d" % addr
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        c.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SOCK_BUF)
        c.setblocking(True)
        alive, started, quit_req = True, False, False
        inbuf, pbuf = b"", b""
        eof = False

        def send(b):
            nonlocal alive
            if not alive:
                return False
            a = time.perf_counter()
            try:
                c.sendall(b)
            except OSError:
                alive = False
                stats["disconnect_t"] = time.perf_counter()
                return False
            stats["max_send_s"] = max(stats["max_send_s"], time.perf_counter() - a)
            return True

        while True:
            rl = [data_r] if not eof else []
            if alive:
                rl.append(c)
            if not rl:
                break
            r, _, _ = select.select(rl, [], [], 10.0 if eof else None)
            if not r and eof:
                break                                      # no final 'Q' within 10 s
            if c in r and alive:
                try:
                    b = c.recv(4096)
                except OSError:
                    b = b""
                if not b:
                    alive = False
                    if stats["disconnect_t"] is None:
                        stats["disconnect_t"] = time.perf_counter()
                inbuf += b
                while inbuf:
                    t = inbuf[:1]
                    if t == b"S":
                        if len(inbuf) < 9:
                            break
                        sid = inbuf[1:9]
                        inbuf = inbuf[9:]
                        send(b"Y" + sid + struct.pack("<d", time.perf_counter()))
                        stats["sync_replies"] += 1
                    elif t == b"G":
                        inbuf = inbuf[1:]
                        if not started:
                            started = True
                            os.write(ready_w, b"1")
                    elif t == b"Q":
                        inbuf = inbuf[1:]
                        quit_req = True
                    else:
                        stats["error"] = "bad client byte %r" % t
                        inbuf = b""
                if quit_req and eof:
                    break
            if data_r in r:
                b = os.read(data_r, 65536)
                if not b:
                    eof = True
                    send(b"E" + struct.pack("<III", stats["records_read"], stats["undelivered"], 0))
                    if not alive or quit_req:
                        break
                    continue
                pbuf += b
                while len(pbuf) >= REC_SIZE:
                    rec, pbuf = pbuf[:REC_SIZE], pbuf[REC_SIZE:]
                    t_rx = time.perf_counter()
                    stats["records_read"] += 1
                    if send(b"R" + rec + struct.pack("<dd", t_rx, time.perf_counter())):
                        stats["sent"] += 1
                    else:
                        stats["undelivered"] += 1
                        if stats["undelivered_first_seq"] is None:
                            stats["undelivered_first_seq"] = struct.unpack(REC_FMT, rec)[0]
        try:
            c.close()
        except Exception:
            pass
    except SystemExit:
        pass
    except Exception as e:                                  # recorded, never hidden
        stats["error"] = repr(e)
    try:
        with open(os.path.join(out_dir, "publisher.json"), "w") as f:
            json.dump(stats, f, indent=2)
    except Exception:
        pass
    os._exit(0)


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


def say(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def cpu_stat():
    with open("/proc/stat") as f:
        return [l.split() for l in f.read().splitlines() if l.startswith("cpu")]


def main():
    bit, set_path, idx_path = sys.argv[1], sys.argv[2], sys.argv[3]
    n_total, warmup, cpu, prio = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
    out_dir, port, pub_cpu, pub_prio = sys.argv[8], int(sys.argv[9]), int(sys.argv[10]), int(sys.argv[11])
    os.makedirs(out_dir)
    data_r, data_w = os.pipe()
    ready_r, ready_w = os.pipe()
    try:
        fcntl.fcntl(data_w, F_SETPIPE_SZ, PIPE_BYTES)
    except OSError:
        pass
    pipe_bytes = fcntl.fcntl(data_w, F_GETPIPE_SZ)
    pid = os.fork()
    if pid == 0:
        os.close(data_w)
        os.close(ready_r)
        publisher(data_r, ready_w, port, out_dir, pub_cpu, pub_prio)
    os.close(data_r)
    os.close(ready_w)
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
    say("TEMP_START=%s" % t0c)
    if t0c is None or t0c >= STOP_C:
        say("ABORT temperature gate")
        sys.exit(5)

    a = time.perf_counter()
    buf = np.fromfile(set_path, dtype="<i4").reshape(-1, 8192)
    ds_idx = np.array([int(x) for x in open(idx_path).read().split()], dtype=np.int64)
    if len(ds_idx) != len(buf):
        say("ABORT indices/set length mismatch")
        sys.exit(7)
    n_unique, load_s = len(buf), time.perf_counter() - a
    from fault_classifier import FaultClassifier, HardwareBackend
    clf = FaultClassifier(backend=HardwareBackend(bit, wait_timeout=10.0))
    be, decide = clf.backend, clf._decide

    say("LISTEN port=%d publisher_pid=%d pipe_bytes=%d records=%d" % (port, pid, pipe_bytes, pipe_bytes // REC_SIZE))
    try:
        r, _, _ = select.select([ready_r], [], [], ACCEPT_TIMEOUT_S + 30)
        go = os.read(ready_r, 1) if r else b""
    except Abort as e:
        say("ABORT %s before start" % e)
        sys.exit(7)
    if go != b"1":
        say("ABORT client did not connect/start (%r)" % go)
        os.close(data_w)
        sys.exit(6)

    _sched(cpu, prio)
    sched = "SCHED_FIFO prio %d, cpu %s" % (prio, sorted(os.sched_getaffinity(0)))
    say("META n=%d seq=1024 ch=8 class_source=%s mode=ram_tcp_e2e_demo n_unique=%d n_total=%d warmup=%d "
        "sched=%s preload_s=%.2f" % (n_total, clf.class_source, n_unique, n_total, warmup,
                                     sched.replace(" ", "_"), load_s))
    os.set_blocking(data_w, False)
    rel = np.zeros(n_total)
    st = np.zeros(n_total)
    dn = np.zeros(n_total)
    qd = np.zeros(n_total)
    cls = np.full(n_total, -1, dtype=np.int8)
    lg = np.zeros((n_total, 4), dtype=np.int16)
    queued = np.zeros(n_total, dtype=np.int8)
    overflow = 0
    aborted, done, maxt = None, 0, t0c
    cpu0 = cpu_stat()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    pack = struct.Struct(REC_FMT).pack
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
            be.dma.recvchannel.transfer(be._out)
            be.dma.sendchannel.transfer(be._in)
            be._wait(be.dma.sendchannel)
            be._wait(be.dma.recvchannel)
            logits = np.array(be._out, dtype=np.int64)
            c = decide(logits)
            s2 = time.perf_counter()
            signal.alarm(0)
            rel[k], st[k], dn[k] = release, s0, s2
            cls[k], lg[k] = c, logits
            done = k + 1
            # hand-off after class-ready; never blocks, never silent
            try:
                os.write(data_w, pack(k, int(ds_idx[k % n_unique]), int(c), int(logits[0]), int(logits[1]),
                                      int(logits[2]), int(logits[3]), release, s2, overflow))
                queued[k] = 1
            except (BlockingIOError, InterruptedError, OSError):
                overflow += 1
            qd[k] = time.perf_counter()
            if k % 250 == 0:
                tc, stale = mon.snapshot()
                if tc is not None:
                    maxt = max(maxt, tc)
                if tc is not None and tc >= STOP_C:
                    aborted = "temperature_%.3f_C_at_window_%d" % (tc, k)
                    break
                if stale is not None and stale >= TEMP_STALE_S:
                    aborted = "no_valid_temperature_at_window_%d" % k
                    break
    except Abort as e:
        signal.alarm(0)
        aborted = "%s_at_window_%d" % (e, done)
    wall = time.perf_counter() - T0
    cpu1 = cpu_stat()
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
    mon.stop_flag = True
    n = done
    np.savez(os.path.join(out_dir, "rt_raw.npz"), release=rel[:n] - T0, start=st[:n] - T0, done=dn[:n] - T0,
             queued_t=qd[:n] - T0, cls=cls[:n], logits=lg[:n], window_index=np.arange(n) % n_unique,
             dataset_index=ds_idx[np.arange(n) % n_unique], queued=queued[:n], T0_board=np.array([T0]))
    acc = (dn[warmup:n] - rel[warmup:n]) * 1e3
    summ = {"mode": "ram_tcp_e2e_demo", "n_total": n_total, "executed": n, "warmup": warmup,
            "aborted": aborted, "sched": sched, "class_source": clf.class_source,
            "acceptance_windows": int(len(acc)), "deadline_misses": int((acc > P * 1e3).sum()),
            "response_ms_max": round(float(acc.max()), 3) if len(acc) else None,
            "board_overflow": int(overflow),
            "board_overflow_seqs": [int(i) for i in np.where(queued[:n] == 0)[0]][:1000],
            "pipe_bytes": pipe_bytes, "pipe_records": pipe_bytes // REC_SIZE,
            "involuntary_ctx_switches": ru1.ru_nivcsw - ru0.ru_nivcsw,
            "proc_stat_cpu_before": cpu0, "proc_stat_cpu_after": cpu1,
            "temp_c": {"start": t0c, "max": maxt, "end": mon.snapshot()[0]}}
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    os.set_blocking(data_w, True)
    os.close(data_w)                     # EOF -> publisher sends 'E', final sync, exits
    try:
        os.waitpid(pid, 0)
    except Exception:
        pass
    say("SUMMARY executed=%d aborted=%s wall_s=%.3f throughput=%.4f temp_start=%s temp_max=%s temp_end=%s "
        "deadline_misses=%d acceptance_windows=%d resp_max_ms=%s board_overflow=%d"
        % (n, aborted, wall, n / wall if wall > 0 else 0.0, t0c, maxt, summ["temp_c"]["end"],
           summ["deadline_misses"], summ["acceptance_windows"], summ["response_ms_max"], overflow))
    sys.exit(7 if aborted else 0)


if __name__ == "__main__":
    main()
