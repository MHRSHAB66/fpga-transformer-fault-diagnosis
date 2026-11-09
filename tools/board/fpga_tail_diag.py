#!/usr/bin/env python3
"""Tail-latency attribution on the board (Python 3.6). Diagnostic only.

Two isolated paths, each timed per window with resource.getrusage deltas
(ru_inblock = block-device input ops, ru_nvcsw = voluntary context switches,
ru_nivcsw = involuntary preemptions, ru_majflt) and process CPU time:

  read <in.i32.gz> <n> <out_dir> [readahead_mb]
      gzip read of n contiguous windows only (no FPGA). With readahead_mb > 0 a
      thread keeps posix_fadvise(WILLNEED) issued for the next readahead_mb of
      the compressed file (kernel page-cache readahead, no data copied).
  dma  <bit> <in.i32.gz> <k> <n> <out_dir>
      load the first k windows into RAM once, then run n inferences cycling
      through them (no file I/O in the loop). Each window is one intact
      32,768-byte MM2S frame; class = the deployed driver's decision (FPGA
      register on the argmax overlay), recorded for comparison.

Writes <out_dir>/diag.npz (per-window arrays) and summary.json.
Safety for `dma`: in-process XADC stop at >= 75 C / 30 s without a valid
reading, SIGALRM DMA watchdog; run it under tools/board/temp_guard.sh.
"""
import gzip
import json
import os
import resource
import signal
import sys
import time

import numpy as np

REC = 1024 * 8 * 4
XADC = "/sys/bus/iio/devices/iio:device0"


def _ru():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return (r.ru_inblock, r.ru_nvcsw, r.ru_nivcsw, r.ru_majflt)


def _temp():
    try:
        raw = float(open(XADC + "/in_temp0_raw").read())
        off = float(open(XADC + "/in_temp0_offset").read())
        scl = float(open(XADC + "/in_temp0_scale").read())
        return (raw + off) * scl / 1000.0
    except Exception:
        return None


class Timeout(Exception):
    pass


def _alarm(*_):
    raise Timeout()


def _pct(a):
    return {"mean": round(float(a.mean()), 3), "p50": round(float(np.percentile(a, 50)), 3),
            "p99": round(float(np.percentile(a, 99)), 3),
            "p99.9": round(float(np.percentile(a, 99.9)), 3), "max": round(float(a.max()), 3)}


def main():
    mode = sys.argv[1]
    ra_mb = 0
    if mode == "read":
        in_path, n, out = sys.argv[2], int(sys.argv[3]), sys.argv[4]
        ra_mb = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    else:
        bit, in_path, k, n, out = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    os.makedirs(out)
    wall = np.zeros(n)
    cpu = np.zeros(n)
    ru = np.zeros((n, 4), dtype=np.int64)
    cls = np.full(n, -1, dtype=np.int8)
    temps = []
    aborted = None
    t_start = time.time()
    if mode == "read":
        with gzip.open(in_path, "rb") as f:
            ra = None
            if ra_mb > 0:
                import threading
                ra_stop = threading.Event()
                raw = f.fileobj

                def warm():
                    fd = os.open(in_path, os.O_RDONLY)
                    done_to = 0
                    try:
                        while not ra_stop.is_set():
                            try:
                                pos = raw.tell()
                            except Exception:
                                pos = done_to
                            target = pos + ra_mb * (1 << 20)
                            if done_to < target:
                                os.posix_fadvise(fd, done_to, target - done_to, os.POSIX_FADV_WILLNEED)
                                done_to = target
                            ra_stop.wait(0.05)
                    finally:
                        os.close(fd)
                ra = threading.Thread(target=warm, daemon=True)
                ra.start()
            for i in range(n):
                r0, c0, a = _ru(), time.process_time(), time.perf_counter()
                buf = f.read(REC)
                np.frombuffer(buf, dtype="<i4")
                wall[i] = time.perf_counter() - a
                cpu[i] = time.process_time() - c0
                ru[i] = np.subtract(_ru(), r0)
                if len(buf) != REC:
                    aborted = "short read at %d" % i
                    n = i
                    break
            if ra is not None:
                ra_stop.set()
    else:
        signal.signal(signal.SIGALRM, _alarm)
        t0 = _temp()
        if t0 is None or t0 >= 75.0:
            print("ABORT temperature gate %s" % t0, flush=True)
            sys.exit(5)
        from fault_classifier import HardwareBackend, FaultClassifier
        clf = FaultClassifier(backend=HardwareBackend(bit, wait_timeout=10.0))
        with gzip.open(in_path, "rb") as f:
            ram = [np.frombuffer(f.read(REC), dtype="<i4").astype(np.int64).reshape(1024, 8)
                   for _ in range(k)]
        last_t = time.monotonic()
        for i in range(n):
            w = ram[i % k]
            signal.alarm(20)
            try:
                r0, c0, a = _ru(), time.process_time(), time.perf_counter()
                c, _ = clf.infer_from_codes(w)
                wall[i] = time.perf_counter() - a
                cpu[i] = time.process_time() - c0
                ru[i] = np.subtract(_ru(), r0)
            except Timeout:
                aborted = "DMA timeout at %d" % i
                n = i
                break
            finally:
                signal.alarm(0)
            cls[i] = c
            if time.monotonic() - last_t >= 2.0:
                last_t = time.monotonic()
                t = _temp()
                temps.append(t)
                if t is None or t >= 75.0:
                    aborted = "temperature %s at %d" % (t, i)
                    n = i + 1
                    break
    wall, cpu, ru, cls = wall[:n] * 1e3, cpu[:n] * 1e3, ru[:n], cls[:n]
    np.savez(os.path.join(out, "diag.npz"), wall_ms=wall, cpu_ms=cpu, ru=ru, cls=cls)
    slow = wall > (20.48 if mode == "dma" else 10.0)
    summ = {"mode": mode, "n": int(n), "aborted": aborted, "wall_s": round(time.time() - t_start, 2),
            "wall_ms": _pct(wall), "cpu_ms": _pct(cpu),
            "slow_threshold_ms": 20.48 if mode == "dma" else 10.0, "slow_count": int(slow.sum()),
            "slow_with_block_input": int((ru[slow, 0] > 0).sum()),
            "slow_with_involuntary_preemption": int((ru[slow, 2] > 0).sum()),
            "slow_with_voluntary_wait": int((ru[slow, 1] > 0).sum()),
            "slow_with_major_fault": int((ru[slow, 3] > 0).sum()),
            "slow_wall_minus_cpu_ms_mean": round(float((wall[slow] - cpu[slow]).mean()), 3) if slow.any() else None,
            "all_block_input_ops": int(ru[:, 0].sum()),
            "temps_c": [round(t, 3) if t is not None else None for t in temps[:: max(1, len(temps) // 20)]],
            "temp_max_c": max([t for t in temps if t is not None], default=None)}
    if mode == "dma":
        summ["k_windows_in_ram"] = k
    else:
        summ["readahead_mb"] = ra_mb
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print("SUMMARY " + json.dumps(summ), flush=True)
    sys.exit(7 if aborted else 0)


if __name__ == "__main__":
    main()
