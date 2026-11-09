#!/usr/bin/env python3
"""Board-side (Python 3.6) checks of the ram_realtime_demo display hand-off.
No overlay, no DMA. Run on the board:  python3.6 check_ram_rt_demo_handoff.py <dir with fpga_ram_rt_demo.py>

1. publisher stalled (its stdout is a pipe nobody reads): the timed side's non-blocking
   write never blocks; lines are dropped and counted once the pipe is full.
2. publisher relays lines verbatim when the reader keeps up.
3. clock_sync with no PC on stdin -> SYNC_SKIPPED, no hang.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, sys.argv[1])
import fpga_ram_rt_demo as d  # noqa: E402


def child_case(case):
    if case == "stalled":
        w, pid = d.start_publisher()
        out = d.Out(w)
        os.set_blocking(w, False)
        worst = 0.0
        for k in range(20000):
            a = time.perf_counter()
            out.nowait(("R idx=%d cls=0 lat_ms=15.000 temp_c=45.00 logits=1,2,3,4 k=%d rel_b=1.0\n" % (k, k)).encode())
            worst = max(worst, time.perf_counter() - a)
        # report via a file: the blocked publisher still holds the inherited stdio
        with open(sys.argv[3], "w") as f:
            f.write("STALLED dropped=%d worst_write_us=%.1f\n" % (out.dropped, worst * 1e6))
        os._exit(0)
    if case == "relay":
        w, pid = d.start_publisher()
        out = d.Out(w)
        os.set_blocking(w, False)
        for k in range(500):
            out.nowait(("R k=%d\n" % k).encode())
            time.sleep(0.001)
        print("RELAY dropped=%d" % out.dropped, file=sys.stderr, flush=True)
        os.close(w)
        os.waitpid(pid, 0)
        os._exit(0)
    if case == "sync":
        w, pid = d.start_publisher()
        out = d.Out(w)
        d.clock_sync(out, "start", 3)
        os.close(w)
        os.waitpid(pid, 0)
        os._exit(0)


def main():
    if len(sys.argv) > 2:
        child_case(sys.argv[2])
        return
    me = os.path.abspath(__file__)
    ok = True
    # 1. stdout = pipe that is never read (stalled SSH)
    r, w = os.pipe()
    res = "/tmp/ram_demo_stalled_%d.txt" % os.getpid()
    p = subprocess.Popen([sys.executable, me, sys.argv[1], "stalled", res], stdout=w,
                         stderr=subprocess.DEVNULL)
    os.close(w)
    p.wait(timeout=60)
    err = open(res).read()
    os.remove(res)
    os.close(r)          # stalled reader goes away -> publisher gets EPIPE, drains, exits on EOF
    dropped = int(err.split("dropped=")[1].split()[0])
    worst_us = float(err.split("worst_write_us=")[1].split()[0])
    t1 = dropped > 0 and worst_us < 5000
    print("test_stalled_publisher_never_blocks:", "PASS" if t1 else "FAIL", err.strip())
    ok &= t1
    # 2. relay
    p = subprocess.run([sys.executable, me, sys.argv[1], "relay"], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=60)
    lines = p.stdout.decode().splitlines()
    t2 = lines == ["R k=%d" % k for k in range(500)] and "dropped=0" in p.stderr.decode()
    print("test_publisher_relays_verbatim:", "PASS" if t2 else "FAIL", len(lines), p.stderr.decode().strip())
    ok &= t2
    # 3. sync without PC (stdin closed)
    a = time.time()
    p = subprocess.run([sys.executable, me, sys.argv[1], "sync"], stdin=subprocess.DEVNULL,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    out = p.stdout.decode()
    t3 = "SYNC_REQ phase=start rounds=3" in out and "SYNC_SKIPPED phase=start at=0" in out and time.time() - a < 10
    print("test_sync_without_pc_skips:", "PASS" if t3 else "FAIL", out.strip().replace("\n", " | "))
    ok &= t3
    print("RESULT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
