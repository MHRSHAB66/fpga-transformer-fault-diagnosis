#!/usr/bin/env python3
"""Board-side (Python 3.6) contract checks of the ram_tcp_e2e_demo publisher. No overlay, no DMA.
Run on the board:  python3.6 check_ram_e2e_publisher.py <dir with fpga_ram_e2e_demo.py> <port>

1 sync replies + start handshake; 2 frames in order with intact fields; 3 PC backpressure:
the writer never blocks, overflow is detected (bounded queue), no gap among queued records
after the reader resumes; 4 disconnect: publisher keeps draining, counts undelivered, exits on
EOF and writes publisher.json.
"""
import fcntl
import json
import os
import socket
import struct
import sys
import tempfile
import time

sys.path.insert(0, sys.argv[1])
import fpga_ram_e2e_demo as d  # noqa: E402

PORT = int(sys.argv[2])
pack = struct.Struct(d.REC_FMT).pack


def recv_exact(s, n):
    b = b""
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c:
            raise EOFError
        b += c
    return b


def read_msg(s):
    t = recv_exact(s, 1)
    if t == b"R":
        return "R", recv_exact(s, d.REC_SIZE + 16)
    if t == b"Y":
        return "Y", recv_exact(s, 16)
    if t == b"E":
        return "E", recv_exact(s, 12)
    raise ValueError(t)


def main():
    out = tempfile.mkdtemp(prefix="e2e_pub_check_")
    data_r, data_w = os.pipe()
    ready_r, ready_w = os.pipe()
    fcntl.fcntl(data_w, d.F_SETPIPE_SZ, d.PIPE_BYTES)
    pipe_bytes = fcntl.fcntl(data_w, d.F_GETPIPE_SZ)
    pid = os.fork()
    if pid == 0:
        os.close(data_w)
        os.close(ready_r)
        d.publisher(data_r, ready_w, PORT, out, -1, 0)
    os.close(data_r)
    os.close(ready_w)
    res = {}
    time.sleep(0.5)
    s = socket.create_connection(("127.0.0.1", PORT))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
    # 1 sync + handshake
    rtts = []
    for i in range(20):
        a = time.perf_counter()
        s.sendall(b"S" + struct.pack("<Q", i))
        k, b = read_msg(s)
        rtts.append(time.perf_counter() - a)
        assert k == "Y" and struct.unpack("<Qd", b)[0] == i
    s.sendall(b"G")
    res["handshake"] = os.read(ready_r, 1) == b"1"
    res["sync_rtt_ms_max_loopback"] = round(max(rtts) * 1e3, 3)
    # 2 ordered frames
    os.set_blocking(data_w, False)
    got = []
    for batch in range(4):                         # 50 records = 2000 B < pipe bound
        for k in range(batch * 50, batch * 50 + 50):
            os.write(data_w, pack(k, k + 7, k % 4, k, -k, 1, 2, 1.0 + k, 2.0 + k, 0))
        for _ in range(50):
            k, b = read_msg(s)
            got.append(struct.unpack(d.REC_FMT, b[:d.REC_SIZE]))
    res["ordered_200"] = [g[0] for g in got] == list(range(200)) and all(
        g[1] == g[0] + 7 and list(g[3:7]) == [g[0], -g[0], 1, 2] for g in got)
    # 3 backpressure: stop reading; writer must not block; overflow must appear
    worst, queued, ovf, seq = 0.0, [], 0, 1000
    for _ in range(20000):
        a = time.perf_counter()
        try:
            os.write(data_w, pack(seq, 0, 0, 0, 0, 0, 0, 0.0, 0.0, ovf))
            queued.append(seq)
        except BlockingIOError:
            ovf += 1
        worst = max(worst, time.perf_counter() - a)
        seq += 1
    res["backpressure_overflow"] = ovf
    res["backpressure_queued"] = len(queued)
    res["writer_worst_us"] = round(worst * 1e6, 1)
    seen = []
    s.settimeout(3.0)
    try:
        while len(seen) < len(queued):
            k, b = read_msg(s)
            seen.append(struct.unpack(d.REC_FMT, b[:d.REC_SIZE])[0])
    except socket.timeout:
        pass
    res["resume_no_gap"] = seen == queued
    # 4 disconnect, then more records, then EOF
    s.close()
    time.sleep(0.3)
    for k in range(50):
        try:
            os.write(data_w, pack(50000 + k, 0, 0, 0, 0, 0, 0, 0.0, 0.0, ovf))
        except BlockingIOError:
            pass
        time.sleep(0.002)
    os.close(data_w)
    t0 = time.time()
    os.waitpid(pid, 0)
    res["publisher_exit_s"] = round(time.time() - t0, 2)
    st = json.load(open(os.path.join(out, "publisher.json")))
    res["publisher_json"] = {k: st[k] for k in ("records_read", "sent", "undelivered", "error", "sched")}
    ok = (res["handshake"] and res["ordered_200"] and ovf > 0 and res["resume_no_gap"]
          and st["undelivered"] > 0 and st["records_read"] == st["sent"] + st["undelivered"]
          and res["publisher_exit_s"] < 15 and pipe_bytes == d.PIPE_BYTES)
    res["pipe_bytes"] = pipe_bytes
    print(json.dumps(res, indent=1))
    print("RESULT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
