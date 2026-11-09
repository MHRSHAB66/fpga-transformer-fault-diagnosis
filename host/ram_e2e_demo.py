"""End-to-end demo mode `ram_tcp_e2e_demo` (PC side). Separate from the legacy SD/gzip
mode and from `ram_realtime_demo` (both unchanged).

Control plane: SSH launches tools/board/fpga_ram_e2e_demo.py under the independent 75 C
guard (same wrapper, preflight, run-dir and Stop as the other live modes).
Data plane: one persistent TCP connection PC -> board:PORT (no SSH, no per-frame process),
TCP_NODELAY, small buffers. Each result carries its immutable sequence id and the board
timestamps t_rel, t_ready, t_pubrx, t_pub; the PC adds t_rx (receiver thread) and the GUI
adds t_render. Clock sync (request/response) runs on the same connection before, every
0.5 s during, and after the run. Criterion: docs/reports/e2e_hard_rt_criterion_20261024.md.

Nothing here classifies: the class comes from the FPGA register on the board.
"""
from __future__ import annotations

import ctypes
import json
import os
import shlex
import socket
import struct
import subprocess
import threading
import time
import uuid
from collections import deque
from typing import Dict, Iterator, List, Optional

from host import live_fpga as lf
from host import ram_rt_demo as rd

MODE = "ram_tcp_e2e_demo"
PORT = 50555
PUB_CPU, PUB_PRIO = 0, 40
WARMUP, CPU, PRIO = rd.WARMUP, rd.CPU, rd.PRIO
SINK_MAX = 1024                  # bounded PC display queue (frames)
RCVBUF = 8192
SYNC_INITIAL, SYNC_FINAL, SYNC_PERIOD_S = 50, 30, 0.5
RUNNER_LOCAL = os.path.join(lf._REPO, "tools", "board", "fpga_ram_e2e_demo.py")
REC_FMT = "<IIb4hddI3x"
REC_SIZE = struct.calcsize(REC_FMT)
FRAME_SIZE = 1 + REC_SIZE + 16
SCOPE_NOTE = ("END-TO-END mode: release on the board -> frame rendered in this window. SSH/TCP, "
              "Ethernet and Windows are NOT presumed hard real-time; see the e2e verdict report.")
_CF = 0x08000000 if os.name == "nt" else 0


def decode_frame(b: bytes) -> Dict:
    """b = REC(40) + t_pubrx + t_pub (no leading type byte)."""
    seq, idx, cls, l0, l1, l2, l3, t_rel, t_ready, ovf = struct.unpack(REC_FMT, b[:REC_SIZE])
    t_pubrx, t_pub = struct.unpack("<dd", b[REC_SIZE:REC_SIZE + 16])
    return {"seq": seq, "idx": idx, "cls": cls, "logits": [l0, l1, l2, l3], "t_rel": t_rel,
            "t_ready": t_ready, "board_overflow_cum": ovf, "t_pubrx": t_pubrx, "t_pub": t_pub}


def encode_frame(f: Dict) -> bytes:
    """Test helper (mirror of the board's packing)."""
    return b"R" + struct.pack(REC_FMT, f["seq"], f["idx"], f["cls"], *f["logits"], f["t_rel"],
                              f["t_ready"], f.get("board_overflow_cum", 0)) + struct.pack(
        "<dd", f.get("t_pubrx", 0.0), f.get("t_pub", 0.0))


class Parser:
    """Incremental stream parser for the board->PC wire format."""

    def __init__(self):
        self.buf = b""

    def feed(self, b: bytes) -> List[tuple]:
        self.buf += b
        out = []
        while self.buf:
            t = self.buf[:1]
            if t == b"R":
                if len(self.buf) < FRAME_SIZE:
                    break
                out.append(("R", decode_frame(self.buf[1:FRAME_SIZE])))
                self.buf = self.buf[FRAME_SIZE:]
            elif t == b"Y":
                if len(self.buf) < 17:
                    break
                sid, tb = struct.unpack("<Qd", self.buf[1:17])
                out.append(("Y", (sid, tb)))
                self.buf = self.buf[17:]
            elif t == b"E":
                if len(self.buf) < 13:
                    break
                out.append(("E", struct.unpack("<III", self.buf[1:13])))
                self.buf = self.buf[13:]
            else:
                raise ValueError("bad frame type %r" % t)
        return out


class SeqTracker:
    """Order/loss accounting by immutable sequence id."""

    def __init__(self):
        self.expected = 0
        self.missing: List[int] = []
        self.duplicate: List[int] = []
        self.out_of_order: List[int] = []
        self.seen = set()

    def add(self, seq: int) -> None:
        if seq in self.seen:
            self.duplicate.append(seq)
            return
        self.seen.add(seq)
        if seq == self.expected:
            self.expected += 1
        elif seq > self.expected:
            self.missing.extend(range(self.expected, seq))
            self.expected = seq + 1
        else:
            self.out_of_order.append(seq)
            if seq in self.missing:
                self.missing.remove(seq)


class Sink:
    """Bounded receiver -> GUI queue. Overflow is recorded by seq, never silent."""

    def __init__(self, maxlen: int = SINK_MAX):
        self.maxlen = maxlen
        self.q: deque = deque()
        self.lock = threading.Lock()
        self.overflow: List[int] = []

    def put(self, f: Dict) -> bool:
        with self.lock:
            if len(self.q) >= self.maxlen:
                self.overflow.append(f["seq"])
                return False
            self.q.append(f)
            return True

    def pop_all(self) -> List[Dict]:
        with self.lock:
            out = list(self.q)
            self.q.clear()
            return out


def _win_rt(enable: bool) -> None:
    """Windows: 1 ms timer resolution + HIGH priority class for THIS mode only."""
    if os.name != "nt":
        return
    try:
        k32 = ctypes.windll.kernel32
        if enable:
            ctypes.windll.winmm.timeBeginPeriod(1)
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x80)      # HIGH_PRIORITY_CLASS
        else:
            ctypes.windll.winmm.timeEndPeriod(1)
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x20)      # NORMAL_PRIORITY_CLASS
    except Exception:
        pass


def _thread_time_critical() -> None:
    if os.name == "nt":
        try:
            ctypes.windll.kernel32.SetThreadPriority(ctypes.windll.kernel32.GetCurrentThread(), 15)
        except Exception:
            pass


def push_runner(host=lf.BOARD_HOST, user=lf.BOARD_USER, key=lf.SSH_KEY, port=22):
    remote_dir = "/home/xilinx/live_runs/e2e_" + uuid.uuid4().hex
    rd._ssh("mkdir -p /home/xilinx/live_runs && mkdir %s" % shlex.quote(remote_dir),
            host=host, user=user, key=key, port=port).check_returncode()
    scp = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-P", str(port), "-i", key]
    for src in (RUNNER_LOCAL, rd.GUARD_LOCAL):
        subprocess.run(scp + [src, "%s@%s:%s/" % (user, host, remote_dir)],
                       check=True, capture_output=True, text=True, creationflags=_CF)
    return remote_dir + "/fpga_ram_e2e_demo.py", remote_dir + "/temp_guard.sh"


def port_and_runner_facts(host=lf.BOARD_HOST, user=lf.BOARD_USER, key=lf.SSH_KEY, port=22) -> Dict:
    r = rd._ssh("ss -ltn | grep -c ':%d ' ; ps -eo pid=,args= | grep -E 'python3[.]6 .*[f]pga_ram_e2e_demo' "
                "| awk '{print $1}' | paste -sd, -" % PORT, host=host, user=user, key=key, port=port)
    lines = r.stdout.splitlines()
    return {"port_listeners": (lines[0].strip() if lines else "?"),
            "running": (lines[1].strip() if len(lines) > 1 else "")}


def evaluate_e2e(facts: Dict) -> List[Dict]:
    def chk(name, ok, detail):
        return {"check": name, "ok": bool(ok), "detail": detail}
    return [chk("e2e_port_free", facts.get("port_listeners") == "0",
                "listeners on %d: %s" % (PORT, facts.get("port_listeners"))),
            chk("no_e2e_runner", not facts.get("running"),
                "clear" if not facts.get("running") else "pids %s" % facts.get("running"))]


class E2EClient(lf.LiveFpgaClient):
    def __init__(self, n_total=1200, runner_remote=None, monitor_remote=None,
                 stall_at_s=None, stall_for_s=0.0, disconnect_at_s=None, **kw):
        if runner_remote is not None:
            kw["runner_remote"] = runner_remote
        super().__init__(n=n_total, monitor_remote=monitor_remote, **kw)
        self.n_total = n_total
        self.sink = Sink()
        self.records: Dict[int, Dict] = {}          # seq -> frame (+ t_rx, t_render)
        self.tracker = SeqTracker()
        self.sync: List[Dict] = []
        self.board_end = None
        self.sock: Optional[socket.socket] = None
        self.data_error: Optional[str] = None
        self.started_t = None
        self.stall_at_s, self.stall_for_s, self.disconnect_at_s = stall_at_s, stall_for_s, disconnect_at_s
        self.stall_log: List[Dict] = []
        self._sync_pending: Dict[int, float] = {}
        self._sid = 0
        self._send_lock = threading.Lock()
        self._data_done = threading.Event()

    def run_dir(self) -> str:
        return os.path.dirname(self.runner_remote).replace("\\", "/")

    def _kill_pattern(self) -> str:
        return "^python3[.]6 -u " + self.run_dir() + "/[f]pga_ram_e2e_demo[.]py"

    def _remote_cmd(self) -> str:
        q = shlex.quote
        drv = "%s/%s" % (self.dep_dir, self.driver_subdir)
        base = ("PYTHONPATH=%s python3.6 -u %s %s %s %s %d %d %d %d %s %d %d %d"
                % (q(drv), q(self.runner_remote), q(self.bit), q(rd.BOARD_SET_DIR + "/set.i32"),
                   q(rd.BOARD_SET_DIR + "/indices.txt"), self.n_total, WARMUP, CPU, PRIO,
                   q(self.run_dir() + "/out"), PORT, PUB_CPU, PUB_PRIO))
        log = self.run_dir() + "/guard.log"
        return ("cd %s || exit 1; test -r %s || { echo 'ABORT guard missing'; exit 1; }; "
                "%s & P=$!; bash %s $P %s 75 1 & G=$!; "
                "while kill -0 $P 2>/dev/null; do "
                "if ! kill -0 $G 2>/dev/null; then kill -TERM $P 2>/dev/null; "
                "wait $P; wait $G; echo 'ABORT independent guard exited'; exit 1; fi; sleep 1; done; "
                "wait $P; R=$?; wait $G; S=$?; "
                "echo WORKLOAD_RC=$R; echo GUARD_RC=$S; if [ $R -ne 0 ] || [ $S -ne 0 ]; then exit 1; fi"
                % (q(drv), q(self.monitor_remote), base, q(self.monitor_remote), q(log)))

    def preflight_checks(self) -> List[Dict]:
        c = rd.RamRealtimeDemoClient(n_total=self.n_total, host=self.host, user=self.user,
                                     key=self.key, port=self.port)
        return c.preflight_checks() + evaluate_e2e(port_and_runner_facts(self.host, self.user, self.key, self.port))

    # -- data plane ---------------------------------------------------------
    def _send(self, b: bytes) -> bool:
        with self._send_lock:
            if self.sock is None:
                return False
            try:
                self.sock.sendall(b)
                return True
            except OSError:
                return False

    def _sync_once(self, phase: str) -> None:
        self._sid += 1
        sid = self._sid
        ts = time.perf_counter()
        self._sync_pending[sid] = (ts, phase)
        self._send(b"S" + struct.pack("<Q", sid))

    def _receiver(self) -> None:
        _thread_time_critical()
        p = Parser()
        try:
            while True:
                if self.stall_at_s is not None and self.started_t is not None and not self.stall_log:
                    if time.perf_counter() - self.started_t >= self.stall_at_s:
                        a = time.perf_counter()
                        time.sleep(self.stall_for_s)            # PC backpressure test hook
                        self.stall_log.append({"start": a, "end": time.perf_counter()})
                b = self.sock.recv(65536)
                t_rx = time.perf_counter()
                if not b:
                    break
                for kind, v in p.feed(b):
                    if kind == "R":
                        v["t_rx"] = t_rx
                        self.tracker.add(v["seq"])
                        if v["seq"] not in self.records:
                            self.records[v["seq"]] = v
                        self.sink.put(v)
                    elif kind == "Y":
                        sid, tb = v
                        ent = self._sync_pending.pop(sid, None)
                        if ent:
                            self.sync.append({"phase": ent[1], "i": sid, "ts": ent[0], "tr": t_rx, "tb": tb})
                    elif kind == "E":
                        self.board_end = {"records_read": v[0], "undelivered": v[1], "t_rx": t_rx}
                        threading.Thread(target=self._final_sync, daemon=True).start()
        except (OSError, ValueError) as e:
            if not self._stop:
                self.data_error = repr(e)
        finally:
            self._data_done.set()

    def _final_sync(self) -> None:
        for _ in range(SYNC_FINAL):
            self._sync_once("end")
            time.sleep(0.005)
        time.sleep(0.2)
        self._send(b"Q")

    def _sync_loop(self) -> None:
        while not self._data_done.is_set() and self.board_end is None:
            if self.disconnect_at_s is not None and self.started_t is not None and \
                    time.perf_counter() - self.started_t >= self.disconnect_at_s:
                self.data_error = "test hook: PC closed the data connection"
                try:
                    self.sock.shutdown(socket.SHUT_RDWR)
                    self.sock.close()
                except OSError:
                    pass
                return
            self._sync_once("run")
            self._data_done.wait(SYNC_PERIOD_S)

    def _connect(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(5.0)
        end = time.monotonic() + 15
        while True:
            try:
                s.connect((self.host, PORT))
                break
            except OSError:
                if time.monotonic() > end:
                    raise
                time.sleep(0.2)
        s.settimeout(None)
        self.sock = s
        threading.Thread(target=self._receiver, daemon=True).start()
        for _ in range(SYNC_INITIAL):
            self._sync_once("start")
            time.sleep(0.005)
        time.sleep(0.1)
        self._send(b"G")
        self.started_t = time.perf_counter()
        threading.Thread(target=self._sync_loop, daemon=True).start()

    # -- control plane ------------------------------------------------------
    def stream(self) -> Iterator[Dict]:
        if self._stop:
            yield {"type": "status", "msg": "stop requested before launch; nothing started"}
            return
        if not self.check_connection():
            yield {"type": "error", "msg": "SSH connect failed to %s@%s" % (self.user, self.host)}
            return
        from host.live_preflight import all_ok
        checks = self.preflight_checks()
        if not all_ok(checks):
            yield {"type": "error", "msg": "preflight failed: " +
                   "; ".join(c["check"] + ": " + c["detail"] for c in checks if not c["ok"])}
            return
        if not self.monitor_remote:
            yield {"type": "error", "msg": "no independent temperature guard configured"}
            return
        if self._stop:
            yield {"type": "status", "msg": "stop requested before launch; nothing started"}
            return
        yield {"type": "status", "msg": "mode=%s; data plane TCP %s:%d; N=%d (warm-up %d)"
                                        % (MODE, self.host, PORT, self.n_total, WARMUP)}
        self.proc = subprocess.Popen(lf._ssh_base(self.host, self.user, self.key, self.port, timeout=15)
                                     + [self._remote_cmd()], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1, creationflags=_CF)
        q: list = []
        lock = threading.Lock()
        done = {"v": False}

        def pump():
            try:
                for line in self.proc.stdout:
                    with lock:
                        q.append(line.rstrip("\n"))
            finally:
                done["v"] = True

        threading.Thread(target=pump, daemon=True).start()
        last_rx = time.monotonic()
        while True:
            if self._stop:
                yield {"type": "status", "msg": "stop requested"}
                break
            with lock:
                lines = q[:]
                del q[:]
            if lines:
                last_rx = time.monotonic()
            for ln in lines:
                if ln.startswith("LISTEN "):
                    try:
                        self._connect()
                        yield {"type": "status", "msg": "data connection open; clock sync %d rounds; start sent"
                                                        % SYNC_INITIAL}
                    except OSError as e:
                        yield {"type": "error", "msg": "data connection failed: %s" % e}
                    continue
                ev = lf.parse_line(ln)
                if ev is None:
                    continue
                if ev["type"] in ("meta", "summary"):
                    ev["mode"] = MODE
                if ev["type"] == "summary":
                    kv = lf._kv(ln[8:])
                    for k in ("deadline_misses", "acceptance_windows", "board_overflow"):
                        if k in kv:
                            ev[k] = int(kv[k])
                    ev["resp_max_ms"] = kv.get("resp_max_ms")
                yield ev
            if done["v"] and not lines:
                break
            if time.monotonic() - last_rx > self.line_timeout and self.started_t is None:
                yield {"type": "error", "msg": "no output for %.0fs (timeout)" % self.line_timeout}
                break
            time.sleep(0.02)
        if self.proc.poll() is None:
            self._end_remote()
        self._data_done.wait(15)
        try:
            if self.sock is not None:
                self.sock.close()
        except OSError:
            pass
        rc = self.proc.poll()
        if rc not in (None, 0) and not self._stop:
            yield {"type": "error", "msg": "runner or independent guard failed (rc=%s)" % rc}
        if self.data_error:
            yield {"type": "error", "msg": "data plane: %s" % self.data_error}
        yield {"type": "status", "msg": "runner exited rc=%s" % rc}

    def stop(self) -> None:
        super().stop()


def write_evidence(out_dir: str, client: E2EClient, events: List[Dict], extra: Optional[Dict] = None) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "frames.tsv"), "w", encoding="utf-8") as f:
        f.write("seq\tidx\tcls\tlogits\tt_rel\tt_ready\tt_pubrx\tt_pub\tboard_overflow_cum\tt_rx\tt_render\n")
        for seq in sorted(client.records):
            r = client.records[seq]
            f.write("%d\t%d\t%d\t%s\t%.7f\t%.7f\t%.7f\t%.7f\t%d\t%.7f\t%s\n" % (
                seq, r["idx"], r["cls"], ",".join(map(str, r["logits"])), r["t_rel"], r["t_ready"],
                r["t_pubrx"], r["t_pub"], r["board_overflow_cum"], r["t_rx"],
                "" if r.get("t_render") is None else "%.7f" % r["t_render"]))
    rec = {"mode": MODE, "scope": SCOPE_NOTE, "n_total": client.n_total, "warmup": WARMUP,
           "port": PORT, "publisher_sched": "SCHED_FIFO %d cpu %d" % (PUB_PRIO, PUB_CPU),
           "board_run_dir": client.run_dir(), "frames_received": len(client.records),
           "sync_rounds": client.sync, "pc_sink_overflow_seqs": client.sink.overflow,
           "seq_missing": client.tracker.missing, "seq_duplicate": client.tracker.duplicate,
           "seq_out_of_order": client.tracker.out_of_order, "board_end": client.board_end,
           "data_error": client.data_error, "stall_log": client.stall_log,
           "disconnect_at_s": client.disconnect_at_s}
    rec.update(extra or {})
    with open(os.path.join(out_dir, "events_control.jsonl"), "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    scp = ["scp", "-r", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-P",
           str(client.port), "-i", client.key]
    r = subprocess.run(scp + ["%s@%s:%s" % (client.user, client.host, client.run_dir()),
                              os.path.join(out_dir, "remote")], capture_output=True, text=True, creationflags=_CF)
    rec["remote_copy_rc"] = r.returncode
    rec["post_run_board_processes"] = rd._ssh(
        "ps -eo pid=,args= | grep -E 'python3[.]6 .*[f]pga_(ram_e2e_demo|ram_rt_demo|live_runner|rt_acceptance)"
        "|[t]emp_guard[.]sh' || echo BOARD_IDLE").stdout.strip()
    with open(os.path.join(out_dir, "pc_evidence.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    return rec
