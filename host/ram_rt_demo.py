"""Deterministic demo mode `ram_realtime_demo` (PC side).

A SEPARATE mode next to the legacy SD/gzip live mode (host.live_fpga, unchanged):
the board preloads a documented finite input set into RAM
(artifacts/ram_demo_sets/<name>/, built by tools/make_ram_demo_set.py), releases one
window every 20.48 ms and runs only its timing-critical main thread with SCHED_FIFO 50
on CPU 1 (tools/board/fpga_ram_rt_demo.py). The FPGA decides every class; the PC only
launches, displays and records.

Two separate measurements:
  * board response latency = board release -> FPGA class ready (board clock). This is
    the quantity of the hard-real-time criterion.
  * end-to-end latency = board release -> PC receipt of the line / -> GUI update, on
    the PC clock, using a clock offset measured over the same SSH channel before and
    after the run (error bound = half the best round-trip). Measured and reported
    only; it is NOT part of any hard-real-time claim.

Headless (physical evidence):
  python -m host.ram_rt_demo --n 30200                 # 30,000 accepted after warm-up 200
  python -m host.ram_rt_demo --stage-only               # copy + hash-check the set only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, Iterator, List, Optional

from host import live_fpga as lf

MODE = "ram_realtime_demo"
SET_NAME = "class_balanced_256_v1"
LOCAL_SET_DIR = os.path.join(lf._REPO, "artifacts", "ram_demo_sets", SET_NAME)
BOARD_SET_DIR = "/home/xilinx/ram_demo_sets/" + SET_NAME
RUNNER_LOCAL = os.path.join(lf._REPO, "tools", "board", "fpga_ram_rt_demo.py")
GUARD_LOCAL = os.path.join(lf._REPO, "tools", "board", "temp_guard.sh")
WARMUP, CPU, PRIO = 200, 1, 50
SYNC_ROUNDS = 20
PERIOD_MS = 20.48
SCOPE_NOTE = ("RAM real-time demo: board response (release -> FPGA class) is the hard-RT "
              "boundary (SCHED_FIFO 50, CPU 1, tested durations only). SSH transport and "
              "the Windows GUI update are measured separately and are NOT hard real-time.")
_CF = 0x08000000 if os.name == "nt" else 0


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_provenance(set_dir: str = LOCAL_SET_DIR) -> Dict:
    with open(os.path.join(set_dir, "provenance.json"), encoding="utf-8") as f:
        return json.load(f)


def _ssh(cmd: str, timeout=60, host=lf.BOARD_HOST, user=lf.BOARD_USER, key=lf.SSH_KEY, port=22):
    return subprocess.run(lf._ssh_base(host, user, key, port) + [cmd], capture_output=True,
                          text=True, timeout=timeout, creationflags=_CF)


def board_set_facts(host=lf.BOARD_HOST, user=lf.BOARD_USER, key=lf.SSH_KEY, port=22) -> Dict[str, str]:
    r = _ssh("cd %s 2>/dev/null && sha256sum set.i32 indices.txt; "
             "ps -eo pid=,args= | grep -E 'python3[.]6 .*[f]pga_ram_rt_demo' | awk '{print $1}' | paste -sd, -"
             % shlex.quote(BOARD_SET_DIR), host=host, user=user, key=key, port=port)
    facts = {"rc": str(r.returncode)}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] in ("set.i32", "indices.txt"):
            facts[parts[1]] = parts[0]
        elif line.strip() and "," in line or line.strip().isdigit():
            facts["running"] = line.strip()
    return facts


def evaluate_set(facts: Dict[str, str], prov: Dict, indices_sha: str, n_total: int) -> List[Dict]:
    """Pure: mode-specific preflight checks (in addition to host.live_preflight)."""
    def chk(name, ok, detail):
        return {"check": name, "ok": bool(ok), "detail": detail}
    got = facts.get("set.i32", "")
    return [
        chk("ram_set_identity", got == prov["set_sha256"],
            "board %s, expected %s" % (got[:12] or "missing", prov["set_sha256"][:12])),
        chk("ram_set_indices_identity", facts.get("indices.txt", "") == indices_sha,
            "indices.txt %s" % ((facts.get("indices.txt") or "missing")[:12])),
        chk("ram_n_total_valid", isinstance(n_total, int) and n_total > WARMUP,
            "n_total=%s (must exceed warm-up %d)" % (n_total, WARMUP)),
        chk("no_ram_demo_running", not facts.get("running"),
            "clear" if not facts.get("running") else "pids %s" % facts.get("running")),
    ]


def stage_set(host=lf.BOARD_HOST, user=lf.BOARD_USER, key=lf.SSH_KEY, port=22) -> Dict[str, str]:
    """Copy the set to its own versioned board directory once (never overwrites a
    differing file: refuses if a file exists with another hash)."""
    prov = load_provenance()
    facts = board_set_facts(host, user, key, port)
    if facts.get("set.i32") == prov["set_sha256"]:
        return facts
    if facts.get("set.i32"):
        raise lf.LiveError("board set exists with a different hash; refusing to overwrite")
    _ssh("mkdir -p %s" % shlex.quote(BOARD_SET_DIR), host=host, user=user, key=key, port=port)
    scp = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-P", str(port), "-i", key]
    for name in ("set.i32", "indices.txt", "provenance.json"):
        subprocess.run(scp + [os.path.join(LOCAL_SET_DIR, name),
                              "%s@%s:%s/%s" % (user, host, BOARD_SET_DIR, name)],
                       check=True, capture_output=True, text=True, creationflags=_CF)
    return board_set_facts(host, user, key, port)


def push_runner(host=lf.BOARD_HOST, user=lf.BOARD_USER, key=lf.SSH_KEY, port=22):
    """Runner + guard into a NEW unique run directory (same convention as live mode)."""
    remote_dir = "/home/xilinx/live_runs/ram_" + uuid.uuid4().hex
    _ssh("mkdir -p /home/xilinx/live_runs && mkdir %s" % shlex.quote(remote_dir),
         host=host, user=user, key=key, port=port).check_returncode()
    scp = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-P", str(port), "-i", key]
    for src in (RUNNER_LOCAL, GUARD_LOCAL):
        subprocess.run(scp + [src, "%s@%s:%s/" % (user, host, remote_dir)],
                       check=True, capture_output=True, text=True, creationflags=_CF)
    return remote_dir + "/fpga_ram_rt_demo.py", remote_dir + "/temp_guard.sh"


class RamRealtimeDemoClient(lf.LiveFpgaClient):
    """Launch/consume the RAM demo runner. Stop/cleanup reuse LiveFpgaClient."""

    def __init__(self, n_total=1200, runner_remote=None, monitor_remote=None, **kw):
        # runner_remote=None: preflight-only client (nothing pushed yet)
        if runner_remote is not None:
            kw["runner_remote"] = runner_remote
        super().__init__(n=n_total, monitor_remote=monitor_remote, **kw)
        self.n_total = n_total
        self.sync: List[Dict] = []      # {"phase", "i", "ts", "tr", "tb"} (PC perf_counter)

    def run_dir(self) -> str:
        return os.path.dirname(self.runner_remote).replace("\\", "/")

    def _kill_pattern(self) -> str:
        return "^python3[.]6 -u " + self.run_dir() + "/[f]pga_ram_rt_demo[.]py"

    def _remote_cmd(self) -> str:
        q = shlex.quote
        drv = "%s/%s" % (self.dep_dir, self.driver_subdir)
        base = ("PYTHONPATH=%s python3.6 -u %s %s %s %s %d %d %d %d %s %d"
                % (q(drv), q(self.runner_remote), q(self.bit), q(BOARD_SET_DIR + "/set.i32"),
                   q(BOARD_SET_DIR + "/indices.txt"), self.n_total, WARMUP, CPU, PRIO,
                   q(self.run_dir() + "/out"), SYNC_ROUNDS))
        log = self.run_dir() + "/guard.log"
        # `<&3`: a background job would otherwise get /dev/null as stdin (clock sync).
        return ("cd %s || exit 1; test -r %s || { echo 'ABORT guard missing'; exit 1; }; exec 3<&0; "
                "%s <&3 & P=$!; bash %s $P %s 75 1 </dev/null & G=$!; "
                "while kill -0 $P 2>/dev/null; do "
                "if ! kill -0 $G 2>/dev/null; then kill -TERM $P 2>/dev/null; "
                "wait $P; wait $G; echo 'ABORT independent guard exited'; exit 1; fi; sleep 1; done; "
                "wait $P; R=$?; wait $G; S=$?; "
                "echo WORKLOAD_RC=$R; echo GUARD_RC=$S; if [ $R -ne 0 ] || [ $S -ne 0 ]; then exit 1; fi"
                % (q(drv), q(self.monitor_remote), base, q(self.monitor_remote), q(log)))

    def preflight_checks(self) -> List[Dict]:
        from host.live_preflight import evaluate, run_probe
        checks = evaluate(run_probe(host=self.host, user=self.user, key=self.key, port=self.port,
                                    dep_dir=self.dep_dir, bit=self.bit, in_gz=self.in_gz), 1)
        prov = load_provenance()
        checks += evaluate_set(board_set_facts(self.host, self.user, self.key, self.port), prov,
                               _sha256(os.path.join(LOCAL_SET_DIR, "indices.txt")), self.n_total)
        return checks

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
        yield {"type": "status", "msg": "mode=%s; preloading %s into board RAM; N=%d (warm-up %d)"
                                        % (MODE, SET_NAME, self.n_total, WARMUP)}
        if self._stop:
            yield {"type": "status", "msg": "stop requested before launch; nothing started"}
            return
        self.proc = subprocess.Popen(lf._ssh_base(self.host, self.user, self.key, self.port, timeout=15)
                                     + [self._remote_cmd()], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, creationflags=_CF)
        q: list = []
        lock = threading.Lock()
        done = {"v": False}

        def pump():
            try:
                while True:
                    line = self.proc.stdout.readline()
                    if not line:
                        break
                    rx = time.perf_counter()
                    line = line.rstrip("\n")
                    if line.startswith("SYNC_REQ "):
                        kv = lf._kv(line[9:])
                        self._sync_rounds(kv.get("phase", "?"), int(kv.get("rounds", "0")))
                    with lock:
                        q.append((rx, line))
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
            for rx, ln in lines:
                ev = lf.parse_line(ln)
                if ev is None:
                    continue
                if ev["type"] == "result":
                    kv = lf._kv(ln[2:])
                    ev["k"], ev["rel_b"], ev["rx_pc"] = int(kv["k"]), float(kv["rel_b"]), rx
                    ev["mode"] = MODE
                elif ev["type"] == "meta":
                    ev["mode"] = MODE
                elif ev["type"] == "summary":
                    kv = lf._kv(ln[8:])
                    for key in ("deadline_misses", "acceptance_windows", "display_dropped"):
                        if key in kv:
                            ev[key] = int(kv[key])
                    ev["resp_max_ms"] = kv.get("resp_max_ms")
                yield ev
            if done["v"] and not lines:
                break
            if time.monotonic() - last_rx > self.line_timeout:
                yield {"type": "error", "msg": "no output for %.0fs (timeout)" % self.line_timeout}
                break
            time.sleep(0.02)
        if self.proc.poll() is None:
            self._end_remote()
        rc = self.proc.poll()
        if rc not in (None, 0) and not self._stop:
            yield {"type": "error", "msg": "runner or independent guard failed (rc=%s)" % rc}
        yield {"type": "status", "msg": "runner exited rc=%s" % rc}

    def _sync_rounds(self, phase: str, rounds: int) -> None:
        """Runs in the reader thread: answer each round immediately."""
        for i in range(rounds):
            ts = time.perf_counter()
            try:
                self.proc.stdin.write("S %d\n" % i)
                self.proc.stdin.flush()
            except OSError:
                return
            line = self.proc.stdout.readline()
            tr = time.perf_counter()
            parts = line.split()
            if len(parts) == 3 and parts[0] == "SYNC" and parts[2].startswith("tb="):
                self.sync.append({"phase": phase, "i": i, "ts": ts, "tr": tr, "tb": float(parts[2][3:])})
            else:
                return


def clock_map(sync: List[Dict]) -> Optional[Dict]:
    """board_t = a + b * pc_t from the best (min-RTT) round of each phase.
    Error bound of each anchor = RTT/2 (any path asymmetry included)."""
    best = {}
    for s in sync:
        rtt = s["tr"] - s["ts"]
        if s["phase"] not in best or rtt < best[s["phase"]][0]:
            best[s["phase"]] = (rtt, s)
    if "start" not in best:
        return None
    r0, s0 = best["start"]
    pc0, b0 = (s0["ts"] + s0["tr"]) / 2, s0["tb"]
    out = {"anchors": 1, "rtt_start_ms": r0 * 1e3, "err_bound_ms": r0 * 1e3 / 2, "drift_ppm": None}
    slope = 1.0
    if "end" in best:
        r1, s1 = best["end"]
        pc1, b1 = (s1["ts"] + s1["tr"]) / 2, s1["tb"]
        if pc1 - pc0 > 1.0:
            slope = (b1 - b0) / (pc1 - pc0)
            out.update(anchors=2, rtt_end_ms=r1 * 1e3, err_bound_ms=max(r0, r1) * 1e3 / 2,
                       drift_ppm=(slope - 1.0) * 1e6)
    out.update(a=b0 - slope * pc0, b=slope)
    return out


def e2e_latencies(results: List[Dict], cmap: Dict, key: str) -> List[float]:
    """PC-side time of `key` minus the board release mapped to the PC clock (ms)."""
    a, b = cmap["a"], cmap["b"]
    return [(r[key] - (r["rel_b"] - a) / b) * 1e3 for r in results if r.get(key) is not None]


def pct(v: List[float], q: float) -> Optional[float]:
    if not v:
        return None
    s = sorted(v)
    return round(s[min(len(s) - 1, int(round(q / 100.0 * (len(s) - 1))))], 3)


def latency_stats(v: List[float]) -> Optional[Dict]:
    if not v:
        return None
    return {"n": len(v), "p50": pct(v, 50), "p99": pct(v, 99), "p99.9": pct(v, 99.9),
            "max": round(max(v), 3), "over_20.48ms": sum(1 for x in v if x > PERIOD_MS)}


def write_evidence(out_dir: str, client: RamRealtimeDemoClient, results: List[Dict],
                   events: List[Dict], extra: Optional[Dict] = None) -> Dict:
    """Save PC-side records + pull the board run dir (raw npz, summary, guard log)."""
    os.makedirs(out_dir, exist_ok=True)
    cmap = clock_map(client.sync)
    rec = {"mode": MODE, "scope": SCOPE_NOTE, "set": SET_NAME, "n_total": client.n_total,
           "warmup": WARMUP, "sched": "SCHED_FIFO %d, CPU %d (board main thread only)" % (PRIO, CPU),
           "board_run_dir": client.run_dir(), "results_received": len(results),
           "clock_map": cmap, "sync_rounds": client.sync}
    if cmap:
        post = [r for r in results if r["k"] >= WARMUP]
        rec["pc_receipt_latency_ms"] = latency_stats(e2e_latencies(post, cmap, "rx_pc"))
        rec["gui_display_latency_ms"] = latency_stats(e2e_latencies(post, cmap, "disp_pc"))
    rec.update(extra or {})
    with open(os.path.join(out_dir, "pc_results.tsv"), "w", encoding="utf-8") as f:
        f.write("k\tidx\tcls\tboard_lat_ms\tlogits\trel_b\trx_pc\tdisp_pc\n")
        for r in results:
            f.write("%d\t%d\t%d\t%.3f\t%s\t%.7f\t%.7f\t%s\n" % (
                r["k"], r["idx"], r["cls"], r["lat_ms"], ",".join(map(str, r["logits"] or [])),
                r["rel_b"], r["rx_pc"], "" if r.get("disp_pc") is None else "%.7f" % r["disp_pc"]))
    with open(os.path.join(out_dir, "events_nonresult.jsonl"), "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    scp = ["scp", "-r", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-P",
           str(client.port), "-i", client.key]
    r = subprocess.run(scp + ["%s@%s:%s" % (client.user, client.host, client.run_dir()),
                              os.path.join(out_dir, "remote")],
                       capture_output=True, text=True, creationflags=_CF)
    rec["remote_copy_rc"] = r.returncode
    procs = _ssh("ps -eo pid=,args= | grep -E 'python3[.]6 .*[f]pga_(ram_rt_demo|live_runner|rt_acceptance)|[t]emp_guard[.]sh' || echo BOARD_IDLE")
    rec["post_run_board_processes"] = procs.stdout.strip()
    with open(os.path.join(out_dir, "pc_evidence.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RAM real-time demo mode (real board).")
    ap.add_argument("--n", type=int, default=1200, help="windows incl. 200 warm-up")
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--stop-after-s", type=float, default=None,
                    help="request Stop after this many seconds (Stop test)")
    ap.add_argument("--out-dir")
    args = ap.parse_args(argv)
    print("staging set:", stage_set())
    if args.stage_only:
        return 0
    out = args.out_dir or os.path.join(lf._REPO, "artifacts", "results",
                                       "ram_rt_demo_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    runner, guard = push_runner()
    client = RamRealtimeDemoClient(n_total=args.n, runner_remote=runner, monitor_remote=guard,
                                   line_timeout=60.0)
    if args.stop_after_s:
        threading.Timer(args.stop_after_s, client.stop).start()
    results, events = [], []
    for ev in client.stream():
        if ev["type"] == "result":
            results.append(ev)
        else:
            events.append(ev)
            print(json.dumps(ev)[:300], flush=True)
    rec = write_evidence(out, client, results, events)
    print(json.dumps({k: rec.get(k) for k in ("results_received", "clock_map", "pc_receipt_latency_ms",
                                              "post_run_board_processes")}, indent=1))
    print("evidence:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
