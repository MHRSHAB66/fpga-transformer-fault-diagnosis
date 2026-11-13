"""S01: interference envelope of the board-side hard real-time path.

Same workload and boundary as the hard real-time acceptance runs (RAM windows
released every 20.48 ms, SCHED_FIFO 50 on CPU1, response = done - release,
4,096 unique raw windows, 30,000 + 200 warm-up), repeated while a bounded
background load runs (on CPU0 where it can be pinned):

  quiet       no added load (same-session control)
  cpu0_busy   Python busy loop pinned to CPU0
  mem_copy    numpy copy of two 32 MB arrays in a loop on CPU0 (DDR bandwidth, shared with DMA)
  cache_thrash random read-modify-write over 64 MB on CPU0 (shared L2 + DDR)
  sd_io       256 MB write with fsync + read back, in a loop on CPU0 (SD card / mmc IRQ)
  net_rx      the PC streams 4 MiB/s into the board over SSH (Ethernet IRQ + sshd);
              link pinned to 100 Mb/s (an unbounded 1 Gb/s stream dropped the link)
  combined    mem_copy + sd_io + net_rx together
Every output is checked against the canonical C-model; the temperature guard
stops at 75 C. Not a WCET proof: a finite stress envelope.

    python -m tools.research.s01_interference --name seq256_vote_z7020_20261110 \
        --out artifacts/research/S01_20261113 [--conditions ...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from tools.raw_core_board_test import (BOARD_RAW_GZ, BOARD_TOOLS, DEPLOYMENTS, HOST, KEY, REPO, fetch, guarded,
                                       scp, ssh, wait_pid)

CMODEL = REPO / "artifacts/results/onchip_preprocess_20261109/csim_raw_argmax/csim_logits.txt"
STRESS = {
    "cpu0_busy": "taskset -c 0 python3.6 -c 'while True: pass'",
    "mem_copy": "taskset -c 0 python3.6 -c 'import numpy as np\na=np.ones(8<<20,np.float32); b=np.empty_like(a)\nwhile True: np.copyto(b,a)'",
    "cache_thrash": "taskset -c 0 python3.6 -c 'import numpy as np\na=np.ones(16<<20,np.int32); i=np.random.randint(0,len(a),1<<20)\nwhile True: a[i]+=1'",
    "sd_io": "taskset -c 0 sh -c 'while :; do dd if=/dev/zero of=/home/xilinx/s01_io.bin bs=1M count=256 conv=fsync 2>/dev/null; dd if=/home/xilinx/s01_io.bin of=/dev/null bs=1M 2>/dev/null; done'",
}
COMBINED = ["mem_copy", "sd_io"]   # + net_rx from the PC
NET_RATE = 4 << 20                  # bytes/s (32 Mb/s) for net_rx / combined


def start_board_stress(names, run):
    pgids = []
    for n in names:
        cmd = "setsid nohup %s > %s/stress_%s.log 2>&1 < /dev/null & echo $!" % (STRESS[n], run, n)
        pgids.append(ssh(cmd).stdout.strip())
    return pgids


def stop_board_stress(pgids):
    for g in pgids:
        ssh("kill -TERM -- -%s 2>/dev/null; kill -TERM %s 2>/dev/null; true" % (g, g), check=False)
    ssh("rm -f /home/xilinx/s01_io.bin", check=False)


class NetStream:
    """PC -> board over SSH at a bounded rate (bytes/s) until stopped. An
    unbounded stream at 1 Gb/s dropped the board's Ethernet link on 2026-11-13
    (known link instability at 1 Gb/s), so the load is bounded and the link is
    pinned to 100 Mb/s before the run."""

    def __init__(self, rate=NET_RATE):
        self.rate = rate
        self.proc = subprocess.Popen(["ssh", "-i", KEY, "-o", "BatchMode=yes", HOST, "cat > /dev/null"],
                                     stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.sent, self.stop = 0, False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        block = bytes(64 << 10)
        t0 = time.time()
        try:
            while not self.stop:
                if self.sent > self.rate * (time.time() - t0):
                    time.sleep(0.005)
                    continue
                self.proc.stdin.write(block)
                self.sent += len(block)
        except (BrokenPipeError, OSError):
            pass

    def close(self):
        self.stop = True
        self.t.join(timeout=10)
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.terminate()
        return self.sent


def run_condition(cond, dep, out_root, n_total):
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = "/home/xilinx/s01_%s_%s" % (cond, tag)
    busy = ssh("pgrep -fa '[f]pga_live_runner|[f]pga_rt_acceptance|[v]erify_argmax_golden' || true").stdout.strip()
    if busy:
        raise SystemExit("board busy:\n" + busy)
    ssh("mkdir %s" % run)
    scp([REPO / "tools" / "board" / t for t in BOARD_TOOLS], run + "/")
    ssh("sed -i 's/\\r$//' %s/temp_guard.sh; chmod +x %s/temp_guard.sh" % (run, run))
    names = COMBINED if cond == "combined" else ([cond] if cond in STRESS else [])
    pgids = start_board_stress(names, run)
    net = NetStream() if cond in ("net_rx", "combined") else None
    time.sleep(5)
    t0 = time.time()
    pid = guarded(dep, run, "python3.6 -u {run}/fpga_rt_acceptance.py {dep}/overlay/design_1_wrapper.bit {gz} "
                  "4096 {n} 200 1 1 50 {run}/out".format(run=run, dep=dep, gz=BOARD_RAW_GZ, n=n_total), "rt", background=True)
    wait_pid(pid, poll=30)
    sent = net.close() if net else 0
    stop_board_stress(pgids)
    out = out_root / cond
    out.mkdir(parents=True, exist_ok=True)
    for f in ("out", "rt.out", "rt_guard.log"):
        fetch("%s/%s" % (run, f), out)
    for n in names:
        fetch("%s/stress_%s.log" % (run, n), out)
    summary = json.loads((out / "out" / "summary.json").read_text())
    r = np.load(out / "out" / "rt_raw.npz")
    ref = np.loadtxt(CMODEL, dtype=np.int64)
    w = r["window_index"]
    resp = (r["done"] - r["release"]) * 1000
    temps = [float(l.split("temp_c=")[1]) for l in (out / "rt_guard.log").read_text().splitlines() if "temp_c=" in l]
    res = {"condition": cond, "board_run": run, "stress": [STRESS[n] for n in names] + (["PC->board SSH stream"] if net else []),
           "net_bytes_sent": sent, "wall_s": round(time.time() - t0, 1),
           "executed": summary["executed"], "aborted": summary["aborted"], "deadline_misses": summary["deadline_misses"],
           "response_ms": summary["response_ms"], "response_ms_max": summary["response_ms_max"],
           "copy_mm2s_ms_p99": round(float(np.percentile((r["send_done"] - r["start"])[200:] * 1000, 99)), 3),
           "core_s2mm_ms_p99": round(float(np.percentile((r["done"] - r["send_done"])[200:] * 1000, 99)), 3),
           "start_jitter_ms_p99": round(float(np.percentile((r["start"] - r["release"])[200:] * 1000, 99)), 3),
           "involuntary_ctx_switches": summary["involuntary_ctx_switches"],
           "class_equal_cmodel": int((r["cls"] == ref[w, 5]).sum()),
           "logits_equal_cmodel": int((r["logits"].astype(np.int64) == ref[w, 1:5]).all(1).sum()),
           "windows": int(len(w)), "temp_max_c": max(temps) if temps else None,
           "response_recomputed_max_ms": round(float(resp[200:].max()), 3)}
    (out / "condition_summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("condition", "deadline_misses", "response_ms_max", "start_jitter_ms_p99",
                                          "copy_mm2s_ms_p99", "core_s2mm_ms_p99", "logits_equal_cmodel", "temp_max_c")}), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True, help="deployment directory name on the board")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-total", type=int, default=30200)
    ap.add_argument("--conditions", nargs="+",
                    default=["quiet", "cpu0_busy", "mem_copy", "cache_thrash", "sd_io", "net_rx", "combined"])
    args = ap.parse_args()
    dep = "%s/%s" % (DEPLOYMENTS, args.name)
    results = {}
    for c in args.conditions:
        results[c] = run_condition(c, dep, args.out, args.n_total)
        (args.out / "summary.json").write_text(json.dumps({"id": "S01", "deployment": args.name, "conditions": results}, indent=1))


if __name__ == "__main__":
    main()
