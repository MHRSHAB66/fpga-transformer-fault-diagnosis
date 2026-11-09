"""End-to-end live-path throughput: board runner -> SSH/Ethernet -> PC parse/frame.

Runs a board-side live runner over the first N CONTIGUOUS test-set windows
(the same input file and deployed bitstream as the validated live demo) and
consumes its stdout on the PC with the production parser
(host.live_fpga.parse_line -> FRAME/CLASS/LABEL/CONF line). The independent
board temperature guard (tools/board/temp_guard.sh, stop at 75 C) watches the
runner PID; nothing is retried after a stop.

Measured on the PC:
  * receive throughput = results / (t_last_result - t_first_result)
  * per-line PC parse+frame time
  * correctness vs the recorded full-test FPGA predictions (fpga_pred.txt)
The GUI (Tk) rendering is NOT included.

Usage:
    python -m tools.measure_live_e2e --runner-remote /home/xilinx/fpga_live_runner.py \
        --n 10000 --tag before --out artifacts/results/fpga_realtime_bench_20261022/e2e
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from host.live_fpga import BIT, BOARD_HOST, BOARD_USER, DEP_DIR, IN_GZ, SSH_KEY, _ssh_base, parse_line

ROOT = Path(__file__).resolve().parents[1]
FT = ROOT / "artifacts" / "results" / "fpga_fulltest_20261008_141335"
GUARD = "/home/xilinx/bench_20261022/temp_guard.sh"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runner-remote", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample-mode", choices=["sequential", "class_balanced"], default="sequential",
                    help="class_balanced uses host.live_sampling.build_plan (same plan as the live demo); "
                         "the index file is written to the bench dir, not the demo's index file")
    args = ap.parse_args()
    out = args.out / args.tag
    out.mkdir(parents=True, exist_ok=False)

    drv = DEP_DIR + "/driver"
    idx_arg = ""
    if args.sample_mode != "sequential":
        from host.live_sampling import build_plan
        labels = np.loadtxt(FT / "labels.txt", dtype=np.int64).tolist()
        plan = build_plan(args.sample_mode, total=len(labels), n=args.n, labels=labels)
        (out / "indices.txt").write_text("\n".join(map(str, plan)) + "\n", encoding="ascii")
        remote_idx = "/home/xilinx/bench_20261022/e2e_%s_indices.txt" % args.tag
        subprocess.run(["scp", "-o", "BatchMode=yes", "-i", SSH_KEY, str(out / "indices.txt"),
                        "%s@%s:%s" % (BOARD_USER, BOARD_HOST, remote_idx)], check=True)
        idx_arg = " %s" % remote_idx
    glog = "/home/xilinx/bench_20261022/e2e_%s_guard.log" % args.tag
    # busy check anchored on the python command so it cannot match this shell
    remote = ("if ps -eo args= | grep -qE '^python3[.]6 .*(fpga_live_runner|fpga_stage_bench)'; "
              "then echo BUSY; exit 3; fi; "
              "cd %s && PYTHONPATH=%s python3.6 -u %s %s %s 1024 8 %d -%s & P=$!; "
              "bash %s $P %s 75 1 & G=$!; wait $P; RC=$?; wait $G; echo \"RUNNER_RC=$RC GUARD_RC=$?\""
              % (drv, drv, args.runner_remote, BIT, IN_GZ, args.n, idx_arg, GUARD, glog))
    argv = _ssh_base(BOARD_HOST, BOARD_USER, SSH_KEY, timeout=15) + [remote]
    recorded = np.loadtxt(FT / "fpga_pred.txt", dtype=np.int64)

    t_rx, parse_s, results, other = [], [], [], []
    t0 = time.perf_counter()
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        now = time.perf_counter()
        a = time.perf_counter()
        ev = parse_line(line)
        parse_s.append(time.perf_counter() - a)
        if ev and ev["type"] == "result":
            t_rx.append(now)
            results.append(ev)
        elif ev:
            other.append(line.strip())
    rc = proc.wait()
    total = time.perf_counter() - t0

    n = len(results)
    idx = np.array([r["idx"] for r in results], dtype=np.int64)
    cls = np.array([r["cls"] for r in results], dtype=np.int64)
    span = (t_rx[-1] - t_rx[0]) if n > 1 else 0.0
    lat = np.array([r["lat_ms"] for r in results])
    gaps = np.diff(np.array(t_rx)) * 1e3 if n > 1 else np.array([0.0])
    confs = [r.get("conf") for r in results]
    summ = {
        "tag": args.tag, "runner_remote": args.runner_remote, "requested": args.n,
        "received_results": n, "ssh_rc": rc,
        "sample_mode": args.sample_mode,
        "contiguous_from_0": bool(n and (idx == np.arange(n)).all()),
        "accuracy_vs_labels": round(float((np.loadtxt(FT / "labels.txt", dtype=np.int64)[idx] == cls).mean()), 6) if n else None,
        "match_recorded_fpga_pred": int((recorded[idx] == cls).sum()) if n else 0,
        "pc_receive_throughput_inf_s": round((n - 1) / span, 3) if span else None,
        "wall_incl_ssh_setup_overlay_load_s": round(total, 3),
        "pc_inter_arrival_ms": {"mean": round(float(gaps.mean()), 3),
                                "p50": round(float(np.percentile(gaps, 50)), 3),
                                "p95": round(float(np.percentile(gaps, 95)), 3),
                                "p99": round(float(np.percentile(gaps, 99)), 3),
                                "max": round(float(gaps.max()), 3)},
        "board_lat_ms": {"mean": round(float(lat.mean()), 3),
                         "p50": round(float(np.percentile(lat, 50)), 3),
                         "p95": round(float(np.percentile(lat, 95)), 3)} if n else None,
        "pc_parse_us": {"mean": round(float(np.mean(parse_s)) * 1e6, 2),
                        "max": round(float(np.max(parse_s)) * 1e6, 2)},
        "conf_present": int(sum(c is not None for c in confs)),
        "board_lines": other[-4:],
        "excluded": "Tk GUI rendering is not part of this measurement",
    }
    (out / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    with open(out / "frames.txt", "w", encoding="utf-8") as f:
        for r in results:
            f.write(r["frame"] + "\n")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
