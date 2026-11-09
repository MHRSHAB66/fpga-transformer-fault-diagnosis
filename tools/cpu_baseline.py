"""CPU baseline for the deployed INT10 model: same C++ source on the board's ARM
and on the PC CPU, compared with the FPGA.

The deployed HLS top (argmax variant) is compiled as ordinary C++ and timed per
window on the 256-window class-balanced RAM demo set. Its logits must equal the
physical board logits of the same dataset windows, so the CPU and FPGA results
compare the same computation.

    python -m tools.cpu_baseline board      # push, build and run on the Z-Turn (needs the board)
    python -m tools.cpu_baseline pc         # build and run on this PC (g++ on PATH)
    python -m tools.cpu_baseline summarize  # verify logits, write summary.json

Outputs: artifacts/results/cpu_baseline_20261109/{board,pc}/ and summary.json.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "results" / "cpu_baseline_20261109"
SET_DIR = REPO / "artifacts" / "ram_demo_sets" / "class_balanced_256_v1"
VITIS = REPO / "hls" / "f1_int10_seq256" / "vitis"
STUBS = REPO / "hls" / "f1_int10" / "gpp_csim" / "stubs"
BENCH = REPO / "hls" / "f1_int10_seq256" / "gpp_csim" / "cpu_baseline_bench.cpp"
TOP = VITIS / "transformer_top_seq256_ingress1024_dsp_margin_argmax.cpp"
BOARD_LOGITS = (REPO / "artifacts" / "results" / "fpga_realtime_bench_20261022"
                / "board_full_pipelined" / "logits_i32.bin")
ARGMAX_FULLTEST = REPO / "artifacts" / "results" / "fpga_argmax_fulltest_20261023" / "live_evidence.json"
HARD_RT_R1 = (REPO / "artifacts" / "results" / "hard_rt_20261024" / "R1_fifo50_cpu1"
              / "remote" / "out" / "summary.json")
HOST, KEY = "root@192.168.2.99", str(Path.home() / ".ssh" / "id_zturn")
REMOTE = "/home/xilinx/cpu_baseline_20261109"
N_WINDOWS, REPEATS = 256, 1


def ssh(cmd, **kw):
    return subprocess.run(["ssh", "-i", KEY, "-o", "BatchMode=yes", HOST, cmd], **kw)


def scp(srcs, dst):
    subprocess.run(["scp", "-q", "-i", KEY] + [str(s) for s in srcs] + [dst], check=True)


def run_board():
    if ssh("test -e %s" % REMOTE).returncode == 0:
        sys.exit("remote dir %s exists; refusing to overwrite" % REMOTE)
    ssh("mkdir -p %s/src %s/stubs" % (REMOTE, REMOTE), check=True)
    scp([TOP, VITIS / "ap_types.h", VITIS / "weights_int10.h", VITIS / "argmax_first_max.h", BENCH],
        "%s:%s/src/" % (HOST, REMOTE))
    scp(sorted(STUBS.glob("*.h")), "%s:%s/stubs/" % (HOST, REMOTE))
    scp([SET_DIR / "set.i32", REPO / "tools" / "board" / "run_cpu_baseline.sh",
         REPO / "tools" / "board" / "temp_guard.sh"], "%s:%s/" % (HOST, REMOTE))
    ssh("cd %s && chmod +x run_cpu_baseline.sh temp_guard.sh && sed -i 's/\\r$//' *.sh && "
        "./run_cpu_baseline.sh %d %d" % (REMOTE, N_WINDOWS, REPEATS), check=True)
    dest = OUT / "board"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["scp", "-q", "-r", "-i", KEY, "%s:%s/out/." % (HOST, REMOTE), str(dest)],
                   check=True)
    print("board results in", dest)


def run_pc():
    gxx = shutil.which("g++")
    if not gxx:
        sys.exit("g++ not on PATH")
    dest = OUT / "pc"
    dest.mkdir(parents=True, exist_ok=True)
    info = [subprocess.run([gxx, "--version"], capture_output=True, text=True).stdout.splitlines()[0]]
    cpu = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True)
    info.append(cpu.stdout.strip())
    for name, flags in (("O2", ["-O2"]), ("O3native", ["-O3", "-march=native"])):
        exe = dest / ("bench_%s.exe" % name)
        subprocess.run([gxx] + flags + ["-std=c++17", "-Wno-unknown-pragmas", "-DWITH_PRED_CLASS",
                        "-I", str(STUBS), "-I", str(VITIS), str(TOP), str(BENCH), "-o", str(exe)],
                       check=True)
        info.append("build %s: g++ %s" % (name, " ".join(flags)))
        with open(dest / ("%s_single.txt" % name), "w") as fo, \
                open(dest / ("%s_single.sum" % name), "w") as fe:
            subprocess.run([str(exe), str(SET_DIR / "set.i32"), str(N_WINDOWS), "4"],
                           stdout=fo, stderr=fe, check=True)
        exe.unlink()
    (dest / "system.txt").write_text("\n".join(info) + "\n")
    print("pc results in", dest)


def parse_sum(path):
    m = re.search(r"SUMMARY (.*)", path.read_text())
    return {k: float(v) for k, v in (kv.split("=") for kv in m.group(1).split())}


def check_logits(path, ref):
    rows = np.loadtxt(path, dtype=np.float64)
    logits = rows[:, 1:5].astype(np.int64)
    pred = rows[:, 5].astype(np.int64)
    ok = int((logits == ref).all(axis=1).sum())
    return {"windows": int(len(rows)), "logits_equal_board": ok,
            "pred_equal_board_argmax": int((pred == ref.argmax(axis=1)).sum())}


def summarize():
    idx = np.loadtxt(SET_DIR / "indices.txt", dtype=np.int64)
    ref = np.fromfile(BOARD_LOGITS, dtype="<i4").reshape(-1, 4)[idx].astype(np.int64)
    fpga_live = json.loads(ARGMAX_FULLTEST.read_text())
    r1 = json.loads(HARD_RT_R1.read_text())
    fpga_ms = r1["response_ms"]["p50"]
    out = {"model": "deployed SEQ=256 INT10 top, argmax variant, compiled as C++",
           "windows": "artifacts/ram_demo_sets/class_balanced_256_v1 (256 windows, 64 per class)",
           "fpga_reference": {
               "board_stage_p50_ms_release_to_class_ready": fpga_ms,
               "dma_plus_core_mean_ms_full_test": fpga_live["latency_ms"]["mean"],
               "note": "FPGA times include the 32 KB DMA transfer; CPU times are compute only"},
           "runs": {}}
    for where in ("board", "pc"):
        d = OUT / where
        if not d.exists():
            continue
        for s in sorted(d.glob("*.sum")):
            name = s.stem
            stats = parse_sum(s)
            entry = {"timing_us": stats, **check_logits(s.with_suffix(".txt"), ref)}
            entry["speedup_fpga_vs_cpu_p50"] = round(stats["p50_us"] / 1000 / fpga_ms, 2)
            out["runs"]["%s/%s" % (where, name)] = entry
        wall = d / "dual_wall.txt"
        if wall.exists():
            for line in wall.read_text().splitlines():
                name, _, sec = line.split()
                n = 2 * N_WINDOWS * REPEATS
                out["runs"]["%s/%s_dual_throughput" % (where, name)] = {
                    "inferences": n, "wall_s": float(sec),
                    "inferences_per_s_incl_warmup": round(2 * 2 * N_WINDOWS / float(sec), 2),
                    "note": "two processes (CPU 0 and CPU 1); wall time includes each process's "
                            "untimed warm-up pass, so the rate counts warm-up + timed passes"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["board", "pc", "summarize"])
    {"board": run_board, "pc": run_pc, "summarize": summarize}[ap.parse_args().what]()


if __name__ == "__main__":
    main()
