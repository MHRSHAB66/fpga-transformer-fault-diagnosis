"""Check the on-FPGA preprocessing C++ (onchip_preprocess.h) against the Python
reference, bit for bit, on real raw windows plus constructed edge cases.

Builds hls/f1_int10_seq256/gpp_csim/dump_raw_ingress.cpp with g++ (functional
C-model, not synthesis), feeds it the windows and compares every one of the
256 x 8 INT10 tokens with
    aggregate_round_div(preprocess_window(raw), 4).

    python -m tools.verify_onchip_preprocess --raw <fpga_testset_raw.i32.gz> --n 2000
    python -m tools.verify_onchip_preprocess --edge-only
"""
import argparse
import gzip
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from src.data.token_aggregation import aggregate_round_div
from src.quantize.onchip_preprocess import CODE_MAX, CODE_MIN, preprocess_window

REPO = Path(__file__).resolve().parents[1]
INPUT_SCALE = 0.028626115354773118   # deployed model_manifest.json input_scale
SRC = REPO / "hls" / "f1_int10_seq256" / "gpp_csim" / "dump_raw_ingress.cpp"
INCLUDES = [REPO / "hls" / "f1_int10" / "gpp_csim" / "stubs", REPO / "hls" / "f1_int10_seq256" / "vitis"]
# Xilinx ap_int headers: with them the C++ is also checked at the real hardware
# bit widths (-DHLS_USE_AP_INT), not only with wide native integers.
XILINX_INCLUDE = Path("G:/Xilinx/Vitis_HLS/2020.2/include")


def edge_windows(rng):
    w = []
    base = rng.normal(0, 2e5, size=(1024, 8)).round().astype(np.int64)
    w.append(base)
    z = base.copy(); z[:, 3] = 12345; w.append(z)                       # constant channel
    w.append(np.zeros((1024, 8), dtype=np.int64))                         # all zero
    sq = np.where(np.arange(1024)[:, None] % 2 == 0, CODE_MAX, CODE_MIN) * np.ones((1, 8), dtype=np.int64)
    w.append(sq)                                                          # full-scale square wave
    tiny = rng.integers(-1, 2, size=(1024, 8)); w.append(tiny)            # +-1 LSB noise
    out = base.copy(); out[0, :] = 1 << 30; out[1, :] = -(1 << 30); w.append(out)  # clamped words
    spike = np.zeros((1024, 8), dtype=np.int64); spike[500, :] = 4_000_000; w.append(spike)
    offset = base + 6_000_000; w.append(np.clip(offset, CODE_MIN, CODE_MAX))
    for _ in range(12):
        scale = 10 ** rng.uniform(0, 6.5)
        w.append(np.clip(rng.normal(rng.uniform(-1e6, 1e6), scale, size=(1024, 8)).round(),
                         CODE_MIN, CODE_MAX).astype(np.int64))
    return np.stack(w)


def read_raw(path, n):
    with gzip.open(path, "rb") as f:
        data = f.read(n * 8192 * 4)
    return np.frombuffer(data, dtype="<i4").reshape(-1, 1024, 8).astype(np.int64)


def build(tmp, ap_int=False):
    gxx = shutil.which("g++")
    if not gxx:
        raise SystemExit("g++ not on PATH")
    exe = Path(tmp) / ("dump_raw_ingress%s.exe" % ("_apint" if ap_int else ""))
    cmd = [gxx, "-O2", "-std=c++14" if ap_int else "-std=c++17", "-w"]
    for inc in INCLUDES:
        cmd += ["-I", str(inc)]
    if ap_int:
        cmd += ["-DHLS_USE_AP_INT", "-I", str(XILINX_INCLUDE)]
    subprocess.run(cmd + [str(SRC), "-o", str(exe)], check=True)
    return exe


def run_cpp(exe, windows, tmp):
    path = Path(tmp) / "windows.i32"
    windows.astype("<i4").tofile(path)
    out = subprocess.run([str(exe), str(path), str(len(windows))], capture_output=True, text=True,
                         check=True).stdout.split("\n")
    return np.array([[int(v) for v in line.split()] for line in out if line.strip()],
                    dtype=np.int64).reshape(len(windows), 256, 8)


def reference(windows):
    return np.stack([aggregate_round_div(preprocess_window(w, INPUT_SCALE)[None], 4)[0]
                     for w in windows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", help="fpga_testset_raw.i32.gz")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--edge-only", action="store_true")
    ap.add_argument("--out", help="write a JSON result here")
    args = ap.parse_args()
    rng = np.random.default_rng(20261109)
    sets = {"edge_cases": edge_windows(rng)}
    if not args.edge_only:
        sets["real_windows"] = read_raw(args.raw, args.n)
    result = {}
    builds = [("native", False)] + ([("ap_int", True)] if XILINX_INCLUDE.exists() else [])
    with tempfile.TemporaryDirectory() as tmp:
        for bname, ap in builds:
            exe = build(tmp, ap)
            for name, w in sets.items():
                got, ref = run_cpp(exe, w, tmp), reference(w)
                bad = int((got != ref).any(axis=(1, 2)).sum())
                result["%s/%s" % (bname, name)] = {
                    "windows": int(len(w)), "windows_with_any_token_mismatch": bad,
                    "tokens_compared": int(ref.size)}
    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    if any(v["windows_with_any_token_mismatch"] for v in result.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
