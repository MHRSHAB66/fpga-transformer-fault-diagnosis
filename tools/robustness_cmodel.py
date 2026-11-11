"""Robustness of the deployed raw-input pipeline to sensor noise and sensor failure.

Runs the g++ C-model of the raw-input HLS top (bit-exact with the physical board
on all 95,404 test windows) over the raw test input with a perturbation applied
to every window BEFORE it enters the core, then scores the predictions:

  clean          no change (must reproduce 97.7705 %)
  snr<k>         additive white Gaussian noise per window and channel at an SNR of
                 k dB relative to that channel's AC power in the window
                 (P_noise = P_signal / 10^(k/10)), then rounded and clipped to the
                 24-bit ADC range; fixed seed per window
  dead<c>        channel c reads 0 for the whole test (dead sensor / cut cable)
  dead<a>+<b>    two channels dead

The model is NOT retrained or augmented; this measures the deployed system.

    python -m tools.robustness_cmodel --exe tmp/csim/batch_raw_argmax.exe \
        --conditions clean snr20 snr10 snr5 snr0 dead0 dead7 --out artifacts/results/robustness_20261110
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, recall_score

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "artifacts" / "results" / "fpga_rawinput_testset_20261109" / "fpga_testset_raw.i32.gz"
LABELS = REPO / "artifacts" / "results" / "fpga_rawinput_testset_20261109" / "labels.txt"
WORDS = 1024 * 8
REC = WORDS * 4
CODE_MIN, CODE_MAX = -(1 << 23), (1 << 23) - 1
CHANNELS = ["tachometer", "acc_axial_b1", "acc_radial_b1", "acc_tangential_b1",
            "acc_axial_b2", "acc_radial_b2", "acc_tangential_b2", "microphone"]


def perturb(windows: np.ndarray, first_index: int, cond: str) -> np.ndarray:
    """windows: [k, 1024, 8] int64 raw codes -> perturbed int32 codes."""
    x = windows.astype(np.float64)
    if cond.startswith("snr"):
        snr_db = float(cond[3:])
        for j in range(len(x)):
            rng = np.random.default_rng([20261110, first_index + j])
            p_sig = x[j].var(axis=0)                                 # AC power per channel
            std = np.sqrt(p_sig / 10 ** (snr_db / 10))
            x[j] += rng.standard_normal(x[j].shape) * std
    elif cond.startswith("dead"):
        for c in (int(v) for v in cond[4:].split("+")):
            x[:, :, c] = 0
    elif cond != "clean":
        raise ValueError(cond)
    return np.clip(np.rint(x), CODE_MIN, CODE_MAX).astype("<i4")


def run_shard(exe, start, count, cond, out):
    proc = subprocess.Popen([str(exe), str(start), str(count)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def feed():
        with gzip.open(RAW, "rb") as f:
            skip = start * REC
            while skip:
                skip -= len(f.read(min(skip, 64 * REC)))
            done = 0
            while done < count:
                k = min(64, count - done)
                w = np.frombuffer(f.read(k * REC), dtype="<i4").astype(np.int64).reshape(k, 1024, 8)
                proc.stdin.write(perturb(w, start + done, cond).tobytes())
                done += k
        proc.stdin.close()

    t = threading.Thread(target=feed)
    t.start()
    stdout = proc.stdout.read().decode()
    t.join()
    proc.wait()
    out.extend(stdout.strip().splitlines())


def score(lines, labels):
    rows = np.array([[int(v) for v in l.split()] for l in lines], dtype=np.int64)
    rows = rows[np.argsort(rows[:, 0])]
    pred = rows[:, 5]
    y = labels[rows[:, 0]]
    return {"windows": int(len(rows)), "accuracy": round(float((pred == y).mean()), 6),
            "macro_f1": round(float(f1_score(y, pred, average="macro")), 6),
            "recall_per_class": [round(float(r), 4) for r in recall_score(y, pred, average=None)]}, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=95404)
    ap.add_argument("--shards", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-note", default="raw-input HLS top, g++ C-model (bit-exact with the board)",
                    help="recorded in summary.json _meta (describe a non-deployed variant here)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    labels = np.loadtxt(LABELS, dtype=np.int64)
    summary_path = args.out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    per = -(-args.n // args.shards)
    for cond in args.conditions:
        t0 = time.time()
        lines, threads = [], []
        for s in range(args.shards):
            start, count = s * per, min(per, args.n - s * per)
            if count <= 0:
                continue
            part = []
            lines.append(part)
            th = threading.Thread(target=run_shard, args=(args.exe, start, count, cond, part))
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        flat = [l for part in lines for l in part]
        result, rows = score(flat, labels)
        result["seconds"] = round(time.time() - t0, 1)
        summary[cond] = result
        np.savetxt(args.out / ("pred_%s.txt" % cond), rows[:, [0, 5]], fmt="%d")
        summary_path.write_text(json.dumps(summary, indent=1))
        print(cond, json.dumps(result), flush=True)
    summary["_meta"] = {"model": args.model_note,
                        "exe": str(args.exe), "channels": CHANNELS,
                        "noise": "AWGN per window/channel, SNR vs AC power, seed [20261110, window]"}
    summary_path.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
