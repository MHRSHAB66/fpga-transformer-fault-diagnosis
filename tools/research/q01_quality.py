"""Q01: input quality and decision reliability (offline, bit-exact C-model).

Acquisition faults are applied to the raw 24-bit ADC codes BEFORE the on-FPGA
normalisation (the deployed raw-input path), on one randomly chosen vibration or
microphone channel per window (the tachometer is never perturbed: its pulses
have long flat segments by design):
  gain2 / offset   positive gain x2 and a +25 % full-scale DC offset, clipped to the
                   24-bit range: invariance controls (z-score removes both unless clipping)
  clip30           clipping at +-30 % of the channel's peak deviation in the window
  bits8            reduced effective ADC precision: 16 low bits zeroed (8 effective bits)
  hold             intermittent sample-and-hold: runs of 16 repeated samples covering ~50 %
  lag32            the channel delayed by 32 samples (0.64 ms) relative to the others
Methods (thresholds from clean VALIDATION only; outputs stay the 4 classes, the
status is metadata):
  none        accept everything
  conf        calibrated confidence (softmax(logits/T), T = 1.4248) >= t
  margin      integer top-1 minus top-2 logit >= t
  flags       pre-normalisation quality flags: clipped fraction, longest constant run,
              low-bit occupancy (per channel, thresholds = clean-validation extremes)
  flags+margin
For each method: clean coverage and risk, accepted wrong outputs and accepted
false-healthy outputs under each corruption.

    python -m tools.research.q01_quality --out artifacts/research/Q01_20261113
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
VAL = REPO / "artifacts/results/fpga_rawinput_valset_20261113"
TEST = REPO / "artifacts/results/fpga_rawinput_testset_20261109"
EXE = (REPO / "tmp/csim/batch_raw_argmax.exe").resolve()
CONTRACT = REPO / "artifacts/preprocessing/raw_input_contract_v1.json"
CODE_MAX, CODE_MIN = (1 << 23) - 1, -(1 << 23)
T_CONF = 1.4248
SEED = 20261113
CORRUPTIONS = ["clean", "gain2", "offset", "clip30", "bits8", "hold", "lag32"]


def read_windows(gz_path, idx):
    """Windows idx (sorted) from a raw gz file (8192 int32 words per window)."""
    out, want, k = [], set(int(i) for i in idx), 0
    with gzip.open(gz_path, "rb") as f:
        while True:
            b = f.read(32768)
            if not b:
                break
            if k in want:
                out.append(np.frombuffer(b, dtype="<i4").reshape(1024, 8).astype(np.int64))
            k += 1
    return np.stack(out)


def perturb(w, cond, ch, fs):
    x = w.copy()
    c = x[:, ch]
    if cond == "gain2":
        c *= 2
    elif cond == "offset":
        c += int(0.25 * (1 << 23))                      # +25 % of the 24-bit code range
    elif cond == "clip30":
        # saturation relative to the actual signal (the 24-bit full scale is far above
        # typical amplitudes, so a fixed fraction of it does not clip at all)
        med = int(np.median(c))
        lim = max(1, int(0.30 * np.abs(c - med).max()))
        c[:] = np.clip(c, med - lim, med + lim)
    elif cond == "bits8":
        c[:] = (c >> 16) << 16
    elif cond == "hold":
        for s in range(0, 1024, 32):
            c[s + 1:s + 16] = c[s]
    elif cond == "lag32":
        c[:] = np.concatenate([np.full(32, c[0]), c[:-32]])
    x[:, ch] = np.clip(c, CODE_MIN, CODE_MAX)
    return x


def quality_features(w):
    """Per-window, per-channel cheap features of the raw codes."""
    clipped = ((w >= CODE_MAX) | (w <= CODE_MIN)).mean(axis=0)
    same = (np.diff(w, axis=0) == 0)
    longest = np.zeros(w.shape[1], dtype=np.int64)
    for ch in range(w.shape[1]):
        pos = np.concatenate([[-1], np.flatnonzero(~same[:, ch]), [same.shape[0]]])
        longest[ch] = int(np.diff(pos).max() - 1)
    lowbits = ((w & 0xFFFF) != 0).mean(axis=0)          # occupancy of the 16 low bits
    return clipped, longest, lowbits


def cmodel(windows):
    out = subprocess.run([str(EXE), "0", str(len(windows))], input=windows.astype("<i4").tobytes(),
                         capture_output=True, check=True).stdout.decode().split("\n")
    return np.array([[int(v) for v in l.split()] for l in out if l.strip()])[:, 1:5]


def evaluate(logits, labels):
    pred = logits.argmax(1)
    srt = np.sort(logits, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    z = logits / T_CONF
    z = np.exp(z - z.max(1, keepdims=True))
    conf = (z / z.sum(1, keepdims=True)).max(1)
    return pred, margin, conf


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-split", type=int, default=2048)
    args = ap.parse_args()
    fs = json.loads(CONTRACT.read_text())["full_scale_per_channel"]
    rng = np.random.default_rng(SEED)
    result = {"id": "Q01", "corruptions": CORRUPTIONS, "per_split_windows": args.per_split}
    data = {}
    for split, d, gz in (("val", VAL, "fpga_valset_raw.i32.gz"), ("test", TEST, "fpga_testset_raw.i32.gz")):
        labels = np.loadtxt(d / "labels.txt", dtype=np.int64)
        idx = np.sort(rng.choice(len(labels), size=args.per_split, replace=False))
        w = read_windows(d / gz, idx)
        y = labels[idx]
        chans = rng.integers(1, 8, size=len(idx))            # never the tachometer (channel 0)
        per = {}
        for cond in CORRUPTIONS:
            wc = w if cond == "clean" else np.stack([perturb(w[i], cond, chans[i], fs) for i in range(len(w))])
            feats = [quality_features(x) for x in wc]
            logits = cmodel(wc)
            pred, margin, conf = evaluate(logits, y)
            per[cond] = {"pred": pred, "margin": margin, "conf": conf, "y": y,
                         "clipped": np.stack([f[0] for f in feats]), "longest": np.stack([f[1] for f in feats]),
                         "lowbits": np.stack([f[2] for f in feats])}
            print(split, cond, "acc %.4f" % (pred == y).mean(), flush=True)
        data[split] = per
    # thresholds from CLEAN VALIDATION only
    cv = data["val"]["clean"]
    thr = {"clipped": cv["clipped"].max(axis=0) + 1e-9, "longest": cv["longest"].max(axis=0),
           "lowbits": np.maximum(cv["lowbits"].min(axis=0) - 1e-9, 0)}
    def flags(p):
        return ((p["clipped"] > thr["clipped"]) | (p["longest"] > thr["longest"]) | (p["lowbits"] < thr["lowbits"])).any(axis=1)
    # margin / confidence thresholds: largest values keeping >= 90 % clean validation coverage
    m_t = float(np.percentile(cv["margin"], 10))
    c_t = float(np.percentile(cv["conf"], 10))
    methods = {"none": lambda p: np.ones(len(p["y"]), bool), "conf": lambda p: p["conf"] >= c_t,
               "margin": lambda p: p["margin"] >= m_t, "flags": lambda p: ~flags(p),
               "flags+margin": lambda p: (~flags(p)) & (p["margin"] >= m_t)}
    report = {}
    for split in ("val", "test"):
        report[split] = {}
        for mname, acc_fn in methods.items():
            row = {}
            for cond in CORRUPTIONS:
                p = data[split][cond]
                a = acc_fn(p)
                wrong = (p["pred"] != p["y"])
                row[cond] = {"coverage": float(a.mean()), "risk_accepted": float(wrong[a].mean()) if a.any() else None,
                             "accepted_wrong": int((a & wrong).sum()),
                             "accepted_false_healthy": int((a & (p["pred"] == 0) & (p["y"] != 0)).sum()),
                             "flagged": int((~a).sum())}
            report[split][mname] = row
    corr = [c for c in CORRUPTIONS if c not in ("clean", "gain2", "offset")]
    gate = {}
    for mname in methods:
        base = sum(report["val"]["none"][c]["accepted_wrong"] for c in corr)
        got = sum(report["val"][mname][c]["accepted_wrong"] for c in corr)
        gate[mname] = {"clean_val_coverage": report["val"][mname]["clean"]["coverage"],
                       "accepted_wrong_reduction_vs_none": (1 - got / base) if base else None,
                       "passes": bool(report["val"][mname]["clean"]["coverage"] >= 0.9 and base and (1 - got / base) >= 0.3)}
    result.update({"thresholds": {"margin": m_t, "conf": c_t, "clipped": thr["clipped"].tolist(),
                                  "longest": thr["longest"].tolist(), "lowbits": thr["lowbits"].tolist()},
                   "report": report, "gate_Q01_on_validation": gate,
                   "notes": "thresholds from clean validation only; test reported once; flags are metadata, not a class"})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(result, indent=1))
    for split in ("val", "test"):
        for mname in methods:
            print(split, "%-13s" % mname, " ".join("%s:cov%.2f/wrong%d" % (c, report[split][mname][c]["coverage"],
                  report[split][mname][c]["accepted_wrong"]) for c in CORRUPTIONS))
    print(json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()
