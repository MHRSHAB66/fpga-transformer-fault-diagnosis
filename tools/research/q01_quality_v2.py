"""Q01 v2 (corrected, 2026-11-14): input quality, calibrated confidence and integer
margin as decision-status metadata (offline, bit-exact canonical C-model).

Supersedes the `conf` method and the gate of tools/research/q01_quality.py
(artifacts/research/Q01_20261113, kept unchanged). Corrections after the
2026-11-13 audit (findings 2 and 5):

* calibrated confidence uses the frozen contract of
  artifacts/calibration/conf_temperature_v1/temperature.json:
  conf = max softmax(s_logits * logits_int / T), both constants read from the file
  (the old code used logits_int / T);
* the SAME validation/test windows and perturbation channels as Q01 (seed 20261113,
  2048 windows per split; identical sampling sequence, saved to sample_manifest.json);
* complete risk/coverage: accepted/rejected counts, accepted wrong, accepted
  false-healthy, correct outputs rejected, per-class coverage, and explicit
  zero-acceptance cases (rejecting every corrupted window is reported as refusing
  to classify, not as better accuracy under that fault);
* severity sweep of every fault plus nuisance operating conditions (white noise at
  20/10 dB SNR on all channels) that a quality monitor should NOT reject;
* ordered fault bursts inside recordings (frozen burst manifest: recording, start,
  onset, duration, fault, severity, channel), identical streams for every method:
  onset detection, rejection duration, accepted wrong during the burst, recovery.
Thresholds use the frozen Q01 rule on CLEAN VALIDATION only (margin/confidence at
the 10th percentile, flags at the clean-validation extremes) and are not changed
after seeing results. Numeric agreement (C-model = board, established for the
canonical core) is separate from diagnostic correctness (prediction vs label).

    python -m tools.research.q01_quality_v2 --out artifacts/research/Q01v2_20261114
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from tools.research.q01_quality import (CODE_MAX, CODE_MIN, CONTRACT, EXE, SEED, TEST, VAL, perturb as perturb_v1,
                                        quality_features, read_windows)
from tools.robustness_cmodel import perturb as perturb_noise

REPO = Path(__file__).resolve().parents[2]
CALIB = REPO / "artifacts/calibration/conf_temperature_v1/temperature.json"
REC = 244
BURST_SEED = 20261117
ORIGINAL = {"clip30": "clip30", "bits8": "bits8", "hold16": "hold", "lag32": "lag32"}   # v2 name -> v1 name
FAULTS = ["clip50", "clip30", "clip10", "bits12", "bits8", "bits4", "hold4", "hold16", "hold31",
          "lag8", "lag32", "lag128"]
CONTROLS = ["gain2", "offset"]
NUISANCE = ["snr20", "snr10"]
CONDITIONS = ["clean"] + CONTROLS + NUISANCE + FAULTS
GATE_SET = ["clip30", "bits8", "hold16", "lag32"]          # the frozen Q01 corruption set
BURST_FAULTS = ["clip50", "clip30", "bits12", "bits8", "hold4", "hold16", "lag8", "lag32"]


def perturb(w, cond, ch, index):
    """One window [1024, 8] int64 raw codes -> perturbed codes (one channel unless snr)."""
    if cond == "clean":
        return w
    if cond.startswith("snr"):
        return perturb_noise(w[None], index, cond)[0].astype(np.int64)
    if cond in ("gain2", "offset"):
        return perturb_v1(w, cond, ch, None)
    x = w.copy()
    c = x[:, ch]
    if cond.startswith("clip"):
        med = int(np.median(c))
        lim = max(1, int(int(cond[4:]) / 100 * np.abs(c - med).max()))
        c[:] = np.clip(c, med - lim, med + lim)
    elif cond.startswith("bits"):
        z = 24 - int(cond[4:])
        c[:] = (c >> z) << z
    elif cond.startswith("hold"):
        r = int(cond[4:])
        for s in range(0, 1024, 32):
            c[s + 1:s + r] = c[s]
    elif cond.startswith("lag"):
        c[:] = np.roll(c, int(cond[3:]))
    else:
        raise ValueError(cond)
    x[:, ch] = np.clip(c, CODE_MIN, CODE_MAX)
    return x


def cmodel(windows, shards=4):
    def run(chunk):
        if not len(chunk):
            return np.zeros((0, 4), np.int64)
        out = subprocess.run([str(EXE), "0", str(len(chunk))], input=chunk.astype("<i4").tobytes(),
                             capture_output=True, check=True).stdout.decode().split("\n")
        r = np.array([[int(v) for v in l.split()] for l in out if l.strip()], np.int64)
        assert len(r) == len(chunk)
        return r[:, 1:5]
    parts = np.array_split(windows, shards)
    with ThreadPoolExecutor(shards) as ex:
        return np.concatenate(list(ex.map(run, parts)))


def calib():
    c = json.loads(CALIB.read_text())
    return float(c["T"]), float(c["s_logits"])


def confidences(logits):
    T, s = calib()
    z = s * logits.astype(np.float64) / T
    z = np.exp(z - z.max(1, keepdims=True))
    conf = (z / z.sum(1, keepdims=True)).max(1)
    zo = logits.astype(np.float64) / T              # the superseded Q01 formula, for the change record
    zo = np.exp(zo - zo.max(1, keepdims=True))
    return conf, (zo / zo.sum(1, keepdims=True)).max(1)


def features(wins):
    f = [quality_features(x) for x in wins]
    return np.stack([a for a, _, _ in f]), np.stack([b for _, b, _ in f]), np.stack([c for _, _, c in f])


def sample_manifest():
    """The exact Q01 sampling sequence (rng consumption order of q01_quality.main)."""
    rng = np.random.default_rng(SEED)
    man = {}
    for split, d in (("val", VAL), ("test", TEST)):
        labels = np.loadtxt(d / "labels.txt", dtype=np.int64)
        idx = np.sort(rng.choice(len(labels), size=2048, replace=False))
        chans = rng.integers(1, 8, size=len(idx))
        man[split] = {"idx": idx.tolist(), "chans": chans.tolist()}
    return man


def burst_manifest(n_windows, rng):
    bursts = []
    for f in BURST_FAULTS:
        for k in range(12):
            rec = int(rng.integers(n_windows // REC))
            start = rec * REC + int(rng.integers(0, REC - 40 + 1))
            bursts.append({"burst_id": len(bursts), "fault": f, "rec": rec, "start": start, "length": 40,
                           "onset": 10, "duration": 20 if k % 2 == 0 else 3, "channel": int(rng.integers(1, 8))})
    return bursts


class Methods:
    def __init__(self, cv):
        self.thr = {"clipped": cv["clipped"].max(axis=0) + 1e-9, "longest": cv["longest"].max(axis=0),
                    "lowbits": np.maximum(cv["lowbits"].min(axis=0) - 1e-9, 0)}
        self.m_t = float(np.percentile(cv["margin"], 10))
        self.c_t = float(np.percentile(cv["conf"], 10))
        self.c_old_t = float(np.percentile(cv["conf_old"], 10))

    def flags(self, p):
        t = self.thr
        return ((p["clipped"] > t["clipped"]) | (p["longest"] > t["longest"]) | (p["lowbits"] < t["lowbits"])).any(axis=1)

    def accept(self, name, p):
        n = len(p["y"])
        return {"none": lambda: np.ones(n, bool), "conf": lambda: p["conf"] >= self.c_t,
                "conf_superseded_q01": lambda: p["conf_old"] >= self.c_old_t,
                "margin": lambda: p["margin"] >= self.m_t, "flags": lambda: ~self.flags(p),
                "flags+margin": lambda: (~self.flags(p)) & (p["margin"] >= self.m_t)}[name]()

    def to_json(self):
        return {"margin_min": self.m_t, "conf_min": self.c_t, "conf_superseded_q01_min": self.c_old_t,
                **{k: v.tolist() for k, v in self.thr.items()}}


METHODS = ["none", "conf", "conf_superseded_q01", "margin", "flags", "flags+margin"]


def pack(wins, y):
    lg = cmodel(wins)
    clipped, longest, lowbits = features(wins)
    conf, conf_old = confidences(lg)
    srt = np.sort(lg, axis=1)
    return {"logits": lg, "pred": lg.argmax(1), "y": y, "margin": srt[:, -1] - srt[:, -2], "conf": conf,
            "conf_old": conf_old, "clipped": clipped, "longest": longest, "lowbits": lowbits}


def risk_row(a, p):
    wrong = p["pred"] != p["y"]
    r = {"windows": int(len(a)), "accepted": int(a.sum()), "rejected": int((~a).sum()), "coverage": float(a.mean()),
         "accepted_wrong": int((a & wrong).sum()), "risk_accepted": float(wrong[a].mean()) if a.any() else None,
         "accepted_false_healthy": int((a & (p["pred"] == 0) & (p["y"] != 0)).sum()),
         "correct_rejected": int((~a & ~wrong).sum()), "wrong_rejected": int((~a & wrong).sum()),
         "accuracy_all": float((~wrong).mean()), "zero_accepted": bool(not a.any()),
         "coverage_per_class": [float(a[p["y"] == c].mean()) if (p["y"] == c).any() else None for c in range(4)]}
    return r


def burst_report(bursts, accept_seq, pred_seq, y_seq):
    out = []
    for b, a, pr, y in zip(bursts, accept_seq, pred_seq, y_seq):
        on, d = b["onset"], b["duration"]
        inb = np.arange(on, on + d)
        rej = np.flatnonzero(~a[inb])
        post = a[on + d:]
        rec = int(np.flatnonzero(post)[0]) if post.any() else None
        out.append({"burst_id": b["burst_id"], "false_reject_pre": int((~a[:on]).sum()),
                    "onset_detect_delay": int(rej[0]) if len(rej) else None,
                    "rejected_in_burst": int(len(rej)), "accepted_wrong_in_burst": int((a[inb] & (pr[inb] != y[inb])).sum()),
                    "accepted_false_healthy_in_burst": int((a[inb] & (pr[inb] == 0) & (y[inb] != 0)).sum()),
                    "recovery_delay_after_burst": rec, "rejected_after_burst": int((~post).sum())})
    return out


def summarize_bursts(rows, bursts):
    res = {}
    for f in BURST_FAULTS:
        for d in (20, 3):
            rs = [r for r, b in zip(rows, bursts) if b["fault"] == f and b["duration"] == d]
            if not rs:
                continue
            det = [r["onset_detect_delay"] for r in rs if r["onset_detect_delay"] is not None]
            rec = [r["recovery_delay_after_burst"] for r in rs if r["recovery_delay_after_burst"] is not None]
            res["%s/d%d" % (f, d)] = {
                "bursts": len(rs), "detected": len(det), "onset_delay_max": max(det) if det else None,
                "rejected_fraction_in_burst": sum(r["rejected_in_burst"] for r in rs) / (d * len(rs)),
                "accepted_wrong_in_burst": sum(r["accepted_wrong_in_burst"] for r in rs),
                "accepted_false_healthy_in_burst": sum(r["accepted_false_healthy_in_burst"] for r in rs),
                "false_reject_pre": sum(r["false_reject_pre"] for r in rs),
                "recovered": len(rec), "recovery_delay_max": max(rec) if rec else None,
                "rejected_after_burst": sum(r["rejected_after_burst"] for r in rs)}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    T, s = calib()
    smp = sample_manifest()
    smp_path = args.out / "sample_manifest.json"
    smp_json = {"seed": SEED, "source": "tools/research/q01_quality.py sampling sequence", **smp}
    if smp_path.exists():
        assert json.loads(smp_path.read_text()) == smp_json, "sample manifest changed"
    else:
        smp_path.write_text(json.dumps(smp_json))
    data, bursts_all = {}, {}
    rng_b = np.random.default_rng(BURST_SEED)
    for split, d, gz in (("val", VAL, "fpga_valset_raw.i32.gz"), ("test", TEST, "fpga_testset_raw.i32.gz")):
        labels = np.loadtxt(d / "labels.txt", dtype=np.int64)
        idx, chans = np.array(smp[split]["idx"]), np.array(smp[split]["chans"])
        w = read_windows(d / gz, idx)
        y = labels[idx]
        # identity with the v1 perturbations for the original severities
        for v2, v1 in list(ORIGINAL.items()) + [("gain2", "gain2"), ("offset", "offset")]:
            for i in range(3):
                assert (perturb(w[i], v2, chans[i], idx[i]) == perturb_v1(w[i], v1, chans[i], None)).all(), v2
        per = {}
        for cond in CONDITIONS:
            wc = np.stack([perturb(w[i], cond, chans[i], int(idx[i])) for i in range(len(w))])
            per[cond] = pack(wc, y)
            print(split, cond, "acc %.4f" % (per[cond]["pred"] == y).mean(), flush=True)
        data[split] = per
        # ordered bursts inside recordings (frozen manifest)
        bursts = burst_manifest(len(labels), rng_b)
        for b in bursts:
            assert b["start"] // REC == b["rec"] and (b["start"] + b["length"] - 1) // REC == b["rec"]
        need = sorted({b["start"] + k for b in bursts for k in range(b["length"])})
        wb = dict(zip(need, read_windows(d / gz, need)))
        seqs = []
        for b in bursts:
            ws = []
            for k in range(b["length"]):
                x = wb[b["start"] + k]
                if b["onset"] <= k < b["onset"] + b["duration"]:
                    x = perturb(x, b["fault"], b["channel"], b["start"] + k)
                ws.append(x)
            seqs.append(np.stack(ws))
        flat = pack(np.concatenate(seqs), np.concatenate([labels[b["start"]:b["start"] + b["length"]] for b in bursts]))
        bursts_all[split] = (bursts, flat)
        print(split, "bursts", len(bursts), flush=True)
    bm = {sp: v[0] for sp, v in bursts_all.items()}
    bm_path = args.out / "burst_manifest.json"
    bm_json = {"seed": BURST_SEED, "recording_windows": REC, "bursts": bm,
               "sha256": hashlib.sha256(json.dumps(bm, sort_keys=True).encode()).hexdigest()}
    if bm_path.exists():
        assert json.loads(bm_path.read_text())["sha256"] == bm_json["sha256"], "burst manifest changed"
    else:
        bm_path.write_text(json.dumps(bm_json, indent=0))
    M = Methods(data["val"]["clean"])
    report = {sp: {m: {c: risk_row(M.accept(m, data[sp][c]), data[sp][c]) for c in CONDITIONS} for m in METHODS}
              for sp in ("val", "test")}
    burst_res = {}
    for sp, (bursts, flat) in bursts_all.items():
        burst_res[sp] = {}
        L = 40
        for m in METHODS:
            a = M.accept(m, flat)
            rows = burst_report(bursts, [a[i * L:(i + 1) * L] for i in range(len(bursts))],
                                [flat["pred"][i * L:(i + 1) * L] for i in range(len(bursts))],
                                [flat["y"][i * L:(i + 1) * L] for i in range(len(bursts))])
            burst_res[sp][m] = summarize_bursts(rows, bursts)
    gate = {}
    for m in METHODS:
        base = sum(report["val"]["none"][c]["accepted_wrong"] for c in GATE_SET)
        got = sum(report["val"][m][c]["accepted_wrong"] for c in GATE_SET)
        cov = report["val"][m]["clean"]["coverage"]
        gate[m] = {"clean_val_coverage": cov, "accepted_wrong_reduction_vs_none": (1 - got / base) if base else None,
                   "zero_accepted_conditions": [c for c in GATE_SET if report["val"][m][c]["zero_accepted"]],
                   "nuisance_coverage": {c: report["val"][m][c]["coverage"] for c in NUISANCE},
                   "passes_frozen_gate": bool(cov >= 0.9 and base and (1 - got / base) >= 0.3)}
    result = {"id": "Q01v2", "supersedes": "artifacts/research/Q01_20261113 (conf baseline and gate)",
              "calibration": {"file": str(CALIB.relative_to(REPO)).replace("\\", "/"), "T": T, "s_logits": s,
                              "formula": "max softmax(s_logits * logits_int / T)"},
              "cmodel": "tmp/csim/batch_raw_argmax.exe (canonical raw-input argmax top, g++)",
              "thresholds_from_clean_validation": M.to_json(), "conditions": CONDITIONS, "gate_set": GATE_SET,
              "report": report, "bursts": burst_res, "gate_Q01_on_validation": gate,
              "notes": ["test split previously explored: reported once, not an independent confirmation",
                        "flags/margin/confidence are status metadata; the class is always the FPGA argmax",
                        "a method with zero_accepted under a fault refuses to classify there; that is not higher accuracy"]}
    (args.out / "summary.json").write_text(json.dumps(result, indent=1))
    for sp in ("val", "test"):
        for m in METHODS:
            print(sp, "%-20s" % m, " ".join("%s:%.2f/%d" % (c, report[sp][m][c]["coverage"], report[sp][m][c]["accepted_wrong"])
                                           for c in ["clean"] + NUISANCE + GATE_SET))
    print(json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()
