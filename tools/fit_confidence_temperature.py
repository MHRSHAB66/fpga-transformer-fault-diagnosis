"""Fit and freeze temperature scaling for the live confidence (validation only).

confidence_uncal(x) = max softmax(s_logits * z)          (current display)
confidence_cal(x)   = max softmax(s_logits * z / T)        (T fitted here)

z = the 4 integer logits produced by the deployed FPGA core; s_logits = the
deployed logits scale. T > 0 is a single scalar, so the arg-max (the FPGA's class)
is unchanged for every window. T is chosen by minimising the validation NLL
(golden-section search on log T); the TEST split is only evaluated afterwards,
never used for fitting.

Inputs (already produced, bit-exact with the board on the test set):
  validation: artifacts/results/board_vs_reference_20261022/confidence/val_numpy_export_logits.txt
  test      : artifacts/results/board_vs_reference_20261022/numpy_reference/numpy_export_logits.txt
              + artifacts/results/fpga_fulltest_20261008_141335/labels.txt

Writes a NEW versioned dir (refuses to overwrite):
  <out>/temperature.json     frozen parameter + provenance (hashes, split, command)
  <out>/evaluation.json      uncalibrated vs calibrated metrics on val and test

Usage: python -m tools.fit_confidence_temperature --out artifacts/calibration/conf_temperature_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "artifacts/results/board_vs_reference_20261022/confidence/val_numpy_export_logits.txt"
TEST = ROOT / "artifacts/results/board_vs_reference_20261022/numpy_reference/numpy_export_logits.txt"
TEST_LABELS = ROOT / "artifacts/results/fpga_fulltest_20261008_141335/labels.txt"
SCALES = ROOT / "artifacts/deploy/seq256_argmax_z7020_20261022/config/integer_config_scales.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def probs(z: np.ndarray, s: float, T: float = 1.0) -> np.ndarray:
    a = z.astype(np.float64) * (s / T)
    a -= a.max(1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(1, keepdims=True)


def nll(z, y, s, T):
    p = probs(z, s, T)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-300, 1)).mean())


def metrics(z, y, s, T, bins=15):
    p = probs(z, s, T)
    pred = z.argmax(1)                      # FPGA rule: lowest index among equal maxima
    conf = p[np.arange(len(y)), pred]
    corr = (pred == y).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    ece, mce, rows = 0.0, 0.0, []
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if not m.any():
            continue
        gap = abs(conf[m].mean() - corr[m].mean())
        ece += m.mean() * gap
        mce = max(mce, gap)
        rows.append({"bin": [round(edges[i], 4), round(edges[i + 1], 4)], "n": int(m.sum()),
                     "mean_conf": round(float(conf[m].mean()), 4), "accuracy": round(float(corr[m].mean()), 4)})
    onehot = np.eye(z.shape[1])[y]
    return {"n": int(len(y)), "accuracy": round(float(corr.mean()), 6),
            "mean_confidence": round(float(conf.mean()), 6),
            "ece_15bin": round(float(ece), 6), "mce_15bin": round(float(mce), 6),
            "nll": round(nll(z, y, s, T), 6),
            "brier": round(float(((p - onehot) ** 2).sum(1).mean()), 6),
            "reliability": rows}


def fit_T(z, y, s, lo=math.log(0.05), hi=math.log(20.0), iters=200):
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(z, y, s, math.exp(c)), nll(z, y, s, math.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(z, y, s, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(z, y, s, math.exp(d))
        if b - a < 1e-10:
            break
    return math.exp((a + b) / 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    s = float(json.loads(SCALES.read_text())["scales"]["logits"])
    v = np.loadtxt(VAL, dtype=np.int64)
    vy, vz = v[:, 0], v[:, 1:]
    t = np.loadtxt(TEST, dtype=np.int64)
    tz = t[:, 1:]
    ty = np.loadtxt(TEST_LABELS, dtype=np.int64)
    assert len(tz) == len(ty) == 95404 and len(vz) == 47580

    T = fit_T(vz, vy, s)                    # VALIDATION ONLY
    param = {"method": "temperature scaling of the FPGA's integer logits",
             "formula": "conf = max softmax(s_logits * logits_int / T)",
             "T": T, "s_logits": s,
             "fitted_on": "validation split (195 files, 47,580 windows), Python INT10 reference logits "
                          "with the deployed export scales (bit-exact with the board on the test set)",
             "objective": "validation NLL (golden-section on log T)",
             "test_used_for_fitting": False,
             "argmax_unchanged": "T>0 scalar: class is unchanged for every window",
             "inputs_sha256": {"validation_logits": _sha(VAL), "scales": _sha(SCALES)},
             "command": "python -m tools.fit_confidence_temperature --out " + str(a.out.relative_to(ROOT)).replace("\\", "/"),
             "python": sys.version.split()[0], "numpy": np.__version__}
    ev = {"T": T,
          "validation": {"uncalibrated": metrics(vz, vy, s, 1.0), "calibrated": metrics(vz, vy, s, T)},
          "test": {"uncalibrated": metrics(tz, ty, s, 1.0), "calibrated": metrics(tz, ty, s, T)},
          "test_inputs_sha256": {"test_logits": _sha(TEST), "test_labels": _sha(TEST_LABELS)}}
    (a.out / "temperature.json").write_text(json.dumps(param, indent=2), encoding="utf-8")
    (a.out / "evaluation.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")
    print("T =", round(T, 6))
    for split in ("validation", "test"):
        for k in ("uncalibrated", "calibrated"):
            m = ev[split][k]
            print("%-10s %-12s ECE %.4f  NLL %.4f  Brier %.4f  MCE %.4f  meanconf %.4f  acc %.4f"
                  % (split, k, m["ece_15bin"], m["nll"], m["brier"], m["mce_15bin"], m["mean_confidence"], m["accuracy"]))


if __name__ == "__main__":
    main()
