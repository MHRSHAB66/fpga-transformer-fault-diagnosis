"""Calibration of the softmax confidence derived from the hardware INT10 logits.

confidence(window) = max_c softmax(s_logits * logits_int)[c]
where logits_int are the 4 signed integers the FPGA core streams out and
s_logits is the exported logits quantization scale of the deployed model.

Validation split: logits computed with the Python INT10 reference (export
scales), which is bit-exact with the HLS C-model / board on the test set.
Test split: per-window logits from an existing file (default: the export-scale
reference logits, == board classes on 95,404/95,404 windows).
Nothing is fitted: no temperature scaling, no threshold tuning.

Writes <out>/confidence_calibration.json (+ val logits text file).

Usage:
    python -m tools.evaluate_confidence_calibration \
        --test-logits artifacts/results/board_vs_reference_20261022/numpy_reference/numpy_export_logits.txt \
        --out artifacts/results/board_vs_reference_20261022/confidence
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCALES = ROOT / "artifacts" / "deploy" / "seq256_z7020_pynq_dma16_py36_20261008" / "config" / "integer_config_scales.json"
LABELS_TEST = ROOT / "artifacts" / "results" / "fpga_fulltest_20261008_141335" / "labels.txt"


def _val_shard(job):
    files, scales = job
    import pandas as pd
    from src.data.make_dataloaders import build_X_y_from_file_index
    from src.data.token_aggregation import aggregate_round_div
    from src.quantize.integer_ops import qint_max
    from tools.run_numpy_reference_fulltest import _model
    df = pd.DataFrame(files)
    X, y, _, _, _ = build_X_y_from_file_index(df, window_size=1024, stride=1024,
                                              aggregation_factor=1)
    s_in = float(scales["input"])
    q = np.clip(np.round(X.astype(np.float64) / s_in), -qint_max(10), qint_max(10)).astype(np.int64)
    del X
    m = _model(scales)
    out = []
    for k in range(0, len(q), 32):
        out.append(m.infer_int(aggregate_round_div(q[k:k + 32], factor=4).astype(np.float64) * s_in))
    return np.concatenate(out), y.astype(np.int64)


def softmax_pmax(logits: np.ndarray, scale: float) -> np.ndarray:
    z = logits.astype(np.float64) * scale
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    return p.max(1)


def calibration(logits: np.ndarray, y: np.ndarray, scale: float, bins: int = 15) -> dict:
    pred = logits.argmax(1)
    conf = softmax_pmax(logits, scale)
    corr = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows, ece, mce = [], 0.0, 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if not m.any():
            continue
        gap = abs(conf[m].mean() - corr[m].mean())
        ece += m.mean() * gap
        mce = max(mce, gap)
        rows.append({"bin": [round(edges[i], 4), round(edges[i + 1], 4)], "n": int(m.sum()),
                     "mean_conf": round(float(conf[m].mean()), 4),
                     "accuracy": round(float(corr[m].mean()), 4)})
    onehot = np.eye(4)[y]
    z = logits.astype(np.float64) * scale
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    ties = int(((logits == logits.max(1, keepdims=True)).sum(1) > 1).sum())
    sel = {}
    for thr in (0.5, 0.8, 0.9, 0.99):
        m = conf >= thr
        sel[str(thr)] = {"coverage": round(float(m.mean()), 4),
                         "accuracy": round(float(corr[m].mean()), 4) if m.any() else None}
    return {"n": int(len(y)), "accuracy": round(float(corr.mean()), 6),
            "mean_confidence": round(float(conf.mean()), 6),
            "ece_15bin": round(float(ece), 6), "mce_15bin": round(float(mce), 6),
            "brier": round(float(((p - onehot) ** 2).sum(1).mean()), 6),
            "nll": round(float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()), 6),
            "tied_max_logit_windows": ties,
            "accuracy_at_confidence_ge": sel, "reliability": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-logits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scales = json.loads(SCALES.read_text())["scales"]
    s_log = float(scales["logits"])

    from src.config import DATA_RAW_DIR, LABEL_MAP
    from src.data.load_data import build_file_index
    from src.data.make_dataloaders import split_file_index
    idx = build_file_index(Path(DATA_RAW_DIR))
    idx = idx[idx["label"].isin(LABEL_MAP.keys())].copy()
    _, val_df, _ = split_file_index(idx)
    t0 = time.time()
    recs = val_df.to_dict("records")
    jobs = [(recs[s::args.workers], scales) for s in range(args.workers) if recs[s::args.workers]]
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        parts = list(ex.map(_val_shard, jobs))
    vl = np.concatenate([p[0] for p in parts])
    vy = np.concatenate([p[1] for p in parts])
    np.savetxt(args.out / "val_numpy_export_logits.txt",
               np.column_stack([vy, vl]), fmt="%d", header="true l0 l1 l2 l3")

    tl = np.loadtxt(args.test_logits, dtype=np.int64)[:, 1:]
    ty = np.loadtxt(LABELS_TEST, dtype=np.int64)
    res = {
        "definition": "confidence = max softmax(s_logits * hardware_int_logits); "
                      "uncalibrated model probability, nothing fitted",
        "s_logits": s_log, "scales_source": str(SCALES.relative_to(ROOT)),
        "validation": dict(calibration(vl, vy, s_log), val_files=len(recs)),
        "test": calibration(tl, ty, s_log),
        "test_logits_source": str(args.test_logits),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (args.out / "confidence_calibration.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    for k in ("validation", "test"):
        r = res[k]
        print(k, "n", r["n"], "acc", r["accuracy"], "meanconf", r["mean_confidence"],
              "ECE", r["ece_15bin"], "MCE", r["mce_15bin"])


if __name__ == "__main__":
    main()
