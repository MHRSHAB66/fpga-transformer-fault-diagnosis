"""Build hls/f1_int10_seq256/golden_raw1024_argmax/ (raw-input core golden set).

Same file format as golden_ingress1024_argmax/ (so the argmax testbench and the
board golden checker are reused unchanged), but every input word is a raw
signed 24-bit ADC code and the IP must do the preprocessing itself.

Vectors (all generated, nothing hand-edited):
  * the 24 real test windows of golden_ingress1024_argmax (ties and per-class
    picks), now taken from the raw test input fpga_testset_raw.i32.gz;
  * the first 4 windows of each true class from the 256-window RAM demo plan;
  * 4 constructed edge windows: one constant channel, all-zero channel set,
    out-of-range words (clamped by the IP), a full-scale square wave.
Expected logits: on-FPGA preprocessing reference (src/quantize/onchip_preprocess.py)
-> aggregate_round_div -> IntegerTransformer.infer_int with the export scales;
expected class = lowest-index argmax.

    python -m tools.make_raw_golden --raw artifacts/results/fpga_rawinput_testset_20261109/fpga_testset_raw.i32.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from src.data.token_aggregation import aggregate_round_div
from src.quantize.onchip_preprocess import CODE_MAX, CODE_MIN, preprocess_window
from tools.run_numpy_reference_fulltest import EXPORT_SCALES, _model

ROOT = Path(__file__).resolve().parents[1]
ARGMAX_GOLDEN = ROOT / "hls" / "f1_int10_seq256" / "golden_ingress1024_argmax" / "provenance.json"
RAM_SET = ROOT / "artifacts" / "ram_demo_sets" / "class_balanced_256_v1" / "indices.txt"
LABELS = ROOT / "artifacts" / "results" / "fpga_fulltest_20261008_141335" / "labels.txt"
DST = ROOT / "hls" / "f1_int10_seq256" / "golden_raw1024_argmax"
REC = 1024 * 8 * 4


def read_windows(path, wanted):
    rows, want = {}, set(wanted)
    with gzip.open(path, "rb") as f:
        for idx in range(max(wanted) + 1):
            buf = f.read(REC)
            if idx in want:
                rows[idx] = np.frombuffer(buf, dtype="<i4").astype(np.int64).reshape(1024, 8)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", type=Path, required=True)
    args = ap.parse_args()
    if DST.exists():
        raise SystemExit("%s exists; refusing to overwrite" % DST)
    labels = np.loadtxt(LABELS, dtype=np.int64)
    picks = [(a["dataset_window"], "argmax golden: " + a["reason"])
             for a in json.loads(ARGMAX_GOLDEN.read_text())["appended"]]
    ram = np.loadtxt(RAM_SET, dtype=np.int64)
    for c in range(4):
        for idx in [int(i) for i in ram if labels[i] == c][:4]:
            picks.append((idx, "RAM demo set, true class %d" % c))
    seen, unique = set(), []
    for idx, why in picks:
        if idx not in seen:
            seen.add(idx)
            unique.append((idx, why))
    rows = read_windows(args.raw, [i for i, _ in unique])

    vectors = [(rows[i], int(labels[i]), why, i) for i, why in unique]
    base = rows[unique[0][0]]
    const = base.copy(); const[:, 2] = 4321
    zeros = base.copy(); zeros[:, 5:] = 0
    wide = base.copy(); wide[::97, :] = 1 << 30; wide[1::97, :] = -(1 << 30)
    square = np.where(np.arange(1024)[:, None] % 2 == 0, CODE_MAX, CODE_MIN) * np.ones((1, 8), dtype=np.int64)
    for w, why in ((const, "edge: channel 2 constant"), (zeros, "edge: channels 5-7 all zero"),
                   (wide, "edge: out-of-range words clamped by the IP"),
                   (square, "edge: full-scale square wave")):
        vectors.append((w, -1, why, None))

    scales = json.loads(EXPORT_SCALES.read_text())["scales"]
    model = _model(scales)
    s_in = float(scales["input"])
    q256 = np.stack([aggregate_round_div(preprocess_window(w, s_in)[None], 4)[0] for w, *_ in vectors])
    logits = model.infer_int(q256.astype(np.float64) * s_in).astype(np.int64)
    pred = logits.argmax(axis=1)

    DST.mkdir(parents=True)
    with open(DST / "golden_in.txt", "w", newline="\n") as f:
        f.write("%d 1024 8\n" % len(vectors))
        for w, *_ in vectors:
            f.write(" ".join(map(str, w.reshape(-1).tolist())) + "\n")
    with open(DST / "golden_expected.txt", "w", newline="\n") as f:
        f.write("# idx true pred logit0 logit1 logit2 logit3\n")
        for k, (_, true, _, _) in enumerate(vectors):
            f.write("%d %d %d %s\n" % (k, true, pred[k], " ".join(map(str, logits[k].tolist()))))
    ties = int(((logits == logits.max(1, keepdims=True)).sum(1) > 1).sum())
    (DST / "provenance.json").write_text(json.dumps({
        "input": "raw signed 24-bit ADC codes (artifacts/preprocessing/raw_input_contract_v1.json)",
        "input_source": str(args.raw),
        "expected_generator": "src.quantize.onchip_preprocess.preprocess_window -> "
                              "aggregate_round_div -> IntegerTransformer.infer_int (export scales)",
        "true_class": "-1 for constructed edge windows",
        "tie_vectors": ties,
        "vectors": [{"vector": k, "dataset_window": idx, "reason": why, "true_class": true,
                     "expected_class": int(pred[k]), "expected_logits": logits[k].tolist()}
                    for k, (_, true, why, idx) in enumerate(vectors)],
    }, indent=1), encoding="utf-8")
    print("wrote %s: %d vectors, %d ties" % (DST, len(vectors), ties))


if __name__ == "__main__":
    main()
