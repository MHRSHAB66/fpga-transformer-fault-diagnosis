"""Golden set of a model variant for the raw-input core: the SAME 41 input
windows as hls/f1_int10_seq256/golden_raw1024_argmax/ (copied byte for byte),
expected logits recomputed with the variant's INT10 reference (on-FPGA
preprocessing reference -> 4:1 aggregation -> IntegerTransformer with the
variant's export scales); expected class = lowest-index argmax.

    python -m tools.make_variant_golden \
        --ckpt artifacts/results/generalization_20261110/random_noise/best.pt \
        --scales artifacts/quantized/f_seq256_int10_noiseaug/integer_config.json \
        --dst artifacts/quantized/f_seq256_int10_noiseaug/golden_raw1024_argmax
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from src.data.token_aggregation import aggregate_round_div
from src.quantize.onchip_preprocess import preprocess_window
from tools.check_variant_cmodel import reference_model

SRC = Path(__file__).resolve().parents[1] / "hls" / "f1_int10_seq256" / "golden_raw1024_argmax"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--scales", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()
    lines = (SRC / "golden_in.txt").read_text().splitlines()
    n = int(lines[0].split()[0])
    windows = np.array([[int(v) for v in l.split()] for l in lines[1:1 + n]], dtype=np.int64).reshape(n, 1024, 8)
    true = [int(l.split()[1]) for l in (SRC / "golden_expected.txt").read_text().splitlines()[1:] if l.strip()]
    scales = json.loads(args.scales.read_text())["scales"]
    s_in = float(scales["input"])
    model = reference_model(args.ckpt, scales)
    q = np.stack([aggregate_round_div(preprocess_window(w, s_in)[None], 4)[0] for w in windows])
    logits = model.infer_int(q.astype(np.float64) * s_in).astype(np.int64)
    pred = logits.argmax(axis=1)
    args.dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC / "golden_in.txt", args.dst / "golden_in.txt")
    with open(args.dst / "golden_expected.txt", "w", newline="\n") as f:
        f.write("# idx true pred logit0 logit1 logit2 logit3\n")
        for k in range(n):
            f.write("%d %d %d %s\n" % (k, true[k], pred[k], " ".join(map(str, logits[k].tolist()))))
    ties = int(((logits == logits.max(1, keepdims=True)).sum(1) > 1).sum())
    (args.dst / "provenance.json").write_text(json.dumps({
        "inputs": "copied from hls/f1_int10_seq256/golden_raw1024_argmax/golden_in.txt",
        "checkpoint": str(args.ckpt), "scales": str(args.scales),
        "reference": "onchip_preprocess.preprocess_window -> aggregate_round_div -> IntegerTransformer.infer_int",
        "vectors": n, "ties": ties,
        "correct_vs_true_label": int(sum(int(p) == t for p, t in zip(pred, true) if t >= 0))}, indent=1))
    print("wrote", args.dst, "vectors", n, "ties", ties)


if __name__ == "__main__":
    main()
