"""Build hls/f1_int10_seq256/golden_ingress1024_argmax/ (argmax-variant golden set).

Content (all generated, nothing hand-edited):
  * vectors 0..15  : the existing golden_ingress1024 set, copied verbatim;
  * then real test-set windows taken from the exact board input file
    (fpga_testset_in.i32.gz), chosen deterministically:
      - every tied-max-logit pattern found in the full test set, up to 3
        windows per pattern (lowest dataset index first);
      - 2 non-tied windows per predicted class (lowest dataset index first).
Expected logits come from the approved Python INT10 reference
(src.quantize.integer_model, export scales) -- the per-window file produced by
tools/run_numpy_reference_fulltest.py --scales export, which was verified
bit-exact against the HLS C-model on all 95,404 windows. The expected class is
numpy.argmax (lowest index wins ties).

Usage:
    python -m tools.make_argmax_golden \
        --logits artifacts/results/board_vs_reference_20261022/numpy_reference/numpy_export_logits.txt
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FT = ROOT / "artifacts" / "results" / "fpga_fulltest_20261008_141335"
SRC = ROOT / "hls" / "f1_int10_seq256" / "golden_ingress1024"
DST = ROOT / "hls" / "f1_int10_seq256" / "golden_ingress1024_argmax"
REC = 1024 * 8 * 4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logits", type=Path, required=True)
    ap.add_argument("--per-pattern", type=int, default=3)
    ap.add_argument("--per-class", type=int, default=2)
    args = ap.parse_args()
    if DST.exists():
        raise SystemExit(f"{DST} exists; refusing to overwrite")

    lg = np.loadtxt(args.logits, dtype=np.int64)
    assert (lg[:, 0] == np.arange(len(lg))).all()
    lg = lg[:, 1:]
    labels = np.loadtxt(FT / "labels.txt", dtype=np.int64)
    pred = lg.argmax(1)
    ismax = lg == lg.max(1, keepdims=True)
    tied = ismax.sum(1) > 1

    chosen, why = [], []
    patterns = {}
    for i in np.where(tied)[0]:
        patterns.setdefault(tuple(int(c) for c in np.where(ismax[i])[0]), []).append(int(i))
    for pat in sorted(patterns):
        for i in patterns[pat][:args.per_pattern]:
            chosen.append(i)
            why.append("tie classes %s" % list(pat))
    for c in range(4):
        for i in np.where((pred == c) & ~tied)[0][:args.per_class]:
            chosen.append(int(i))
            why.append("non-tie predicted class %d" % c)
    order = sorted(range(len(chosen)), key=lambda k: chosen[k])
    chosen = [chosen[k] for k in order]
    why = [why[k] for k in order]

    want = set(chosen)
    rows = {}
    with gzip.open(FT / "fpga_testset_in.i32.gz", "rb") as f:
        for idx in range(max(chosen) + 1):
            buf = f.read(REC)
            if idx in want:
                rows[idx] = np.frombuffer(buf, dtype="<i4").astype(np.int64)

    src_in = (SRC / "golden_in.txt").read_text().splitlines()
    src_exp = (SRC / "golden_expected.txt").read_text().splitlines()
    n0 = int(src_in[0].split()[0])
    DST.mkdir(parents=True)
    with open(DST / "golden_in.txt", "w", newline="\n") as f:
        f.write("%d 1024 8\n" % (n0 + len(chosen)))
        for line in src_in[1:1 + n0]:
            f.write(line + "\n")
        for idx in chosen:
            f.write(" ".join(map(str, rows[idx].tolist())) + "\n")
    with open(DST / "golden_expected.txt", "w", newline="\n") as f:
        f.write(src_exp[0] + "\n")
        for line in src_exp[1:1 + n0]:
            f.write(line + "\n")
        for k, idx in enumerate(chosen):
            f.write("%d %d %d %s\n" % (n0 + k, labels[idx], pred[idx],
                                        " ".join(map(str, lg[idx].tolist()))))
    (DST / "provenance.json").write_text(json.dumps({
        "vectors_0_to_%d" % (n0 - 1): "verbatim copy of golden_ingress1024 (unchanged)",
        "appended": [{"vector": n0 + k, "dataset_window": idx, "reason": why[k],
                      "true_class": int(labels[idx]), "expected_class": int(pred[idx]),
                      "expected_logits": lg[idx].tolist()} for k, idx in enumerate(chosen)],
        "input_source": str((FT / "fpga_testset_in.i32.gz").relative_to(ROOT)),
        "expected_source": str(args.logits),
        "expected_generator": "src.quantize.integer_model.IntegerTransformer.infer_int, "
                              "scales artifacts/quantized/f_seq256_int10/integer_config.json",
        "tie_policy": "lowest class index among equal maxima (numpy.argmax)",
        "tie_patterns_in_full_test": {str(list(k)): len(v) for k, v in sorted(patterns.items())},
    }, indent=2), encoding="utf-8")
    print(f"wrote {DST} : {n0} original + {len(chosen)} appended vectors")


if __name__ == "__main__":
    main()
