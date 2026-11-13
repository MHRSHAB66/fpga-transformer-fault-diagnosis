"""Bit-exactness check of a model variant's g++ C-model against the Python INT10
reference, on raw test windows (optionally perturbed as in tools/robustness_cmodel).

Reference chain (same as tools/make_raw_golden.py): on-FPGA preprocessing
reference (src/quantize/onchip_preprocess.py) -> aggregate_round_div ->
IntegerTransformer.infer_int with the variant's export scales.

    python -m tools.check_variant_cmodel \
        --ckpt artifacts/results/generalization_20261110/random_noise/best.pt \
        --scales artifacts/quantized/f_seq256_int10_noiseaug/integer_config.json \
        --exe tmp/csim/batch_raw_argmax_noiseaug.exe --n 400 --conditions clean snr0
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from pathlib import Path

import numpy as np

from src.data.token_aggregation import aggregate_round_div
from src.quantize.onchip_preprocess import preprocess_window
from tools.robustness_cmodel import RAW, perturb
from tools.run_numpy_reference_fulltest import CFG


def reference_model(ckpt, scales):
    import torch
    from src.quantize.evaluate_integer_bits import set_bits
    from src.quantize.integer_model import IntegerTransformer
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    set_bits(10)
    m = IntegerTransformer({k: v.cpu().numpy() for k, v in ck["model_state_dict"].items()}, CFG)
    m.set_scales(scales)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--scales", type=Path, required=True)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--conditions", nargs="+", default=["clean"])
    ap.add_argument("--research-bits", default=None,
                    help="research numerics 'W,A,P,unsigned' (tools/research/precision_ablation.py), e.g. 10,10,8,1")
    args = ap.parse_args()
    scales = json.loads(args.scales.read_text())["scales"]
    s_in = float(scales["input"])
    if args.research_bits:
        from tools.research.precision_ablation import load_state, make_model
        w, a, pb, u = (int(v) for v in args.research_bits.split(","))
        model = make_model(load_state(), w, a, pb, bool(u), scales=scales)
    else:
        model = reference_model(args.ckpt, scales)
    with gzip.open(RAW, "rb") as f:
        raw = np.frombuffer(f.read(args.n * 8192 * 4), dtype="<i4").reshape(-1, 1024, 8).astype(np.int64)
    result = {}
    for cond in args.conditions:
        w = perturb(raw, 0, cond).astype(np.int64)
        q = np.stack([aggregate_round_div(preprocess_window(x, s_in)[None], 4)[0] for x in w])
        ref = model.infer_int(q.astype(np.float64) * s_in).astype(np.int64)
        out = subprocess.run([str(args.exe), "0", str(len(w))], input=w.astype("<i4").tobytes(),
                             capture_output=True, check=True).stdout.decode().split("\n")
        rows = np.array([[int(v) for v in line.split()] for line in out if line.strip()], dtype=np.int64)
        equal = int((rows[:, 1:5] == ref).all(axis=1).sum())
        cls_ok = int((rows[:, 5] == ref.argmax(axis=1)).sum())
        result[cond] = {"windows": len(w), "logits_equal": equal, "class_register_equal": cls_ok}
        print(cond, result[cond], flush=True)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
