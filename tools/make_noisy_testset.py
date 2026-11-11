"""Write a perturbed copy of the raw test input (fpga_testset_raw.i32.gz) for a
board run, with EXACTLY the perturbation of tools/robustness_cmodel.py
(perturb(windows, first_index, cond); per-window seeds), so the board can be
compared window by window with the C-model sweep.

    python -m tools.make_noisy_testset --cond snr0 \
        --out artifacts/results/noisy_testsets_20261111/fpga_testset_raw_snr0.i32.gz
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.robustness_cmodel import RAW, REC, perturb


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cond", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=95404)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    done = 0
    with gzip.open(RAW, "rb") as fin, gzip.open(args.out, "wb", compresslevel=1) as fout:
        while done < args.n:
            k = min(512, args.n - done)
            w = np.frombuffer(fin.read(k * REC), dtype="<i4").astype(np.int64).reshape(k, 1024, 8)
            b = perturb(w, done, args.cond).tobytes()
            h.update(b)
            fout.write(b)
            done += k
    meta = {"source": str(RAW), "condition": args.cond, "windows": done,
            "perturbation": "tools/robustness_cmodel.perturb (seed [20261110, window index])",
            "sha256_uncompressed": h.hexdigest()}
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
