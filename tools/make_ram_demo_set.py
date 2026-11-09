"""Build the documented finite input set for the RAM real-time demo mode.

Selects REAL test-set windows (class_balanced plan from host.live_sampling, i.e.
round-robin over the ground-truth classes -- used only to pick which windows to
show; the FPGA still decides every class) and copies their INT10 input codes,
byte for byte, from the staged test-set input into one uncompressed file that the
board preloads into RAM. No value is changed.

Output (versioned, never overwritten): artifacts/ram_demo_sets/<name>/
  set.i32          n records x 32768 bytes (<i4, 1024 x 8, same layout as the gz)
  indices.txt      dataset index of each record, in order
  provenance.json  source file + sha256, plan, per-class counts, sha256 of set.i32,
                   recorded FPGA class (2026-10-08) and INT10 reference logits per record

Usage: python -m tools.make_ram_demo_set --n 256 --name class_balanced_256_v1
Raw-input core (preprocessing in the FPGA): same windows from the raw 24-bit input,
    python -m tools.make_ram_demo_set --name class_balanced_256_raw_v1 \n        --source artifacts/results/fpga_rawinput_testset_20261109/fpga_testset_raw.i32.gz \n        --reference artifacts/results/onchip_preprocess_20261109/numpy_reference_raw/numpy_export_logits.txt
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from host.live_sampling import build_plan

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "artifacts/results/fpga_fulltest_20261008_141335"
SRC = FULL / "fpga_testset_in.i32.gz"
REF = ROOT / "artifacts/results/board_vs_reference_20261022/numpy_reference/numpy_export_logits.txt"
REC = 1024 * 8 * 4


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--name", default="class_balanced_256_v1")
    ap.add_argument("--source", type=Path, default=SRC, help="input gz (INT10 codes or raw codes)")
    ap.add_argument("--reference", type=Path, default=REF, help="reference logits for that input")
    args = ap.parse_args()
    src, ref_path = args.source.resolve(), args.reference.resolve()
    out = ROOT / "artifacts/ram_demo_sets" / args.name
    out.mkdir(parents=True, exist_ok=False)
    labels = [int(x) for x in (FULL / "labels.txt").read_text().split()]
    pred = np.loadtxt(FULL / "fpga_pred.txt", dtype=np.int64)
    ref = np.loadtxt(ref_path, dtype=np.int64)[:, 1:]
    idx = build_plan("class_balanced", total=len(labels), n=args.n, labels=labels)
    want = {i: k for k, i in enumerate(idx)}
    recs = [None] * len(idx)
    with gzip.open(src, "rb") as f:
        for i in range(max(idx) + 1):
            b = f.read(REC)
            if len(b) != REC:
                raise SystemExit("short input at %d" % i)
            if i in want:
                recs[want[i]] = b
    with open(out / "set.i32", "wb") as f:
        for b in recs:
            f.write(b)
    (out / "indices.txt").write_text("\n".join(str(i) for i in idx) + "\n", encoding="ascii")
    counts = {c: sum(1 for i in idx if labels[i] == c) for c in range(4)}
    prov = {"name": args.name, "n": len(idx), "record_bytes": REC, "layout": "<i4, 1024 samples x 8 channels",
            "plan": "host.live_sampling.build_plan('class_balanced', seed=0)",
            "source": str(src.relative_to(ROOT)).replace("\\", "/"), "source_sha256": sha256(src),
            "reference": str(ref_path.relative_to(ROOT)).replace("\\", "/"),
            "set_sha256": sha256(out / "set.i32"), "true_class_counts": counts,
            "indices": idx,
            "recorded_fpga_class_20261008": [int(pred[i]) for i in idx],
            "reference_logits_int10": [[int(v) for v in ref[i]] for i in idx]}
    (out / "provenance.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    print(json.dumps({k: prov[k] for k in ("name", "n", "set_sha256", "true_class_counts")}))


if __name__ == "__main__":
    main()
