"""Per-window NumPy INT10 reference logits on the EXACT board input file.

Reads the INT10 ingress codes the physical board consumed
(fpga_testset_in.i32.gz, 1024x8 int32 per window, row-major [sample, channel]),
applies the reference 4:1 aggregate_round_div, and runs
src.quantize.integer_model.IntegerTransformer.infer_int -- the same numeric path
as src/quantize/evaluate_seq256_ingress_int10_shard.py and pynq NumpyBackend.

Two scale sets are supported (both read-only; neither changes any artifact):

  --scales export   artifacts/quantized/f_seq256_int10/integer_config.json
                    (20-file train calibration; the set the HLS weights header
                    and the deployed bitstream were generated from)
  --scales calib12  recomputed exactly as evaluate_seq256_ingress_int10_shard.py
                    does (12 train files, seed 42); this is the calibration
                    behind the reported 97.7359 % "software INT10 reference"

Writes (new dir, nothing overwritten):
    <out>/numpy_<scales>_logits.txt   "<idx> l0 l1 l2 l3"
    <out>/numpy_<scales>_scales.json  the scale set actually used
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = (PROJECT_ROOT / "artifacts" / "experiments" / "transformer_f1_seq256_pool4"
              / "kaggle_v3" / "seq256_pool4" / "scratch" / "best.pt")
EXPORT_SCALES = PROJECT_ROOT / "artifacts" / "quantized" / "f_seq256_int10" / "integer_config.json"
CFG = {"input_dim": 8, "seq_len": 256, "num_classes": 4, "d_model": 32, "nhead": 4,
       "num_layers": 1, "dim_feedforward": 64, "dropout": 0.1}
REC = 1024 * 8


def _model(scales: dict):
    import torch
    from src.quantize.evaluate_integer_bits import set_bits
    from src.quantize.integer_model import IntegerTransformer
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = {k: v.cpu().numpy() for k, v in ck["model_state_dict"].items()}
    set_bits(10)
    m = IntegerTransformer(state, CFG)
    m.set_scales(scales)
    return m


def calib12_scales() -> dict:
    """Exactly the calibration of evaluate_seq256_ingress_int10_shard.py."""
    import torch
    from src.config import DATA_RAW_DIR, LABEL_MAP
    from src.data.load_data import build_file_index
    from src.data.make_dataloaders import split_file_index, build_X_y_from_file_index
    from src.quantize.evaluate_integer_bits import calibrate_bits, set_bits
    from src.quantize.integer_model import IntegerTransformer
    index_df = build_file_index(Path(DATA_RAW_DIR))
    index_df = index_df[index_df["label"].isin(LABEL_MAP.keys())].copy()
    train_df, _, _ = split_file_index(index_df)
    calib_df = train_df.sample(n=12, random_state=42).reset_index(drop=True)
    Xc, _, _, _, _ = build_X_y_from_file_index(calib_df, window_size=1024, stride=1024,
                                               aggregation_factor=4)
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = {k: v.cpu().numpy() for k, v in ck["model_state_dict"].items()}
    set_bits(10)
    m = IntegerTransformer(state, CFG)
    return {k: float(v) for k, v in calibrate_bits(m, Xc, 10).items()}


def _shard(job):
    gz, start, count, scales, chunk = job
    from src.data.token_aggregation import aggregate_round_div
    m = _model(scales)
    s_in = float(scales["input"])
    out = []
    with gzip.open(gz, "rb") as f:
        skip = start * REC * 4
        while skip:
            b = f.read(min(skip, 64 * REC * 4))
            if not b:
                break
            skip -= len(b)
        done = 0
        while done < count:
            k = min(chunk, count - done)
            raw = f.read(k * REC * 4)
            codes = np.frombuffer(raw, dtype="<i4").astype(np.int64).reshape(k, 1024, 8)
            q256 = aggregate_round_div(codes, factor=4)
            lg = m.infer_int(q256.astype(np.float64) * s_in)
            for i in range(k):
                out.append("%d %d %d %d %d" % (start + done + i, *lg[i]))
            done += k
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--scales", choices=["export", "calib12"], required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.scales == "export":
        scales = json.loads(EXPORT_SCALES.read_text())["scales"]
    else:
        scales = calib12_scales()
    (args.out / f"numpy_{args.scales}_scales.json").write_text(
        json.dumps(scales, indent=2), encoding="utf-8")

    t0 = time.time()
    per = (args.n + args.workers - 1) // args.workers
    jobs = [(args.input, s * per, min(per, args.n - s * per), scales, args.chunk)
            for s in range(args.workers) if s * per < args.n]
    lines = []
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        for part in ex.map(_shard, jobs):
            lines.extend(part)
    assert len(lines) == args.n, (len(lines), args.n)
    (args.out / f"numpy_{args.scales}_logits.txt").write_text("\n".join(lines) + "\n",
                                                              encoding="ascii")
    print(f"wrote {len(lines)} windows, scales={args.scales}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
