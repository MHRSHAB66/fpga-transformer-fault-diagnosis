"""Raw-sample input for the on-FPGA preprocessing core (no normalization on the PC).

Two steps:

  fullscale  scan the TRAINING split only and fix one full scale per channel
             (next power of two >= 1.25 x max |x|) for a signed 24-bit ADC model;
             writes artifacts/preprocessing/raw_input_contract_v1.json
  export     write the test split as raw 24-bit codes, in exactly the window order
             and labels of the existing board input
             (artifacts/results/fpga_fulltest_20261008_141335/), so every window id
             means the same window as before

The windows are cut exactly like src.data.make_dataloaders.build_X_y_from_file_index
(same file filters, preprocess_dataframe, 1024/1024 windowing) but are NOT
z-scored or quantized: that now happens inside the FPGA
(src/quantize/onchip_preprocess.py is its bit-exact reference).

    python -m tools.export_fpga_testset_raw fullscale
    python -m tools.export_fpga_testset_raw export --out artifacts/results/fpga_rawinput_testset_20261109
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np

from src.config import DATA_RAW_DIR, EXPECTED_NUM_COLUMNS, LABEL_MAP
from src.data.load_data import build_file_index, load_one_csv
from src.data.make_dataloaders import split_file_index
from src.data.prepare_dataset import create_windows_from_dataframe, preprocess_dataframe
from src.quantize.onchip_preprocess import ADC_BITS, CODE_MAX, CODE_MIN, to_adc_codes

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "artifacts" / "preprocessing" / "raw_input_contract_v1.json"
OLD_BOARD_INPUT = REPO / "artifacts" / "results" / "fpga_fulltest_20261008_141335"
WINDOW = 1024
HEADROOM = 1.25


def file_windows(index_df):
    """Yield (label_id, float32 windows [n, 1024, 8]) per kept file, in index order."""
    for _, row in index_df.iterrows():
        if row["label"] not in LABEL_MAP:
            continue
        df = preprocess_dataframe(load_one_csv(row["filepath"]))
        if df is None or df.shape[1] != EXPECTED_NUM_COLUMNS or len(df) < WINDOW:
            continue
        values = df.to_numpy(copy=True).astype(np.float32, copy=False)
        windows = create_windows_from_dataframe(values, window_size=WINDOW, stride=WINDOW)
        if windows:
            yield LABEL_MAP[row["label"]], np.asarray(windows, dtype=np.float32)


def splits():
    index_df = build_file_index(Path(DATA_RAW_DIR))
    index_df = index_df[index_df["label"].isin(LABEL_MAP.keys())].copy()
    return split_file_index(index_df)


def cmd_fullscale(_args):
    train_df, _, _ = splits()
    peak = np.zeros(EXPECTED_NUM_COLUMNS)
    n_files = 0
    for _, w in file_windows(train_df):
        peak = np.maximum(peak, np.abs(w).max(axis=(0, 1)))
        n_files += 1
    fs = [float(2 ** math.ceil(math.log2(HEADROOM * p))) for p in peak]
    contract = {
        "name": "raw_input_contract_v1",
        "adc_bits": ADC_BITS,
        "code": "clip(round(x / FS_c * 2^23), -2^23, 2^23 - 1), little-endian int32 words",
        "layout": "8192 words per window, row-major [sample, channel], channel order of src/config.py",
        "full_scale_per_channel": fs,
        "lsb_per_channel": [f / (1 << (ADC_BITS - 1)) for f in fs],
        "train_peak_abs_per_channel": [round(float(p), 6) for p in peak],
        "rule": "FS_c = next power of two >= %.2f x max|x| over the training split only" % HEADROOM,
        "train_files_scanned": n_files,
        "note": "The FPGA's per-window z-score is scale invariant, so FS_c only sets the "
                "resolution of the simulated ADC; the hardware never uses it.",
    }
    CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT.write_text(json.dumps(contract, indent=1), encoding="utf-8")
    print(json.dumps(contract, indent=1))


def cmd_export(args):
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    fs = contract["full_scale_per_channel"]
    _, _, test_df = splits()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    labels, clipped, n = [], np.zeros(len(fs), dtype=np.int64), 0
    with gzip.open(out / "fpga_testset_raw.i32.gz", "wb", compresslevel=6) as gz:
        for label, w in file_windows(test_df):
            codes = to_adc_codes(w, fs)
            clipped += ((codes == CODE_MIN) | (codes == CODE_MAX)).sum(axis=(0, 1))
            gz.write(codes.astype("<i4").tobytes())
            labels.extend([label] * len(w))
            n += len(w)
    labels = np.asarray(labels, dtype=np.int64)
    old = np.loadtxt(OLD_BOARD_INPUT / "labels.txt", dtype=np.int64)
    same = bool(len(old) == n and (old == labels).all())
    np.savetxt(out / "labels.txt", labels, fmt="%d")
    meta = {"n_windows": n, "words_per_window": WINDOW * EXPECTED_NUM_COLUMNS,
            "contract": str(CONTRACT.relative_to(REPO)).replace("\\", "/"),
            "full_scale_per_channel": fs,
            "clipped_samples_per_channel": clipped.tolist(),
            "labels_equal_previous_board_input": same,
            "previous_board_input": str(OLD_BOARD_INPUT.relative_to(REPO)).replace("\\", "/")}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    if not same:
        raise SystemExit("window order/labels differ from the previous board input")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fullscale")
    e = sub.add_parser("export")
    e.add_argument("--out", required=True)
    args = ap.parse_args()
    {"fullscale": cmd_fullscale, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    main()
