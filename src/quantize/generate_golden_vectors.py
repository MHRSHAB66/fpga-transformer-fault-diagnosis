"""
Stage 6d / 7 — Golden vector generation.

Produces a small, class-balanced set of reference input/output pairs from the
int16 fixed-point simulator. These are the ground truth the FPGA testbench
(Stage 8) checks against: the HLS/RTL core must reproduce the predicted class
for every golden input, with logits within the approved tolerance.

For each selected test window we save:
- the RAW window ``[1024, 8]`` (pre-normalization) — exercises the full hardware
  path including the per-window z-score the FPGA must perform;
- the NORMALIZED window ``[1024, 8]`` (what actually enters the model) — lets the
  testbench validate the core alone, without the normalization block;
- the fixed-point logits ``[4]`` and predicted class id (the golden output);
- the float32 logits/class for reference.

Ordering contract (must match hardware): windows are flattened row-major as
``[sequence, channel]`` — i.e. for each of the 1024 time steps, the 8 channel
values in EXPECTED_COLUMNS order. Class ids follow src.config.LABEL_MAP.

Outputs (artifacts/golden/):
- ``golden_vectors.npz``            — all arrays.
- ``golden_input_normalized.txt``   — N lines x 8192 floats (model input).
- ``golden_input_raw.txt``          — N lines x 8192 floats (pre-normalization).
- ``golden_expected.txt``           — per line: idx label_id pred_id l0 l1 l2 l3.
- ``golden_manifest.json``          — format, ordering, tolerance, class map.

Run
---
    python -m src.quantize.generate_golden_vectors --per-class 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import (
    DATA_RAW_DIR,
    PROJECT_ROOT,
    LABEL_MAP,
    EXPECTED_COLUMNS,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    NORMALIZATION_METHOD,
    RANDOM_SEED,
)
from src.data.load_data import build_file_index
from src.data.make_dataloaders import split_file_index
from src.data.preprocessed_cache import load_or_create_preprocessed_array
from src.data.prepare_dataset import create_windows_from_dataframe, normalize_windows_array
from src.quantize.evaluate_quantized import DEFAULT_CHECKPOINT, load_model
from src.quantize.fixed_point_model import FixedPointTransformer, FixedPointConfig

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "golden"
LABEL_NAMES_BY_ID = {v: k for k, v in LABEL_MAP.items()}
# Golden-output match tolerance for the HLS testbench (Stage 8). Class must match
# exactly; logits are compared with this absolute tolerance.
LOGIT_ABS_TOLERANCE = 0.05


def select_windows(per_class: int, seed: int):
    """
    Pick ``per_class`` raw windows for each class from the real test split.

    Returns lists of (raw_window, label_id, source_file, window_index).
    """
    index_df = build_file_index(DATA_RAW_DIR)
    index_df = index_df[index_df["label"].isin(LABEL_MAP.keys())].copy()
    _, _, test_index_df = split_file_index(index_df)

    rng = np.random.default_rng(seed)
    selected = []
    for label_name, label_id in LABEL_MAP.items():
        files = test_index_df[test_index_df["label"] == label_name]
        files = files.sample(frac=1, random_state=seed).reset_index(drop=True)
        collected = 0
        for _, row in files.iterrows():
            if collected >= per_class:
                break
            values = load_or_create_preprocessed_array(row["filepath"])
            if values is None:
                continue
            windows = create_windows_from_dataframe(
                values, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE
            )
            if not windows:
                continue
            # Take one window from a deterministic position in this file.
            widx = int(rng.integers(0, len(windows)))
            selected.append((
                np.asarray(windows[widx], dtype=np.float32),
                label_id,
                Path(row["filepath"]).name,
                widx,
            ))
            collected += 1
    return selected


def main():
    parser = argparse.ArgumentParser(description="Generate golden vectors (Stage 6d/7).")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--weight-bits", type=int, default=16)
    parser.add_argument("--act-bits", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_model, float_state, model_cfg = load_model(args.checkpoint, device)
    fp_model = FixedPointTransformer(
        float_state, model_cfg,
        FixedPointConfig(weight_bits=args.weight_bits, act_bits=args.act_bits),
    )

    selected = select_windows(args.per_class, args.seed)
    n = len(selected)
    print(f"[info] selected {n} golden windows "
          f"({args.per_class}/class x {len(LABEL_MAP)} classes)")

    raw = np.stack([s[0] for s in selected])            # (n, 1024, 8)
    labels = np.array([s[1] for s in selected], dtype=np.int64)
    # Per-window z-score normalization, identical to training.
    normalized = normalize_windows_array(
        raw.copy(), method=NORMALIZATION_METHOD, scope="per_window"
    ).astype(np.float32)

    with torch.no_grad():
        xb = torch.from_numpy(normalized).float().to(device)
        fp_logits = fp_model.forward(xb).cpu().numpy()
        ref_logits = ref_model(xb).cpu().numpy()
    fp_pred = fp_logits.argmax(axis=1)
    ref_pred = ref_logits.argmax(axis=1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- npz ---------------------------------------------------------------
    np.savez_compressed(
        OUTPUT_DIR / "golden_vectors.npz",
        raw_input=raw,
        normalized_input=normalized,
        label_id=labels,
        fixed_point_logits=fp_logits.astype(np.float32),
        fixed_point_pred=fp_pred.astype(np.int64),
        float_logits=ref_logits.astype(np.float32),
        float_pred=ref_pred.astype(np.int64),
    )

    # --- testbench text files (row-major [seq, channel]) -------------------
    def write_flat(path, arr3d):
        with open(path, "w", encoding="ascii") as f:
            for w in arr3d:
                f.write(" ".join(f"{v:.8g}" for v in w.reshape(-1)) + "\n")

    write_flat(OUTPUT_DIR / "golden_input_normalized.txt", normalized)
    write_flat(OUTPUT_DIR / "golden_input_raw.txt", raw)

    with open(OUTPUT_DIR / "golden_expected.txt", "w", encoding="ascii") as f:
        f.write("# idx label_id pred_id logit0 logit1 logit2 logit3\n")
        for i in range(n):
            logits = " ".join(f"{v:.8g}" for v in fp_logits[i])
            f.write(f"{i} {labels[i]} {fp_pred[i]} {logits}\n")

    # --- manifest ----------------------------------------------------------
    fp_correct = int((fp_pred == labels).sum())
    fp_vs_float = int((fp_pred == ref_pred).sum())
    manifest = {
        "stage": "6d_golden_vectors",
        "checkpoint": str(args.checkpoint),
        "num_vectors": n,
        "per_class": args.per_class,
        "weight_bits": args.weight_bits,
        "act_bits": args.act_bits,
        "window_shape": [WINDOW_SIZE, len(EXPECTED_COLUMNS)],
        "flatten_order": "row-major [sequence, channel]; channel order = EXPECTED_COLUMNS",
        "channel_order": EXPECTED_COLUMNS,
        "class_map": LABEL_MAP,
        "normalization": {"scope": "per_window", "method": NORMALIZATION_METHOD},
        "golden_output_source": "int16 fixed-point simulator (Stage 6b)",
        "match_criteria": {
            "class": "exact",
            "logit_abs_tolerance": LOGIT_ABS_TOLERANCE,
        },
        "self_check": {
            "fixed_point_correct": fp_correct,
            "fixed_point_vs_float_agree": fp_vs_float,
        },
        "files": {
            "arrays": "golden_vectors.npz",
            "input_normalized": "golden_input_normalized.txt",
            "input_raw": "golden_input_raw.txt",
            "expected": "golden_expected.txt",
        },
    }
    with open(OUTPUT_DIR / "golden_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[info] fixed-point correct: {fp_correct}/{n}, "
          f"agree with float: {fp_vs_float}/{n}")
    print(f"[info] golden vectors saved to: {OUTPUT_DIR}")
    print("\n idx | true            | fp_pred         | correct")
    print("-" * 55)
    for i in range(n):
        print(f" {i:>3} | {LABEL_NAMES_BY_ID[labels[i]]:<15} | "
              f"{LABEL_NAMES_BY_ID[fp_pred[i]]:<15} | "
              f"{'yes' if fp_pred[i] == labels[i] else 'NO'}")


if __name__ == "__main__":
    main()
