"""
Dataset-wide channel-order consistency check for MAFAULDA.

`src.tools.channel_analysis` verifies the 8-column identity (col 0 = tachometer,
col 7 = microphone) on a SINGLE representative file. This script extends that
check to EVERY CSV file in the dataset, so the claim "all files share the same
channel order" is backed by evidence rather than one sample.

Method
------
Absolute zero-crossing thresholds do NOT work dataset-wide: an accelerometer's
crossing rate depends heavily on the operating condition and overlaps both the
tachometer (low) and microphone (high) ranges. The physically reliable invariant
is instead the RELATIVE ordering *within each file*: because all channels of a
file are recorded together, the tachometer is always the slowest-crossing signal
and the microphone always the fastest.

So per file we compute the zero-crossing rate of all 8 columns and require:
  - the column with the FEWEST crossings is col 0  (tachometer)
  - the column with the MOST crossings  is col 7  (microphone)
This tests "is the channel order consistent" directly, with no fragile absolute
thresholds. The raw zc vector is recorded for any file that fails.

For speed, only the first second (MAFAULDA_SAMPLE_RATE samples) of each file is
read; that is enough for these statistics and matches how the zero-crossing rate
is already measured in channel_analysis (first-second segment).

Usage
-----
    python -m src.tools.verify_channel_order

Output
------
    artifacts/results/channel_order_consistency.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import (
    DATA_RAW_DIR,
    EXPECTED_COLUMNS,
    EXPECTED_NUM_COLUMNS,
    MAFAULDA_SAMPLE_RATE,
    PROJECT_ROOT,
)
from src.data.load_data import build_file_index, project_relative_or_absolute

OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "results" / "channel_order_consistency.json"
SR = MAFAULDA_SAMPLE_RATE  # 50 000 Hz

TACHO_COL = 0
MIC_COL = EXPECTED_NUM_COLUMNS - 1  # 7


def zero_crossings(sig: np.ndarray) -> int:
    """Zero-crossings relative to the mean over the first second."""
    mu = float(np.mean(sig))
    seg = sig[:SR]
    return int(np.sum(np.diff(np.sign(seg - mu)) != 0))


def check_file(rel_path: str) -> Dict:
    """Verify col 0 has the min and col 7 has the max zero-crossing rate."""
    df = pd.read_csv(PROJECT_ROOT / rel_path, header=None, nrows=SR)

    result: Dict = {"filepath": rel_path, "n_columns": int(df.shape[1])}

    if df.shape[1] != EXPECTED_NUM_COLUMNS:
        result["status"] = "WRONG_COLUMN_COUNT"
        result["consistent"] = False
        return result

    zc = [zero_crossings(df.iloc[:, c].to_numpy(dtype=np.float64))
          for c in range(EXPECTED_NUM_COLUMNS)]

    tacho_is_min = int(np.argmin(zc)) == TACHO_COL
    mic_is_max = int(np.argmax(zc)) == MIC_COL

    result["consistent"] = tacho_is_min and mic_is_max
    result["status"] = "OK" if result["consistent"] else "MISMATCH"
    if not result["consistent"]:
        result["zero_crossings_per_col"] = zc
        result["argmin_col"] = int(np.argmin(zc))
        result["argmax_col"] = int(np.argmax(zc))
    return result


def main() -> int:
    print("=" * 65)
    print("MAFAULDA dataset-wide channel-order consistency check")
    print("=" * 65)

    index_df = build_file_index(DATA_RAW_DIR)
    total = len(index_df)
    if total == 0:
        print("[ERROR] No CSV files found.")
        return 1

    print(f"[INFO] Checking {total} files "
          f"(first {SR} samples each, expecting {EXPECTED_NUM_COLUMNS} columns)\n")

    t0 = time.perf_counter()
    consistent = 0
    problems: List[Dict] = []

    for i, rel_path in enumerate(index_df["filepath"].tolist(), start=1):
        try:
            res = check_file(rel_path)
        except Exception as exc:  # malformed/unreadable file
            res = {"filepath": rel_path, "status": f"ERROR: {exc}",
                   "consistent": False}

        if res.get("consistent"):
            consistent += 1
        else:
            problems.append(res)
            print(f"  [{res['status']}] {rel_path}")

        if i % 200 == 0:
            print(f"  ... {i}/{total} checked")

    elapsed = time.perf_counter() - t0
    all_ok = consistent == total

    summary = {
        "dataset_dir": str(project_relative_or_absolute(DATA_RAW_DIR)),
        "expected_column_order": list(EXPECTED_COLUMNS),
        "expected_num_columns": EXPECTED_NUM_COLUMNS,
        "method": (
            "Relative zero-crossing ordering within each file (first second). A "
            "file is consistent when col 0 has the fewest crossings (tachometer) "
            "and col 7 has the most (microphone). Robust to operating-condition "
            "variation, unlike absolute thresholds."
        ),
        "total_files": total,
        "consistent_files": consistent,
        "inconsistent_files": total - consistent,
        "all_consistent": all_ok,
        "elapsed_seconds": round(elapsed, 1),
        "problems": problems,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 65)
    if all_ok:
        print(f"RESULT: all {total}/{total} files have a CONSISTENT channel order.")
    else:
        print(f"RESULT: {consistent}/{total} consistent, "
              f"{total - consistent} problematic (see JSON).")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"-> {project_relative_or_absolute(OUTPUT_PATH)}")
    print("=" * 65)
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
