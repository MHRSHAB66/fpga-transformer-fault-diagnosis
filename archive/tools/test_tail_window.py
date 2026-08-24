"""
Unit / sanity checks for the backshift tail-window option.

Covers create_windows_from_dataframe and count_normal_windows across:
- num_rows < window_size
- num_rows == window_size
- num_rows exactly divisible by window_size (stride == window_size)
- num_rows not divisible by window_size
- different window_size and stride values (overlapping windows)

The tests use synthetic arrays filled with row indices so the exact start/end
of every window can be verified, including that the backshift tail window:
- keeps the fixed [window_size, num_channels] shape,
- uses only real samples (ends exactly at the last row),
- is never a duplicate of the last standard window.

Usage:
    python -m src.tools.test_tail_window
"""

from __future__ import annotations

import sys

import numpy as np

from src.data.prepare_dataset import (
    create_windows_from_dataframe,
    count_normal_windows,
)


_NUM_CHANNELS = 8


def _make_array(num_rows: int) -> np.ndarray:
    """Array where row i has every channel equal to i, for index checks."""
    col = np.arange(num_rows, dtype=np.float32).reshape(-1, 1)
    return np.repeat(col, _NUM_CHANNELS, axis=1)


class _Checker:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, condition: bool, message: str) -> None:
        status = "PASS" if condition else "FAIL"
        if not condition:
            self.failures += 1
        print(f"  [{status}] {message}")


def _all_full_size(windows, window_size) -> bool:
    return all(w.shape == (window_size, _NUM_CHANNELS) for w in windows)


def main() -> int:
    print("=" * 60)
    print("Tail-window sanity checks")
    print("=" * 60)

    chk = _Checker()

    # --- Case 1: num_rows < window_size -> always empty ---
    print("\n[1] num_rows < window_size (300 < 1024)")
    arr = _make_array(300)
    off = create_windows_from_dataframe(arr, 1024, 1024, include_tail=False)
    on = create_windows_from_dataframe(arr, 1024, 1024, include_tail=True)
    chk.check(len(off) == 0, f"tail off -> 0 windows (got {len(off)})")
    chk.check(len(on) == 0, f"tail on  -> 0 windows (got {len(on)})")
    chk.check(count_normal_windows(300, 1024, 1024) == 0, "count_normal_windows == 0")

    # --- Case 2: num_rows == window_size -> exactly 1, no tail ---
    print("\n[2] num_rows == window_size (1024 == 1024)")
    arr = _make_array(1024)
    off = create_windows_from_dataframe(arr, 1024, 1024, include_tail=False)
    on = create_windows_from_dataframe(arr, 1024, 1024, include_tail=True)
    chk.check(len(off) == 1, f"tail off -> 1 window (got {len(off)})")
    chk.check(len(on) == 1, f"tail on  -> 1 window, no tail added (got {len(on)})")
    chk.check(_all_full_size(on, 1024), "window has fixed [1024, 8] shape")

    # --- Case 3: divisible (stride == window_size) -> no tail ---
    print("\n[3] num_rows divisible by window_size (4096, ws=1024, stride=1024)")
    arr = _make_array(4096)
    off = create_windows_from_dataframe(arr, 1024, 1024, include_tail=False)
    on = create_windows_from_dataframe(arr, 1024, 1024, include_tail=True)
    chk.check(len(off) == 4, f"tail off -> 4 windows (got {len(off)})")
    chk.check(len(on) == 4, f"tail on  -> 4 windows, no tail added (got {len(on)})")
    chk.check(count_normal_windows(4096, 1024, 1024) == 4, "count_normal_windows == 4")

    # --- Case 4: not divisible -> +1 tail ending at last row, no duplicate ---
    print("\n[4] num_rows NOT divisible (250000, ws=1024, stride=1024)")
    num_rows = 250000
    arr = _make_array(num_rows)
    off = create_windows_from_dataframe(arr, 1024, 1024, include_tail=False)
    on = create_windows_from_dataframe(arr, 1024, 1024, include_tail=True)
    n_normal = count_normal_windows(num_rows, 1024, 1024)
    chk.check(len(off) == n_normal, f"tail off -> {n_normal} windows (got {len(off)})")
    chk.check(len(on) == n_normal + 1, f"tail on  -> {n_normal + 1} windows (got {len(on)})")
    chk.check(_all_full_size(on, 1024), "all windows fixed [1024, 8] shape")
    # Tail ends exactly at the last row, and uses only real samples.
    tail = on[-1]
    chk.check(float(tail[-1, 0]) == float(num_rows - 1), "tail window ends at last row index")
    chk.check(float(tail[0, 0]) == float(num_rows - 1024), "tail window starts at num_rows - window_size")
    # Tail is not a duplicate of the previous (last standard) window.
    chk.check(not np.array_equal(on[-1], on[-2]), "tail window != last standard window")

    # --- Case 5: overlapping windows (stride < window_size) ---
    print("\n[5] overlap + remainder (ws=256, stride=128, num_rows=1000)")
    num_rows = 1000
    arr = _make_array(num_rows)
    n_normal = count_normal_windows(num_rows, 256, 128)  # (1000-256)//128 + 1 = 6
    off = create_windows_from_dataframe(arr, 256, 128, include_tail=False)
    on = create_windows_from_dataframe(arr, 256, 128, include_tail=True)
    chk.check(len(off) == n_normal, f"tail off -> {n_normal} windows (got {len(off)})")
    chk.check(len(on) == n_normal + 1, f"tail on  -> {n_normal + 1} windows (got {len(on)})")
    chk.check(_all_full_size(on, 256), "all windows fixed [256, 8] shape")
    chk.check(float(on[-1][-1, 0]) == float(num_rows - 1), "tail ends at last row index")

    # --- Case 6: overlap where last standard window already reaches the end ---
    print("\n[6] overlap, last window already reaches end (ws=256, stride=128, num_rows=1024)")
    # (1024-256)//128 + 1 = 7 windows; last starts at 768, ends at 1024 -> no tail.
    num_rows = 1024
    arr = _make_array(num_rows)
    on = create_windows_from_dataframe(arr, 256, 128, include_tail=True)
    chk.check(len(on) == 7, f"tail on -> 7 windows, no tail added (got {len(on)})")

    # --- Case 7: unsupported tail_mode raises ---
    print("\n[7] unsupported tail_mode raises ValueError")
    arr = _make_array(2000)
    try:
        create_windows_from_dataframe(arr, 1024, 1024, include_tail=True, tail_mode="pad")
        chk.check(False, "expected ValueError for tail_mode='pad'")
    except ValueError:
        chk.check(True, "ValueError raised for tail_mode='pad'")

    print("\n" + "=" * 60)
    if chk.failures == 0:
        print("All tail-window checks PASSED.")
        print("=" * 60)
        return 0
    print(f"{chk.failures} check(s) FAILED.")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())
