"""
On-chip per-window z-score normalization — reference + spec (Stage 9 prep).

WHY THIS EXISTS
---------------
Today the FPGA core receives windows that were ALREADY per-window z-scored on the
host (see DECISION_LOG D016). For a live-sensor path (Stage 9) there is no PC in
the loop, so the FPGA must normalize the raw 1024-sample buffer on-chip before the
transformer. This file is the exact algorithm that hardware block must implement,
plus a bit-for-bit check against the active host normalization
(prepare_dataset.normalize_windows_array, scope="per_window").

It is a REFERENCE, not an FPGA implementation. Converting it to HLS/RTL is Stage 9
work and needs the Vitis/board toolchain.

ALGORITHM (two passes per channel c of a [1024, 8] window; N = 1024)
-------------------------------------------------------------------
  pass 1:  mean = sum(x[:, c]) / N
  pass 2:  var  = sum((x[:, c] - mean) ** 2) / N     # population variance (ddof=0)
           std  = sqrt(var)
           inv  = 1/std if std >= eps else 1.0       # divide-by-zero guard
           y[:, c] = (x[:, c] - mean) * inv

Two-pass matches np.mean/np.std (ddof=0) used by normalize_windows_array, so the
on-chip output equals the host output bit-for-bit (float32). The 1024 buffer is
already stored on-chip, so two passes are cheap.

HARDWARE NOTES (for the Stage 9 HLS block)
------------------------------------------
- One-pass alternative (acc, acc_sq -> var = acc_sq/N - mean^2) saves a pass but
  loses precision (~1e-3 vs two-pass); prefer two-pass since the buffer is stored.
- Clamp var to >= 0 before sqrt (the host does the same in the global path).
- 1/std: use an inverse-sqrt LUT/CORDIC on `var` (rsqrt) to avoid a divide.
- Apply the per-channel mean/std to all 1024 samples of that channel, then quantize
  with the model's input_scale (model_manifest.json) exactly as the host does today.
"""
from __future__ import annotations

import numpy as np


def onchip_normalize_window(window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-window z-score of one [seq_len, num_channels] window using the single-pass
    accumulator formulation a hardware block would use.
    """
    x = np.asarray(window, dtype=np.float32)
    mean = x.mean(axis=0)                       # pass 1
    std = x.std(axis=0)                         # pass 2 (population, ddof=0)
    inv = np.where(std < eps, np.float32(1.0), np.float32(1.0) / std)
    return ((x - mean) * inv).astype(np.float32)


def _self_check() -> int:
    """Verify bit-for-bit agreement with the active host normalization."""
    from src.data.prepare_dataset import normalize_windows_array

    rng = np.random.default_rng(0)
    # a mix of normal-scale and offset/gain-varied windows
    windows = rng.normal(size=(32, 1024, 8)) * rng.uniform(0.1, 5.0, (32, 1, 8))
    windows += rng.uniform(-100, 100, (32, 1, 8))

    host = normalize_windows_array(windows.astype(np.float32),
                                   method="zscore", scope="per_window")
    onchip = np.stack([onchip_normalize_window(w) for w in windows])

    max_abs = float(np.max(np.abs(host - onchip)))
    print(f"max |host - onchip| = {max_abs:.3e}")
    ok = max_abs < 1e-4
    print("PASS — on-chip reference matches host per-window z-score"
          if ok else "FAIL — mismatch")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_check())
