"""Non-overlapping 4:1 (configurable) token aggregation for the SEQ=256 experiment.

ISOLATED, additive module. It does NOT change the approved SEQ=1024 path: callers
opt in explicitly with an aggregation factor > 1. The physical observation window
(1024 samples, 8 channels, 50 kHz, 20.48 ms) is preserved -- aggregation only
groups already-normalized samples into fewer transformer tokens; no sample and no
channel is discarded.

Order contract (must match everywhere): apply the existing approved per-window,
per-channel z-score normalization FIRST, then aggregate 4:1.

    [L, C] normalized  --non-overlapping mean over `factor` steps-->  [L/factor, C]
    token[t, c] = mean( x[factor*t + 0 .. factor*t + factor-1, c] )

Two exact variants are provided and MUST agree with their consumers:

* aggregate_mean_float : exact arithmetic mean (float64) -- for the Python float
  model / training.
* aggregate_round_div  : integer signed-rounding division used by the int10
  reference and the HLS aggregation block. For factor=4 the rounding rule is
  ROUND-HALF-AWAY-FROM-ZERO implemented with an efficient shift:

        s = a + b + c + d                       # exact integer sum
        q = (s + 2) >> 2         if s >= 0      # +factor/2 then arithmetic >>2
        q = -((-s + 2) >> 2)     if s <  0      # symmetric for negatives

  This is bit-exact between the NumPy reference and the HLS block (same add, same
  bias factor/2, same arithmetic right shift, same sign handling). factor must be a
  power of two so the divide lowers to (>> log2(factor)).
"""
from __future__ import annotations

import numpy as np


def _check(x: np.ndarray, factor: int) -> None:
    if factor < 1:
        raise ValueError("aggregation factor must be >= 1")
    if x.ndim < 2:
        raise ValueError("expected at least [L, C]; got shape %r" % (x.shape,))
    L = x.shape[-2]
    if L % factor != 0:
        raise ValueError(
            "sequence length %d is not divisible by aggregation factor %d" % (L, factor)
        )
    if factor & (factor - 1) != 0:
        raise ValueError("aggregation factor must be a power of two (got %d)" % factor)


def aggregate_mean_float(
    x: np.ndarray, factor: int = 4, output_dtype: np.dtype = np.float32
) -> np.ndarray:
    """Non-overlapping arithmetic mean over the time axis (axis -2).

    Accepts [..., L, C]; returns [..., L//factor, C]. The mean is ACCUMULATED in
    float64 for deterministic numerical behaviour, then cast to ``output_dtype``
    (default float32) because the DataLoader/model use float32 weights -- returning
    float64 would raise a dtype mismatch in the forward pass.

    factor==1 is a no-op copy that PRESERVES the input values AND dtype, so the
    approved SEQ=1024 path is untouched when callers do not opt in.
    """
    _check(x, factor)
    if factor == 1:
        return np.array(x, copy=True)  # preserves values and dtype
    xf = np.asarray(x, dtype=np.float64)
    *lead, L, C = xf.shape
    grouped = xf.reshape(*lead, L // factor, factor, C)
    return grouped.mean(axis=-2).astype(output_dtype)


def aggregate_round_div(x: np.ndarray, factor: int = 4) -> np.ndarray:
    """Integer signed round-half-away-from-zero mean (matches the HLS block).

    token = signed_round( sum(group) / factor ) via (|s| + factor/2) >> log2(factor)
    with the sign reapplied. Inputs/outputs are integers (int64 accumulation).
    """
    _check(x, factor)
    if factor == 1:
        return np.array(x, copy=True)
    xi = np.asarray(x, dtype=np.int64)
    *lead, L, C = xi.shape
    s = xi.reshape(*lead, L // factor, factor, C).sum(axis=-2)  # exact integer sum
    shift = int(factor).bit_length() - 1  # log2(factor)
    half = factor >> 1
    mag = (np.abs(s) + half) >> shift
    return np.where(s < 0, -mag, mag).astype(np.int64)


def aggregated_seq_len(seq_len: int, factor: int = 4) -> int:
    if seq_len % factor != 0:
        raise ValueError("seq_len %d not divisible by factor %d" % (seq_len, factor))
    return seq_len // factor
