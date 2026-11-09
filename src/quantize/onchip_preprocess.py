"""Integer per-window z-score + INT10 quantization, as done INSIDE the FPGA.

This is the bit-exact reference of the raw-input ingress of the HLS top
`transformer_top_seq256_raw1024_dsp_margin_argmax.cpp`. With it, the board
receives raw sensor samples (signed 24-bit ADC codes) and performs the whole
preprocessing itself; the PC/ARM no longer normalizes or quantizes.

Raw input contract (artifacts/preprocessing/raw_input_contract_v1.json):
  code = clip(round(x / FS_c * 2^23), -2^23, 2^23 - 1)   per channel c,
i.e. a 24-bit ADC whose full scale FS_c is a fixed power of two per sensor
channel. The z-score is scale invariant, so the hardware never needs FS_c.

Per window (N = 1024 samples) and channel, integer operations only:
  S1  = sum(x)                            exact, int64
  mu  = (S1 + 512) >> 10                  integer mean, round half up
  d   = x - mu                            |d| < 2^24
  Q   = sum(d * d)                        exact, < 2^58
  Q == 0  ->  every code of the channel is 0 (the float path gives 0 too)
  k   = (62 - bitlen(Q)) // 2             Q << 2k lies in [2^60, 2^62)
  r   = isqrt(Q << 2k)                    floor square root, 31 bits
  m   = (C_FX << 31) // r                 C_FX = round(32 / s_in * 2^20)
  sh  = 51 - k
  q   = sign(d) * ((|d| * m + 2^(sh-1)) >> sh), clamped to +-511
which is q = round(d / (s_in * sigma)) with sigma = sqrt(Q / 1024), the
population standard deviation, and s_in the model's INT10 input scale.
The 4:1 token aggregation that follows is unchanged (aggregate_round_div).
"""
from __future__ import annotations

import math

import numpy as np

WINDOW = 1024
ADC_BITS = 24
CODE_MIN = -(1 << (ADC_BITS - 1))
CODE_MAX = (1 << (ADC_BITS - 1)) - 1
ACT_QMAX = 511
SQRT_N = 32               # sqrt(WINDOW)
C_FRAC = 20


def c_fx(input_scale: float) -> int:
    """Fixed-point constant 32 / s_in in Q20 (shared with the HLS header)."""
    return int(round(SQRT_N / input_scale * (1 << C_FRAC)))


def to_adc_codes(x: np.ndarray, full_scale) -> np.ndarray:
    """Float samples [..., 1024, 8] -> signed 24-bit codes (int32), per-channel FS."""
    fs = np.asarray(full_scale, dtype=np.float64)
    codes = np.rint(np.asarray(x, dtype=np.float64) / fs * (1 << (ADC_BITS - 1)))
    return np.clip(codes, CODE_MIN, CODE_MAX).astype(np.int32)


def channel_factors(q_sum: int, input_scale: float):
    """(m, sh) for one channel from Q = sum(d^2); m = 0 means an all-zero channel."""
    if q_sum == 0:
        return 0, 0
    k = (62 - int(q_sum).bit_length()) // 2
    r = math.isqrt(int(q_sum) << (2 * k))
    m = (c_fx(input_scale) << 31) // r
    return m, 51 - k


def preprocess_window(codes: np.ndarray, input_scale: float) -> np.ndarray:
    """One raw window [1024, 8] of 24-bit codes -> INT10 codes [1024, 8] (int64)."""
    x = np.asarray(codes, dtype=np.int64)
    if x.shape[0] != WINDOW or x.ndim != 2:
        raise ValueError("expected a [1024, channels] window, got %s" % (x.shape,))
    x = np.clip(x, CODE_MIN, CODE_MAX)        # the IP clamps out-of-range words too
    s1 = x.sum(axis=0)
    mu = (s1 + (WINDOW >> 1)) >> 10                     # numpy >> on int64 = floor
    d = x - mu
    q_sum = (d * d).sum(axis=0)
    out = np.zeros_like(d)
    for c in range(x.shape[1]):
        m, sh = channel_factors(int(q_sum[c]), input_scale)
        if m == 0:
            continue
        mag = (np.abs(d[:, c]) * m + (1 << (sh - 1))) >> sh
        out[:, c] = np.minimum(mag, ACT_QMAX) * np.sign(d[:, c])
    return out


def preprocess_windows(codes: np.ndarray, input_scale: float) -> np.ndarray:
    """Stack of raw windows [n, 1024, 8] -> INT10 codes [n, 1024, 8]."""
    return np.stack([preprocess_window(w, input_scale) for w in np.asarray(codes)])
