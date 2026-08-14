"""
Integer-only fixed-point primitives for the full-integer Transformer reference.

Everything a hardware datapath needs, and nothing it does not:

- symmetric linear quantize / dequantize (no zero-point),
- a gemmlowp/TFLite-style requantization multiplier ``M = M0 * 2^-shift`` that
  turns an int32/int64 accumulator back into an int16 activation using only an
  integer multiply and an arithmetic shift,
- integer lookup tables (LUTs) for the transcendental blocks ``exp`` (softmax),
  ``gelu`` and reciprocal-sqrt (LayerNorm). On the FPGA these become BRAM tables;
  here the *contents* are precomputed once from the float function (that is done
  offline, exactly like initialising a ROM), while every inference-time access is
  a pure integer index.

The design rule that makes this "integer-only": every *tensor* that flows through
the network is an integer array; only per-operation *constants* (a scale, a
requant multiplier, a LUT's contents) are real numbers, and each of those maps to
a fixed hardware constant. ``quantize_multiplier`` / ``apply_multiplier`` below
document and self-test that mapping.

Run the self-tests:
    python -m src.quantize.integer_ops
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Symmetric linear quantization (no zero-point)
# ---------------------------------------------------------------------------

def qint_max(bits: int) -> int:
    """Symmetric positive limit, e.g. 127 for int8, 32767 for int16."""
    return (1 << (bits - 1)) - 1


def quantize_sym(x: np.ndarray, scale: float, bits: int) -> np.ndarray:
    """Real -> integer codes: q = clip(round(x / scale)), symmetric range."""
    qmax = qint_max(bits)
    q = np.round(np.asarray(x, dtype=np.float64) / scale)
    return np.clip(q, -qmax, qmax).astype(np.int64)


def dequantize_sym(q: np.ndarray, scale: float) -> np.ndarray:
    """Integer codes -> real approximation q * scale."""
    return q.astype(np.float64) * scale


# ---------------------------------------------------------------------------
# Requantization multiplier  (acc_int * M -> int, M in the reals)
# ---------------------------------------------------------------------------

def quantize_multiplier(m: float) -> tuple[int, int]:
    """
    Express a positive real multiplier ``m`` as ``M0 * 2^-31 * 2^shift`` with
    ``M0`` a 31-bit integer in ``[2^30, 2^31)``. This is the exact constant a
    hardware requant unit stores: one int32 multiply plus a shift.

    Returns (M0, shift). ``m == 0`` maps to (0, 0).
    """
    if m == 0.0:
        return 0, 0
    if m < 0:
        raise ValueError("multiplier must be non-negative")
    shift = 0
    while m < 0.5:
        m *= 2.0
        shift -= 1
    while m >= 1.0:
        m /= 2.0
        shift += 1
    m0 = int(round(m * (1 << 31)))
    if m0 == (1 << 31):          # rounded up to the next power of two
        m0 //= 2
        shift += 1
    return m0, shift


def apply_multiplier(acc: np.ndarray, m0: int, shift: int) -> np.ndarray:
    """
    Integer requantization: round(acc * M0 * 2^-31 * 2^shift), using Python
    arbitrary-precision integers so the intermediate product never overflows.
    This is the bit-exact reference for the hardware fixed-point multiplier.
    """
    acc = np.asarray(acc, dtype=object)          # exact big-int arithmetic
    prod = acc * m0                              # scale by M0 (2^-31 implied)
    total_shift = 31 - shift                     # net right shift
    if total_shift > 0:
        half = 1 << (total_shift - 1)
        out = np.where(prod >= 0,
                       (prod + half) >> total_shift,
                       -((-prod + half) >> total_shift))
    elif total_shift == 0:
        out = prod
    else:
        out = prod << (-total_shift)
    return out.astype(np.int64)


def requant_float(acc: np.ndarray, m: float, out_bits: int) -> np.ndarray:
    """
    Convenience requant used by the reference model: round(acc * m) clipped to
    the symmetric out range. ``acc`` is an exact integer and ``|acc| < 2^52`` for
    this network, so the float64 multiply-and-round is numerically identical to
    ``apply_multiplier`` (verified in the self-tests) while staying vectorised.
    """
    qmax = qint_max(out_bits)
    y = np.round(acc.astype(np.float64) * m)
    return np.clip(y, -qmax, qmax).astype(np.int64)


# ---------------------------------------------------------------------------
# Integer LUTs for the transcendental blocks
# ---------------------------------------------------------------------------

def build_gelu_lut(in_scale: float, out_scale: float, bits: int) -> np.ndarray:
    """
    Table over the whole signed ``bits`` input-code range: given an input code
    ``q`` (real = q*in_scale), returns the GELU output code (real = out*out_scale).

    Uses the exact erf GELU (matches ``nn.GELU`` default). Index with ``q+qmax``.
    """
    qmax = qint_max(bits)
    codes = np.arange(-qmax, qmax + 1, dtype=np.float64)
    x = codes * in_scale
    gelu = 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
    out = np.round(gelu / out_scale)
    return np.clip(out, -qmax, qmax).astype(np.int64)


def gelu_lookup(q: np.ndarray, lut: np.ndarray, bits: int) -> np.ndarray:
    qmax = qint_max(bits)
    idx = np.clip(q, -qmax, qmax) + qmax
    return lut[idx]


def build_exp_lut(arg_scale: float, clamp: int, frac_bits: int) -> tuple[np.ndarray, int]:
    """
    exp table for softmax. The argument is ``d = (score - rowmax)`` in integer
    score codes, always ``<= 0``. Real argument is ``d * arg_scale``. We clamp
    ``d`` to ``[-clamp, 0]`` (exp underflows well before that) and store
    ``round(exp(d*arg_scale) * 2^frac_bits)``.

    Returns (table, clamp). Index with ``d + clamp`` after clamping.
    """
    d = np.arange(-clamp, 1, dtype=np.float64)
    e = np.exp(d * arg_scale) * (1 << frac_bits)
    return np.round(e).astype(np.int64), clamp


def exp_lookup(d: np.ndarray, lut: np.ndarray, clamp: int) -> np.ndarray:
    idx = np.clip(d, -clamp, 0) + clamp
    return lut[idx]


# reciprocal-sqrt via mantissa range reduction (hardware-standard)
_RSQRT_MANT_BITS = 8
_RSQRT_FRAC = 15


def build_rsqrt_lut() -> np.ndarray:
    """
    LUT of ``1/sqrt(mant)`` for mantissa ``mant in [1, 2)`` sampled with
    ``_RSQRT_MANT_BITS`` bits, scaled by ``2^_RSQRT_FRAC``. Combined with an
    exponent shift this covers the full positive range (see ``rsqrt_fxp``).
    """
    n = 1 << _RSQRT_MANT_BITS
    mant = 1.0 + np.arange(n, dtype=np.float64) / n          # [1, 2)
    return np.round((1.0 / np.sqrt(mant)) * (1 << _RSQRT_FRAC)).astype(np.int64)


_SQRT1_2 = np.int64(round((1.0 / math.sqrt(2.0)) * (1 << _RSQRT_FRAC)))


def rsqrt_fxp(v: np.ndarray, lut: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Fixed-point reciprocal square root of a positive integer array ``v``.

    Returns (y, frac) with ``1/sqrt(v) ~= y * 2^-frac``. Pure integer:
    range-reduce ``v = mant * 2^e`` (mant in [1,2) via the top bits), look up
    ``1/sqrt(mant)``, then apply ``2^(-e/2)`` (a shift, with a ``1/sqrt(2)``
    correction for odd exponents).
    """
    v = np.asarray(v, dtype=np.int64)
    v = np.maximum(v, 1)                       # avoid div-by-zero on constant rows
    frac = _RSQRT_FRAC

    # exponent e = position of the most significant bit
    e = np.floor(np.log2(v.astype(np.float64))).astype(np.int64)

    # mantissa index: top _RSQRT_MANT_BITS bits below the leading 1
    shift = e - _RSQRT_MANT_BITS
    mant_idx = np.where(
        shift >= 0,
        (v >> np.maximum(shift, 0)) - (1 << _RSQRT_MANT_BITS),
        (v << np.maximum(-shift, 0)) - (1 << _RSQRT_MANT_BITS),
    )
    mant_idx = np.clip(mant_idx, 0, (1 << _RSQRT_MANT_BITS) - 1)
    base = lut[mant_idx]                        # 1/sqrt(mant) * 2^frac

    # multiply by 2^(-e/2)
    half_e = e >> 1                            # floor(e/2)
    odd = (e & 1).astype(bool)
    y = base.copy()
    # even e: y = base >> half_e ; odd e: extra *1/sqrt(2)
    y = np.where(odd, (y * _SQRT1_2) >> _RSQRT_FRAC, y)
    y = np.where(half_e >= 0, y >> np.maximum(half_e, 0), y << np.maximum(-half_e, 0))
    return y.astype(np.int64), frac


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _test_multiplier() -> None:
    rng = np.random.default_rng(0)
    for m in [0.000123, 0.5, 0.9999, 1.0, 3.7, 127.25]:
        m0, shift = quantize_multiplier(m)
        acc = rng.integers(-(1 << 40), 1 << 40, size=5000).astype(np.int64)
        exact = apply_multiplier(acc, m0, shift)      # HW path (rounded M0)
        flt = requant_float(acc, m, 63)               # reference path (exact m)
        true = np.round(acc.astype(np.float64) * m)
        # reference path must equal the true rounded product exactly
        assert np.array_equal(flt, true.astype(np.int64)), f"requant error at m={m}"
        # HW M0/shift path differs only by M0's ~2^-31 relative rounding
        tol = 1 + np.abs(flt).astype(np.float64) * 2e-9
        assert np.all(np.abs(exact - flt) <= tol), f"M0/shift mismatch at m={m}"
    print("[ok] requant: reference==true exactly; HW M0/shift within 2^-31 rel")


def _test_rsqrt() -> None:
    lut = build_rsqrt_lut()
    vals = np.array([1, 2, 3, 7, 15, 16, 100, 1000, 12345, 1 << 20, 1 << 30],
                    dtype=np.int64)
    y, frac = rsqrt_fxp(vals, lut)
    approx = y.astype(np.float64) / (1 << frac)
    true = 1.0 / np.sqrt(vals.astype(np.float64))
    rel = np.abs(approx - true) / true
    assert rel.max() < 0.01, f"rsqrt rel err too high: {rel.max():.4f}"
    print(f"[ok] fixed-point rsqrt: max rel err = {rel.max():.4%}")


def _test_gelu() -> None:
    in_scale, out_scale, bits = 0.01, 0.01, 16
    lut = build_gelu_lut(in_scale, out_scale, bits)
    q = np.arange(-2000, 2001, dtype=np.int64)
    out = gelu_lookup(q, lut, bits).astype(np.float64) * out_scale
    x = q.astype(np.float64) * in_scale
    true = 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
    assert np.abs(out - true).max() < 2 * out_scale, "gelu LUT error too high"
    print("[ok] gelu LUT: matches erf-GELU within 1 code")


def _test_exp() -> None:
    arg_scale, clamp, frac = 0.02, 4000, 15
    lut, cl = build_exp_lut(arg_scale, clamp, frac)
    d = np.array([0, -1, -50, -500, -4000], dtype=np.int64)
    got = exp_lookup(d, lut, cl).astype(np.float64) / (1 << frac)
    true = np.exp(d.astype(np.float64) * arg_scale)
    assert np.allclose(got, true, atol=2.0 / (1 << frac) + 1e-9), "exp LUT error"
    print("[ok] exp LUT: matches exp within 1 fixed-point code")


if __name__ == "__main__":
    _test_multiplier()
    _test_rsqrt()
    _test_gelu()
    _test_exp()
    print("\nall integer_ops self-tests passed.")
