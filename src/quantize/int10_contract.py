"""
Exact Python-integer oracle for the DEPLOYED SEQ=256 INT10 HLS numeric contract.

Day-2 validation module (additive; nothing on the inference path imports it).

The deployed core (``hls/f1_int10_seq256/vitis/transformer_top_seq256_ingress1024_dsp_margin.cpp``)
does NOT use the legacy ``integer_ops.quantize_multiplier`` / ``apply_multiplier``
M0*2^-31*2^shift representation, and it does NOT use the float64
``integer_ops.requant_float`` of the Python reference. It stores every real
affine coefficient as a signed Q40 integer and rounds with ``div_round_even``:

    Mq      = round_half_even(Decimal(text_of_M) * 2^40)      (gen_weights_header.py)
    requant = clamp511( div_round_even(acc * Mq, 2^40) )

``div_round_even`` is C++ truncating division followed by a ties-to-even
correction. Everything here uses Python arbitrary-precision ``int`` so it is a
width-independent oracle; ``*_np`` variants are vectorised int64 copies used for
exhaustive sweeps and are themselves checked against the scalar oracle.

Rounding policies of the deployed system (they are deliberately different):

    ingress quantisation   q = clip(np.round(x / s_in), +-511)       half-to-even (NumPy)
    4:1 aggregation        sign(s) * ((|s| + 2) >> 2)                 half-away-from-zero
    Q40 requant / residual / LayerNorm affine / LN mean & variance    half-to-even
    softmax normalisation  (num*511 + denom/2) // denom  (num>=0)     half-up
    rsqrt / exp / GELU     ROM tables, integer index, no run-time rounding
"""
from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

import numpy as np

ACT_QMAX = 511
ATTN_QMAX = 511
SCALE_FRAC_BITS = 40
Q40_ONE = 1 << SCALE_FRAC_BITS
RSQRT_MANT_BITS = 8
RSQRT_FRAC = 15
SQRT1_2_Q15 = 23170          # value hard-coded in the HLS source

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOYED_HLS_DIR = PROJECT_ROOT / "hls" / "f1_int10_seq256"
DEPLOYED_TOP = DEPLOYED_HLS_DIR / "vitis" / "transformer_top_seq256_ingress1024_dsp_margin.cpp"
DEPLOYED_HEADER = DEPLOYED_HLS_DIR / "vitis" / "weights_int10.h"
DEPLOYED_DATA = DEPLOYED_HLS_DIR / "data"


# ---------------------------------------------------------------------------
# scalar exact oracle (Python int)
# ---------------------------------------------------------------------------

def clamp_act(y: int) -> int:
    return ACT_QMAX if y > ACT_QMAX else (-ACT_QMAX if y < -ACT_QMAX else int(y))


def c_trunc_div(n: int, d: int) -> tuple[int, int]:
    """C/C++ ``/`` and ``%`` for d > 0: quotient truncates toward zero."""
    q = abs(n) // d
    q = -q if n < 0 else q
    return q, n - q * d


def div_round_even(n: int, den: int) -> int:
    """Mirror of the HLS ``div_round_even<DEN>`` template, line for line."""
    q, r = c_trunc_div(int(n), int(den))
    ar = -r if r < 0 else r
    half = den // 2                      # C++ integer DEN / 2
    if ar > half or (ar == half and (q & 1)):
        q += -1 if n < 0 else 1
    return q


def to_q40(text_or_float) -> int:
    """Exactly what gen_weights_header.py emits for one value of a data/*.txt file."""
    dec = Decimal(text_or_float if isinstance(text_or_float, str) else repr(float(text_or_float)))
    return int((dec * Decimal(Q40_ONE)).to_integral_value(rounding=ROUND_HALF_EVEN))


def requant_q40(acc: int, mq: int) -> int:
    return clamp_act(div_round_even(int(acc) * int(mq), Q40_ONE))


def residual_q40(a: int, b: int, r0q: int, r1q: int) -> int:
    return clamp_act(div_round_even(int(a) * r0q + int(b) * r1q, Q40_ONE))


def rsqrt_fxp_int(v: int, lut) -> int:
    """HLS rsqrt_fxp: loop-based MSB search (no float log2)."""
    v = max(int(v), 1)
    e, t = 0, v
    for _ in range(31):
        if t > 1:
            t >>= 1
            e += 1
    shift = e - RSQRT_MANT_BITS
    mant_idx = (v >> shift) - (1 << RSQRT_MANT_BITS) if shift >= 0 \
        else (v << (-shift)) - (1 << RSQRT_MANT_BITS)
    mant_idx = min(max(mant_idx, 0), (1 << RSQRT_MANT_BITS) - 1)
    y = int(lut[mant_idx])
    if e & 1:
        y = (y * SQRT1_2_Q15) >> RSQRT_FRAC
    return y >> (e >> 1)


def layernorm_q40(row, gm_q40, bt_q40, epsq: int, rsqrt_lut) -> list[int]:
    d = len(row)
    mean = div_round_even(sum(int(x) for x in row), d)
    c = [int(x) - mean for x in row]
    var = div_round_even(sum(x * x for x in c), d)
    inv = rsqrt_fxp_int(var + int(epsq), rsqrt_lut)
    return [clamp_act(div_round_even(c[j] * inv * int(gm_q40[j]) + int(bt_q40[j]), Q40_ONE))
            for j in range(d)]


def gelu_lut_lookup(q: int, lut) -> int:
    idx = -ACT_QMAX if q < -ACT_QMAX else (ACT_QMAX if q > ACT_QMAX else int(q))
    return int(lut[idx + ACT_QMAX])


def softmax_q(scores, exp_lut, clamp: int) -> list[int]:
    """HLS df_head softmax on one row of integer scores."""
    rowmax = max(int(s) for s in scores)
    num = []
    for s in scores:
        dd = min(max(int(s) - rowmax, -clamp), 0)
        num.append(int(exp_lut[dd + clamp]))
    denom = max(sum(num), 1)
    return [(n * ATTN_QMAX + denom // 2) // denom for n in num]


def attention_head_q40(qh, kh, vh, m_scores_q40: int, m_ctx_q40: int,
                       exp_lut, clamp: int) -> list[int]:
    """One head of HLS df_head: qh[HD], kh[SEQ][HD], vh[SEQ][HD] -> ctx[HD]."""
    sc = [requant_q40(sum(int(a) * int(b) for a, b in zip(qh, kj)), m_scores_q40) for kj in kh]
    p = softmax_q(sc, exp_lut, clamp)
    hd = len(qh)
    return [requant_q40(sum(p[j] * int(vh[j][d]) for j in range(len(vh))), m_ctx_q40)
            for d in range(hd)]


def aggregate4(group) -> int:
    """HLS ingress 4:1: half-away-from-zero, then clamp511."""
    s = sum(int(x) for x in group)
    mag = (abs(s) + 2) >> 2
    return clamp_act(-mag if s < 0 else mag)


# ---------------------------------------------------------------------------
# vectorised int64 versions (for exhaustive sweeps; verified vs the oracle)
# ---------------------------------------------------------------------------

def div_round_even_np(n: np.ndarray, den: int) -> np.ndarray:
    n = np.asarray(n, dtype=np.int64)
    a = np.abs(n)
    q = a // den
    r = a - q * den
    half = den // 2
    up = (r > half) | ((r == half) & ((q & 1) == 1))
    q = q + up.astype(np.int64)
    return np.where(n < 0, -q, q)


def requant_q40_np(acc: np.ndarray, mq: int) -> np.ndarray:
    acc = np.asarray(acc, dtype=np.int64)
    bound = np.abs(acc).max(initial=0)
    if bound and abs(int(mq)) > (2 ** 63 - 1) // int(bound):
        raise OverflowError("acc * Mq would overflow int64")
    return np.clip(div_round_even_np(acc * np.int64(mq), Q40_ONE), -ACT_QMAX, ACT_QMAX)


# ---------------------------------------------------------------------------
# deployed constants
# ---------------------------------------------------------------------------

_ARR_RE = re.compile(r"static const (?:long long|long|double|int) (\w+)(?:\[(\d+)\])?\s*=\s*\{?([^;}]*)\}?;")


def load_header_constants(path: Path = DEPLOYED_HEADER) -> dict:
    """Parse weights_int10.h into {name: list[int|str]} (doubles kept as text)."""
    out = {}
    for m in _ARR_RE.finditer(path.read_text(encoding="ascii")):
        name, _, body = m.group(1), m.group(2), m.group(3)
        toks = [t.strip() for t in body.split(",") if t.strip()]
        if name.endswith("_q40") or not any(c in "".join(toks) for c in ".eE"):
            out[name] = [int(t) for t in toks]
        else:
            out[name] = toks
    return out


def read_data_file(name: str, data_dir: Path = DEPLOYED_DATA) -> list[str]:
    toks = (data_dir / name).read_text(encoding="ascii").split()
    n = int(toks[0])
    assert len(toks) - 1 == n, name
    return toks[1:]
