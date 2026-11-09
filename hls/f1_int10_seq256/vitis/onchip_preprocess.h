// On-FPGA preprocessing for the raw-input core (2026-11-09).
//
// The IP receives one window of raw sensor samples (8192 int32 words, row-major
// [sample, channel], each a signed 24-bit ADC code) and produces the INT10 token
// buffer the transformer already consumes: per-window, per-channel z-score,
// INT10 quantization with the model's input scale, then the unchanged exact 4:1
// aggregation. Integer arithmetic only; bit-exact reference:
// src/quantize/onchip_preprocess.py (its docstring derives every constant).
//
//   pass 1  store the window, S1[c] = sum x
//   mean    mu[c] = (S1[c] + 512) >> 10
//   pass 2  Q[c] = sum (x - mu)^2
//   factors k = (62 - bitlen(Q)) / 2, r = isqrt(Q << 2k), m = (C_FX << 31) / r,
//           sh = 51 - k   (sequential, 8 channels per window)
//   pass 3  q = sign(d) * min(511, (|d| * m + 2^(sh-1)) >> sh), 4:1 aggregation
//
// Words outside the signed 24-bit range are clamped to it (the reference does
// the same). Cost: three 8,192-cycle passes plus the factors, ~0.26 ms at 100 MHz.
// Its two per-sample products use selects and adds instead of multipliers
// (all DSP48s are taken by the transformer), see pp_mul_digits.
#ifndef ONCHIP_PREPROCESS_H
#define ONCHIP_PREPROCESS_H

#include <hls_stream.h>
#include <ap_axi_sdata.h>

// Bit widths follow the bounds of the reference: |x| < 2^23, |d| < 2^24,
// d^2 < 2^48, Q < 2^58, m < 2^33, |d| * m < 2^57. g++ C-model: wide natives.
#if defined(__SYNTHESIS__) || defined(HLS_USE_AP_INT)
  #include "ap_int.h"
  typedef ap_int<24>  raw_t;       // stored raw sample (signed 24-bit ADC code)
  typedef ap_int<36>  pp_s1_t;     // sum of 1024 samples
  typedef ap_int<26>  pp_d_t;      // x - mu
  typedef ap_uint<25> pp_mag_t;    // |d|
  typedef ap_uint<50> pp_sq_t;     // |d|^2 (exact product width)
  typedef ap_uint<60> pp_q_t;      // sum of d^2
  typedef ap_uint<34> pp_m_t;      // per-channel multiplier
  typedef ap_uint<59> pp_mul_t;    // |d| * m (exact product width)
  typedef ap_uint<60> pp_prod_t;   // |d| * m + rounding term
#else
  typedef long long raw_t;
  typedef long long pp_s1_t;
  typedef long long pp_d_t;
  typedef unsigned long long pp_mag_t;
  typedef unsigned long long pp_sq_t;
  typedef unsigned long long pp_q_t;
  typedef unsigned long long pp_m_t;
  typedef unsigned long long pp_mul_t;
  typedef unsigned long long pp_prod_t;
#endif

#if defined(__SYNTHESIS__) || defined(HLS_USE_AP_INT)
// Exact unsigned product a * b without a multiplier: b is taken two bits at a
// time and each term is 0, a, 2a or 3a, then all terms are added. All DSP48s
// are used by the transformer (MAX_DSP 210); a LUT multiplier of this width
// does not fit the 10 ns clock and DSP-bound pieces take DSPs from the
// transformer. Selects and adds let the HLS scheduler pipeline the product.
template<int AW, int BW>
static ap_uint<AW + BW> pp_mul_digits(ap_uint<AW> a, ap_uint<BW> b) {
#pragma HLS INLINE
    const ap_uint<AW + 2> a1 = a;
    const ap_uint<AW + 2> a2 = (ap_uint<AW + 2>)a << 1;
    const ap_uint<AW + 2> a3 = a1 + a2;
    ap_uint<AW + BW> acc = 0;
    for (int k = 0; k < (BW + 1) / 2; ++k) {
#pragma HLS UNROLL
        const ap_uint<2> digit = (ap_uint<2>)(b >> (2 * k));
        const ap_uint<AW + 2> term = digit == 0 ? (ap_uint<AW + 2>)0
                                   : digit == 1 ? a1 : digit == 2 ? a2 : a3;
        acc += (ap_uint<AW + BW>)term << (2 * k);
    }
    return acc;
}
#define PP_MUL(AW, BW, a, b) pp_mul_digits<AW, BW>((a), (b))
#else
#define PP_MUL(AW, BW, a, b) ((a) * (b))
#endif

static const long long PP_CODE_MIN = -(1LL << 23);
static const long long PP_CODE_MAX = (1LL << 23) - 1;
// round(32 / input_scale * 2^20), input_scale = 0.028626115354773118
static const unsigned long long PP_C_FX = 1172161559ULL;
static const long long PP_QMAX = 511;

// floor(sqrt(v)) for v < 2^64, fixed 32 iterations
static unsigned long long pp_isqrt(unsigned long long v) {
    unsigned long long res = 0;
    unsigned long long bit = 1ULL << 62;
    for (int i = 0; i < 32; ++i) {
#pragma HLS PIPELINE off
        if (v >= res + bit) {
            v -= res + bit;
            res = (res >> 1) + bit;
        } else {
            res >>= 1;
        }
        bit >>= 2;
    }
    return res;
}

// floor(num / den) for den < 2^32, restoring division, fixed 64 iterations
static unsigned long long pp_div(unsigned long long num, unsigned long long den) {
    unsigned long long rem = 0, quo = 0;
    for (int i = 63; i >= 0; --i) {
#pragma HLS PIPELINE off
        rem = (rem << 1) | ((num >> i) & 1ULL);
        if (rem >= den) {
            rem -= den;
            quo |= 1ULL << i;
        }
    }
    return quo;
}

static int pp_bitlen(unsigned long long v) {
    int n = 0;
    for (int i = 0; i < 64; ++i) {
#pragma HLS PIPELINE off
        if ((v >> i) & 1ULL) n = i + 1;
    }
    return n;
}

static inline long long pp_quant(pp_d_t d, pp_m_t m, int sh) {
    if (m == 0) return 0;
    const pp_mag_t mag_in = (pp_mag_t)(d < 0 ? (pp_d_t)(-d) : d);
    const pp_mul_t product = PP_MUL(25, 34, mag_in, m);
    const pp_prod_t prod = (pp_prod_t)product + ((pp_prod_t)1 << (sh - 1));
    long long mag = (long long)(unsigned long long)(prod >> sh);
    if (mag > PP_QMAX) mag = PP_QMAX;
    return d < 0 ? -mag : mag;
}

template<typename AXIS_T, typename ACT_T, int IN_SEQ, int OUT_SEQ, int NCH>
static void raw_ingress(hls::stream<AXIS_T>& in, ACT_T* tokens) {
    static raw_t raw[IN_SEQ * NCH];
    pp_s1_t s1[NCH];
    pp_d_t mu[NCH];
    pp_q_t qsum[NCH];
    pp_m_t mfac[NCH];
    int shift[NCH];
#pragma HLS ARRAY_PARTITION variable=s1 complete
#pragma HLS ARRAY_PARTITION variable=mu complete
#pragma HLS ARRAY_PARTITION variable=qsum complete
#pragma HLS ARRAY_PARTITION variable=mfac complete
#pragma HLS ARRAY_PARTITION variable=shift complete
    for (int c = 0; c < NCH; ++c) {
#pragma HLS UNROLL
        s1[c] = 0;
        qsum[c] = 0;
    }

PP_READ:
    for (int i = 0; i < IN_SEQ * NCH; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=s1 inter false
        AXIS_T t = in.read();
        long long v = (int)t.data;
        if (v < PP_CODE_MIN) v = PP_CODE_MIN;
        if (v > PP_CODE_MAX) v = PP_CODE_MAX;
        raw[i] = v;
        s1[i % NCH] += v;
    }
    for (int c = 0; c < NCH; ++c) {
#pragma HLS UNROLL
        mu[c] = (pp_d_t)((s1[c] + (IN_SEQ >> 1)) >> 10);
    }

PP_VAR:
    for (int i = 0; i < IN_SEQ * NCH; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=qsum inter false
        const pp_d_t d = (pp_d_t)(raw[i] - mu[i % NCH]);
        const pp_mag_t md = (pp_mag_t)(d < 0 ? (pp_d_t)(-d) : d);
        const pp_sq_t sq = PP_MUL(25, 25, md, md);
        qsum[i % NCH] += (pp_q_t)sq;
    }

PP_FACTORS:
    for (int c = 0; c < NCH; ++c) {
#pragma HLS PIPELINE off
        if (qsum[c] == 0) {
            mfac[c] = 0;
            shift[c] = 0;
        } else {
            const unsigned long long q = (unsigned long long)qsum[c];
            const int k = (62 - pp_bitlen(q)) / 2;
            const unsigned long long r = pp_isqrt(q << (2 * k));
            mfac[c] = (pp_m_t)pp_div(PP_C_FX << 31, r);
            shift[c] = 51 - k;
        }
    }

PP_QUANT:
    // One flat II=1 loop over all samples. Sample j belongs to token j / 32 and
    // channel j % 8; a channel's 4:1 group is complete at its 4th sample, so the
    // token is written right there (same signed round-half-away-from-zero mean
    // and +-511 clamp as the INT10-input core).
    long long sums[NCH];
#pragma HLS ARRAY_PARTITION variable=sums complete
    for (int c = 0; c < NCH; ++c) {
#pragma HLS UNROLL
        sums[c] = 0;
    }
    for (int j = 0; j < IN_SEQ * NCH; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=sums inter false
        const int c = j % NCH;
        const int group = IN_SEQ / OUT_SEQ;
        const pp_d_t d = (pp_d_t)(raw[j] - mu[c]);
        const long long s = sums[c] + pp_quant(d, mfac[c], shift[c]);
        if ((j / NCH) % group == group - 1) {
            const long long mag = ((s < 0 ? -s : s) + 2) >> 2;
            long long v = s < 0 ? -mag : mag;
            if (v > PP_QMAX) v = PP_QMAX;
            if (v < -PP_QMAX) v = -PP_QMAX;
            tokens[(j / (NCH * group)) * NCH + c] = v;
            sums[c] = 0;
        } else {
            sums[c] = s;
        }
    }
}

#endif  // ONCHIP_PREPROCESS_H
