// Fixed-point type aliases for the F_1layer @ int10 core.
//
// Under Vitis HLS synthesis (__SYNTHESIS__ defined) these become arbitrary-
// precision ap_int; under plain g++ (C-sim / desktop verify) they fall back to
// wide native ints so the SAME source builds and runs both ways.
//
// Widths are the STARTING point for synthesis. The int10 reference clamps
// activations to +/-511 (10-bit signed) and attention weights to +/-511; the MAC
// accumulators must be wide enough for the worst-case dot product:
//   proj:  CH=8   terms of 10b*10b  -> <= 8*(2^18)  ~ 21 bits
//   attn/ffn: up to D=32 or DFF=64 terms          -> <= 64*(2^18) ~ 24 bits
// so ACC 32-bit is safe. Narrow further only with overflow analysis.
#ifndef AP_TYPES_H
#define AP_TYPES_H

#ifdef __SYNTHESIS__
  #include "ap_int.h"
  typedef ap_int<10> act_t;    // activation / weight codes (int10, +/-511)
  typedef ap_int<32> acc_t;    // MAC accumulator
  typedef ap_int<20> lut_t;    // LUT entries (exp/gelu/rsqrt fit < 2^18)
#else
  #include <cstdint>
  typedef int64_t act_t;
  typedef int64_t acc_t;
  typedef int64_t lut_t;
#endif

// Requant / affine multipliers are float64 in the reference (bit-exact). For a
// first functional synthesis they can stay `double` (FP cores); the optimization
// pass converts each to an ap_fixed or an integer (M0 * 2^-shift) multiplier.
typedef double scale_t;

#endif // AP_TYPES_H
