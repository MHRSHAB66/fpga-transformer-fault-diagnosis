// A04 research operator (2026-11-13): exact softmax normalisation divider with
// a 9-bit quotient. NOT used by the canonical design.
//
// Canonical expression (df_head in transformer_top_seq256_raw1024_*.cpp):
//     q = (num * 511 + denom / 2) / denom        (acc_t, C truncation)
// with num = exp_l0[...] in [0, 32768] and denom = sum of the 256 num of the row.
// Proven ranges (A03, see a04_proof.md):
//   num >= 0 and num <= denom (denom is a sum of non-negative terms that
//   includes num), and the row maximum always contributes exp_l0[clamp] = 32768,
//   so 32768 <= denom <= 256 * 32768 = 2^23   -> 24-bit unsigned
//   numer = num*511 + floor(denom/2) <= 32768*511 + 2^22 = 20,938,752 < 2^25
//   numer < 512 * denom                         -> quotient in [0, 511], 9 bits
// For non-negative operands C truncation equals floor, so a 9-step restoring
// division (bits 8..0) returns exactly the canonical quotient.
#ifndef SOFTMAX_DIV_Q9_H
#define SOFTMAX_DIV_Q9_H

#if defined(__SYNTHESIS__) || defined(HLS_USE_AP_INT)
#include <ap_int.h>
typedef ap_uint<25> q9_numer_t;
typedef ap_uint<24> q9_denom_t;
typedef ap_uint<9> q9_quot_t;
typedef ap_uint<33> q9_wide_t;
#else
typedef unsigned long long q9_numer_t;
typedef unsigned long long q9_denom_t;
typedef unsigned long long q9_quot_t;
typedef unsigned long long q9_wide_t;
#endif

// Precondition (asserted by the exhaustive test, not checked in hardware):
// denom >= 1 and numer < 512 * denom.
static inline q9_quot_t div_q9(q9_numer_t numer, q9_denom_t denom) {
#pragma HLS INLINE
    q9_wide_t rem = numer;
    q9_quot_t q = 0;
    for (int b = 8; b >= 0; --b) {
#pragma HLS UNROLL
        const q9_wide_t t = (q9_wide_t)denom << b;
        if (rem >= t) {
            rem -= t;
            q |= (q9_quot_t)1 << b;
        }
    }
    return q;
}

#endif  // SOFTMAX_DIV_Q9_H
