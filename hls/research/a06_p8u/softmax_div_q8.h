// A06 research operator (2026-11-13): unsigned 8-bit attention probabilities
// (qmax 255) with an exact 8-bit-quotient softmax divider. NOT canonical.
//   q = (num * 255 + denom / 2) / denom,  num in [0, 32768], 32768 <= denom <= 2^23,
//   num <= denom  ->  numer <= 255*denom + denom/2 < 256*denom  ->  q in [0, 255].
// Precondition checked exhaustively by test_div_q8_exhaustive.cpp.
#ifndef SOFTMAX_DIV_Q8_H
#define SOFTMAX_DIV_Q8_H

#if defined(__SYNTHESIS__) || defined(HLS_USE_AP_INT)
#include <ap_int.h>
typedef ap_uint<24> q8_numer_t;
typedef ap_uint<24> q8_denom_t;
typedef ap_uint<8> q8_t;
typedef ap_uint<32> q8_wide_t;
#else
typedef unsigned long long q8_numer_t;
typedef unsigned long long q8_denom_t;
typedef unsigned int q8_t;          // promotes to signed long long in products with act_t
typedef unsigned long long q8_wide_t;
#endif

static inline q8_t div_q8(q8_numer_t numer, q8_denom_t denom) {
#pragma HLS INLINE
    q8_wide_t rem = numer;
    q8_t q = 0;
    for (int b = 7; b >= 0; --b) {
#pragma HLS UNROLL
        const q8_wide_t t = (q8_wide_t)denom << b;
        if (rem >= t) {
            rem -= t;
            q |= (q8_t)1 << b;
        }
    }
    return q;
}

#endif  // SOFTMAX_DIV_Q8_H
