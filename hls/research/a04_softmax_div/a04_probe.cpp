// A04 single-operator probe: one attention row's softmax normalisation exactly
// as in df_head (denominator accumulation + 2 divisions per iteration, II=1),
// with the canonical C division (default) or the 9-bit-quotient divider
// (-DA04_DIVQ9). Synthesised alone so HLS and Vivado report the operator's cost.
#include "ap_types.h"
#include "softmax_div_q9.h"

#define SEQ 256
static const long ATTN_QMAX = 511;

void a04_probe(const lut_t num[SEQ], act_t qattn[SEQ]) {
#pragma HLS INTERFACE ap_memory port=num
#pragma HLS INTERFACE ap_memory port=qattn
#pragma HLS ARRAY_PARTITION variable=num cyclic factor=2
#pragma HLS ARRAY_PARTITION variable=qattn cyclic factor=2
    acc_t denom0 = 0, denom1 = 0;
    for (int j = 0; j < SEQ; j += 2) {
#pragma HLS PIPELINE
        denom0 += num[j];
        denom1 += num[j + 1];
    }
    acc_t denom = denom0 + denom1;
    if (denom < 1) denom = 1;
    for (int j = 0; j < SEQ; j += 2) {
#pragma HLS PIPELINE
        acc_t numerator0 = (acc_t)num[j] * ATTN_QMAX + denom / 2;
        acc_t numerator1 = (acc_t)num[j + 1] * ATTN_QMAX + denom / 2;
#ifdef A04_DIVQ9
        qattn[j]     = (act_t)div_q9((q9_numer_t)numerator0, (q9_denom_t)denom);
        qattn[j + 1] = (act_t)div_q9((q9_numer_t)numerator1, (q9_denom_t)denom);
#else
        qattn[j]     = numerator0 / denom;
        qattn[j + 1] = numerator1 / denom;
#endif
    }
}
