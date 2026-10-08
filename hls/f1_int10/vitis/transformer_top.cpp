// ---------------------------------------------------------------------------
// Synthesizable top for the F_1layer @ int10 core (Vitis HLS).
//
// Functionally a faithful mirror of the verified reference (../transformer.cpp
// == src/quantize/integer_model.py): SAME integer arithmetic, LAYERS=1 inlined,
// parameters from the flat compile-time header (weights_int10.h) instead of the
// std::map/file loader (which is not synthesizable). Datapath types come from
// ap_types.h.
//
// PURPOSE OF THIS FILE: get a BASELINE csynth + resource/latency report in the
// real tool. Full-sequence K/V and input projections remain, while Q, context,
// FFN and normalization intermediates are processed one token at a time.
// Attention still evaluates all SEQ keys per query; this is not local attention.
// Further numeric and throughput optimization is needed for a small device.
// Validate this top with run_top_csim.tcl against the 16 golden vectors.
//
// Interface: one AXI-Stream in (SEQ*CH int codes, row-major [seq,channel]) and
// one AXI-Stream out (NUM_CLASSES logits), matching a single axi_dma on PYNQ
// (pynq/fault_classifier.py HardwareBackend).
// ---------------------------------------------------------------------------
#include "ap_types.h"
#include "weights_int10.h"
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <cstdint>

#define SEQ 1024
#define CH  8
#define D   32
#define NHEAD 4
#define HD  (D / NHEAD)
#define DFF 64
#define CLS_HID 16
#define NUM_CLASSES 4

static const long ACT_QMAX  = 511;   // int10
static const long ATTN_QMAX = 511;
static const int  RSQRT_MANT_BITS = 8;
static const int  RSQRT_FRAC = 15;
static const long SQRT1_2 = 23170;  // truncated sqrt(1/2) * 2^15
static const int SCALE_FRAC_BITS = 40;

typedef ap_axis<32, 0, 0, 0> axis_t;

// ---- primitives (identical to the reference) ------------------------------
static inline long clamp_act(long long y) {
    if (y >  ACT_QMAX) return  ACT_QMAX;
    if (y < -ACT_QMAX) return -ACT_QMAX;
    return (long)y;
}
// Round a signed integer ratio to nearest, ties to even (like numpy.rint).
// DEN is a compile-time constant (always a power of two here: 2^SCALE_FRAC_BITS
// or D), so HLS lowers the divide/modulo to shift+mask instead of a 64-bit
// divider. The arithmetic is identical to a runtime-denominator version.
template<long long DEN>
static inline long long div_round_even(long long numerator) {
    long long quotient = numerator / DEN;
    long long remainder = numerator % DEN;
    long long abs_remainder = remainder < 0 ? -remainder : remainder;
    if (abs_remainder > DEN / 2 ||
        (abs_remainder == DEN / 2 && (quotient & 1)))
        quotient += numerator < 0 ? -1 : 1;
    return quotient;
}
static inline long requant(long acc, scale_t Mq) {
    return clamp_act(div_round_even<1LL << SCALE_FRAC_BITS>((long long)acc * Mq));
}
static inline long gelu_lut(long q, const long* lut) {
    long idx = q < -ACT_QMAX ? -ACT_QMAX : (q > ACT_QMAX ? ACT_QMAX : q);
    return lut[idx + ACT_QMAX];
}
static long rsqrt_fxp(long v, const long* lut) {
    if (v < 1) v = 1;
    int e = 0;
    long t = v;
    // A positive 32-bit value reaches one after at most 30 shifts.
    for (int bit = 0; bit < 31; ++bit) {
#pragma HLS PIPELINE off
        if (t > 1) { t >>= 1; ++e; }
    }
    int shift = e - RSQRT_MANT_BITS;
    long mant_idx = (shift >= 0) ? (v >> shift) - (1L << RSQRT_MANT_BITS)
                                 : (v << (-shift)) - (1L << RSQRT_MANT_BITS);
    if (mant_idx < 0) mant_idx = 0;
    if (mant_idx > (1L << RSQRT_MANT_BITS) - 1) mant_idx = (1L << RSQRT_MANT_BITS) - 1;
    long y = lut[mant_idx];
    if (e & 1) y = (y * SQRT1_2) >> RSQRT_FRAC;
    y = y >> (e >> 1);
    return y;
}
// out[rows*Dout] = requant(in @ W^T + b, M)
static void linear(const act_t* in, int rows, int Din, int Dout,
                   const long* W, const long* b, scale_t Mq, act_t* out) {
    for (int r = 0; r < rows; ++r)
        for (int o = 0; o < Dout; ++o) {
#pragma HLS PIPELINE
            const long* wo = &W[(long)o * Din];
            const act_t* xr = &in[(long)r * Din];
            acc_t acc = 0;
            for (int i = 0; i < Din; ++i) acc += xr[i] * wo[i];
            out[(long)r * Dout + o] = requant(acc + b[o], Mq);
        }
}
static void layernorm(const act_t* in, int rows, const scale_t* gm, const scale_t* bt,
                      long epsq, const long* rsqrt, act_t* out) {
    for (int r = 0; r < rows; ++r) {
        const act_t* xr = &in[(long)r * D];
        acc_t sum = 0; for (int j = 0; j < D; ++j) sum += xr[j];
        long mean = (long)div_round_even<D>((long)sum);
        long c[D]; acc_t sumsq = 0;
        for (int j = 0; j < D; ++j) { c[j] = xr[j] - mean; sumsq += c[j] * c[j]; }
        long var = (long)div_round_even<D>((long)sumsq);
        long inv = rsqrt_fxp(var + epsq, rsqrt);
        act_t* orow = &out[(long)r * D];
        for (int j = 0; j < D; ++j)
            orow[j] = clamp_act(div_round_even<1LL << SCALE_FRAC_BITS>(
                                    (long long)c[j] * inv * gm[j] + bt[j]));
    }
}
static void residual(const act_t* a, const act_t* b_, int n,
                     scale_t r0, scale_t r1, act_t* out) {
    for (int i = 0; i < n; ++i)
        out[i] = clamp_act(div_round_even<1LL << SCALE_FRAC_BITS>(
                               (long long)a[i] * r0 + (long long)b_[i] * r1));
}

// ---- layer-0 attention (single layer inlined) -----------------------------
static void attention0_row(const act_t* x, const act_t* k, const act_t* v,
                           int i, act_t* attn_out) {
    act_t q[D], ctx[D];
    linear(&x[(long)i * D], 1, D, D, W_l0_wq, b_l0_wq, M_l0_wq_q40[0], q);
    for (int h = 0; h < NHEAD; ++h) {
        int off = h * HD;
        const act_t* qi = &q[off];
        act_t sc[SEQ];
        for (int j = 0; j < SEQ; ++j) {
#pragma HLS PIPELINE
            const act_t* kj = &k[(long)j * D + off];
            acc_t acc = 0; for (int d = 0; d < HD; ++d) acc += qi[d] * kj[d];
            sc[j] = requant(acc, M_l0_scores_q40[0]);
        }
        act_t rowmax = sc[0];
        for (int j = 1; j < SEQ; ++j) {
#pragma HLS PIPELINE
            if (sc[j] > rowmax) rowmax = sc[j];
        }
        lut_t num[SEQ]; acc_t denom = 0;
        for (int j = 0; j < SEQ; ++j) {
#pragma HLS PIPELINE
            long dd = sc[j] - rowmax;
            if (dd < -exp_clamp_l0) dd = -exp_clamp_l0;
            if (dd > 0) dd = 0;
            num[j] = exp_l0[dd + exp_clamp_l0];
            denom += num[j];
        }
        if (denom < 1) denom = 1;
        act_t qattn[SEQ];
        for (int j = 0; j < SEQ; ++j) {
#pragma HLS PIPELINE
            // exp_l0 <= 32768, so for SEQ=1024 both denominator and
            // numerator fit in 32 bits. Keep division at that width.
            acc_t numerator = (acc_t)num[j] * ATTN_QMAX + denom / 2;
            qattn[j] = numerator / denom;
        }
        for (int d = 0; d < HD; ++d) {
            acc_t acc = 0;
            for (int j = 0; j < SEQ; ++j) {
#pragma HLS PIPELINE
                acc += qattn[j] * v[(long)j * D + off + d];
            }
            ctx[off + d] = requant(acc, M_l0_ctx_q40[0]);
        }
    }
    linear(ctx, 1, D, D, W_l0_out, b_l0_out, M_l0_out_q40[0], attn_out);
}

static void attention0_kv(const act_t* x, act_t* k, act_t* v) {
    linear(x, SEQ, D, D, W_l0_wk, b_l0_wk, M_l0_wk_q40[0], k);
    linear(x, SEQ, D, D, W_l0_wv, b_l0_wv, M_l0_wv_q40[0], v);
}

// Keep the post-attention work as a separate HLS module per token.
static void finish_token(const act_t* xrow, const act_t* attn_out, act_t* outrow) {
#pragma HLS INLINE off
    act_t res[D], ln1[D], ff[DFF], ff2[D];
    residual(xrow, attn_out, D, add_l0_res1_q40[0], add_l0_res1_q40[1], res);
    layernorm(res, 1, ln_gm_l0_ln1_q40, ln_bt_l0_ln1_q40, ln_epsq_l0_ln1, rsqrt_lut, ln1);
    linear(ln1, 1, D, DFF, W_l0_ff0, b_l0_ff0, M_l0_ff0_q40[0], ff);
    for (int j = 0; j < DFF; ++j) ff[j] = gelu_lut(ff[j], gelu_l0);
    linear(ff, 1, DFF, D, W_l0_ff3, b_l0_ff3, M_l0_ff3_q40[0], ff2);
    residual(ln1, ff2, D, add_l0_res2_q40[0], add_l0_res2_q40[1], res);
    layernorm(res, 1, ln_gm_l0_ln2_q40, ln_bt_l0_ln2_q40, ln_epsq_l0_ln2, rsqrt_lut,
              outrow);
}

// ---- full forward: input codes (SEQ*CH) -> NUM_CLASSES logits --------------
static void transformer_core(const act_t* input, act_t logits[NUM_CLASSES]) {
    static act_t x[SEQ * D], k[SEQ * D], v[SEQ * D];
    linear(input, SEQ, CH, D, W_proj, b_proj, M_proj_q40[0], x);
    for (int i = 0; i < SEQ * D; ++i) {
#pragma HLS PIPELINE
        x[i] = clamp_act((long)x[i] + pe_codes[i]);
    }

    // K and V cover the whole sequence. Each query and subsequent stage uses
    // one token, without changing the full-sequence attention calculation.
    attention0_kv(x, k, v);
    for (int i = 0; i < SEQ; ++i) {
        act_t attn_out[D];
        attention0_row(x, k, v, i, attn_out);
        finish_token(&x[(long)i * D], attn_out, &x[(long)i * D]);
    }
    act_t pooled[D];
    for (int j = 0; j < D; ++j) {
        acc_t sum = 0;
        for (int i = 0; i < SEQ; ++i) {
#pragma HLS PIPELINE
            sum += x[(long)i * D + j];
        }
        pooled[j] = requant(sum, M_pool_q40[0]);
    }
    act_t cls_ln[D], c1[CLS_HID];
    layernorm(pooled, 1, ln_gm_cls_ln_q40, ln_bt_cls_ln_q40, ln_epsq_cls_ln, rsqrt_lut, cls_ln);
    linear(cls_ln, 1, D, CLS_HID, W_cls1, b_cls1, M_cls1_q40[0], c1);
    for (int o = 0; o < CLS_HID; ++o) c1[o] = gelu_lut(c1[o], gelu_cls);
    linear(c1, 1, CLS_HID, NUM_CLASSES, W_cls4, b_cls4, M_cls4_q40[0], logits);
}

// ---- AXI-Stream top -------------------------------------------------------
void transformer_top(hls::stream<axis_t>& in, hls::stream<axis_t>& out) {
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
    static act_t input[SEQ * CH];
    for (int i = 0; i < SEQ * CH; ++i) {
#pragma HLS PIPELINE
        axis_t t = in.read();
        input[i] = (int)t.data;
    }
    act_t logits[NUM_CLASSES];
    transformer_core(input, logits);
    for (int c = 0; c < NUM_CLASSES; ++c) {
        axis_t t;
        t.data = (int)logits[c];
        t.keep = -1; t.strb = -1;
        t.last = (c == NUM_CLASSES - 1) ? 1 : 0;
        out.write(t);
    }
}
