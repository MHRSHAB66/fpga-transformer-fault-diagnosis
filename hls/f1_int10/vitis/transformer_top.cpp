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
// real tool. It is NOT yet optimized for a final target device: the SEQ*D
// activation buffers and the O(SEQ^2) attention need
// the streaming/tiling + type-narrowing pass described in SYNTHESIS_GUIDE.md.
// Validate FUNCTIONALITY first with `run_hls.tcl` C-sim on ../transformer.cpp
// (prints PASS 16/16); this file is the starting point for the HW build.
//
// Interface: one AXI-Stream in (SEQ*CH int codes, row-major [seq,channel]) and
// one AXI-Stream out (NUM_CLASSES logits), matching a single axi_dma on PYNQ
// (pynq/fault_classifier.py HardwareBackend).
// ---------------------------------------------------------------------------
#include "ap_types.h"
#include "weights_int10.h"
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <cmath>

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
static const long SQRT1_2 = (long)(0.7071067811865476 * (1 << RSQRT_FRAC));

typedef ap_axis<32, 0, 0, 0> axis_t;

// ---- primitives (identical to the reference) ------------------------------
static inline long clamp_act(double y) {
    if (y >  (double)ACT_QMAX) return  ACT_QMAX;
    if (y < -(double)ACT_QMAX) return -ACT_QMAX;
    return (long)y;
}
static inline long requant(long acc, double M) {
    return clamp_act(std::rint((double)acc * M));
}
static inline long gelu_lut(long q, const long* lut) {
    long idx = q < -ACT_QMAX ? -ACT_QMAX : (q > ACT_QMAX ? ACT_QMAX : q);
    return lut[idx + ACT_QMAX];
}
static long rsqrt_fxp(long v, const long* lut) {
    if (v < 1) v = 1;
    int e = 0; { long t = v; while (t > 1) { t >>= 1; ++e; } }
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
static void linear(const long* in, int rows, int Din, int Dout,
                   const long* W, const long* b, double M, long* out) {
    for (int r = 0; r < rows; ++r)
        for (int o = 0; o < Dout; ++o) {
            const long* wo = &W[(long)o * Din];
            const long* xr = &in[(long)r * Din];
            long acc = 0;
            for (int i = 0; i < Din; ++i) acc += xr[i] * wo[i];
            out[(long)r * Dout + o] = requant(acc + b[o], M);
        }
}
static void layernorm(const long* in, int rows, const double* gm, const double* bt,
                      long epsq, const long* rsqrt, long* out) {
    for (int r = 0; r < rows; ++r) {
        const long* xr = &in[(long)r * D];
        long sum = 0; for (int j = 0; j < D; ++j) sum += xr[j];
        long mean = (long)std::rint((double)sum / D);
        long c[D]; long sumsq = 0;
        for (int j = 0; j < D; ++j) { c[j] = xr[j] - mean; sumsq += c[j] * c[j]; }
        long var = (long)std::rint((double)sumsq / D);
        long inv = rsqrt_fxp(var + epsq, rsqrt);
        long* orow = &out[(long)r * D];
        for (int j = 0; j < D; ++j)
            orow[j] = clamp_act(std::rint((double)(c[j] * inv) * gm[j] + bt[j]));
    }
}
static void residual(const long* a, const long* b_, int n, double r0, double r1, long* out) {
    for (int i = 0; i < n; ++i)
        out[i] = clamp_act(std::rint((double)a[i] * r0 + (double)b_[i] * r1));
}

// ---- layer-0 attention (single layer inlined) -----------------------------
static void attention0(const long* x, long* attn_out) {
    static long q[SEQ * D], k[SEQ * D], v[SEQ * D], ctx[SEQ * D];
    linear(x, SEQ, D, D, W_l0_wq, b_l0_wq, M_l0_wq[0], q);
    linear(x, SEQ, D, D, W_l0_wk, b_l0_wk, M_l0_wk[0], k);
    linear(x, SEQ, D, D, W_l0_wv, b_l0_wv, M_l0_wv[0], v);
    for (int h = 0; h < NHEAD; ++h) {
        int off = h * HD;
        for (int i = 0; i < SEQ; ++i) {
            const long* qi = &q[(long)i * D + off];
            long sc[SEQ];
            for (int j = 0; j < SEQ; ++j) {
                const long* kj = &k[(long)j * D + off];
                long acc = 0; for (int d = 0; d < HD; ++d) acc += qi[d] * kj[d];
                sc[j] = requant(acc, M_l0_scores[0]);
            }
            long rowmax = sc[0];
            for (int j = 1; j < SEQ; ++j) if (sc[j] > rowmax) rowmax = sc[j];
            long num[SEQ]; long denom = 0;
            for (int j = 0; j < SEQ; ++j) {
                long dd = sc[j] - rowmax;
                if (dd < -exp_clamp_l0) dd = -exp_clamp_l0;
                if (dd > 0) dd = 0;
                num[j] = exp_l0[dd + exp_clamp_l0];
                denom += num[j];
            }
            if (denom < 1) denom = 1;
            long qattn[SEQ];
            for (int j = 0; j < SEQ; ++j)
                qattn[j] = (num[j] * ATTN_QMAX + denom / 2) / denom;
            for (int d = 0; d < HD; ++d) {
                long acc = 0;
                for (int j = 0; j < SEQ; ++j) acc += qattn[j] * v[(long)j * D + off + d];
                ctx[(long)i * D + off + d] = requant(acc, M_l0_ctx[0]);
            }
        }
    }
    linear(ctx, SEQ, D, D, W_l0_out, b_l0_out, M_l0_out[0], attn_out);
}

// ---- full forward: input codes (SEQ*CH) -> NUM_CLASSES logits --------------
static void transformer_core(const long* input, long logits[NUM_CLASSES]) {
    static long x[SEQ * D], attn_out[SEQ * D], res[SEQ * D], ln1[SEQ * D],
                ff[SEQ * DFF], ff2[SEQ * D];
    linear(input, SEQ, CH, D, W_proj, b_proj, M_proj[0], x);
    for (int i = 0; i < SEQ * D; ++i) x[i] = clamp_act((double)(x[i] + pe_codes[i]));

    attention0(x, attn_out);
    residual(x, attn_out, SEQ * D, add_l0_res1[0], add_l0_res1[1], res);
    layernorm(res, SEQ, ln_gm_l0_ln1, ln_bt_l0_ln1, ln_epsq_l0_ln1, rsqrt_lut, ln1);
    linear(ln1, SEQ, D, DFF, W_l0_ff0, b_l0_ff0, M_l0_ff0[0], ff);
    for (int i = 0; i < SEQ * DFF; ++i) ff[i] = gelu_lut(ff[i], gelu_l0);
    linear(ff, SEQ, DFF, D, W_l0_ff3, b_l0_ff3, M_l0_ff3[0], ff2);
    residual(ln1, ff2, SEQ * D, add_l0_res2[0], add_l0_res2[1], res);
    layernorm(res, SEQ, ln_gm_l0_ln2, ln_bt_l0_ln2, ln_epsq_l0_ln2, rsqrt_lut, x);

    long pooled[D];
    for (int j = 0; j < D; ++j) {
        long sum = 0; for (int i = 0; i < SEQ; ++i) sum += x[(long)i * D + j];
        pooled[j] = requant(sum, M_pool[0]);
    }
    long cls_ln[D], c1[CLS_HID];
    layernorm(pooled, 1, ln_gm_cls_ln, ln_bt_cls_ln, ln_epsq_cls_ln, rsqrt_lut, cls_ln);
    linear(cls_ln, 1, D, CLS_HID, W_cls1, b_cls1, M_cls1[0], c1);
    for (int o = 0; o < CLS_HID; ++o) c1[o] = gelu_lut(c1[o], gelu_cls);
    linear(c1, 1, CLS_HID, NUM_CLASSES, W_cls4, b_cls4, M_cls4[0], logits);
}

// ---- AXI-Stream top -------------------------------------------------------
void transformer_top(hls::stream<axis_t>& in, hls::stream<axis_t>& out) {
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
    static long input[SEQ * CH];
    for (int i = 0; i < SEQ * CH; ++i) {
        axis_t t = in.read();
        input[i] = (long)(int)t.data;
    }
    long logits[NUM_CLASSES];
    transformer_core(input, logits);
    for (int c = 0; c < NUM_CLASSES; ++c) {
        axis_t t;
        t.data = (int)logits[c];
        t.keep = -1; t.strb = -1;
        t.last = (c == NUM_CLASSES - 1) ? 1 : 0;
        out.write(t);
    }
}
