// ---------------------------------------------------------------------------
// VARIANT E (Zynq-7010 LUT reduction) — derived from Variant B (8 lanes + shared
// layernorm). Baseline and v3 NOT touched. Adds maximal safe operator sharing:
// requant and residual are made non-inlined modules and capped to ONE instance
// each via ALLOCATION, so their 64-bit multiply + round-to-even datapaths are
// time-shared instead of duplicated. All bit-exact (values/order/weights/scales/
// LUTs/interface/golden unchanged). linear_fixed stays a non-inlined shared
// module (weights single-port; no 382-BRAM regression). Goal: LUT <= 17,600.
//
// --- original v3 header retained below ---
// BRAM-ORIENTED optimization variant (isolated) for the F_1layer @ int10 core.
// Derived from transformer_top_opt.cpp (the measured "v1"). Baseline
// transformer_top.cpp is NOT touched.
//
// GOAL: keep the MEASURED context-loop speedup of v1 while removing the
// weight-ROM replication that made v1 overflow Zynq-7020 BRAM (382/280 = 136%).
//
// ROOT CAUSE of the v1 overflow (confirmed from the v1 synthesis hierarchy):
// the generic `linear` had `#pragma HLS INLINE`, and its outer loop had
// `#pragma HLS PIPELINE`, which fully UNROLLS the inner `Din` MAC loop. That
// requires `Din` weight reads per cycle, so each weight ROM is REPLICATED
// ~Din/2 times (e.g. W_l0_ff3 = 126 BRAM, W_l0_ff0 = 62, W_cls1/out/wq = 31 each).
// The `v` cyclic partition itself is cheap (24 BRAM).
//
// FIX (this file):
//   1. Replace the runtime-dimension `linear` with a COMPILE-TIME-SPECIALIZED
//      template `linear_fixed<ROWS,DIN,DOUT>`. Constant loop bounds let HLS
//      bound latency WITHOUT force-inlining the MAC body at every call site
//      (so HLS instantiates one shared module per distinct shape).
//   2. In linear_fixed, PIPELINE the INNER reduction loop (not the output loop).
//      This requests a single MAC datapath that reads ONE weight and ONE input
//      per cycle (II=1 accumulation), so each weight ROM stays SINGLE-PORT and
//      is NOT replicated. Latency of a linear is ROWS*DOUT*DIN cycles (rolled),
//      which is slower than v1's fully-unrolled linears, but the linears are not
//      the latency bottleneck (full-sequence attention is), and fitting the
//      device has priority over maximum speed.
//   3. Keep the j-outer / HD=8-unrolled CONTEXT loop and the cyclic `v`
//      partition exactly as v1 (the actual speedup, and bit-exact).
//   4. Inline the attention helpers ONLY (attention0_row reads, attention0_kv
//      writes) so the partitioned `v` is accessed within one scope. The `v`
//      projection is written by a small in-scope loop (NOT through a non-inlined
//      function pointer) to avoid the Vitis-2020.2 "code transformation" hang
//      seen when a cyclic-partitioned array crosses a non-inlined function
//      boundary (the previously recorded "v2 no-inline HANG").
//   linear_fixed is NOT force-inlined.
//
// BIT-EXACTNESS: every accumulation keeps the same integer terms in the same
// order (ascending index) as transformer_top.cpp; only loop nesting / hardware
// sharing changes. acc_t width, requant(), LUTs, scales, class IDs, tensor
// layout and the AXI interface are all unchanged. No numeric contract altered.
//
// xc7z020clg400-1 is a PROVISIONAL experimental target, not a selected board.
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
static long requant(long acc, scale_t Mq) {
    // VARIANT E: requant is a real (non-inlined) module so its 64-bit multiply +
    // round-to-even datapath can be allocated once and time-shared (ALLOCATION in
    // transformer_core) instead of duplicated at every call site. Bit-exact.
#pragma HLS INLINE off
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

// Compile-time-specialized linear layer. ROWS/DIN/DOUT are template constants,
// so HLS sees constant loop bounds and instantiates ONE shared module per shape
// without needing the body force-inlined at every call site. The inner (Din)
// reduction loop is pipelined at II=1: one weight read and one input read per
// cycle -> weight ROMs stay single-port (no replication). The outer row/output
// loops are left rolled.
template<int ROWS, int DIN, int DOUT>
static void linear_fixed(const act_t* in, const long* W, const long* b,
                         scale_t Mq, act_t* out) {
    for (int r = 0; r < ROWS; ++r)
        for (int o = 0; o < DOUT; ++o) {
            const long* wo = &W[(long)o * DIN];
            const act_t* xr = &in[(long)r * DIN];
            acc_t acc = 0;
            for (int i = 0; i < DIN; ++i) {
#pragma HLS PIPELINE
                acc += xr[i] * wo[i];
            }
            out[(long)r * DOUT + o] = requant(acc + b[o], Mq);
        }
}

static void layernorm(const act_t* in, int rows, const scale_t* gm, const scale_t* bt,
                      long epsq, const long* rsqrt, act_t* out) {
    // VARIANT B: keep layernorm a real (non-inlined) module so its (LUT-heavy)
    // datapath -- including the rsqrt reciprocal-sqrt logic -- can be allocated
    // ONCE and time-shared by all three sequential layernorm calls (finish_token
    // ln1, ln2, and the classifier ln). See ALLOCATION in transformer_core.
#pragma HLS INLINE off
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
    // VARIANT E: non-inlined module so the two sequential residual calls share one
    // datapath (ALLOCATION in transformer_core). Bit-exact.
#pragma HLS INLINE off
    for (int i = 0; i < n; ++i)
        out[i] = clamp_act(div_round_even<1LL << SCALE_FRAC_BITS>(
                               (long long)a[i] * r0 + (long long)b_[i] * r1));
}

// ---- layer-0 attention (single layer inlined) -----------------------------
static void attention0_row(const act_t* x, const act_t* k, const act_t* v,
                           int i, act_t* attn_out) {
    // INLINE (genuinely needed): brings the cyclic ARRAY_PARTITION on `v` into
    // scope so the HD=8 context lanes can be read in one cycle.
#pragma HLS INLINE
    act_t q[D], ctx[D];
    linear_fixed<1, D, D>(&x[(long)i * D], W_l0_wq, b_l0_wq, M_l0_wq_q40[0], q);
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
            acc_t numerator = (acc_t)num[j] * ATTN_QMAX + denom / 2;
            qattn[j] = numerator / denom;
        }
        // --- CONTEXT ACCUMULATION (kept identical to v1) --------------------
        // j-outer, HD lanes unrolled. Per d, sum_{j} qattn[j]*v[j*D+off+d] in
        // ascending j (bit-exact). v is cyclic-partitioned (factor 8) so the 8
        // lane reads (off multiple of 8) land in distinct banks 0..7.
        acc_t cacc[HD];
#pragma HLS ARRAY_PARTITION variable=cacc complete dim=1
        for (int d = 0; d < HD; ++d) {
#pragma HLS UNROLL
            cacc[d] = 0;
        }
        for (int j = 0; j < SEQ; ++j) {
#pragma HLS PIPELINE
            act_t qv = qattn[j];
            const act_t* vj = &v[(long)j * D + off];
            for (int d = 0; d < HD; ++d) {
#pragma HLS UNROLL
                cacc[d] += qv * vj[d];
            }
        }
        // VARIANT A: share the context-output requant. The HD accumulators are
        // already complete (cacc[d]); requantizing them does NOT need HD parallel
        // requant units. Rolling this loop (no UNROLL) makes HLS reuse ONE
        // requant datapath across the HD outputs, removing 7 of the 8 parallel
        // requant units measured on 7010 (~7*555 = ~3.9k LUT). Same integer
        // values, same order; only the requant hardware is time-shared. ~+HD
        // cycles per head (negligible vs 44.9M).
        for (int d = 0; d < HD; ++d) {
            ctx[off + d] = requant(cacc[d], M_l0_ctx_q40[0]);
        }
        // -------------------------------------------------------------------
    }
    linear_fixed<1, D, D>(ctx, W_l0_out, b_l0_out, M_l0_out_q40[0], attn_out);
}

static void attention0_kv(const act_t* x, act_t* k, act_t* v) {
    // INLINE (genuinely needed): the `v` write below targets the cyclic-
    // partitioned array; keeping it in scope avoids passing a partitioned array
    // through a non-inlined function pointer (the recorded v2 2020.2 hang).
#pragma HLS INLINE
    // K projection goes through the shared linear_fixed module (k is NOT
    // partitioned, so no pointer-partition hazard).
    linear_fixed<SEQ, D, D>(x, W_l0_wk, b_l0_wk, M_l0_wk_q40[0], k);
    // V projection written in-scope (NOT via linear_fixed) so the write targets
    // the partitioned `v` locally. Inner MAC loop pipelined -> W_l0_wv stays
    // single-port; the v write address advances by 1 per output (cyclic banks).
    for (int r = 0; r < SEQ; ++r)
        for (int o = 0; o < D; ++o) {
            const long* wo = &W_l0_wv[(long)o * D];
            const act_t* xr = &x[(long)r * D];
            acc_t acc = 0;
            for (int i = 0; i < D; ++i) {
#pragma HLS PIPELINE
                acc += xr[i] * wo[i];
            }
            v[(long)r * D + o] = requant(acc + b_l0_wv[o], M_l0_wv_q40[0]);
        }
}

// Post-attention work kept as a separate module per token (as in the baseline).
static void finish_token(const act_t* xrow, const act_t* attn_out, act_t* outrow) {
    // VARIANT B: INLINE finish_token into transformer_core so its two layernorm
    // calls land in the same scope as the classifier layernorm; a single
    // ALLOCATION(layernorm, limit=1) then forces all three to share one datapath.
    // Inlining does NOT replicate finish_token (the token loop is not unrolled)
    // and does NOT touch linear_fixed (still a shared module -> weights stay
    // single-port, no BRAM regression).
#pragma HLS INLINE
    act_t res[D], ln1[D], ff[DFF], ff2[D];
    residual(xrow, attn_out, D, add_l0_res1_q40[0], add_l0_res1_q40[1], res);
    layernorm(res, 1, ln_gm_l0_ln1_q40, ln_bt_l0_ln1_q40, ln_epsq_l0_ln1, rsqrt_lut, ln1);
    linear_fixed<1, D, DFF>(ln1, W_l0_ff0, b_l0_ff0, M_l0_ff0_q40[0], ff);
    for (int j = 0; j < DFF; ++j) ff[j] = gelu_lut(ff[j], gelu_l0);
    linear_fixed<1, DFF, D>(ff, W_l0_ff3, b_l0_ff3, M_l0_ff3_q40[0], ff2);
    residual(ln1, ff2, D, add_l0_res2_q40[0], add_l0_res2_q40[1], res);
    layernorm(res, 1, ln_gm_l0_ln2_q40, ln_bt_l0_ln2_q40, ln_epsq_l0_ln2, rsqrt_lut,
              outrow);
}

// ---- full forward: input codes (SEQ*CH) -> NUM_CLASSES logits --------------
static void transformer_core(const act_t* input, act_t logits[NUM_CLASSES]) {
    static act_t x[SEQ * D], k[SEQ * D], v[SEQ * D];
    // Bank v across D so the context loop reads HD=8 lanes of one head per cycle.
#pragma HLS ARRAY_PARTITION variable=v cyclic factor=8 dim=1
    // VARIANT B: one shared layernorm datapath for all three sequential calls
    // (finish_token ln1/ln2 + classifier ln). finish_token is inlined so its two
    // calls are visible in this scope alongside the classifier call.
#pragma HLS ALLOCATION function instances=layernorm limit=1
    // VARIANT E: also time-share the residual and requant datapaths.
#pragma HLS ALLOCATION function instances=residual limit=1
#pragma HLS ALLOCATION function instances=requant limit=1
    linear_fixed<SEQ, CH, D>(input, W_proj, b_proj, M_proj_q40[0], x);
    for (int i = 0; i < SEQ * D; ++i) {
#pragma HLS PIPELINE
        x[i] = clamp_act((long)x[i] + pe_codes[i]);
    }

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
    linear_fixed<1, D, CLS_HID>(cls_ln, W_cls1, b_cls1, M_cls1_q40[0], c1);
    for (int o = 0; o < CLS_HID; ++o) c1[o] = gelu_lut(c1[o], gelu_cls);
    linear_fixed<1, CLS_HID, NUM_CLASSES>(c1, W_cls4, b_cls4, M_cls4_q40[0], logits);
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
