// ---------------------------------------------------------------------------
// PHASE 2+4 CANDIDATE rt_df_2head: rt_dataflow2 + head-loop UNROLL factor=2 +
// k/v cyclic partition factor 16 (two heads read disjoint banks -> 2 parallel
// per-head attention pipelines). Bit-exact.
// single reader/writer per array; x_in read only by df_load, k/v only by df_attn,
// x_out written only by df_store). Fixes the rt_dataflow x_in dual-reader error.
// NOTE: this deliberately puts partitioned k/v across a non-inlined boundary to
// measure whether 2020.2 dataflow tolerates it. (rt5 header:)
// PHASE 3 CANDIDATE rt5_splitbuf — T2 + separate immutable x_in / write-only
// x_out buffers (removes the false in-place WAR dependency on x, enabling later
// query-level DATAFLOW). Bit-exact; adds one SEQ*D activation buffer. (T2 header:)
// THROUGHPUT VARIANT T2 (T1 K-banking + shared context-output requant) — the
// context-output requant loop is rolled so one requant datapath is time-shared
// (frees ~7x555 LUT for later parallelism). All else identical to T1. Bit-exact.
// (T1 header retained below)
// THROUGHPUT VARIANT T1 (K-banking) — derived from the measured v3
// (transformer_top_opt_bram.cpp). Baseline transformer_top.cpp and v3 are NOT
// touched. ONLY change vs v3: `k` is cyclic-partitioned (factor 8) and written by
// an in-scope loop (like `v`), so the attention SCORE dot-product can read HD=8
// k-lanes per cycle and reach II=1 (v3 measured score II=4, K-port limited).
// Bit-exact: same integer values/order/weights/scales/LUTs/interface/golden.
// (original v3 header retained below)
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

// ---- DATAFLOW2 stages: load -> attn -> finish -> store --------------------
// df_attn reads ONLY its query row (xrow) + shared read-only k/v, and FORWARDS
// xrow to the finish stage so x_in has a single reader (df_load). This satisfies
// the Vitis dataflow single-reader/single-writer rule that rt_dataflow violated.
// Bit-exact (same arithmetic as attention0_row, query row now passed by value).
static void df_attn(const act_t xrow[D], const act_t* k, const act_t* v,
                    act_t attn_out[D], act_t xrow_fwd[D]) {
#pragma HLS INLINE off
    act_t q[D], ctx[D];
    for (int d = 0; d < D; ++d) {
#pragma HLS PIPELINE
        xrow_fwd[d] = xrow[d];   // forward query row to the finish stage
    }
    linear_fixed<1, D, D>(xrow, W_l0_wq, b_l0_wq, M_l0_wq_q40[0], q);
    for (int h = 0; h < NHEAD; ++h) {
        // PHASE 2: process two heads concurrently. With k/v cyclic-partitioned
        // factor 16, heads h and h+1 (offsets multiples of 8) read disjoint banks,
        // so HLS can run two independent per-head pipelines in parallel, ~halving
        // the head-loop latency. Each head keeps its own weights/scales/LUTs and
        // its own reduction order -> bit-exact.
#pragma HLS UNROLL factor=2
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
        // T2: roll the context-output requant loop so ONE shared requant
        // datapath is reused across the HD outputs instead of 8 parallel units
        // (v3/T1 measured 8x555 = ~4,440 LUT here). Same integer values/order;
        // ~+HD cycles per head (negligible vs 32.8M).
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
    // T1: K projection written in-scope (NOT via linear_fixed) so the write
    // targets the now cyclic-partitioned `k` locally, exactly as the v write does.
    // This lets the score dot-product read HD=8 k-lanes per cycle (score II -> 1)
    // while avoiding the v2 2020.2 partitioned-pointer transformation stall.
    // Same integer values/order as the linear_fixed K projection (bit-exact).
    for (int r = 0; r < SEQ; ++r)
        for (int o = 0; o < D; ++o) {
            const long* wko = &W_l0_wk[(long)o * D];
            const act_t* xr = &x[(long)r * D];
            acc_t acck = 0;
            for (int i = 0; i < D; ++i) {
#pragma HLS PIPELINE
                acck += xr[i] * wko[i];
            }
            k[(long)r * D + o] = requant(acck + b_l0_wk[o], M_l0_wk_q40[0]);
        }
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
#pragma HLS INLINE off
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

// DATAFLOW2 load/store stages: isolate the single read of x_in and single write
// of x_out into their own processes so the dataflow single-reader/writer rule
// holds. Bit-exact copies.
static void df_load(const act_t* x_in, int i, act_t xrow[D]) {
#pragma HLS INLINE off
    for (int d = 0; d < D; ++d) {
#pragma HLS PIPELINE
        xrow[d] = x_in[(long)i * D + d];
    }
}
static void df_store(const act_t outrow[D], act_t* x_out, int i) {
#pragma HLS INLINE off
    for (int d = 0; d < D; ++d) {
#pragma HLS PIPELINE
        x_out[(long)i * D + d] = outrow[d];
    }
}

// ---- full forward: input codes (SEQ*CH) -> NUM_CLASSES logits --------------
static void transformer_core(const act_t* input, act_t logits[NUM_CLASSES]) {
    // PHASE 3 (split buffers): x_in is the IMMUTABLE projected+PE input read by
    // K/V generation and by every query; x_out holds finish_token results read by
    // pooling. This removes the in-place WAR on a single `x` array (a false HLS
    // cross-query dependency) while preserving exact semantics: query i reads only
    // original x_in[i] and writes only x_out[i]; no query reads another query's
    // updated result (verified in Phase 0). Cost: one extra SEQ*D activation
    // buffer (~+20 BRAM_18K).
    static act_t x_in[SEQ * D], x_out[SEQ * D], k[SEQ * D], v[SEQ * D];
    // Bank v across D so the context loop reads HD=8 lanes of one head per cycle.
#pragma HLS ARRAY_PARTITION variable=v cyclic factor=16 dim=1
    // T1: bank k the same way so the score dot-product reads HD=8 k-lanes per
    // cycle (target score-loop II=1 instead of the measured II=4).
#pragma HLS ARRAY_PARTITION variable=k cyclic factor=16 dim=1
    linear_fixed<SEQ, CH, D>(input, W_proj, b_proj, M_proj_q40[0], x_in);
    for (int i = 0; i < SEQ * D; ++i) {
#pragma HLS PIPELINE
        x_in[i] = clamp_act((long)x_in[i] + pe_codes[i]);
    }

    attention0_kv(x_in, k, v);
    // PHASE 4: query-level DATAFLOW. With separate x_in (read-only) / x_out
    // (write-only, disjoint rows) and read-only k/v, consecutive queries are
    // independent, so DATAFLOW inside the loop overlaps finish_token(i) with
    // attention0_row(i+1). Target loop II ~= max(attention, finish) instead of
    // their sum. attn_out is the producer->consumer channel (ping-pong).
    for (int i = 0; i < SEQ; ++i) {
#pragma HLS DATAFLOW
        act_t xrow[D], attn_out[D], xrow_fwd[D], outrow[D];
        df_load(x_in, i, xrow);                       // sole reader of x_in
        df_attn(xrow, k, v, attn_out, xrow_fwd);      // sole reader of k/v; forwards xrow
        finish_token(xrow_fwd, attn_out, outrow);     // reads forwarded row + attn
        df_store(outrow, x_out, i);                   // sole writer of x_out
    }
    act_t pooled[D];
    for (int j = 0; j < D; ++j) {
        acc_t sum = 0;
        for (int i = 0; i < SEQ; ++i) {
#pragma HLS PIPELINE
            sum += x_out[(long)i * D + j];
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
