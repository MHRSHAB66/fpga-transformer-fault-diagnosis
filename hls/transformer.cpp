// ---------------------------------------------------------------------------
// Full-integer Transformer core — bit-exact transcription of
// src/quantize/integer_model.py (IntegerTransformer.infer_int).
//
// Rounding note: the Python reference uses numpy round-half-to-even. std::rint
// with the default rounding mode is also round-half-to-even, so the two match.
// ---------------------------------------------------------------------------
#include "transformer.h"

#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

using std::vector;
using std::string;

// SQRT(1/2) in Q0.RSQRT_FRAC — matches integer_ops._SQRT1_2
static const int64_t SQRT1_2 = (int64_t)std::llround((1.0 / std::sqrt(2.0)) * (1 << RSQRT_FRAC));

// ----- small helpers -------------------------------------------------------
static inline int64_t clamp_act(double y) {
    if (y >  (double)ACT_QMAX) return  ACT_QMAX;
    if (y < -(double)ACT_QMAX) return -ACT_QMAX;
    return (int64_t)y;
}
static inline int64_t requant(acc_t acc, double M) {
    return clamp_act(std::rint((double)acc * M));
}
static inline int64_t gelu_lut(int64_t q, const vector<int64_t>& lut) {
    int64_t idx = std::min<int64_t>(std::max<int64_t>(q, -ACT_QMAX), ACT_QMAX);
    return lut[idx + ACT_QMAX];
}

// fixed-point reciprocal sqrt — mirrors integer_ops.rsqrt_fxp (returns y, frac=RSQRT_FRAC)
static int64_t rsqrt_fxp(int64_t v, const vector<int64_t>& lut) {
    if (v < 1) v = 1;
    int e = 0; { int64_t t = v; while (t > 1) { t >>= 1; ++e; } }   // floor(log2 v)
    int shift = e - RSQRT_MANT_BITS;
    int64_t mant_idx = (shift >= 0) ? (v >> shift) - (1LL << RSQRT_MANT_BITS)
                                    : (v << (-shift)) - (1LL << RSQRT_MANT_BITS);
    mant_idx = std::min<int64_t>(std::max<int64_t>(mant_idx, 0), (1LL << RSQRT_MANT_BITS) - 1);
    int64_t y = lut[mant_idx];
    int half_e = e >> 1;
    if (e & 1) y = (y * SQRT1_2) >> RSQRT_FRAC;
    y = y >> half_e;                       // e >= 0 so half_e >= 0
    return y;
}

// ----- integer linear:  out[rows][Dout] = requant(in @ W^T + b, M) ---------
static void linear(const int64_t* in, int rows, int Din, int Dout,
                   const vector<int64_t>& W, const vector<int64_t>& b,
                   double M, int64_t* out) {
    for (int r = 0; r < rows; ++r) {
        const int64_t* xr = in + (size_t)r * Din;
        for (int o = 0; o < Dout; ++o) {
            const int64_t* wo = &W[(size_t)o * Din];
            acc_t acc = 0;
            for (int i = 0; i < Din; ++i) acc += xr[i] * wo[i];
            out[(size_t)r * Dout + o] = requant(acc + b[o], M);
        }
    }
}

// ----- integer LayerNorm over the last dim (size D) ------------------------
static void layernorm(const int64_t* in, int rows, const string& name,
                      const Params& p, int64_t* out) {
    const vector<double>& gm = p.ln_gm.at(name);
    const vector<double>& bt = p.ln_bt.at(name);
    int64_t epsq = p.ln_epsq.at(name);
    for (int r = 0; r < rows; ++r) {
        const int64_t* xr = in + (size_t)r * D;
        acc_t sum = 0;
        for (int j = 0; j < D; ++j) sum += xr[j];
        int64_t mean = (int64_t)std::rint((double)sum / D);
        acc_t sumsq = 0;
        int64_t centered[D];
        for (int j = 0; j < D; ++j) { centered[j] = xr[j] - mean; sumsq += centered[j] * centered[j]; }
        int64_t var = (int64_t)std::rint((double)sumsq / D);
        int64_t inv = rsqrt_fxp(var + epsq, p.rsqrt);
        int64_t* orow = out + (size_t)r * D;
        for (int j = 0; j < D; ++j) {
            int64_t norm = centered[j] * inv;
            orow[j] = clamp_act(std::rint((double)norm * gm[j] + bt[j]));
        }
    }
}

// ----- residual add: out = clamp(round(a*r0 + b*r1)) -----------------------
static void residual(const int64_t* a, const int64_t* b_, int n,
                     double r0, double r1, int64_t* out) {
    for (int i = 0; i < n; ++i)
        out[i] = clamp_act(std::rint((double)a[i] * r0 + (double)b_[i] * r1));
}

// ----- one attention sub-layer ---------------------------------------------
static void attention(const int64_t* x, int l, const Params& p, int64_t* attn_out) {
    const string L = "l" + std::to_string(l);
    vector<int64_t> q(SEQ * D), k(SEQ * D), v(SEQ * D);
    linear(x, SEQ, D, D, p.W.at(L + ".wq"), p.b.at(L + ".wq"), p.M.at(L + ".wq"), q.data());
    linear(x, SEQ, D, D, p.W.at(L + ".wk"), p.b.at(L + ".wk"), p.M.at(L + ".wk"), k.data());
    linear(x, SEQ, D, D, p.W.at(L + ".wv"), p.b.at(L + ".wv"), p.M.at(L + ".wv"), v.data());

    const double M_scores = p.M.at(L + ".scores");
    const double M_ctx    = p.M.at(L + ".ctx");
    const vector<int64_t>& elut = p.exp_lut[l];
    const int clamp = p.exp_clamp[l];

    vector<int64_t> ctx(SEQ * D, 0);
    vector<int64_t> sc(SEQ), num(SEQ), qattn(SEQ);

    for (int h = 0; h < NHEAD; ++h) {
        const int off = h * HD;
        for (int i = 0; i < SEQ; ++i) {
            const int64_t* qi = &q[(size_t)i * D + off];
            // scores row
            for (int j = 0; j < SEQ; ++j) {
                const int64_t* kj = &k[(size_t)j * D + off];
                acc_t acc = 0;
                for (int d = 0; d < HD; ++d) acc += qi[d] * kj[d];
                sc[j] = requant(acc, M_scores);
            }
            // softmax (integer): row-max subtract, exp LUT, integer normalise
            int64_t rowmax = sc[0];
            for (int j = 1; j < SEQ; ++j) rowmax = std::max(rowmax, sc[j]);
            acc_t denom = 0;
            for (int j = 0; j < SEQ; ++j) {
                int64_t dd = sc[j] - rowmax;
                if (dd < -clamp) dd = -clamp; if (dd > 0) dd = 0;
                num[j] = elut[dd + clamp];
                denom += num[j];
            }
            if (denom < 1) denom = 1;
            for (int j = 0; j < SEQ; ++j)
                qattn[j] = (num[j] * ATTN_QMAX + denom / 2) / denom;
            // context: qattn @ vh
            for (int d = 0; d < HD; ++d) {
                acc_t acc = 0;
                for (int j = 0; j < SEQ; ++j) acc += qattn[j] * v[(size_t)j * D + off + d];
                ctx[(size_t)i * D + off + d] = requant(acc, M_ctx);
            }
        }
    }
    linear(ctx.data(), SEQ, D, D, p.W.at(L + ".out"), p.b.at(L + ".out"),
           p.M.at(L + ".out"), attn_out);
}

// ----- one encoder layer (post-norm) ---------------------------------------
static void encoder(int64_t* x, int l, const Params& p) {
    const string L = "l" + std::to_string(l);
    vector<int64_t> attn_out(SEQ * D), res(SEQ * D), ln1(SEQ * D), ff(SEQ * DFF), ff2(SEQ * D);

    attention(x, l, p, attn_out.data());
    const vector<double>& a1 = p.add.at(L + ".res1");
    residual(x, attn_out.data(), SEQ * D, a1[0], a1[1], res.data());
    layernorm(res.data(), SEQ, L + ".ln1", p, ln1.data());

    linear(ln1.data(), SEQ, D, DFF, p.W.at(L + ".ff0"), p.b.at(L + ".ff0"), p.M.at(L + ".ff0"), ff.data());
    for (int i = 0; i < SEQ * DFF; ++i) ff[i] = gelu_lut(ff[i], p.gelu[l]);
    linear(ff.data(), SEQ, DFF, D, p.W.at(L + ".ff3"), p.b.at(L + ".ff3"), p.M.at(L + ".ff3"), ff2.data());

    const vector<double>& a2 = p.add.at(L + ".res2");
    residual(ln1.data(), ff2.data(), SEQ * D, a2[0], a2[1], res.data());
    layernorm(res.data(), SEQ, L + ".ln2", p, x);          // write back into x
}

// ----- full forward --------------------------------------------------------
void transformer_infer(const int64_t* input, const Params& p, int64_t logits[NUM_CLASSES]) {
    vector<int64_t> x(SEQ * D);
    linear(input, SEQ, CH, D, p.W.at("proj"), p.b.at("proj"), p.M.at("proj"), x.data());
    for (int i = 0; i < SEQ * D; ++i) x[i] = clamp_act((double)(x[i] + p.pe[i]));  // +PE, same scale

    for (int l = 0; l < LAYERS; ++l) encoder(x.data(), l, p);

    // mean pool over sequence
    int64_t pooled[D];
    for (int j = 0; j < D; ++j) {
        acc_t sum = 0;
        for (int i = 0; i < SEQ; ++i) sum += x[(size_t)i * D + j];
        pooled[j] = requant(sum, p.M.at("pool"));
    }

    // classifier: LN -> Linear -> GELU -> Linear
    int64_t cls_ln[D], c1[CLS_HID];
    layernorm(pooled, 1, "cls_ln", p, cls_ln);
    linear(cls_ln, 1, D, CLS_HID, p.W.at("cls1"), p.b.at("cls1"), p.M.at("cls1"), c1);
    for (int o = 0; o < CLS_HID; ++o) c1[o] = gelu_lut(c1[o], p.gelu_cls);
    linear(c1, 1, CLS_HID, NUM_CLASSES, p.W.at("cls4"), p.b.at("cls4"), p.M.at("cls4"), logits);
}
