// ---------------------------------------------------------------------------
// Full-integer Transformer fault classifier — HLS / portable C++ core.
//
// This is a bit-exact C++ transcription of the verified Python reference
// (src/quantize/integer_model.py). Every tensor is an integer; only the requant
// and affine *constants* are doubles (exactly as in the Python reference — on a
// real board these become fixed-point multipliers, the deferred optimisation).
//
// Written in portable C++: it compiles and runs under plain g++ for functional
// verification against the golden vectors, and is the starting point for a Vitis
// HLS build (swap the typedefs for ap_int / ap_fixed and add pragmas per board).
// ---------------------------------------------------------------------------
#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <cstdint>
#include <vector>
#include <string>
#include <map>

// ----- model geometry (matches the trained checkpoint) --------------------
constexpr int SEQ         = 1024;   // sequence length (samples per window)
constexpr int CH          = 8;      // input channels
constexpr int D           = 32;     // d_model
constexpr int NHEAD       = 4;      // attention heads
constexpr int HD          = D / NHEAD;   // head dim = 8
constexpr int DFF         = 64;     // feed-forward hidden
constexpr int LAYERS      = 2;      // encoder layers
constexpr int CLS_HID     = 16;     // classifier hidden
constexpr int NUM_CLASSES = 4;

// ----- fixed-point constants (must match integer_ops.py) ------------------
constexpr int  ACT_BITS   = 16;
constexpr int  ATTN_BITS  = 16;
constexpr long ACT_QMAX   = (1L << (ACT_BITS - 1)) - 1;    // 32767
constexpr long ATTN_QMAX  = (1L << (ATTN_BITS - 1)) - 1;   // 32767
constexpr int  RSQRT_MANT_BITS = 8;
constexpr int  RSQRT_FRAC = 15;

// ----- data types ---------------------------------------------------------
// Portable defaults; for Vitis HLS replace with ap_int<16>/ap_int<64> etc.
typedef int64_t data_t;   // activation codes (fit in int16 range, stored wide)
typedef int64_t acc_t;    // MAC accumulator

// ----- all model constants, loaded from hls/data/ -------------------------
struct Params {
    std::map<std::string, std::vector<int64_t>> W;    // weight matrices  (Dout*Din)
    std::map<std::string, std::vector<int64_t>> b;    // folded biases    (Dout)
    std::map<std::string, double>               M;    // requant multipliers
    std::map<std::string, std::vector<double>>  add;  // residual scale ratios (2)
    std::map<std::string, std::vector<double>>  ln_gm;
    std::map<std::string, std::vector<double>>  ln_bt;
    std::map<std::string, int64_t>              ln_epsq;
    std::vector<int64_t>                        pe;   // SEQ*D positional codes
    std::vector<int64_t>                        gelu[LAYERS];
    std::vector<int64_t>                        gelu_cls;
    std::vector<int64_t>                        exp_lut[LAYERS];
    int                                         exp_clamp[LAYERS];
    std::vector<int64_t>                        rsqrt;
};

// Load every constant from a data directory (produced by export_hls.py).
Params load_params(const std::string& data_dir);

// Run one window: input is SEQ*CH int codes (row-major [seq, channel]);
// writes NUM_CLASSES integer logits. Prediction = argmax(logits).
void transformer_infer(const int64_t* input, const Params& p,
                       int64_t logits[NUM_CLASSES]);

#endif // TRANSFORMER_H
