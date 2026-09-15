// ---------------------------------------------------------------------------
// Full-integer Transformer fault classifier — HLS / portable C++ core.
// F_1layer @ int10 variant (1 encoder layer, 10-bit activations/weights/attn).
//
// Bit-exact C++ transcription of the verified Python reference
// (src/quantize/integer_model.py) for the FPGA-winner architecture. Every tensor
// is an integer; only the requant/affine *constants* are doubles (exactly as in
// the Python reference — on a real board these become fixed-point multipliers).
//
// Portable C++: compiles under plain g++ for functional verification against the
// int10 golden vectors, and is the starting point for a Vitis HLS build. For
// synthesis, swap data_t/acc_t for ap_int<10>/ap_int<32> (or ap_fixed for the
// requant constants) and add interface + pipeline pragmas per target board.
//
// Geometry/bit-width differ from the day14 A_baseline core ONLY in the three
// constants below (LAYERS, ACT_BITS, ATTN_BITS); the arithmetic is identical.
// ---------------------------------------------------------------------------
#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <cstdint>
#include <vector>
#include <string>
#include <map>

// ----- model geometry (matches the F_1layer checkpoint) -------------------
constexpr int SEQ         = 1024;   // sequence length (samples per window)
constexpr int CH          = 8;      // input channels
constexpr int D           = 32;     // d_model
constexpr int NHEAD       = 4;      // attention heads
constexpr int HD          = D / NHEAD;   // head dim = 8
constexpr int DFF         = 64;     // feed-forward hidden
constexpr int LAYERS      = 1;      // encoder layers (F_1layer)
constexpr int CLS_HID     = 16;     // classifier hidden
constexpr int NUM_CLASSES = 4;

// ----- fixed-point constants (must match integer_ops.py at int10) ---------
constexpr int  ACT_BITS   = 10;
constexpr int  ATTN_BITS  = 10;
constexpr long ACT_QMAX   = (1L << (ACT_BITS - 1)) - 1;    // 511
constexpr long ATTN_QMAX  = (1L << (ATTN_BITS - 1)) - 1;   // 511
constexpr int  RSQRT_MANT_BITS = 8;
constexpr int  RSQRT_FRAC = 15;

// ----- data types ---------------------------------------------------------
// Portable defaults; for Vitis HLS replace with ap_int<10>/ap_int<32> etc.
typedef int64_t data_t;   // activation codes (fit in int10 range, stored wide)
typedef int64_t acc_t;    // MAC accumulator

// ----- all model constants, loaded from hls/f1_int10/data/ ----------------
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

// Load every constant from a data directory (produced by export_hls_f_int10.py).
Params load_params(const std::string& data_dir);

// Run one window: input is SEQ*CH int codes (row-major [seq, channel]);
// writes NUM_CLASSES integer logits. Prediction = argmax(logits).
void transformer_infer(const int64_t* input, const Params& p,
                       int64_t logits[NUM_CLASSES]);

#endif // TRANSFORMER_H
