// Fixed-point type aliases for the F_1layer @ int10 core.
//
// Under synthesis, or when HLS_USE_AP_INT is set for C-simulation, use the
// actual hardware widths. Plain desktop g++ can use wide native integers.
//
// Widths are the STARTING point for synthesis. The int10 reference clamps
// activations to +/-511 (10-bit signed) and attention weights to +/-511; the MAC
// accumulators must be wide enough for the worst-case dot product:
//   proj:  CH=8   terms of 10b*10b  -> <= 8*(2^18)  ~ 21 bits
//   attention QK: HD=8 terms; FFN: DFF=64 terms
//   attention context: up to SEQ=1024 terms of 511*511 < 2^28
// The current exported biases have magnitude <= 22128, so 32-bit signed
// accumulators cover these bounds. Re-check after changing model/export data.
#ifndef AP_TYPES_H
#define AP_TYPES_H

#if defined(__SYNTHESIS__) || defined(HLS_USE_AP_INT)
  #include "ap_int.h"
  typedef ap_int<10> act_t;    // activation / weight codes (int10, +/-511)
  typedef ap_int<32> acc_t;    // MAC accumulator
  typedef ap_int<20> lut_t;    // LUT entries (exp/gelu/rsqrt fit < 2^18)
#else
  #include <cstdint>
  typedef int64_t act_t;
  typedef int64_t acc_t;
  typedef int64_t lut_t;
#endif

// Hardware-facing affine coefficients are exported as signed Q40 integers.
// The original float64 arrays remain in weights_int10.h for reference.
typedef long long scale_t;

#endif // AP_TYPES_H
